import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value

import copy


class ACMPOAgent(flax.struct.PyTreeNode):
    """
    Maximum a Posteriori Policy Optimisation (MPO) agent with action chunking. 
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def get_q_loss_clip(self, batch, dtype):
        """Return the q loss clip value supplied by the training loop."""
        return jnp.asarray(batch.get('q_loss_clip', self.config.get("q_loss_clip", -1.0)), dtype=dtype)

    def calql_loss(self, batch, batch_actions, q, grad_params, rng):
        batch_size = batch_actions.shape[0]
        action_dim = batch_actions.shape[-1]
        num_actions = self.config["calql_num_actions"]
        temp = self.config["calql_temp"]

        rng, random_rng, current_rng, next_rng = jax.random.split(rng, 4)
        observations = batch["observations"]
        next_observations = batch["next_observations"][..., -1, :]

        random_actions = jax.random.uniform(
            random_rng, (batch_size, num_actions, action_dim), minval=-1.0, maxval=1.0
        )
        observations_repeat = jnp.repeat(observations[:, None, :], num_actions, axis=1)
        next_observations_repeat = jnp.repeat(next_observations[:, None, :], num_actions, axis=1)

        current_noises = jax.random.normal(current_rng, (batch_size, num_actions, action_dim))
        next_noises = jax.random.normal(next_rng, (batch_size, num_actions, action_dim))
        current_actions = jax.lax.stop_gradient(self.compute_flow_actions(observations_repeat, current_noises))
        next_actions = jax.lax.stop_gradient(self.compute_flow_actions(next_observations_repeat, next_noises))

        q_random = self.network.select("critic")(observations_repeat, actions=random_actions, params=grad_params)
        q_current = self.network.select("critic")(observations_repeat, actions=current_actions, params=grad_params)
        q_next = self.network.select("critic")(next_observations_repeat, actions=next_actions, params=grad_params)

        mc_returns = jnp.asarray(batch.get("mc_returns", jnp.zeros((batch_size,))), dtype=q.dtype)
        lower_bounds = mc_returns[None, :, None]
        bound_rate_current = jnp.mean(q_current < lower_bounds)
        bound_rate_next = jnp.mean(q_next < lower_bounds)

        if self.config["enable_calql"]:
            q_current = jnp.maximum(q_current, lower_bounds)
            q_next = jnp.maximum(q_next, lower_bounds)

        cat_q = jnp.concatenate([q_random, q_current, q_next], axis=-1)
        cql_ood = jax.nn.logsumexp(cat_q / temp, axis=-1) * temp
        cql_diff = cql_ood - q
        cql_diff = jnp.clip(
            cql_diff,
            self.config["calql_clip_diff_min"],
            self.config["calql_clip_diff_max"],
        )
        calql_loss = self.config["calql_alpha"] * cql_diff.mean()

        return calql_loss, {
            "calql_loss": calql_loss,
            "calql_diff": cql_diff.mean(),
            "calql_ood_q_mean": cql_ood.mean(),
            "calql_random_q_mean": q_random.mean(),
            "calql_current_q_mean": q_current.mean(),
            "calql_next_q_mean": q_next.mean(),
            "calql_mc_return_mean": mc_returns.mean(),
            "calql_bound_rate_current": bound_rate_current,
            "calql_bound_rate_next": bound_rate_next,
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            # take the first action
            batch_actions = batch["actions"][..., 0, :]

        # TD loss
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

        next_qs = self.network.select(f'target_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'][..., -1] + (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q
        q = self.network.select('critic')(batch['observations'], actions=batch_actions, params=grad_params)
        td_loss = jnp.square(q - target_q) * batch['valid'][..., -1]
        critic_loss = td_loss.mean()

        calql_info = {}
        if self.config["enable_calql"]:
            rng, calql_rng = jax.random.split(rng)
            calql_loss, calql_info = self.calql_loss(batch, batch_actions, q, grad_params, calql_rng)
            critic_loss = critic_loss + calql_loss

        info = {
            'critic_loss': critic_loss,
            'td_critic_loss': td_loss.mean(),
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }
        info.update(calql_info)
        return critic_loss, info

    def actor_bc_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        if self.config["action_chunking"]:
            # fold in horizon_length together with action_dim
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]  # take the first one
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        # only bc on the valid chunk indices
        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2,
                    (batch_size, self.config["horizon_length"], self.config["action_dim"])
                ) * batch["valid"][..., None]  # * weights[..., None]
            )
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))
        
        actor_loss = bc_flow_loss

        return actor_loss, {
            'actor_loss': actor_loss,
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        if self.config["action_chunking"]:
            # fold in horizon_length together with action_dim
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]  # take the first one
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        # only bc on the valid chunk indices
        if self.config["action_chunking"]:
            ori_bc_flow_loss = jnp.reshape(
                    (pred - vel) ** 2,
                    (batch_size, self.config["horizon_length"], self.config["action_dim"])
                ) * batch["valid"][..., None]  # * weights[..., None] 
            
        else:
            ori_bc_flow_loss = jnp.square(pred - vel)

        ori_bc_flow_loss = ori_bc_flow_loss.mean(axis=(1,2)) # (batch_size,)

        bc_flow_loss = jnp.mean(ori_bc_flow_loss)
        ori_bc_flow_loss_mean = jnp.mean(ori_bc_flow_loss)
        ori_bc_flow_loss_max = jnp.max(ori_bc_flow_loss)
        ori_bc_flow_loss_min = jnp.min(ori_bc_flow_loss)
        
        # Q loss.
        rng, noise_rng, x_rng1, t_rng1 = jax.random.split(rng, 4)
        noises = jax.random.normal(noise_rng, (
            *batch['observations'].shape[: -len(self.config['ob_dims'])], self.config["num_action_samples"], action_dim),)
        observations = jnp.repeat(batch['observations'][..., None, :], self.config["num_action_samples"], axis=-2)
        target_flow_actions = self.compute_flow_actions(observations, noises=noises)
        target_flow_actions = jnp.clip(target_flow_actions, -1, 1)  
        
        if self.config['q_agg'] == 'min': 
            target_q_values = self.network.select(f'target_critic')(observations, actions=target_flow_actions).min(axis=0)
        else:
            target_q_values = self.network.select(f'target_critic')(observations, actions=target_flow_actions).mean(axis=0) 
                
        # indices = jnp.argmax(target_q_values, axis=-1)
        # bshape = indices.shape
        # indices = indices.reshape(-1)
        # bsize = len(indices)
        # target_flow_actions_max = jnp.reshape(target_flow_actions, (-1, self.config["num_action_samples"], action_dim))[jnp.arange(bsize), indices, :].reshape(
        #     bshape + (action_dim,))
        
        # x_0_ = jax.random.normal(x_rng1, (batch_size, action_dim))
        # x_1_ = target_flow_actions_max
        # t_ = jax.random.uniform(t_rng1, (batch_size, 1))
        # x_t_ = (1 - t_) * x_0_ + t_ * x_1_
        # vel_ = x_1_ - x_0_
        
        # pred_ = self.network.select('actor_bc_flow')(batch['observations'], x_t_, t_, params=grad_params)
        # # only bc on the valid chunk indices
        # if self.config["action_chunking"]:
        #     q_loss = jnp.mean(
        #         jnp.reshape(
        #             (pred_ - vel_) ** 2,
        #             (batch_size, self.config["horizon_length"], self.config["action_dim"])
        #         ) * batch["valid"][..., None]  # * weights[..., None]
        #     )
        # else:
        #     q_loss = jnp.mean(jnp.square(pred_ - vel_))
        
        if self.config.get("is_q_weighted", True):
            max_q = jnp.max(target_q_values, axis=1, keepdims=True)
            weights = jax.nn.softmax((target_q_values - max_q) / self.config["temperature"], axis=-1)
        else:
            weights = jnp.ones_like(target_q_values) / target_q_values.shape[-1]
        # weights = weights[..., None]
        
        x_0_ = jax.random.normal(x_rng1, (batch_size, self.config["num_action_samples"], action_dim))
        x_1_ = target_flow_actions
        t_ = jax.random.uniform(t_rng1, (batch_size, self.config["num_action_samples"], 1))
        x_t_ = (1 - t_) * x_0_ + t_ * x_1_
        vel_ = x_1_ - x_0_
        
        pred_ = self.network.select('actor_bc_flow')(observations, x_t_, t_, params=grad_params)
        
        # Q loss computation following torch implementation
        if self.config["action_chunking"]:
            # Step 1: Compute MSE loss (pred_ - vel_)^2
            # Shape: (batch, num_samples, horizon*dim)
            mse_loss = (pred_ - vel_) ** 2      #(256, 32, 35) 
            
            # Step 2: Reshape to (batch, num_samples, horizon, dim)
            mse_loss = jnp.reshape(
                mse_loss,
                (batch_size, self.config["num_action_samples"], self.config["horizon_length"], self.config["action_dim"])
            )

            # (256, 32, 5, 7)
            
            # Step 3: Multiply by valids
            # valids shape: (batch, horizon, 1) -> need to broadcast to (batch, 1, horizon, 1)
            valids = batch["valid"][:, None, :, None]  # (batch, 1, horizon, 1) (256, 1, 5, 1)
            mse_loss = mse_loss * valids  #  (256, 32, 5, 7)
            
            # Step 5: Reshape to (batch, num_samples, -1) where -1 = horizon*dim
            mse_loss = jnp.reshape(
                mse_loss,
                (batch_size, self.config["num_action_samples"], -1)
            )
            # (256, 32, 35)

            mse_loss = mse_loss.mean(axis=2) # 对 horzion 和 action_dim 做平均  # mse_loss: (256, 32)

            # Step 6: Multiply by weights
            # weights shape: (batch, num_samples, 1)
            
            # Calculate statistics before weighting (256, 32, 35)
            # Mean over (num_samples, horizon*dim) for each batch element, then take max/min
            ori_q_loss_mean = jnp.mean(mse_loss)  # Global mean
            # ori_loss_per_batch = jnp.mean(mse_loss, axis=(2))  # (batch, num_samples)
            ori_q_loss_max = jnp.max(mse_loss)
            ori_q_loss_min = jnp.min(mse_loss)

            # Step 4: Clip per-element (only if q_loss_clip > 0)
            q_loss_clip = self.get_q_loss_clip(batch, mse_loss.dtype)
            clip_enabled = q_loss_clip > 0
            q_clip_ratio = jnp.where(clip_enabled, jnp.mean(mse_loss > q_loss_clip), 0.0)
            mse_loss = jnp.where(clip_enabled, jnp.minimum(mse_loss, q_loss_clip), mse_loss)

            mse_loss = mse_loss * weights  # (batch, num_samples, horizon*dim)

            weights_ori_q_loss_mean = jnp.mean(mse_loss)
            weights_ori_q_loss_max = jnp.max(mse_loss)
            weights_ori_q_loss_min = jnp.min(mse_loss)
            
            # Step 7: Sum over num_samples dimension (axis=1)
            mse_loss = jnp.sum(mse_loss, axis=1)  # (batch, horizon*dim)
            
            # Step 8: Mean over all dimensions
            q_loss = jnp.mean(mse_loss)

        else:
            # Non-chunking mode: simpler computation
            mse_loss = (pred_ - vel_) ** 2  # (batch, num_samples, action_dim)

            # Calculate statistics before weighting (batch, num_samples, action_dim)
            ori_q_loss_mean = jnp.mean(mse_loss)  # Global mean
            ori_q_loss_max = jnp.max(mse_loss)
            ori_q_loss_min = jnp.min(mse_loss)
            
            # Clip per-element (only if q_loss_clip > 0)
            q_loss_clip = self.get_q_loss_clip(batch, mse_loss.dtype)
            clip_enabled = q_loss_clip > 0
            q_clip_ratio = jnp.where(clip_enabled, jnp.mean(mse_loss > q_loss_clip), 0.0)
            mse_loss = jnp.where(clip_enabled, jnp.minimum(mse_loss, q_loss_clip), mse_loss)

            # Multiply by weights and sum over num_samples
            mse_loss = mse_loss * weights[..., None]  # (batch, num_samples, action_dim)

            weights_ori_q_loss_mean = jnp.mean(mse_loss)
            weights_ori_q_loss_max = jnp.max(mse_loss)
            weights_ori_q_loss_min = jnp.min(mse_loss)

            mse_loss = jnp.sum(mse_loss, axis=1)  # (batch, action_dim)
            
            # Mean over all dimensions
            q_loss = jnp.mean(mse_loss)

        # Total loss.
        bc_loss_coef = self.config.get('bc_loss_coef', 1.0)
        actor_loss = bc_loss_coef * bc_flow_loss + self.config['beta'] * q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'bc_loss_coef': bc_loss_coef,
            'q_loss': q_loss,
            'q_loss_clip': q_loss_clip,
            'q_clip_ratio': q_clip_ratio,
            'is_q_weighted': jnp.asarray(self.config.get("is_q_weighted", True), dtype=jnp.float32),
            'ori_q_loss_mean': ori_q_loss_mean,
            'ori_q_loss_max': ori_q_loss_max,
            'ori_q_loss_min': ori_q_loss_min,
            'weights_ori_q_loss_mean': weights_ori_q_loss_mean,
            'weights_ori_q_loss_max': weights_ori_q_loss_max,
            'weights_ori_q_loss_min': weights_ori_q_loss_min,
            'ori_bc_flow_loss_mean': ori_bc_flow_loss_mean,
            'ori_bc_flow_loss_max': ori_bc_flow_loss_max,
            'ori_bc_flow_loss_min': ori_bc_flow_loss_min,
            # 'q_loss_max': q_loss_max,
            # 'q_loss_min': q_loss_min,
            'target_q_mean': target_q_values.mean(),
            'target_q_std': target_q_values.std(),
            'target_q_max': target_q_values.max(),
            'target_q_min': target_q_values.min(),
            'weights_mean': weights.mean(),
            'weights_std': weights.std(),
            'weights_max': weights.max(),
            'weights_min': weights.min(),
        }


    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(
            batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    @jax.jit
    def total_loss_bc(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(
            batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_bc_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] +
            tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @staticmethod
    def _update_actor(agent, batch):
        """Update the actor BC model."""
        new_rng, rng = jax.random.split(agent.rng)
        def loss_fn(grad_params):
            return agent.actor_bc_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_actor(self, batch):
        return self._update_actor(self, batch)

    @staticmethod
    def _update_actor_mpo(agent, batch):
        """Update the actor with the MPO actor loss while keeping critic params fixed."""
        new_rng, rng = jax.random.split(agent.rng)

        critic_params = agent.network.params['modules_critic']
        target_critic_params = agent.network.params['modules_target_critic']

        def loss_fn(grad_params):
            return agent.actor_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        new_network.params['modules_critic'] = critic_params
        new_network.params['modules_target_critic'] = target_critic_params
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_actor_mpo(self, batch):
        return self._update_actor_mpo(self, batch)

    @staticmethod
    def _update_critic(agent, batch):
        """Update the critic model."""
        new_rng, rng = jax.random.split(agent.rng)
        actor_params = agent.network.params['modules_actor_bc_flow']

        def loss_fn(grad_params):
            return agent.critic_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        new_network.params['modules_actor_bc_flow'] = actor_params
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_critic(self, batch):
        return self._update_critic(self, batch)

    @staticmethod
    def _update_bc(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss_bc(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_bc(self, batch):
        return self._update_bc(self, batch)
    

    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def batch_update_actor_mpo(self, batch):
        """Update only the actor with the MPO actor loss."""
        agent, infos = jax.lax.scan(self._update_actor_mpo, self, batch)
        return agent, {f'actor/{k}': v.mean() for k, v in infos.items()}

    @jax.jit
    def batch_update_critic(self, batch):
        """Update only the critic."""
        agent, infos = jax.lax.scan(self._update_critic, self, batch)
        return agent, {f'critic/{k}': v.mean() for k, v in infos.items()}
    
    @jax.jit
    def batch_update_separate(self, critic_batch, actor_batch):
        """Update critic and actor separately with different batches."""
        # Update critic with critic_batch
        agent = self
        agent, critic_infos = jax.lax.scan(self._update_critic, agent, critic_batch)
        
        # Update actor with actor_batch  
        agent, actor_infos = jax.lax.scan(self._update_actor, agent, actor_batch)
        
        # Merge infos
        infos = {}
        for k, v in critic_infos.items():
            infos[f'critic/{k}'] = v.mean()
        for k, v in actor_infos.items():
            infos[f'actor/{k}'] = v.mean()
        
        return agent, infos

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        noises = jax.random.normal(rng, (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'] * (self.config['horizon_length'] if self.config["action_chunking"] else 1),),)
        actions = self.compute_flow_actions(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        
        # action_dim = self.config['action_dim'] * \
        #     (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        # noises = jax.random.normal(rng, (
        #         *observations.shape[: -len(self.config['ob_dims'])],
        #         self.config["num_action_samples"], action_dim),)
        # observations = jnp.repeat(observations[..., None, :], self.config["num_action_samples"], axis=-2)
        # actions = self.compute_flow_actions(observations, noises)
        # actions = jnp.clip(actions, -1, 1)
        
        # if self.config["q_agg"] == "mean":
        #     q = self.network.select("critic")(observations, actions).mean(axis=0)
        # else:
        #     q = self.network.select("critic")(observations, actions).min(axis=0)
        # indices = jnp.argmax(q, axis=-1)

        # bshape = indices.shape
        # indices = indices.reshape(-1)
        # bsize = len(indices)
        # actions = jnp.reshape(actions, (-1, self.config["num_action_samples"], action_dim))[jnp.arange(bsize), indices, :].reshape(
        #     bshape + (action_dim,))

        return actions
    
    def sample_actions_(
        self,
        observations,
        rng=None,
    ):
        noises = jax.random.normal(rng, (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'] * (self.config['horizon_length'] if self.config["action_chunking"] else 1),),)
        actions = self.compute_flow_actions(observations, noises)
        actions = jnp.clip(actions, -1, 1)

        return actions


    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate(
                [ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = copy.deepcopy(encoder_module())

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
            # use_tanh=config.get('value_use_tanh', False),
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def,
                           (ex_observations, full_actions, ex_times)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def),
                           (ex_observations, full_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (
                encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(
                learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params

        params[f'modules_target_critic'] = params[f'modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():

    config = ml_collections.ConfigDict(
        dict(
            agent_name='acmpo',  # Agent name.
            # Observation dimensions (will be set automatically).
            ob_dims=ml_collections.config_dict.placeholder(list),
            # Action dimension (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            # Actor network hidden dimensions.
            actor_hidden_dims=(512, 512, 512, 512),
            # Value network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,  # Whether to use layer normalization.
            # Whether to use layer normalization for the actor.
            actor_layer_norm=False,
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='min',  # Aggregation method for target Q values.
            beta=1.0,
            temperature=0.1,
            num_qs=2,  # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            # Optional: bound critic/value outputs.
            value_use_tanh=False,
            # Visual encoder name (None, 'impala_small', etc.).
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),  # will be set
            action_chunking=True,  # False means n-step return
            num_action_samples=32,
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            use_bc_online=False,  # Whether to use epsilon-greedy with frozen BC actor online
            bc_epsilon_start=0.5,  # Starting epsilon for BC actor
            bc_epsilon_end=0.0,  # Ending epsilon for BC actor
            bc_epsilon_decay_steps=100000,  # Number of steps to decay epsilon
            bc_loss_coef=1.0,
            is_q_weighted=True,  # Whether to weight sampled action MSE by target Q softmax.
            q_loss_clip=-1.0,  # Clip value for q_loss elements (-1.0 = no clipping)
            q_clip_start=-1.0,  # Explicit annealed q_loss_clip start value (-1.0 = disabled).
            q_clip_end=-1.0,  # Explicit annealed q_loss_clip end value (-1.0 = disabled).
            q_clip_min=-1.0,  # Annealed q_loss_clip lower bound (-1.0 = disabled).
            q_clip_max=-1.0,  # Annealed q_loss_clip initial value (-1.0 = disabled).
            q_clip_anneal_steps=-1,  # <= 0 means use the current training phase length.
            enable_calql=False,
            calql_alpha=10.0,
            calql_num_actions=10,
            calql_temp=1.0,
            calql_clip_diff_min=float("-inf"),
            calql_clip_diff_max=float("inf"),
        )
    )
    return config

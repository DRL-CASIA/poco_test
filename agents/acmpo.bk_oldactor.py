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


class ACMPOAgent(flax.struct.PyTreeNode):
    """
    Maximum a Posteriori Policy Optimisation (MPO) agent with action chunking. 
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

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
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        if self.config["action_chunking"]:
            # fold in horizon_length together with action_dim
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]  # take the first one
        batch_size, action_dim = batch_actions.shape
        # rng, x_rng, t_rng = jax.random.split(rng, 3)
        rng, noise_rng, x_rng, t_rng = jax.random.split(rng, 4)

        # BC flow loss.
        observations = batch['observations']
        noises = jax.random.normal(noise_rng, (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'] * (self.config['horizon_length'] if self.config["action_chunking"] else 1),),)
        batch_actions_old = self.compute_flow_actions_old(observations, noises)
        
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        # x_1 = batch_actions
        x_1 = batch_actions_old
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
        
        # Q loss.
        rng, noise_rng, x_rng1, t_rng1 = jax.random.split(rng, 4)
        noises = jax.random.normal(noise_rng, (
            *batch['observations'].shape[: -len(self.config['ob_dims'])], self.config["actor_num_samples"], action_dim),)
        observations = jnp.repeat(batch['observations'][..., None, :], self.config["actor_num_samples"], axis=-2)
        
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
        # target_flow_actions_max = jnp.reshape(target_flow_actions, (-1, self.config["actor_num_samples"], action_dim))[jnp.arange(bsize), indices, :].reshape(
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
        
        mean_q = jnp.max(target_q_values, axis=1, keepdims=True)
        weights = jax.nn.softmax((target_q_values - mean_q) / self.config["temperature"], axis=-1)
        weights = weights[..., None]
        
        x_0_ = jax.random.normal(x_rng1, (batch_size, self.config["actor_num_samples"], action_dim))
        x_1_ = target_flow_actions
        t_ = jax.random.uniform(t_rng1, (batch_size, self.config["actor_num_samples"], 1))
        x_t_ = (1 - t_) * x_0_ + t_ * x_1_
        vel_ = x_1_ - x_0_
        
        pred_ = self.network.select('actor_bc_flow')(observations, x_t_, t_, params=grad_params)
        
        # only bc on the valid chunk indices
        if self.config["action_chunking"]:
            q_loss = jnp.mean(
                jnp.reshape(
                    jnp.sum(((pred_ - vel_) ** 2) * weights, axis=1),
                    (batch_size, self.config["horizon_length"], self.config["action_dim"])
                ) * batch["valid"][..., None])
        else:
            q_loss = jnp.mean(weights * jnp.sum(((pred_ - vel_) ** 2) * weights, axis=1))

        # Total loss.
        actor_loss = bc_flow_loss + self.config['beta'] * q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'q_loss': q_loss,
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

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] +
            tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params
        
    def old_actor_update(self, network):
        """Hard update: old_actor = actor_bc_flow."""
        old_actor_params = jax.tree_util.tree_map(
            lambda p: p,
            self.network.params['modules_actor_bc_flow']
        )
        network.params['modules_actor_bc_flow_old'] = old_actor_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        agent.old_actor_update(new_network)
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

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
        #         self.config["actor_num_samples"], action_dim),)
        # observations = jnp.repeat(observations[..., None, :], self.config["actor_num_samples"], axis=-2)
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
        # actions = jnp.reshape(actions, (-1, self.config["actor_num_samples"], action_dim))[jnp.arange(bsize), indices, :].reshape(
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
    
    @jax.jit
    def compute_flow_actions_old(
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
            vels = self.network.select('actor_bc_flow_old')(observations, actions, t, is_encoded=True)
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
            encoders['actor_bc_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
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
            actor_bc_flow_old=(copy.deepcopy(actor_bc_flow_def),
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
            q_agg='mean',  # Aggregation method for target Q values.
            beta=1.0,
            temperature=0.1,
            num_qs=2,  # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            # Visual encoder name (None, 'impala_small', etc.).
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),  # will be set
            action_chunking=True,  # False means n-step return
            actor_num_samples=32,  
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
        )
    )
    return config

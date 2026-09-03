import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value


class ACMPOOriginalAgent(flax.struct.PyTreeNode):
    """MPO agent with a Gaussian policy head and one-step actions."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _batch_actions(self, batch):
        if self.config["action_chunking"]:
            return jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        return batch["actions"][..., 0, :]

    def _num_action_samples(self):
        return self.config.get("num_action_samples", self.config.get("actor_num_samples", 32))

    def _clip_for_log_prob(self, actions):
        if self.config["tanh_squash"]:
            return jnp.clip(actions, -1.0 + 1e-5, 1.0 - 1e-5)
        return actions

    def _sample_policy_actions(self, observations, rng, params=None, num_samples=None):
        if num_samples is None:
            dist = self.network.select("actor")(observations, params=params)
            return jnp.clip(dist.sample(seed=rng), -1.0, 1.0)

        observations = jnp.repeat(observations[..., None, :], num_samples, axis=-2)
        dist = self.network.select("actor")(observations, params=params)
        actions = dist.sample(seed=rng)
        return observations, jnp.clip(actions, -1.0, 1.0), dist

    def _dist_action_mean_and_std(self, dist):
        action_mean = dist.mode()
        if self.config["tanh_squash"]:
            std = dist.distribution.stddev()
        else:
            std = dist.stddev()
        return action_mean, std

    def _bc_loss(self, dist, batch_actions):
        if self.config.get("bc_loss_type", "nll") == "mse":
            action_mean, std = self._dist_action_mean_and_std(dist)
            action_mse = jnp.square(action_mean - batch_actions).mean()
            std_target = self.config.get("bc_std_target", 0.1)
            std_mse = jnp.square(std - std_target).mean()
            std_loss_coef = self.config.get("bc_std_loss_coef", 1.0)
            bc_loss = action_mse + std_loss_coef * std_mse

            return bc_loss, {
                "bc_loss": bc_loss,
                "bc_action_mse": action_mse,
                "bc_std_mse": std_mse,
                "bc_std_target": std_target,
                "bc_std_mean": std.mean(),
                "bc_mean_abs": jnp.abs(action_mean).mean(),
            }

        batch_actions = self._clip_for_log_prob(batch_actions)
        bc_loss = -dist.log_prob(batch_actions).mean()
        return bc_loss, {
            "bc_loss": bc_loss,
        }

    def critic_loss(self, batch, grad_params, rng):
        batch_actions = self._batch_actions(batch)

        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch["next_observations"][..., -1, :], rng=sample_rng)
        next_qs = self.network.select("target_critic")(
            batch["next_observations"][..., -1, :], actions=next_actions
        )
        if self.config["q_agg"] == "min":
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = (
            batch["rewards"][..., -1]
            + (self.config["discount"] ** self.config["horizon_length"])
            * batch["masks"][..., -1]
            * next_q
        )
        q = self.network.select("critic")(batch["observations"], actions=batch_actions, params=grad_params)
        critic_loss = (jnp.square(q - target_q) * batch["valid"][..., -1]).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def actor_bc_loss(self, batch, grad_params, rng):
        batch_actions = self._batch_actions(batch)
        dist = self.network.select("actor")(batch["observations"], params=grad_params)
        bc_loss, bc_info = self._bc_loss(dist, batch_actions)

        return bc_loss, {
            "actor_loss": bc_loss,
            **bc_info,
        }

    def actor_loss(self, batch, grad_params, rng):
        batch_actions = self._batch_actions(batch)
        rng, sample_rng = jax.random.split(rng)

        dist = self.network.select("actor")(batch["observations"], params=grad_params)
        bc_loss, bc_info = self._bc_loss(dist, batch_actions)

        num_action_samples = self._num_action_samples()
        observations, sampled_actions, sampled_dist = self._sample_policy_actions(
            batch["observations"], sample_rng, params=grad_params, num_samples=num_action_samples
        )
        target_actions = jax.lax.stop_gradient(self._clip_for_log_prob(sampled_actions))

        target_qs = self.network.select("target_critic")(observations, actions=target_actions)
        if self.config["q_agg"] == "min":
            target_q_values = target_qs.min(axis=0)
        else:
            target_q_values = target_qs.mean(axis=0)

        max_q = jnp.max(target_q_values, axis=-1, keepdims=True)
        weights = jax.nn.softmax((target_q_values - max_q) / self.config["temperature"], axis=-1)
        weights = jax.lax.stop_gradient(weights)

        log_probs = sampled_dist.log_prob(target_actions)
        q_loss = -(weights * log_probs).sum(axis=-1).mean()

        bc_loss_coef = self.config.get("bc_loss_coef", 1.0)
        actor_loss = bc_loss_coef * bc_loss + self.config["beta"] * q_loss

        return actor_loss, {
            "actor_loss": actor_loss,
            **bc_info,
            "bc_loss_coef": bc_loss_coef,
            "q_loss": q_loss,
            "target_q_mean": target_q_values.mean(),
            "target_q_std": target_q_values.std(),
            "target_q_max": target_q_values.max(),
            "target_q_min": target_q_values.min(),
            "weights_mean": weights.mean(),
            "weights_std": weights.std(),
            "weights_max": weights.max(),
            "weights_min": weights.min(),
            "log_probs_mean": log_probs.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        return critic_loss + actor_loss, info

    @jax.jit
    def total_loss_bc(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_bc_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        return critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "critic")
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @staticmethod
    def _update_bc(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss_bc(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "critic")
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_bc(self, batch):
        return self._update_bc(self, batch)

    @staticmethod
    def _update_actor(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)
        critic_params = agent.network.params["modules_critic"]
        target_critic_params = agent.network.params["modules_target_critic"]

        def loss_fn(grad_params):
            return agent.actor_bc_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        new_network.params["modules_critic"] = critic_params
        new_network.params["modules_target_critic"] = target_critic_params
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_actor(self, batch):
        return self._update_actor(self, batch)

    @staticmethod
    def _update_actor_mpo(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)
        critic_params = agent.network.params["modules_critic"]
        target_critic_params = agent.network.params["modules_target_critic"]

        def loss_fn(grad_params):
            return agent.actor_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        new_network.params["modules_critic"] = critic_params
        new_network.params["modules_target_critic"] = target_critic_params
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_actor_mpo(self, batch):
        return self._update_actor_mpo(self, batch)

    @staticmethod
    def _update_critic(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)
        actor_params = agent.network.params["modules_actor"]

        def loss_fn(grad_params):
            return agent.critic_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "critic")
        new_network.params["modules_actor"] = actor_params
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_critic(self, batch):
        return self._update_critic(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def batch_update_actor_mpo(self, batch):
        agent, infos = jax.lax.scan(self._update_actor_mpo, self, batch)
        return agent, {f"actor/{k}": v.mean() for k, v in infos.items()}

    @jax.jit
    def batch_update_critic(self, batch):
        agent, infos = jax.lax.scan(self._update_critic, self, batch)
        return agent, {f"critic/{k}": v.mean() for k, v in infos.items()}

    @jax.jit
    def sample_actions(self, observations, rng=None):
        rng = rng if rng is not None else self.rng
        if self.config.get("eval_mode", False):
            return jnp.clip(self.network.select("actor")(observations).mode(), -1.0, 1.0)
        return self._sample_policy_actions(observations, rng)

    @jax.jit
    def sample_actions_(self, observations, rng=None):
        rng = rng if rng is not None else self.rng
        if self.config.get("eval_mode", False):
            return jnp.clip(self.network.select("actor")(observations).mode(), -1.0, 1.0)
        return self._sample_policy_actions(observations, rng)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        encoders = {}
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["actor"] = copy.deepcopy(encoder_module())

        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["num_qs"],
            encoder=encoders.get("critic"),
        )
        actor_def = Actor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            log_std_min=config["log_std_min"],
            log_std_max=config["log_std_max"],
            tanh_squash=config["tanh_squash"],
            state_dependent_std=config["state_dependent_std"],
            const_std=config["const_std"],
            encoder=encoders.get("actor"),
        )

        network_info = {
            "actor": (actor_def, (ex_observations,)),
            "critic": (critic_def, (ex_observations, full_actions)),
            "target_critic": (copy.deepcopy(critic_def), (ex_observations, full_actions)),
        }
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.0:
            network_tx = optax.adamw(learning_rate=config["lr"], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        network.params["modules_target_critic"] = network.params["modules_critic"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="acmpo_original",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg="mean",
            beta=1.0,
            temperature=0.001,
            num_qs=2,
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=False,
            num_action_samples=32,
            actor_num_samples=32,
            bc_loss_coef=1.0,
            tanh_squash=True,
            state_dependent_std=True,
            const_std=False,
            log_std_min=-5.0,
            log_std_max=2.0,
            bc_loss_type="nll",
            bc_std_target=0.1,
            bc_std_loss_coef=1.0,
            eval_mode=False,
            use_bc_online=False,
            bc_epsilon_start=0.5,
            bc_epsilon_end=0.0,
            bc_epsilon_decay_steps=100000,
            encoder=ml_collections.config_dict.placeholder(str),
            weight_decay=0.0,
        )
    )
    return config

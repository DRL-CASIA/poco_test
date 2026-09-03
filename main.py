
import jax
import time
import random
import json
import pickle
import wandb
# import swanlab
import tqdm
import glob
import numpy as np
from agents import agents
from evaluation import evaluate, evaluate_
from utils.datasets import Dataset, ReplayBuffer
from utils.flax_utils import save_agent, restore_agent_with_file
from envs.robomimic_utils import is_robomimic_env
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.env_utils import make_env_and_datasets
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger
from ml_collections import config_flags
from absl import app, flags
import os
from flax.core import unfreeze

from agents.acmpo import ACMPOAgent

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

os.environ["MUJOCO_GL"] = "egl"

# os.environ["EGL_DEVICE_ID"] = "4"
# os.environ["MUJOCO_EGL_DEVICE_ID"] = "4"


if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_bool('retain_offline_data', True, 'whether retain offline data in the replay buffer')
flags.DEFINE_bool('debug', False, 'whether debug')
flags.DEFINE_string('run_group', 'Default', 'Run group.')
flags.DEFINE_string('exp_name', None, 'experiment name.')
flags.DEFINE_string('project_name', 'qc', 'experiment name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
# flags.DEFINE_integer('offline_q_steps', 2000, 'Number of offline Q steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 500, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Evaluation interval.') # 
flags.DEFINE_integer('offline_eval_interval', 50000, 'Evaluation interval.') # 2w
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('offline_save_interval', 100000, 'Save interval.')
flags.DEFINE_bool('save_final_online_ckpt', False, 'whether save the final online checkpoint')
flags.DEFINE_bool('save_q_functions', False, 'whether save final offline/online critic Q function parameters')
flags.DEFINE_bool('save_online_eval_q_functions', True, 'whether save critic Q function parameters after each online eval')
flags.DEFINE_bool('critic_warmup', False, 'Update only the critic before start_training during online training.')
flags.DEFINE_bool('disable_q_finetune', False, 'Disable critic/Q-function updates during online training.')
flags.DEFINE_integer('q_finetune_until_online_step', -1, 'Freeze critic/Q-function updates after this online env step. <= 0 means no step-based freeze.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')
flags.DEFINE_bool('load_offline_ckpt', True, 'whether load the offline BC checkpoint before online training')
flags.DEFINE_string('offline_ckpt_path', None, 'Optional offline checkpoint path to load before online training. Supports {seed}.')
flags.DEFINE_string('load_q_function_path', None, 'Optional Q-function checkpoint path with critic/target_critic params. Supports {seed}.')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")
flags.DEFINE_integer('Q_update_decay', 1, 'update Q once every N online update steps; 1 updates Q every step')

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/acmpo.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")
flags.DEFINE_bool('is_subset', False, "whether use subset dataset")

class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log(
            {f'{prefix}/{k}': v for k, v in data.items()}, step=step)


def main(_):
    if FLAGS.exp_name is None:
        FLAGS.exp_name = get_exp_name(FLAGS.seed)
    
    if FLAGS.run_group == 'Default':
        FLAGS.run_group = FLAGS.env_name
        
    run = setup_wandb(project=FLAGS.project_name, group=FLAGS.run_group, name=FLAGS.exp_name, debug=FLAGS.debug)

    FLAGS.save_dir = os.path.join( FLAGS.save_dir, FLAGS.project_name, FLAGS.run_group, FLAGS.env_name, FLAGS.exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
            FLAGS.env_name, is_subset=FLAGS.is_subset)

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0

    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    def get_current_q_loss_clip(update_step, default_anneal_steps):
        q_clip_start = config.get("q_clip_start", -1.0)
        q_clip_end = config.get("q_clip_end", -1.0)
        q_clip_max = config.get("q_clip_max", -1.0)
        q_clip_min = config.get("q_clip_min", -1.0)
        if q_clip_start > 0 and q_clip_end >= 0:
            anneal_steps = config.get("q_clip_anneal_steps", -1)
            if anneal_steps <= 0:
                anneal_steps = default_anneal_steps
            progress = min(max(update_step, 0) / max(anneal_steps, 1), 1.0)
            return q_clip_start + (q_clip_end - q_clip_start) * progress
        if q_clip_max > 0 and q_clip_min >= 0:
            anneal_steps = config.get("q_clip_anneal_steps", -1)
            if anneal_steps <= 0:
                anneal_steps = default_anneal_steps
            progress = min(max(update_step, 0) / max(anneal_steps, 1), 1.0)
            return q_clip_max + (q_clip_min - q_clip_max) * progress
        return config.get("q_loss_clip", -1.0)

    def add_q_loss_clip(batch, q_loss_clip, leading_shape=()):
        batch = dict(batch)
        batch["q_loss_clip"] = np.full(leading_shape, q_loss_clip, dtype=np.float32) if leading_shape else np.float32(q_loss_clip)
        return batch

    def add_mc_returns(ds):
        rewards = np.asarray(ds["rewards"])
        terminals = np.asarray(ds["terminals"])
        mc_returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0.0
        for idx in reversed(range(len(rewards))):
            running_return = rewards[idx] + discount * running_return * (1.0 - terminals[idx])
            mc_returns[idx] = running_return

        ds_dict = {k: v for k, v in ds.items()}
        ds_dict["mc_returns"] = mc_returns
        return Dataset.create(**ds_dict)

    # handle dataset
    def process_train_dataset(ds):
        """
        Process the train dataset to 
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )

        pos_rewards = ds["rewards"]
        ds_dict = {k: v for k, v in ds.items()}
        ds_dict["rewards"] = pos_rewards
        ds = Dataset.create(**ds_dict)
        
        # if is_robomimic_env(FLAGS.env_name):
        #     penalty_rewards = ds["rewards"] - 1.0
        #     ds_dict = {k: v for k, v in ds.items()}
        #     ds_dict["rewards"] = penalty_rewards
        #     ds = Dataset.create(**ds_dict)

        # Add step penalty of -0.01 to all rewards
        penalty_rewards = ds["rewards"] - 0.01
        ds_dict = {k: v for k, v in ds.items()}
        ds_dict["rewards"] = penalty_rewards
        ds = Dataset.create(**ds_dict)

        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return add_mc_returns(ds)

    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())

    agent_class = agents[config['agent_name']]
    agent: ACMPOAgent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Setup logging.
    prefixes = ["eval", "eval_", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) for prefix in prefixes},
        wandb_logger=wandb,
        # wandb_logger=swanlab,
    )

    def with_eval_mode(eval_agent, eval_mode):
        if "eval_mode" not in eval_agent.config:
            return eval_agent
        return eval_agent.replace(config=eval_agent.config.copy({"eval_mode": eval_mode}))

    def run_eval(eval_agent, step, label):
        eval_info, _, _ = evaluate(
            agent=with_eval_mode(eval_agent, True),
            env=eval_env,
            action_dim=example_batch["actions"].shape[-1],
            num_eval_episodes=FLAGS.eval_episodes,
            num_video_episodes=FLAGS.video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
        )
        logger.log(eval_info, "eval", step=step)

        eval_info_, _, _ = evaluate_(
            agent=with_eval_mode(eval_agent, False),
            env=eval_env,
            action_dim=example_batch["actions"].shape[-1],
            num_eval_episodes=FLAGS.eval_episodes,
            num_video_episodes=FLAGS.video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
        )
        logger.log(eval_info_, "eval_", step=step)
        print(f"{label} deterministic eval info: {eval_info}", flush=True)
        print(f"{label} stochastic eval_ info: {eval_info_}", flush=True)

    def save_q_function(q_agent, suffix):
        if not FLAGS.save_q_functions:
            return
        params = q_agent.network.params
        q_params = {
            "critic": params["modules_critic"],
            "target_critic": params["modules_target_critic"],
        }
        save_path = os.path.join(FLAGS.save_dir, f"q_function_{suffix}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(jax.device_get(q_params), f)
        print(f"Saved Q function to {save_path}", flush=True)

    def load_q_function(q_agent, q_path):
        q_path = q_path.format(seed=FLAGS.seed)
        if not os.path.exists(q_path):
            raise FileNotFoundError(f"Q function checkpoint does not exist: {q_path}")
        with open(q_path, "rb") as f:
            q_params = pickle.load(f)
        if "critic" not in q_params:
            raise KeyError(f"Q function checkpoint must contain a 'critic' key: {q_path}")

        params = unfreeze(q_agent.network.params)
        params["modules_critic"] = q_params["critic"]
        params["modules_target_critic"] = q_params.get("target_critic", q_params["critic"])
        q_agent = q_agent.replace(network=q_agent.network.replace(params=params))
        print(f"Loaded Q function from {q_path}", flush=True)
        return q_agent

    ################################################################################################
    ################################################################################################
    # update actor
    offline_init_time = time.time()

    checkpoint_path = os.path.join(FLAGS.save_dir, "params_final_offline.pkl")

    explicit_offline_ckpt = FLAGS.offline_ckpt_path is not None
    if FLAGS.env_name == "square-mh-low_dim":
        load_ckpt = "/home/jzn/workspace/qc_latest/exp/ckpt/square_ippo_bc.pkl"
    elif FLAGS.env_name == "can-mh-low_dim" and not FLAGS.is_subset:
        load_ckpt = "/home/jzn/workspace/qc_latest/exp/ckpt/can_ippo_bc.pkl"
    elif FLAGS.env_name == "can-mh-low_dim" and FLAGS.is_subset:
        load_ckpt = f"/home/jzn/workspace/qc_latest/exp/ckpt/can_subset_ippo_bc_seed{FLAGS.seed}.pkl"
    elif FLAGS.env_name == "scene-play-singletask-task1-v0":
        load_ckpt = f"/home/jzn/workspace/qc_latest/exp/qc_online/scene1_QC-MPO-beta1.0-temp0.001-v10/scene-play-singletask-task1-v0/seed-{FLAGS.seed}-scene1_QC-MPO-beta1.0-temp0.001-v10/params_final_offline.pkl"
    elif FLAGS.env_name == "puzzle-3x3-play-singletask-task1-v0":
        load_ckpt = f"/home/jzn/workspace/qc_latest/exp/qc_online/puzzle1_QC-MPO-beta1.0-temp0.001-v10/puzzle-3x3-play-singletask-task1-v0/seed-{FLAGS.seed}-puzzle1_QC-MPO-beta1.0-temp0.001-v10/params_final_offline.pkl"
    else:
        load_ckpt = checkpoint_path
    if explicit_offline_ckpt:
        load_ckpt = FLAGS.offline_ckpt_path.format(seed=FLAGS.seed)
        # if FLAGS.load_offline_ckpt and FLAGS.offline_steps > 0 and not os.path.exists(load_ckpt):
        #     raise FileNotFoundError(f"Explicit offline checkpoint does not exist: {load_ckpt}")

    if FLAGS.load_offline_ckpt and os.path.exists(load_ckpt) and FLAGS.offline_steps > 0:
        agent = restore_agent_with_file(agent, load_ckpt)
        print(f"Loaded checkpoint from {load_ckpt}")
        log_step += FLAGS.offline_steps
    elif FLAGS.offline_steps > 0:

        for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
            log_step += 1

            if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
                dataset_idx = (dataset_idx + 1) % len(dataset_paths)
                print(
                    f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
                train_dataset, val_dataset = make_ogbench_env_and_datasets(
                    FLAGS.env_name,
                    dataset_path=dataset_paths[dataset_idx],
                    compact_dataset=False,
                    dataset_only=True,
                    cur_env=env,
                )
                train_dataset = process_train_dataset(train_dataset)

            batch = train_dataset.sample_sequence(
                config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)
            q_loss_clip = get_current_q_loss_clip(i - 1, FLAGS.offline_steps)
            batch = add_q_loss_clip(batch, q_loss_clip)
            agent, offline_info = agent.update_actor(batch)

            if i % FLAGS.log_interval == 0 or i == 1:
                logger.log(offline_info, "offline_agent", step=log_step)

            # saving
            # if FLAGS.offline_save_interval > 0 and i % FLAGS.offline_save_interval == 0:
            #     save_agent(agent, FLAGS.save_dir, log_step)

            # eval
            if FLAGS.offline_eval_interval != 0 and i % FLAGS.offline_eval_interval == 0:
                run_eval(agent, log_step, f"{i} offline")
                save_q_function(agent, f"offline_eval_step{log_step}")
        
        # Save final agent after offline training (for potential BC online)
        if FLAGS.offline_steps!=0:
            save_agent(agent, FLAGS.save_dir, "final_offline")

    if FLAGS.load_q_function_path is not None:
        agent = load_q_function(agent, FLAGS.load_q_function_path)
        
    # Final evaluation after offline initialization.
    run_eval(agent, log_step, "offline-final")
    save_q_function(agent, "final_offline")

    #  ------------------------ begin of online training ------------------------
    
    # Save frozen BC actor for epsilon-greedy if enabled
    frozen_bc_agent = None
    if config.get("use_bc_online", False):
        import copy
        frozen_bc_agent = copy.deepcopy(agent)
        print("Saved frozen BC actor for epsilon-greedy exploration.", flush=True)

    if FLAGS.retain_offline_data:
        # Sample 1/10 of trajectories while keeping them complete
        
        # transition from offline to online with reduced dataset
        replay_buffer = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1))
    else:
        replay_buffer = ReplayBuffer.create(example_batch, size=FLAGS.buffer_size)

    ob, _ = env.reset()

    action_queue = []
    action_dim = example_batch["actions"].shape[-1]

    ################################################################################################
    ################################################################################################
    # Online RL
    update_info = {}

    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()
    
    # Epsilon-greedy parameters
    use_bc_online = config.get("use_bc_online", False)
    bc_epsilon_start = config.get("bc_epsilon_start", 0.5)
    bc_epsilon_end = config.get("bc_epsilon_end", 0.0)
    bc_epsilon_decay_steps = config.get("bc_epsilon_decay_steps", 100000)
    bc_actor_used_count = 0  # Track how many times BC actor is used
    
    if use_bc_online and frozen_bc_agent is not None:
        print(f"\n{'='*60}", flush=True)
        print("Epsilon-greedy BC exploration enabled:", flush=True)
        print(f"  - Initial epsilon: {bc_epsilon_start}", flush=True)
        print(f"  - Final epsilon: {bc_epsilon_end}", flush=True)
        print(f"  - Decay steps: {bc_epsilon_decay_steps}", flush=True)
        print(f"{'='*60}\n", flush=True)
    else:
        print("\nEpsilon-greedy BC exploration disabled.\n", flush=True)
    if FLAGS.critic_warmup:
        print("Critic warmup enabled before online actor training.", flush=True)
    if FLAGS.disable_q_finetune:
        print("Q finetuning disabled during online training.", flush=True)
    elif FLAGS.q_finetune_until_online_step > 0:
        print(f"Q finetuning enabled through online step {FLAGS.q_finetune_until_online_step}; frozen afterward.", flush=True)
    
    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)

        # Calculate current epsilon (linear decay)
        if use_bc_online and frozen_bc_agent is not None:
            progress = min(i / bc_epsilon_decay_steps, 1.0)
            current_epsilon = bc_epsilon_start + (bc_epsilon_end - bc_epsilon_start) * progress
        else:
            current_epsilon = 0.0
        
        # during online rl, the action chunk is executed fully
        if len(action_queue) == 0:
            # Epsilon-greedy: use frozen BC actor with probability epsilon
            online_rng, epsilon_key = jax.random.split(online_rng)
            use_bc_actor = (use_bc_online and 
                          frozen_bc_agent is not None and 
                          np.random.random() < current_epsilon)
            
            if use_bc_actor:
                action = frozen_bc_agent.sample_actions(observations=ob, rng=key)
                bc_actor_used_count += 1
            else:
                action = agent.sample_actions(observations=ob, rng=key)

            action_chunk = np.array(action).reshape(-1, action_dim)

            for action in action_chunk:
                action_queue.append(action)
        action = action_queue.pop(0)

        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if FLAGS.save_all_online_states:
            state = env.get_state()
            data["steps"].append(i)
            data["obs"].append(np.copy(next_ob))
            data["qpos"].append(np.copy(state["qpos"]))
            data["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                data["button_states"].append(np.copy(state["button_states"]))

        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        # if 'antmaze' in FLAGS.env_name and (
        #     'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        # ):
        #     # Adjust reward for D4RL antmaze.
        #     int_reward = int_reward - 1.0
        # elif is_robomimic_env(FLAGS.env_name):
        #     # Adjust online (0, 1) reward for robomimic
        #     int_reward = int_reward - 1.0

        # Add step penalty of -0.01
        int_reward = int_reward - 0.01

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
            mc_returns=0.0,
        )
        replay_buffer.add_transition(transition)

        # done
        if done:
            env_info = {}
            env_info['running_success'] = int(info.get('success', 0.0))
            logger.log(env_info, 'env', step=log_step)
            ob, _ = env.reset()
            action_queue = []  # reset the action queue
        else:
            ob = next_ob

        freeze_q_now = FLAGS.disable_q_finetune or (
            FLAGS.q_finetune_until_online_step > 0 and i > FLAGS.q_finetune_until_online_step
        )

        if i >= FLAGS.start_training:
            if freeze_q_now:
                actor_batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio,
                                                            sequence_length=FLAGS.horizon_length, discount=discount)
                actor_batch = jax.tree.map(lambda x: x.reshape((
                    FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), actor_batch)
                q_loss_clip = get_current_q_loss_clip(i - FLAGS.start_training, FLAGS.online_steps - FLAGS.start_training)
                actor_batch = add_q_loss_clip(actor_batch, q_loss_clip, leading_shape=(FLAGS.utd_ratio,))
                agent, update_info["online_agent"] = agent.batch_update_actor_mpo(actor_batch)
            elif FLAGS.Q_update_decay <= 1:
                batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio,
                                                      sequence_length=FLAGS.horizon_length, discount=discount)
                batch = jax.tree.map(lambda x: x.reshape((
                    FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)
                q_loss_clip = get_current_q_loss_clip(i - FLAGS.start_training, FLAGS.online_steps - FLAGS.start_training)
                batch = add_q_loss_clip(batch, q_loss_clip, leading_shape=(FLAGS.utd_ratio,))

                agent, update_info["online_agent"] = agent.batch_update(batch)
            else:
                actor_batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio,
                                                            sequence_length=FLAGS.horizon_length, discount=discount)
                actor_batch = jax.tree.map(lambda x: x.reshape((
                    FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), actor_batch)
                q_loss_clip = get_current_q_loss_clip(i - FLAGS.start_training, FLAGS.online_steps - FLAGS.start_training)
                actor_batch = add_q_loss_clip(actor_batch, q_loss_clip, leading_shape=(FLAGS.utd_ratio,))
                agent, decayed_update_info = agent.batch_update_actor_mpo(actor_batch)

                if i % FLAGS.Q_update_decay == 0:
                    critic_batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio,
                                                                 sequence_length=FLAGS.horizon_length, discount=discount)
                    critic_batch = jax.tree.map(lambda x: x.reshape((
                        FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), critic_batch)
                    agent, critic_info = agent.batch_update_critic(critic_batch)
                    decayed_update_info.update(critic_info)

                update_info["online_agent"] = decayed_update_info
        else:
            batch = replay_buffer.sample_sequence(config['batch_size'],
                                                  sequence_length=FLAGS.horizon_length, discount=discount)
            q_loss_clip = get_current_q_loss_clip(i - 1, FLAGS.start_training)
            batch = add_q_loss_clip(batch, q_loss_clip)
            if FLAGS.critic_warmup:
                agent, update_info["online_agent"] = agent.update_critic(batch)
            elif freeze_q_now:
                agent, update_info["online_agent"] = agent.update_actor(batch)
            else:
                agent, update_info["online_agent"] = agent.update_bc(batch)

        if i % FLAGS.log_interval == 0 or i == 1:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            
            # Log epsilon-greedy stats
            if use_bc_online and frozen_bc_agent is not None:
                bc_stats = {
                    'bc_epsilon': current_epsilon,
                    'bc_actor_usage_rate': bc_actor_used_count / i if i > 0 else 0.0,
                }
                logger.log(bc_stats, "online_agent", step=log_step)
            
            update_info = {}

        if i == FLAGS.online_steps - 1 or \
                (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0) or i==FLAGS.start_training-1:
            run_eval(agent, log_step, f"{i} online")
            if FLAGS.critic_warmup and i == FLAGS.start_training - 1:
                save_q_function(agent, f"after_q_warmup_step{log_step}")
            elif FLAGS.save_online_eval_q_functions:
                save_q_function(agent, f"online_eval_step{log_step}")

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

    end_time = time.time()
    if FLAGS.save_final_online_ckpt and FLAGS.online_steps > 0:
        save_agent(agent, FLAGS.save_dir, "final_online")
        print(f"Saved final online checkpoint to {FLAGS.save_dir}", flush=True)
    if FLAGS.online_steps > 0:
        save_q_function(agent, "final_online")
    
    # Print BC actor usage statistics
    if use_bc_online and frozen_bc_agent is not None:
        print(f"\n{'='*60}", flush=True)
        print("Epsilon-greedy BC Actor Statistics:", flush=True)
        print(f"  - Total steps: {FLAGS.online_steps}", flush=True)
        print(f"  - BC actor used: {bc_actor_used_count} times", flush=True)
        print(f"  - Usage rate: {bc_actor_used_count / FLAGS.online_steps * 100:.2f}%", flush=True)
        print(f"{'='*60}\n", flush=True)

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {"steps": np.array(data["steps"]),
                  "qpos": np.stack(data["qpos"], axis=0),
                  "qvel": np.stack(data["qvel"], axis=0),
                  "obs": np.stack(data["obs"], axis=0),
                  "offline_time": online_init_time - offline_init_time,
                  "online_time": end_time - online_init_time,
                  }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url or "")


if __name__ == '__main__':
    app.run(main)

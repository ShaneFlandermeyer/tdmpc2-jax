import os
from collections import defaultdict
from functools import partial

import flax.linen as nn
import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
# Tensorboard: Prevent tf from allocating full GPU memory
import tensorflow as tf
import tqdm
from flax.metrics import tensorboard
from flax.training.train_state import TrainState

from tdmpc2_jax import TDMPC2, WorldModel
from tdmpc2_jax.common.activations import mish, simnorm
from tdmpc2_jax.data import SequentialReplayBuffer
from tdmpc2_jax.envs.dmcontrol import make_dmc_env
from tdmpc2_jax.networks import NormedLinear

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


@hydra.main(config_name='config', config_path='.', version_base=None)
def train(cfg: dict):
  env_config = cfg['env']
  encoder_config = cfg['encoder']
  model_config = cfg['world_model']
  tdmpc_config = cfg['tdmpc2']
  bmpc_config = cfg['bmpc']

  ##############################
  # Logger setup
  ##############################
  output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
  writer = tensorboard.SummaryWriter(os.path.join(output_dir, 'tensorboard'))
  writer.hparams(cfg)

  ##############################
  # Environment setup
  ##############################
  def make_env(env_config, seed):
    def make_gym_env(env_id, seed):
      env = gym.make(env_id)
      env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env = gym.wrappers.Autoreset(env)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env

    if env_config.backend == "gymnasium":
      return make_gym_env(env_config.env_id, seed)
    elif env_config.backend == "dmc":
      env = make_dmc_env(env_config.env_id, seed, env_config.dmc.obs_type)
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env = gym.wrappers.Autoreset(env)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env
    else:
      raise ValueError("Environment not supported:", env_config)

  if env_config.asynchronous:
    vector_env_cls = gym.vector.AsyncVectorEnv
  else:
    vector_env_cls = gym.vector.SyncVectorEnv
  env = vector_env_cls(
      [
          partial(make_env, env_config, seed)
          for seed in range(cfg.seed, cfg.seed+env_config.num_envs)
      ]
  )
  np.random.seed(cfg.seed)
  rng = jax.random.PRNGKey(cfg.seed)

  ##############################
  # Agent setup
  ##############################
  dtype = jnp.dtype(model_config.dtype)
  rng, model_key, encoder_key = jax.random.split(rng, 3)
  encoder_module = nn.Sequential(
      [
          NormedLinear(
              encoder_config.encoder_dim, activation=mish, dtype=dtype
          )
          for _ in range(encoder_config.num_encoder_layers-1)
      ] + [
          NormedLinear(
              model_config.latent_dim,
              activation=partial(
                  simnorm, simplex_dim=model_config.simnorm_dim
              ),
              dtype=dtype
          )
      ]
  )

  if encoder_config.tabulate:
    print("Encoder")
    print("--------------")
    print(
        encoder_module.tabulate(
            jax.random.key(0),
            env.observation_space.sample(),
            compute_flops=True
        )
    )

  ##############################
  # Replay buffer setup
  ##############################
  dummy_obs, _ = env.reset()
  dummy_action = env.action_space.sample()
  dummy_next_obs, dummy_reward, dummy_term, dummy_trunc, _ = env.step(
      dummy_action
  )
  replay_buffer = SequentialReplayBuffer(
      capacity=cfg.max_steps//env_config.num_envs,
      num_envs=env_config.num_envs,
      seed=cfg.seed,
      dummy_input=dict(
          observation=dummy_obs,
          action=dummy_action,
          reward=dummy_reward,
          next_observation=dummy_next_obs,
          terminated=dummy_term,
          truncated=dummy_trunc,
          expert_mean=np.zeros_like(dummy_action),
          expert_std=np.ones_like(dummy_action),
          last_reanalyze=np.zeros(env_config.num_envs, dtype=int),
      )
  )

  encoder = TrainState.create(
      apply_fn=encoder_module.apply,
      params=encoder_module.init(encoder_key, dummy_obs)['params'],
      tx=optax.chain(
          optax.zero_nans(),
          optax.clip_by_global_norm(model_config.max_grad_norm),
          optax.adam(encoder_config.learning_rate),
      )
  )

  model = WorldModel.create(
      action_dim=np.prod(env.single_action_space.shape),
      encoder=encoder,
      **model_config,
      key=model_key
  )
  if model.action_dim >= 20:
    tdmpc_config.mppi_iterations += 2

  agent = TDMPC2.create(world_model=model, **tdmpc_config)
  global_step = 0

  options = ocp.CheckpointManagerOptions(
      max_to_keep=1, save_interval_steps=cfg['save_interval_steps']
  )
  checkpoint_path = os.path.join(output_dir, 'checkpoint')
  with ocp.CheckpointManager(
      checkpoint_path,
      options=options,
      item_names=('agent', 'global_step', 'buffer_state')
  ) as mngr:
    if mngr.latest_step() is not None:
      print('Checkpoint folder found, restoring from', mngr.latest_step())
      abstract_buffer_state = jax.tree.map(
          ocp.utils.to_shape_dtype_struct, replay_buffer.get_state()
      )
      restored = mngr.restore(
          mngr.latest_step(),
          args=ocp.args.Composite(
              agent=ocp.args.StandardRestore(agent),
              global_step=ocp.args.JsonRestore(),
              buffer_state=ocp.args.StandardRestore(abstract_buffer_state),
          )
      )
      agent, global_step = restored.agent, restored.global_step
      replay_buffer.restore(restored.buffer_state)
    else:
      print('No checkpoint folder found, starting from scratch')
      mngr.save(
          global_step,
          args=ocp.args.Composite(
              agent=ocp.args.StandardSave(agent),
              global_step=ocp.args.JsonSave(global_step),
              buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
          ),
      )
      mngr.wait_until_finished()

    ##############################
    # Training loop
    ##############################
    ep_count = np.zeros(env_config.num_envs, dtype=int)
    prev_logged_step = global_step
    plan = None
    observation, _ = env.reset(seed=cfg.seed)

    T = 500
    seed_steps = int(
        max(5*T, 1000) * env_config.num_envs * env_config.utd_ratio
    )
    total_num_updates = 0
    total_reanalyze_steps = 0
    pbar = tqdm.tqdm(initial=global_step, total=cfg.max_steps)
    done = np.zeros(env_config.num_envs, dtype=bool)
    for global_step in range(global_step, cfg.max_steps, env_config.num_envs):
      if global_step <= seed_steps:
        action = env.action_space.sample()
        expert_mean, expert_std = np.zeros_like(action), np.ones_like(action)
      else:
        rng, action_key = jax.random.split(rng)
        action, plan = agent.act(
            observation, prev_plan=plan, deterministic=False, key=action_key
        )
        expert_mean = plan[0][..., 0, :]
        expert_std = plan[1][..., 0, :]

      next_observation, reward, terminated, truncated, info = env.step(action)

      if np.any(~done):
        replay_buffer.insert(
            dict(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
                expert_mean=expert_mean,
                expert_std=expert_std,
                last_reanalyze=np.full(
                    env_config.num_envs, total_reanalyze_steps
                ),
            ),
            env_mask=~done
        )
      observation = next_observation

      # Handle terminations/truncations
      done = np.logical_or(terminated, truncated)
      if np.any(done):
        if plan is not None:
          plan = (
              plan[0].at[done].set(0),
              plan[1].at[done].set(agent.max_plan_std)
          )
        for ienv in range(env_config.num_envs):
          if done[ienv]:
            r = info['episode']['r'][ienv]
            l = info['episode']['l'][ienv]
            print(
                f"Episode {ep_count[ienv]}: r = {r:.2f}, l = {l}"
            )
            writer.scalar(f'episode/return', r, global_step + ienv)
            writer.scalar(f'episode/length', l, global_step + ienv)
            ep_count[ienv] += 1

      if global_step >= seed_steps:
        if global_step == seed_steps:
          print('Pre-training on seed data...')
          num_updates = seed_steps
          pretrain = True
        else:
          num_updates = max(1, int(env_config.num_envs * env_config.utd_ratio))
          pretrain = False

        rng, *update_keys = jax.random.split(rng, num_updates+1)
        log_this_step = global_step >= prev_logged_step + \
            cfg['log_interval_steps']
        if log_this_step:
          all_train_info = defaultdict(list)
          prev_logged_step = global_step

        for iupdate in range(num_updates):
          batch, batch_inds = replay_buffer.sample(
              agent.batch_size, agent.horizon, return_inds=True
          )
          agent, train_info = agent.update_world_model(
              observations=batch['observation'],
              actions=batch['action'],
              rewards=batch['reward'],
              next_observations=batch['next_observation'],
              terminated=batch['terminated'],
              truncated=batch['truncated'],
              key=update_keys[iupdate]
          )
          total_num_updates += 1

          # Reanalyze
          true_zs = train_info.pop('true_zs')
          if total_num_updates % bmpc_config.reanalyze_interval == 0:
            total_reanalyze_steps += 1
            rng, reanalyze_key = jax.random.split(rng)
            b = bmpc_config.reanalyze_batch_size
            h = bmpc_config.reanalyze_horizon
            _, reanalyzed_plan = agent.plan(
                z=true_zs[:, :b, :],
                horizon=h,
                deterministic=True,
                key=reanalyze_key
            )
            reanalyze_mean = reanalyzed_plan[0][..., 0, :]
            reanalyze_std = reanalyzed_plan[1][..., 0, :]
            # Update expert policy in buffer
            # Reshape for buffer: (T, B, A) -> (B, T, A)
            env_inds = batch_inds[0][:b, None]
            seq_inds = batch_inds[1][:b]
            replay_buffer.data['expert_mean'][seq_inds, env_inds] = \
                np.swapaxes(reanalyze_mean, 0, 1)
            replay_buffer.data['expert_std'][seq_inds, env_inds] = \
                np.swapaxes(reanalyze_std, 0, 1)
            replay_buffer.data['last_reanalyze'][seq_inds, env_inds] = \
                total_reanalyze_steps
            batch['expert_mean'][:, :b, :] = reanalyze_mean
            batch['expert_std'][:, :b, :] = reanalyze_std

          # Update policy with reanalyzed samples
          if not pretrain:
            reanalyze_age = total_reanalyze_steps - batch['last_reanalyze']
            bmpc_scale = bmpc_config.discount**reanalyze_age
            rng, policy_key = jax.random.split(rng)
            agent, policy_info = agent.update_policy(
                zs=true_zs,
                expert_mean=batch['expert_mean'],
                expert_std=batch['expert_std'],
                bmpc_scale=bmpc_scale,
                key=policy_key
            )
            train_info.update(policy_info)

          if log_this_step:
            for k, v in train_info.items():
              all_train_info[k].append(np.array(v))

        if log_this_step:
          for k, v in all_train_info.items():
            writer.scalar(f'train/{k}_mean', np.mean(v), global_step)
            writer.scalar(f'train/{k}_std', np.std(v), global_step)

        mngr.save(
            global_step,
            args=ocp.args.Composite(
                agent=ocp.args.StandardSave(agent),
                global_step=ocp.args.JsonSave(global_step),
                buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
            ),
        )

      pbar.update(env_config.num_envs)
    pbar.close()


if __name__ == '__main__':
  train()

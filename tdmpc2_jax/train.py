from collections import defaultdict
import gymnasium as gym
import numpy as np
import jax
import flax.linen as nn
from flax.metrics import tensorboard
import tqdm
from tdmpc2_jax.networks import NormedLinear
from tdmpc2_jax.common.activations import mish, simnorm
from functools import partial
from tdmpc2_jax import WorldModel, TDMPC2
from tdmpc2_jax.data import SequentialReplayBuffer
import os
import hydra
import jax.numpy as jnp
import orbax.checkpoint as ocp



@hydra.main(config_name='config', config_path='.', version_base=None)
def train(cfg: dict):
  env_config = cfg['env']
  encoder_config = cfg['encoder']
  model_config = cfg['world_model']
  tdmpc_config = cfg['tdmpc2']

  ##############################
  # Logger setup
  ##############################
  output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
  writer = tensorboard.SummaryWriter(os.path.join(output_dir, 'tensorboard'))
  writer.hparams(cfg)

  ##############################
  # Environment setup
  ##############################
  def make_env(env_id, seed):
    env = gym.make(env_id)
    env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

  vector_env_cls = gym.vector.AsyncVectorEnv if env_config.asynchronous else gym.vector.SyncVectorEnv
  env = vector_env_cls(
      [
          partial(make_env, env_config.env_id, seed)
          for seed in range(cfg.seed, cfg.seed+env_config.num_envs)
      ])
  np.random.seed(cfg.seed)
  rng = jax.random.PRNGKey(cfg.seed)

  ##############################
  # Agent setup
  ##############################
  dtype = jnp.dtype(model_config.dtype)
  rng, model_key = jax.random.split(rng, 2)
  encoder = nn.Sequential([
      NormedLinear(encoder_config.encoder_dim, activation=mish, dtype=dtype)
      for _ in range(encoder_config.num_encoder_layers-1)] + [
      NormedLinear(
          model_config.latent_dim,
          activation=partial(simnorm, simplex_dim=model_config.simnorm_dim),
          dtype=dtype)
  ])
  ##############################
  # Replay buffer setup
  ##############################
  dummy_obs, _ = env.reset()
  dummy_action = env.action_space.sample()
  dummy_next_obs, dummy_reward, dummy_term, dummy_trunc, _ = \
      env.step(dummy_action)
  replay_buffer = SequentialReplayBuffer(
      capacity=cfg.max_steps//env_config.num_envs,
      num_envs=env.num_envs,
      seed=cfg.seed,
      dummy_input=dict(
          observation=dummy_obs,
          action=dummy_action,
          reward=dummy_reward,
          next_observation=dummy_next_obs,
          terminated=dummy_term,
          truncated=dummy_trunc)
  )

  model = WorldModel.create(
      observation_space=env.get_wrapper_attr('single_observation_space'),
      action_space=env.get_wrapper_attr('single_action_space'),
      encoder_module=encoder,
      **model_config,
      key=model_key)
  agent = TDMPC2.create(world_model=model, **tdmpc_config)
  global_step = 0

  options = ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=cfg['save_interval_steps'])
  checkpoint_path = os.path.join(output_dir, 'checkpoint')
  with ocp.CheckpointManager(
      checkpoint_path, options=options, item_names=('agent', 'global_step')
  ) as mngr:
    if mngr.latest_step() is not None:
      print('Checkpoint folder found, restoring from', mngr.latest_step())
      mngr.wait_until_finished()
      restored = mngr.restore(mngr.latest_step())
      agent, global_step = restored.agent, restored.global_step
    else:
      print('No checkpoint folder found, starting from scratch')
      mngr.save(
        global_step,
        args=ocp.args.Composite(
            agent=ocp.args.StandardSave(agent),
            global_step=ocp.args.JsonSave(global_step),
        ),
      )
      mngr.wait_until_finished()

    ##############################
    # Training loop
    ##############################
    ep_info = {}
    ep_count = np.zeros(env.num_envs, dtype=int)
    prev_logged_step = global_step
    prev_plan = (
        jnp.zeros((env.num_envs, agent.horizon, agent.model.action_dim)),
        jnp.full((env.num_envs, agent.horizon,
                agent.model.action_dim), agent.max_plan_std)
    )
    observation, _ = env.reset(seed=cfg.seed)

    T = 500
    seed_steps = int(max(5*T, 1000) * env_config.num_envs * env_config.utd_ratio)
    for global_step in tqdm.tqdm(range(global_step, cfg.max_steps, env_config.num_envs)):

      if global_step <= seed_steps:
        action = env.action_space.sample()
      else:
        rng, action_key = jax.random.split(rng)
        prev_plan = (prev_plan[0],
                    jnp.full_like(prev_plan[1], agent.max_plan_std))
        action, prev_plan = agent.act(
            observation, prev_plan=prev_plan, train=True, key=action_key)

      next_observation, reward, terminated, truncated, info = env.step(action)

      # Get real final observation and store transition
      real_next_observation = next_observation.copy()
      for ienv, trunc in enumerate(truncated):
        if trunc:
          real_next_observation[ienv] = info['final_observation'][ienv]
      replay_buffer.insert(dict(
          observation=observation,
          action=action,
          reward=reward,
          next_observation=real_next_observation,
          terminated=terminated,
          truncated=truncated))
      observation = next_observation

      # Handle terminations/truncations
      done = np.logical_or(terminated, truncated)
      if np.any(done):
        prev_plan = (
            prev_plan[0].at[done].set(0),
            prev_plan[1].at[done].set(agent.max_plan_std)
        )
      if "final_info" in info:
        for ienv, final_info in enumerate(info["final_info"]):
          if final_info is None:
            continue
          print(
              f"Episode {ep_count[ienv]}: {final_info['episode']['r']}, {final_info['episode']['l']}")
          writer.scalar(f'episode/return', final_info['episode']['r'], global_step + ienv)
          writer.scalar(f'episode/length', final_info['episode']['l'], global_step + ienv)
          ep_count[ienv] += 1

      if global_step >= seed_steps:
        if global_step == seed_steps:
          print('Pre-training on seed data...')
          num_updates = seed_steps
        else:
          num_updates = max(1, int(env_config.num_envs * env_config.utd_ratio))

        rng, *update_keys = jax.random.split(rng, num_updates+1)
        log_this_step = global_step >= prev_logged_step + cfg['log_interval_steps']
        if log_this_step:
          all_train_info = defaultdict(list)
          prev_logged_step = global_step
          
        for iupdate in range(num_updates):
          batch = replay_buffer.sample(agent.batch_size, agent.horizon)
          agent, train_info = agent.update(
              observations=batch['observation'],
              actions=batch['action'],
              rewards=batch['reward'],
              next_observations=batch['next_observation'],
              terminated=batch['terminated'],
              truncated=batch['truncated'],
              key=update_keys[iupdate])

          if log_this_step:
            for k, v in train_info.items():  
              all_train_info[k].append(v)

        if log_this_step:
          for k, v in all_train_info.items(): 
            writer.scalar(f'train/{k}_mean', np.mean(v), global_step)
            writer.scalar(f'train/{k}_std', np.std(v), global_step)

        mngr.save(
            global_step,
            args=ocp.args.Composite(
                agent=ocp.args.StandardSave(agent),
                global_step=ocp.args.JsonSave(global_step),
            ),
        )


if __name__ == '__main__':
  train()

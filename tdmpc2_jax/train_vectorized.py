import gymnasium as gym
import numpy as np
import jax
import flax.linen as nn
import tqdm
from tdmpc2_jax.networks import NormedLinear
from tdmpc2_jax.common.activations import mish, simnorm
from functools import partial
from tdmpc2_jax import WorldModel, TDMPC2
from tdmpc2_jax.data import SequentialReplayBuffer
import os
import hydra
from tdmpc2_jax.wrappers.action_scale import RescaleActions
import jax.numpy as jnp

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
  seed = cfg['seed']
  max_steps = cfg['max_steps']
  encoder_config = cfg['encoder']
  model_config = cfg['world_model']
  tdmpc_config = cfg['tdmpc2']

  ##############################
  # Environment setup
  ##############################
  def make_env(env_id, seed):
    def thunk():
      env = gym.make(env_id)
      env = RescaleActions(env)
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env

    return thunk

  num_envs = 4
  steps_per_update = 4
  T = 500
  seed_steps = max(5*T, 1000) * num_envs
  env = gym.vector.AsyncVectorEnv(
      [make_env('Humanoid-v4', seed) for seed in range(seed, seed+num_envs)])
  np.random.seed(seed)
  rng = jax.random.PRNGKey(seed)

  ##############################
  # Agent setup
  ##############################
  dtype = jnp.dtype(model_config['dtype'])
  rng, model_key = jax.random.split(rng, 2)
  encoder = nn.Sequential(
      [
          NormedLinear(encoder_config['encoder_dim'],
                       activation=mish, dtype=dtype)
          for _ in range(encoder_config['num_encoder_layers']-1)
      ] +
      [
          NormedLinear(
              model_config['latent_dim'],
              activation=partial(
                  simnorm, simplex_dim=model_config['simnorm_dim']),
              dtype=dtype)
      ])

  model = WorldModel.create(
      observation_space=env.single_observation_space,
      action_space=env.single_action_space,
      encoder_module=encoder,
      **model_config,
      key=model_key)
  agent = TDMPC2.create(world_model=model, **tdmpc_config)

  ##############################
  # Replay buffer setup
  ##############################
  dummy_obs, _ = env.reset()
  dummy_action = env.action_space.sample()
  dummy_next_obs, dummy_reward, dummy_term, dummy_trunc, _ = \
      env.step(dummy_action)
  replay_buffer = SequentialReplayBuffer(
      capacity=max_steps//num_envs,
      num_envs=env.num_envs,
      dummy_input=dict(
          observation=dummy_obs,
          action=dummy_action,
          reward=dummy_reward,
          next_observation=dummy_next_obs,
          terminated=dummy_term,
          truncated=dummy_trunc,
      ),
      seed=seed)

  ##############################
  # Training loop
  ##############################
  ep_info = {}
  ep_count = np.zeros(env.num_envs, dtype=int)
  prev_mean = jnp.zeros((env.num_envs, agent.horizon, agent.model.action_dim))
  observation, _ = env.reset(seed=seed)
  for i in tqdm.tqdm(range(0, max_steps, num_envs), smoothing=0.1):

    if i <= seed_steps:
      action = env.action_space.sample()
    else:
      rng, action_key = jax.random.split(rng)
      action, prev_mean = agent.act(
          observation, prev_mean, train=True, key=action_key)

    next_observation, reward, terminated, truncated, info = env.step(action)

    # Get real final observation and store transition
    real_next_observation = next_observation.copy()
    for itrunc in range(len(truncated)):
      if truncated[itrunc]:
        real_next_observation[itrunc] = info['final_observation'][itrunc]
    replay_buffer.insert(dict(
        observation=observation,
        action=action,
        reward=reward,
        next_observation=real_next_observation,
        terminated=terminated,
        truncated=truncated))
    observation = next_observation

    # Handle terminations/truncations
    if "final_info" in info:
      for ienv, final_info in enumerate(info["final_info"]):
        if final_info is None:
          continue
        # Reset the plan warm start for this env
        prev_mean = prev_mean.at[ienv].set(0)
        print(
            f"Episode {ep_count[ienv]}: {final_info['episode']['r']}, {final_info['episode']['l']}")
        ep_count[ienv] += 1

    if i >= seed_steps:
      if i == seed_steps:
        print('Pre-training on seed data...')
        num_updates = seed_steps
      else:
        num_updates = max(1, num_envs // steps_per_update)

      rng, *update_keys = jax.random.split(rng, num_updates+1)
      for j in range(num_updates):
        batch = replay_buffer.sample(agent.batch_size, agent.horizon)
        agent, train_info = agent.update(
            observations=batch['observation'],
            actions=batch['action'],
            rewards=batch['reward'],
            next_observations=batch['next_observation'],
            terminated=batch['terminated'],
            truncated=batch['truncated'],
            key=update_keys[j])
        # TODO: Log train info


if __name__ == '__main__':
  train()

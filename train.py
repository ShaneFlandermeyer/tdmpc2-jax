import gymnasium as gym
import numpy as np
import jax
import flax.linen as nn
import tqdm
from tdmpc2_jax.networks import NormedLinear
from tdmpc2_jax.common.activations import mish, simnorm
from functools import partial
from tdmpc2_jax import WorldModel, TDMPC2
from tdmpc2_jax.data import EpisodicReplayBuffer
import os

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

if __name__ == '__main__':
  # Args
  seed = np.random.randint(2 ** 32 - 1)
  max_steps = int(1e6)

  # Make env
  T = 1000
  seed_steps = max(5*T, 1000)
  env = gym.make("HalfCheetah-v4", max_episode_steps=T)
  env = gym.wrappers.RecordEpisodeStatistics(env)
  env.action_space.seed(seed)
  env.observation_space.seed(seed)
  np.random.seed(seed)
  rng = jax.random.PRNGKey(seed)

  # Make agent
  model_config = {
      'observation_space': env.observation_space,
      'action_space': env.action_space,
      'encoder_dim': 256,
      'mlp_dim': 512,
      'latent_dim': 512,
      'value_dropout': 0.01,
      'num_value_nets': 5,
      'num_bins': 101,
      'symlog_min': -10,
      'symlog_max': 10,
      'simnorm_dim': 8,
      'learning_rate': 3e-4,
      'encoder_learning_rate': 1e-4,
      'tabulate': True,
  }
  tdmpc_config = {
      # Planning
      'horizon': 3,
      'mppi_iterations': 6,
      'population_size': 512,
      'policy_prior_samples': 24,
      'num_elites': 64,
      'min_plan_std': 0.05,
      'max_plan_std': 2,
      'temperature': 0.5,
      # Optimization
      'batch_size': 256,
      'discount': 0.99,
      'rho': 0.5,
      'consistency_coef': 10,
      'reward_coef': 0.1,
      'value_coef': 0.1,
      'entropy_coef': 1e-4,
      'tau': 0.01,
  }
  rng, model_key = jax.random.split(rng, 2)

  encoder = nn.Sequential([
      NormedLinear(model_config['encoder_dim'], activation=mish),
      NormedLinear(model_config['latent_dim'],
                   activation=partial(simnorm,
                                      simplex_dim=model_config['simnorm_dim']))
  ])

  model = WorldModel.create(
      **model_config, encoder_module=encoder, key=model_key)
  agent = TDMPC2.create(world_model=model, **tdmpc_config)

  replay_buffer = EpisodicReplayBuffer(
      capacity=max_steps,
      dummy_input=dict(
          observations=env.observation_space.sample(),
          actions=env.action_space.sample(),
          rewards=1.0,
      ),
      seed=seed)

  # Training loop
  ep_count = 0
  prev_plan = None
  observation, _ = env.reset(seed=seed)
  for i in tqdm.tqdm(range(max_steps), smoothing=0.1):
    if i <= seed_steps:
      action = env.action_space.sample()
    else:
      rng, action_key = jax.random.split(rng)
      action, prev_plan = agent.act(
          observation, prev_plan, train=True, key=action_key)

    observation, reward, terminated, truncated, info = env.step(action)
    replay_buffer.insert(dict(
        observations=observation,
        actions=action,
        rewards=reward),
        episode_index=ep_count)

    if terminated or truncated:
      observation, _ = env.reset(seed=seed)
      prev_plan = None

      print("Episode reward:", info['episode']['r'])
      ep_count += 1

    if i >= seed_steps:
      if i == seed_steps:
        print('Pre-training on seed data...')
        num_updates = seed_steps
      else:
        num_updates = 1

      rng, *update_keys = jax.random.split(rng, num_updates+1)
      for j in range(num_updates):
        batch = replay_buffer.sample(
            tdmpc_config['batch_size'], tdmpc_config['horizon']+1)
        obs = batch['observations']
        action = batch['actions'][1:]
        reward = batch['rewards'][1:]
        agent, train_info = agent.update(
            obs, action, reward, key=update_keys[j])

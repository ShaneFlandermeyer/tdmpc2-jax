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
import hydra
from tdmpc2_jax.wrappers.action_repeat import RepeatAction

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
  seed = cfg['seed']
  max_steps = cfg['max_steps']
  encoder_config = cfg['encoder']
  model_config = cfg['world_model']
  tdmpc_config = cfg['tdmpc2']

  T = 500
  seed_steps = max(5*T, 1000)
  env = gym.make("HalfCheetah-v4")
  # env = RepeatAction(env, repeat=2)
  env = gym.wrappers.RecordEpisodeStatistics(env)
  env.action_space.seed(seed)
  env.observation_space.seed(seed)
  np.random.seed(seed)
  rng = jax.random.PRNGKey(seed)

  rng, model_key = jax.random.split(rng, 2)

  encoder = nn.Sequential(
      [NormedLinear(encoder_config['encoder_dim'], activation=mish)
       for _ in range(encoder_config['num_encoder_layers']-1)] + [
          NormedLinear(model_config['latent_dim'],
                       activation=partial(simnorm,
                                          simplex_dim=model_config['simnorm_dim']))
      ])

  model = WorldModel.create(
      observation_space=env.observation_space,
      action_space=env.action_space,
      encoder_module=encoder,
      **model_config,
      key=model_key)
  agent = TDMPC2.create(world_model=model, **tdmpc_config)

  replay_buffer = EpisodicReplayBuffer(
      capacity=max_steps,
      dummy_input=dict(
          observations=env.observation_space.sample(),
          actions=env.action_space.sample(),
          rewards=1.0,
          dones=True,
      ),
      seed=seed)

  # Training loop
  c_loss, r_loss, v_loss, loss = 0, 0, 0, 0
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
        rewards=reward,
        dones=terminated),
        episode_index=ep_count)

    if terminated or truncated:
      observation, _ = env.reset(seed=seed)
      prev_plan = None

      r = info['episode']['r']
      l = info['episode']['l']
      print(f"Episode: r = {r}, l = {l}")
      print(
          f"Losses: c = {c_loss/l} r = {r_loss/l} v = {v_loss/l} total = {loss/l}")

      c_loss, r_loss, v_loss, loss = 0, 0, 0, 0
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
        done = batch['dones']
        action = batch['actions'][1:]
        reward = batch['rewards'][1:]
        agent, train_info = agent.update(
            obs, action, reward, done, key=update_keys[j])
        c_loss += train_info['consistency_loss']
        r_loss += train_info['reward_loss']
        v_loss += train_info['value_loss']
        loss += train_info['total_loss']


if __name__ == '__main__':
  train()

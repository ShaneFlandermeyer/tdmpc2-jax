import gymnasium as gym
import numpy as np
from typing import *
import jax
from collections import deque


class SequentialReplayBuffer():

  def __init__(self,
               capacity: int,
               dummy_input: Dict,
               num_envs: int = 1,
               seed: Optional[int] = None,
               ):
    """
    Sequential replay buffer with support for parallel environments. 

    To simplify the implementation and speed up sampling, episode boundaries are NOT respected. i.e., the sampled subsequences may span multiple episodes. Any code using this buffer should handle this with termination/truncation signals

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store PER PARALLEL ENVIRONMENT
    dummy_input : Dict
        Example input from the environment. Used to determine the shape and dtype of the data to store
    num_envs : int, optional
        Number of parallel environments used for data collection, by default 1
    seed : Optional[int], optional
        Seed for sampling, by default None
    """
    self.capacity = capacity
    self.num_envs = num_envs
    self.data = jax.tree_map(lambda x: np.empty(
        (capacity,) + np.asarray(x).shape, np.asarray(x).dtype), dummy_input)

    self.size = 0
    self.current_ind = 0

    self.np_random = np.random.RandomState(seed=seed)

  def __len__(self):
    return self.size

  def insert(self, data: Dict) -> None:
    # Insert the data
    jax.tree_map(lambda x, y: x.__setitem__(self.current_ind, y),
                 self.data, data)

    self.current_ind = (self.current_ind + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

  def sample(self, batch_size: int, sequence_length: int) -> Dict:
    """
    Sample a batch uniformly across environments and time steps

    Parameters
    ----------
    batch_size : int
    sequence_length : int

    Returns
    -------
    Dict
    """
    env_inds = self.np_random.randint(0, self.num_envs, size=(batch_size, 1))

    if self.size < self.capacity:
      # This requires special handling to avoid sampling from the empty part of the buffer. Once the buffer is full, we can sample to our heart's content
      buffer_starts = self.np_random.randint(
          0, self.size - sequence_length, size=(batch_size, 1))
      sequence_inds = buffer_starts + np.arange(sequence_length)
    else: 
      buffer_starts = self.np_random.randint(
          0, self.size, size=(batch_size, 1))
      sequence_inds = buffer_starts + np.arange(sequence_length)
      sequence_inds = sequence_inds % self.capacity

    # Sample from buffer and convert from (batch, time, *) to (time, batch, *)
    batch = jax.tree_map(lambda x: np.swapaxes(
        x[sequence_inds, env_inds], 0, 1), self.data)

    return batch


if __name__ == '__main__':
  # def make_env():
  #   def thunk():
  #     return gym.make('CartPole-v1')
  env = gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v1')] * 2)
  dummy_input = {'obs': env.observation_space.sample()}
  rb = SequentialReplayBuffer(100, dummy_input, num_envs=2)

  obs, _ = env.reset()
  ep_count = np.zeros(env.num_envs, dtype=int)
  for i in range(10):
    action = env.action_space.sample()
    obs, reward, term, trunc, _ = env.step(action)
    rb.insert({'obs': obs})
    if i > 3:
      print(rb.sample(2, 3)['obs'].shape)

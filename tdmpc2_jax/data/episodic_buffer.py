import gymnasium as gym
import numpy as np
from typing import *
import jax


class EpisodicReplayBuffer():

  def __init__(self, capacity: int, dummy_input: Dict, seed: Optional[int] = None):
    self.capacity = capacity
    self.data = jax.tree_map(lambda x: np.empty(
        (capacity,) + np.asarray(x).shape, np.asarray(x).dtype), dummy_input)
    self.episode_inds = np.empty(capacity, dtype=int)

    self.size = 0
    self.current_ind = 0

    self.np_random = np.random.RandomState(seed=seed)

  def __len__(self):
    return self.size

  def insert(self, data: Dict, episode_index: int) -> None:
    jax.tree_map(lambda x, y: x.__setitem__(self.current_ind, y),
                 self.data, data)
    self.episode_inds[self.current_ind] = episode_index
    self.current_ind = (self.current_ind + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

  def sample(self, batch_size: int, sequence_length: int) -> Dict:
    # Sample subsequences uniformly between episodes
    ids, inds, counts = np.unique(
        self.episode_inds[:self.size], return_index=True, return_counts=True)

    # Uniformly sample episodes, then sample a start index for the sequence within each episode
    episode_inds = self.np_random.choice(
        ids[counts >= sequence_length], size=batch_size)
    episode_starts = np.round(
        self.np_random.rand(batch_size) * (counts[episode_inds] - sequence_length + 1)).astype(int)
    buffer_starts = inds[episode_inds] + episode_starts
    sequence_inds = buffer_starts[:, None] + np.arange(sequence_length)

    return jax.tree_map(lambda x: np.swapaxes(x[sequence_inds], 0, 1),
                        self.data)


if __name__ == '__main__':
  env = gym.make('CartPole-v1')
  dummy_input = {'obs': env.observation_space.sample()}

  rb = EpisodicReplayBuffer(
      capacity=10, dummy_input=dummy_input)

  obs, _ = env.reset()
  action = env.action_space.sample()
  rb.insert({'obs': obs}, 0)
  rb.insert({'obs': env.step(action)[0]}, 0)
  rb.insert({'obs': env.step(action)[0]}, 1)
  rb.insert({'obs': env.step(action)[0]}, 1)
  rb.insert({'obs': env.step(action)[0]}, 2)

  rb.sample(3, 2)

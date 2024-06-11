import gymnasium as gym
import numpy as np
from typing import *
import jax
from collections import deque


class EpisodicReplayBuffer():

  def __init__(self,
               capacity: int,
               dummy_input: Dict,
               seed: Optional[int] = None,
               respect_episode_boundaries: bool = True):
    self.capacity = capacity
    self.data = jax.tree.map(lambda x: np.empty(
        (capacity,) + np.asarray(x).shape, np.asarray(x).dtype), dummy_input)

    self.respect_episode_boundaries = respect_episode_boundaries
    self.last_episode_ind = None
    self.episode_starts = deque()
    self.episode_counts = deque()

    self.size = 0
    self.current_ind = 0

    self.np_random = np.random.RandomState(seed=seed)

  def __len__(self):
    return self.size

  def insert(self, data: Dict, episode_index: int) -> None:
    # Insert the data
    jax.tree.map(lambda x, y: x.__setitem__(self.current_ind, y),
                 self.data, data)

    # Remove the oldest episode information if it gets overwritten
    if self.size == self.capacity:
      self.episode_counts[0] -= 1
      if self.episode_counts[0] == 0:
        self.episode_starts.popleft()
        self.episode_counts.popleft()

    # Increment episode information
    if episode_index == self.last_episode_ind:
      self.episode_counts[-1] += 1
    else:
      self.last_episode_ind = episode_index
      self.episode_starts.append(self.current_ind)
      self.episode_counts.append(1)

    self.current_ind = (self.current_ind + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

  def sample(self, batch_size: int, sequence_length: int) -> Dict:
    if self.respect_episode_boundaries:
      episode_starts, counts = np.array(
          self.episode_starts), np.array(self.episode_counts)
      valid = counts >= sequence_length
      episode_starts, counts = episode_starts[valid], counts[valid]

      # Sample subsequences uniformly within episodes
      episode_inds = self.np_random.randint(0, len(counts), size=batch_size)
      sequence_starts = np.round(self.np_random.rand(batch_size) *
                                 (counts[episode_inds] - sequence_length)
                                 ).astype(int)
      buffer_starts = episode_starts[episode_inds] + sequence_starts
    else:
      buffer_starts = self.np_random.randint(0, self.size, size=batch_size)

    sequence_inds = buffer_starts[:, None] + np.arange(sequence_length)
    sequence_inds = sequence_inds % self.capacity

    return jax.tree.map(lambda x: np.swapaxes(x[sequence_inds], 0, 1),
                        self.data)


if __name__ == '__main__':
  env = gym.make('Walker2d-v4')
  dummy_input = {'obs': env.observation_space.sample()}
  rb = EpisodicReplayBuffer(100, dummy_input)

  obs, _ = env.reset()
  ep_count = 0
  while ep_count < 5:
    action = env.action_space.sample()
    obs, reward, term, trunc, _ = env.step(action)
    rb.insert({'obs': obs}, ep_count)
    if term or trunc:
      obs, _ = env.reset()
      rb.sample(10, 3)
      ep_count += 1

  rb.sample(3, 2)

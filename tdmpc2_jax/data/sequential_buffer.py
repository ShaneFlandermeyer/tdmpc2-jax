import gymnasium as gym
import numpy as np
from typing import *
import jax
from collections import deque
from jaxtyping import PyTree


class SequentialReplayBuffer():

  def __init__(self,
               capacity: int,
               dummy_input: Dict,
               num_envs: int = 1,
               vectorized: bool = True,
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
    self.vectorized = vectorized
    if vectorized:
      self.num_envs = num_envs
    else:
      assert num_envs == 1, "num_envs must be 1 for non-vectorized buffer"
      self.num_envs = 1

    self.data = jax.tree.map(
        lambda x: np.zeros(
            (capacity,) + np.asarray(x).shape, np.asarray(x).dtype
        ), dummy_input
    )

    # Size and index counter for each environment buffer
    if vectorized:
      self.size = np.zeros(num_envs, dtype=int)
      self.current_ind = np.zeros(num_envs, dtype=int)
    else:
      self.size = 0
      self.current_ind = 0

    self.np_random = np.random.default_rng(seed=seed)

  def insert(self,
             data: PyTree,
             env_mask: Optional[np.ndarray] = None
             ) -> None:
    """
    Insert data into the buffer

    Parameters
    ----------
    data : PyTree
        Data to insert
    env_mask : Optional[np.ndarray], optional
        A boolean mask of size self.num_envs, which specifies which env buffers receive new data. If None, all envs receive data, by default None
    """
    if self.vectorized:
      self._insert_vectorized(data, env_mask)
    else:
      self._insert(data)

  def _insert(self, data: PyTree) -> None:
    # Insert data for the specified envs
    jax.tree.map(
        lambda x, y: x.__setitem__(self.current_ind, y), self.data, data
    )

    # Update buffer state
    self.current_ind = (self.current_ind + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

  def _insert_vectorized(self,
                         data: PyTree,
                         env_mask: Optional[np.ndarray] = None
                         ) -> None:
    # Insert data for the specified envs
    if env_mask is None:
      env_mask = np.ones(self.num_envs, dtype=bool)

    def masked_set(x, y):
      x[self.current_ind, env_mask] = y[env_mask]
    jax.tree.map(masked_set, self.data, data)

    # Update buffer state
    self.current_ind[env_mask] = (
        self.current_ind[env_mask] + 1
    ) % self.capacity
    self.size[env_mask] = np.clip(self.size[env_mask] + 1, 0, self.capacity)

  def sample(
      self,
      batch_size: int,
      sequence_length: int,
      return_inds: bool = False,
  ) -> Union[PyTree, Tuple[PyTree, Tuple[np.ndarray]]]:
    """
    Sample a batch of sequences from the buffer.

    Sequences are drawn uniformly from each environment buffer, and they may cross episode boundaries.

    Parameters
    ----------
    batch_size : int
    sequence_length : int
    return_inds : bool
        If True, also returns

    Returns
    -------
    Union[PyTree, Tuple[PyTree, Tuple[np.ndarray]]]
        The sampled batch. If return_inds is True, also returns the sampled indices in the batch/time dimensions
    """

    if self.vectorized:
      batch, inds = self._sample_vectorized(batch_size, sequence_length)
    else:
      batch, inds = self._sample(batch_size, sequence_length)

    if return_inds:
      return batch, inds
    else:
      return batch

  def _sample(self, batch_size: int, sequence_length: int) -> PyTree:
    # Sample envs and start indices
    start_inds = self.np_random.integers(
        low=0, high=self.size - sequence_length,
        size=batch_size,
    )
    # Handle wrapping: For wrapped buffers, we define the current pointer index as 0 to avoid stepping into an unrelated trajectory
    start_inds = (
        start_inds - (self.size - self.current_ind)
    ) % self.capacity

    # Sample from buffer and convert from (batch, time, *) to (time, batch, *)
    sequence_inds = (
        start_inds[:, None] + np.arange(sequence_length)
    ) % self.capacity
    batch = jax.tree.map(
        lambda x: np.swapaxes(x[sequence_inds], 0, 1),
        self.data
    )

    return batch, (sequence_inds)

  def _sample_vectorized(self, batch_size: int, sequence_length: int) -> PyTree:
    # Sample envs and start indices
    env_inds = self.np_random.integers(
        low=0, high=self.num_envs,
        size=batch_size
    )
    start_inds = self.np_random.integers(
        low=0, high=self.size[env_inds] - sequence_length,
        size=batch_size,
    )
    # Handle wrapping: For wrapped buffers, we define the current pointer index as 0 to avoid stepping into an unrelated trajectory
    start_inds = (
        start_inds - (self.size[env_inds] - self.current_ind[env_inds])
    ) % self.capacity

    # Sample from buffer and convert from (batch, time, *) to (time, batch, *)
    sequence_inds = (
        start_inds[:, None] + np.arange(sequence_length)
    ) % self.capacity
    batch = jax.tree.map(
        lambda x: np.swapaxes(x[sequence_inds, env_inds[:, None]], 0, 1),
        self.data
    )

    return batch, (env_inds, sequence_inds)

  def get_state(self) -> Dict:
    return {
        'current_ind': self.current_ind,
        'size': self.size,
        'data': self.data,
    }

  def restore(self, state: Dict) -> None:
    self.current_ind = state['current_ind']
    self.size = state['size']
    self.data = state['data']

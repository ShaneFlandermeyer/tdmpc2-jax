import gymnasium as gym
import numpy as np
from typing import *
import jax
import jax.numpy as jnp
from collections import deque
from functools import partial
from jaxtyping import PyTree


@partial(
    jax.jit,
    static_argnames=(
        'num_updates',
        'batch_size',
        'sequence_length',
        'strict_episode_boundaries',
        'max_sample_attempts',
    ),
)
def sample_many_from_state(
    state: Dict[str, PyTree],
    *,
    num_updates: int,
    batch_size: int,
    sequence_length: int,
    strict_episode_boundaries: bool = True,
    max_sample_attempts: int = 8,
) -> Tuple[Dict[str, PyTree], PyTree]:
  """Pure JAX sampler for vectorized replay buffer state."""
  data = state['data']
  size = state['size']
  current_ind = state['current_ind']
  rng_key = state['rng_key']

  capacity = jax.tree.leaves(data)[0].shape[0]
  num_envs = size.shape[0]
  next_rng_key, key_env, key_start = jax.random.split(rng_key, 3)

  attempt_shape = (int(max_sample_attempts), num_updates, batch_size)
  env_candidates = jax.random.randint(
      key_env,
      shape=attempt_shape,
      minval=0,
      maxval=num_envs,
  )
  available = jnp.maximum(size[env_candidates] - sequence_length + 1, 1)
  logical_start_candidates = jnp.floor(
      jax.random.uniform(key_start, shape=attempt_shape) * available
  ).astype(jnp.int32)
  start_candidates = (
      logical_start_candidates -
      (size[env_candidates] - current_ind[env_candidates])
  ) % capacity
  sequence_candidates = (
      start_candidates[..., None] + jnp.arange(sequence_length, dtype=jnp.int32)
  ) % capacity

  has_done_keys = (
      isinstance(data, dict) and
      'terminated' in data and
      'truncated' in data
  )
  if strict_episode_boundaries and has_done_keys:
    done_candidates = jnp.logical_or(
        data['terminated'][sequence_candidates, env_candidates[..., None]],
        data['truncated'][sequence_candidates, env_candidates[..., None]],
    )
    valid_candidates = ~jnp.any(done_candidates[..., :-1], axis=-1)
    first_valid_attempt = jnp.argmax(valid_candidates, axis=0)
  else:
    first_valid_attempt = jnp.zeros(
        (num_updates, batch_size),
        dtype=jnp.int32,
    )

  gather_attempt = first_valid_attempt[None, ...]
  env_inds = jnp.take_along_axis(env_candidates, gather_attempt, axis=0)[0]
  start_inds = jnp.take_along_axis(start_candidates, gather_attempt, axis=0)[0]
  sequence_inds = (
      start_inds[..., None] + jnp.arange(sequence_length, dtype=jnp.int32)
  ) % capacity
  batch = jax.tree.map(
      lambda x: jnp.swapaxes(
          x[sequence_inds, env_inds[..., None]],
          1,
          2,
      ),
      data,
  )

  new_state = {
      'current_ind': current_ind,
      'size': size,
      'data': data,
      'rng_key': next_rng_key,
  }
  return new_state, batch


@partial(
    jax.jit,
    static_argnames=(
        'batch_size',
        'sequence_length',
        'strict_episode_boundaries',
        'max_sample_attempts',
    ),
)
def sample_from_state(
    state: Dict[str, PyTree],
    *,
    batch_size: int,
    sequence_length: int,
    strict_episode_boundaries: bool = True,
    max_sample_attempts: int = 8,
) -> Tuple[Dict[str, PyTree], PyTree]:
  state, batch = sample_many_from_state(
      state,
      num_updates=1,
      batch_size=batch_size,
      sequence_length=sequence_length,
      strict_episode_boundaries=strict_episode_boundaries,
      max_sample_attempts=max_sample_attempts,
  )
  return state, jax.tree.map(lambda x: x[0], batch)


@jax.jit
def insert_into_state(
    state: Dict[str, PyTree],
    data: PyTree,
    mask: jax.Array,
) -> Dict[str, PyTree]:
  buffer_data = state['data']
  current_ind = state['current_ind']
  size = state['size']
  mask = jnp.asarray(mask, dtype=bool)
  env_inds = jnp.arange(current_ind.shape[0], dtype=jnp.int32)
  capacity = jax.tree.leaves(buffer_data)[0].shape[0]

  def broadcast_mask(mask_array: jax.Array, target_ndim: int) -> jax.Array:
    return jnp.reshape(
        mask_array,
        mask_array.shape + (1,) * (target_ndim - mask_array.ndim),
    )

  def masked_set(x, y):
    y = jnp.asarray(y)
    current_values = x[current_ind, env_inds]
    updates = jnp.where(
        broadcast_mask(mask, y.ndim),
        y,
        current_values,
    )
    return x.at[current_ind, env_inds].set(updates)

  new_buffer_data = jax.tree.map(masked_set, buffer_data, data)
  new_current_ind = jnp.where(
      mask,
      (current_ind + 1) % capacity,
      current_ind,
  )
  new_size = jnp.where(
      mask,
      jnp.minimum(size + 1, capacity),
      size,
  )
  return {
      'current_ind': new_current_ind,
      'size': new_size,
      'data': new_buffer_data,
      'rng_key': state['rng_key'],
  }


class SequentialReplayBuffer():

  def __init__(self,
               capacity: int,
               dummy_input: Dict,
               num_envs: int = 1,
               vectorized: bool = False,
               seed: Optional[int] = None,
               ):
    """
    Sequential replay buffer with support for parallel environments.

    To simplify the implementation and speed up sampling, episode boundaries are NOT respected. i.e., the sampled subsequences may span multiple episodes. Any code using this buffer should handle this with termination/truncation signals

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store in the overall buffer
    dummy_input : Dict
        Example input from the environment. Used to determine the shape and dtype of the data to store
    num_envs : int, optional
        Number of parallel environments used for data collection, by default 1
    seed : Optional[int], optional
        Seed for sampling, by default None
    """

    self.vectorized = vectorized
    self.num_envs = num_envs
    self.capacity = capacity // num_envs
    self.size = jnp.zeros(num_envs, dtype=jnp.int32)
    self.current_ind = jnp.zeros(num_envs, dtype=jnp.int32)

    self.data = jax.tree.map(
        lambda x: jnp.zeros(
            (self.capacity,) + np.asarray(x).shape, np.asarray(x).dtype
        ), dummy_input
    )
    self.rng_key = jax.random.PRNGKey(0 if seed is None else int(seed))

  def _split_rng(self, num: int = 1):
    keys = jax.random.split(self.rng_key, num + 1)
    self.rng_key = keys[0]
    if num == 1:
      return keys[1]
    return keys[1:]

  @staticmethod
  def _broadcast_mask(mask: jax.Array, target_ndim: int) -> jax.Array:
    return jnp.reshape(mask, mask.shape + (1,) * (target_ndim - mask.ndim))

  def insert(self,
             data: PyTree,
             mask: Optional[np.ndarray] = None
             ) -> None:
    """
    Insert data into the buffer

    Parameters
    ----------
    data : PyTree
        Data to insert
    mask : Optional[np.ndarray], optional
        A boolean mask of size self.num_envs, which specifies which env buffers receive new data. If None, all envs receive data, by default None
    """
    # Insert data for the specified envs
    if mask is None:
      mask = jnp.ones(self.num_envs, dtype=bool)
    else:
      mask = jnp.asarray(mask, dtype=bool)
    env_inds = jnp.arange(self.num_envs, dtype=jnp.int32)

    if self.vectorized:
      def masked_set(x, y):
        y = jnp.asarray(y)
        current_values = x[self.current_ind, env_inds]
        updates = jnp.where(
            self._broadcast_mask(mask, y.ndim),
            y,
            current_values,
        )
        return x.at[self.current_ind, env_inds].set(updates)
      self.data = jax.tree.map(masked_set, self.data, data)
    else:
      self.data = jax.tree.map(
          lambda x, y: x.at[self.current_ind[0]].set(jnp.asarray(y)),
          self.data,
          data,
      )

    # Update buffer state
    self.current_ind = jnp.where(
        mask,
        (self.current_ind + 1) % self.capacity,
        self.current_ind,
    )
    self.size = jnp.where(
        mask,
        jnp.minimum(self.size + 1, self.capacity),
        self.size,
    )

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
    key = self._split_rng()
    available = jnp.maximum(self.size[0] - sequence_length + 1, 1)
    start_inds = jnp.floor(
        jax.random.uniform(key, shape=(batch_size,)) * available
    ).astype(jnp.int32)
    # Handle wrapping: For wrapped buffers, we define the current pointer index as the start of the buffer to avoid stepping into invalid data
    start_inds = (start_inds - (self.size[0] - self.current_ind[0])) % self.capacity

    # Sample from buffer and convert from (batch, time, *) to (time, batch, *)
    sequence_inds = (
        start_inds[:, None] + jnp.arange(sequence_length, dtype=jnp.int32)
    ) % self.capacity
    batch = jax.tree.map(
        lambda x: jnp.swapaxes(x[sequence_inds], 0, 1),
        self.data
    )

    return batch, (sequence_inds)

  def _sample_vectorized(self, batch_size: int, sequence_length: int) -> PyTree:
    batch, inds = self._sample_vectorized_many(
        num_updates=1,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    batch = jax.tree.map(lambda x: x[0], batch)
    env_inds, sequence_inds = inds
    return batch, (env_inds[0], sequence_inds[0])

  def _sample_vectorized_many(self,
                              num_updates: int,
                              batch_size: int,
                              sequence_length: int):
    key_env, key_start = self._split_rng(2)
    env_inds = jax.random.randint(
        key_env,
        shape=(num_updates, batch_size),
        minval=0,
        maxval=self.num_envs,
    )
    available = jnp.maximum(self.size[env_inds] - sequence_length + 1, 1)
    start_inds = jnp.floor(
        jax.random.uniform(key_start, shape=(num_updates, batch_size)) * available
    ).astype(jnp.int32)
    # Handle wrapping: For wrapped buffers, we define the current pointer index as the start of the buffer to avoid stepping into invalid data
    start_inds = (
        start_inds - (self.size[env_inds] - self.current_ind[env_inds])
    ) % self.capacity

    # Sample from buffer and convert from (batch, time, *) to (time, batch, *)
    sequence_inds = (
        start_inds[..., None] + jnp.arange(sequence_length, dtype=jnp.int32)
    ) % self.capacity
    batch = jax.tree.map(
        lambda x: jnp.swapaxes(
            x[sequence_inds, env_inds[..., None]],
            1,
            2,
        ),
        self.data
    )

    return batch, (env_inds, sequence_inds)

  def sample_many(
      self,
      num_updates: int,
      batch_size: int,
      sequence_length: int,
  ) -> PyTree:
    if self.vectorized:
      batch, _ = self._sample_vectorized_many(
          num_updates=num_updates,
          batch_size=batch_size,
          sequence_length=sequence_length,
      )
      return batch
    batch, _ = self._sample(batch_size=batch_size, sequence_length=sequence_length)
    return jax.tree.map(lambda x: jnp.repeat(x[None, ...], num_updates, axis=0), batch)

  def get_state(self) -> Dict:
    return {
        'current_ind': self.current_ind,
        'size': self.size,
        'data': self.data,
        'rng_key': self.rng_key,
    }

  def restore(self, state: Dict) -> None:
    self.current_ind = state['current_ind']
    self.size = state['size']
    self.data = state['data']
    self.rng_key = state.get('rng_key', self.rng_key)

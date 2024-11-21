from functools import partial
from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
import jax


class NormedLinear(nn.Module):
  features: int
  activation: Callable[[jax.Array], jax.Array] = None
  dropout_rate: Optional[float] = None
  norm: nn.Module = nn.LayerNorm

  kernel_init: Callable = nn.initializers.truncated_normal(stddev=0.02)
  dtype: jnp.dtype = jnp.float32  # Switch this to bfloat16 for speed
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               x: jax.Array,
               train: bool = True) -> jax.Array:
    x = nn.Dense(features=self.features,
                 kernel_init=self.kernel_init,
                 bias_init=nn.initializers.zeros_init(),
                 dtype=self.dtype,
                 param_dtype=self.param_dtype)(x)

    x = self.norm(dtype=self.dtype)(x)
    if self.activation is not None:
      x = self.activation(x)

    if self.dropout_rate is not None and self.dropout_rate > 0:
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    return x

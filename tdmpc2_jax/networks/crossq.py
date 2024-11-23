from typing import Callable, Optional

import flax
import flax.linen as nn
import jax.numpy as jnp
from flax.training.train_state import TrainState

from tdmpc2_jax.common.activations import mish
from tdmpc2_jax.common.layers import BatchRenorm


class BatchNormTrainState(TrainState):  # type: ignore[misc]
  batch_stats: flax.core.FrozenDict  # type: ignore[misc]


class CrossQCritic(nn.Module):
  mlp_dim: int
  num_layers: int
  activation: Optional[Callable] = None
  dropout_rate: Optional[float] = None
  kernel_init: Callable = nn.initializers.xavier_normal()

  @nn.compact
  def __call__(self, z: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    x = z
    for _ in range(self.num_layers):
      x = nn.Dense(self.mlp_dim, kernel_init=self.kernel_init)(x)
      x = BatchRenorm(use_running_average=not train)(x)
      x = self.activation(x)
      if self.dropout_rate is not None:
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    x = nn.Dense(1, kernel_init=nn.initializers.zeros)(x)

    return x

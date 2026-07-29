import flax.linen as nn


class Ensemble(nn.Module):
  base_module: nn.Module
  num: int = 2

  @nn.compact
  def __call__(self, x, train: bool = True):
    ensemble = nn.vmap(
        self.base_module,
        variable_axes={'params': 0},
        split_rngs={
            'params': True,
            'dropout': True
        },
        in_axes=None,
        out_axes=0,
        axis_size=self.num
    )
    # Flax lifted vmap ignores kwargs, so pass train positionally.
    return ensemble()(x, train)

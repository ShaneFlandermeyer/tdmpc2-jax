import jax.numpy as jnp
import jax
import optax

def percentile(x: jax.Array, q: jax.Array) -> jax.Array:
  x_dtype, x_shape = x.dtype, x.shape
  x = x.reshape(x.shape[0], -1)
  in_sorted = jnp.sort(x, axis=0)
  positions = q * (x.shape[0] - 1) / 100
  floored = jnp.floor(positions)
  ceiled = floored + 1
  # Replace below with jnp.where
  ceiled = jnp.where(ceiled > x.shape[0] - 1, x.shape[0] - 1, ceiled)
  weight_ceiled = positions
  weight_floored = 1.0 - weight_ceiled
  d0 = in_sorted[floored.astype(jnp.int32), :] * weight_floored[:, None]
  d1 = in_sorted[ceiled.astype(jnp.int32), :] * weight_ceiled[:, None]
  return (d0 + d1).reshape((-1, *x_shape[1:])).astype(x_dtype)


# Normalize input values using a running scale of the range between a given range of percentiles.
def percentile_normalization(x: jax.Array,
                             prev_scale: jax.Array,
                             percentile_range: jax.Array = jnp.array([5, 95]),
                             tau: float = 0.01) -> jax.Array:
  # Compute percentiles for the input values.
  percentiles = percentile(x, percentile_range)
  scale = percentiles[1] - percentiles[0]

  return optax.incremental_update(scale, prev_scale, tau)


def mean_std_normalization(x: jax.Array,
                           prev_scale: jax.Array,
                           tau: float = 0.01) -> jax.Array:
  mean = jnp.mean(x)
  std = jnp.std(x)
  scale = jnp.array([mean, std])

  return optax.incremental_update(scale, prev_scale, tau)


if __name__ == '__main__':
  x = jnp.ones(256)
  prev_scale = jnp.array([1])
  percentile(x, jnp.array([5, 95]))
  # print(mean_std_normalization(x, prev_scale))

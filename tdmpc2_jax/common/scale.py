import jax.numpy as jnp
import jax


# Normalize input values using a running scale of the range between a given range of percentiles.
def percentile_normalization(x: jax.Array,
                             prev_scale: jax.Array,
                             percentile_range: jax.Array = jnp.array([5, 95]),
                             tau: float = 0.01) -> jax.Array:
  # Compute percentiles for the input values.
  percentiles = jnp.percentile(x, percentile_range)
  scale = percentiles[1] - percentiles[0]

  return tau * scale + (1 - tau) * prev_scale


def mean_std_normalization(x: jax.Array,
                           prev_scale: jax.Array,
                           tau: float = 0.01) -> jax.Array:
  mean = jnp.mean(x)
  std = jnp.std(x)
  scale = jnp.array([mean, std])

  return tau * scale + (1 - tau) * prev_scale

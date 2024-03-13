import jax.numpy as jnp
import jax

from tdmpc2_jax.common.util import two_hot


def huber_loss(x: jax.Array, delta: float = 1.0) -> jax.Array:
  return jnp.where(jnp.abs(x) < delta, 0.5 * x**2, delta * (jnp.abs(x) - 0.5 * delta))


def mse_loss(x: jax.Array, y: jax.Array) -> jax.Array:
  return jnp.mean((x - y)**2)


def mae_loss(x: jax.Array, y: jax.Array) -> jax.Array:
  return jnp.mean(jnp.abs(x - y))


def soft_crossentropy(x_logits: jax.Array, y: jax.Array,
                      low: float, high: float, num_bins: int) -> jax.Array:
  pred = jax.nn.log_softmax(x_logits, axis=-1)
  target = jax.lax.stop_gradient(two_hot(y, low, high, num_bins))
  return -(pred * target).sum(axis=-1)

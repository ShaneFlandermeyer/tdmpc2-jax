import jax.numpy as jnp
import jax
import optax

from tdmpc2_jax.common.util import two_hot


def huber_loss(x: jax.Array, delta: float = 1.0) -> jax.Array:
  return jnp.where(jnp.abs(x) < delta, 0.5 * x**2, delta * (jnp.abs(x) - 0.5 * delta))


def mse_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
  return jnp.mean((pred - target)**2)


def mae_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
  return jnp.mean(jnp.abs(pred - target))


def soft_crossentropy(pred_logits: jax.Array, target: jax.Array,
                      low: float, high: float, num_bins: int) -> jax.Array:
  pred = jax.nn.log_softmax(pred_logits, axis=-1)
  target = two_hot(target, low, high, num_bins)
  return -(pred * target).sum(axis=-1)


def binary_crossentropy(pred_logits: jax.Array, target: jax.Array) -> jax.Array:
  pred = jax.nn.sigmoid(pred_logits)
  return -jnp.mean(target * jnp.log(pred) + (1 - target) * jnp.log(1 - pred))

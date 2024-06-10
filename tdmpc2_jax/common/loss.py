import jax.numpy as jnp
import jax

from tdmpc2_jax.common.util import two_hot

def soft_crossentropy(pred_logits: jax.Array, target: jax.Array,
                      low: float, high: float, num_bins: int) -> jax.Array:
  pred = jax.nn.log_softmax(pred_logits, axis=-1)
  target = two_hot(target, low, high, num_bins)
  return -(pred * target).sum(axis=-1)

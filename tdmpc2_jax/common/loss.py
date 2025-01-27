import jax.numpy as jnp
import jax

from tdmpc2_jax.common.util import two_hot

def soft_crossentropy(pred: jax.Array, 
                      target: jax.Array,
                      apply_softmax: bool = True,
                      ) -> jax.Array:
  if apply_softmax:
    pred = jax.nn.log_softmax(pred, axis=-1)
  else:
    pred = jnp.log(pred)
  return -(pred * target).sum(axis=-1)

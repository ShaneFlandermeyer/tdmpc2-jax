from einops import rearrange
import jax.numpy as jnp
import jax



def mish(x: jax.Array) -> jax.Array:
  return x * jnp.tanh(jnp.log(1 + jnp.exp(x)))


def simnorm(x: jax.Array, simplex_dim: int = 8) -> jax.Array:
  x = rearrange(x, '...(L V) -> ... L V', V=simplex_dim)
  x = jax.nn.softmax(x, axis=-1)
  return rearrange(x, '... L V -> ... (L V)')

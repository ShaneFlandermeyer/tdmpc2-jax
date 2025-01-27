from typing import Any, Dict, Tuple
import jax.numpy as jnp
import flax.linen as nn
import jax
from flax.training.train_state import TrainState


class Temperature(nn.Module):
  initial_value: float = 1.0

  @nn.compact
  def __call__(self) -> jnp.ndarray:
    log_temp = self.param(
        name="log_temp",
        init_fn=lambda key: jnp.full(
            shape=(), fill_value=jnp.log(self.initial_value)
        ),
    )
    return jnp.exp(log_temp)


def update_temperature(
    temperature_model: TrainState,
    entropy: float,
    target_entropy: float
) -> Tuple[TrainState, Dict[str, Any]]:
  def loss_fn(
      temperature_params: nn.FrozenDict,
  ) -> Tuple[float, Dict[str, Any]]:
    temperature = temperature_model.apply_fn({"params": temperature_params})
    loss = temperature * (entropy - target_entropy).mean()
    info = {
        "temperature": temperature,
        "temperature_loss": loss,
    }

    return loss, info

  grads, info = jax.grad(loss_fn, has_aux=True)(temperature_model.params)
  temperature_model = temperature_model.apply_gradients(grads=grads)

  return temperature_model, info

from __future__ import annotations
from functools import partial
from flax import struct
import flax
import jax
from jaxtyping import PRNGKeyArray, PyTree
import optax

from tdmpc2_jax.world_model import WorldModel
import jax.numpy as jnp
from tdmpc2_jax.common.loss import soft_crossentropy
import numpy as np
from typing import Any, Dict, Optional, Tuple
from tdmpc2_jax.common.scale import percentile_normalization
from tdmpc2_jax.common.util import sg


class TDMPC2(struct.PyTreeNode):
  model: WorldModel
  scale: jax.Array

  # Planning
  mpc: bool
  horizon: int = struct.field(pytree_node=False)
  mppi_iterations: int = struct.field(pytree_node=False)
  population_size: int = struct.field(pytree_node=False)
  policy_prior_samples: int = struct.field(pytree_node=False)
  num_elites: int = struct.field(pytree_node=False)
  min_plan_std: float
  max_plan_std: float
  temperature: float
  # Optimization
  batch_size: int = struct.field(pytree_node=False)
  discount: float
  rho: float
  consistency_coef: float
  reward_coef: float
  value_coef: float
  continue_coef: float
  entropy_coef: float
  tau: float

  @classmethod
  def create(cls,
             world_model: WorldModel,
             # Planning
             mpc: bool,
             horizon: int,
             mppi_iterations: int,
             population_size: int,
             policy_prior_samples: int,
             num_elites: int,
             min_plan_std: float,
             max_plan_std: float,
             temperature: float,
             # Optimization
             discount: float,
             batch_size: int,
             rho: float,
             consistency_coef: float,
             reward_coef: float,
             value_coef: float,
             continue_coef: float,
             entropy_coef: float,
             tau: float
             ) -> TDMPC2:

    return cls(model=world_model,
               mpc=mpc,
               horizon=horizon,
               mppi_iterations=mppi_iterations,
               population_size=population_size,
               policy_prior_samples=policy_prior_samples,
               num_elites=num_elites,
               min_plan_std=min_plan_std,
               max_plan_std=max_plan_std,
               temperature=temperature,
               discount=discount,
               batch_size=batch_size,
               rho=rho,
               consistency_coef=consistency_coef,
               reward_coef=reward_coef,
               value_coef=value_coef,
               continue_coef=continue_coef,
               entropy_coef=entropy_coef,
               tau=tau,
               scale=jnp.array([1.0]),
               )

  def act(self,
          obs: PyTree,
          prev_plan: Optional[Tuple[jax.Array, jax.Array]] = None,
          train: bool = True,
          *,
          key: PRNGKeyArray
          ) -> Tuple[np.ndarray, Optional[Tuple[jax.Array]]]:
    encoder_key, action_key = jax.random.split(key, 2)
    z = self.model.encode(obs, self.model.encoder.params, key=encoder_key)

    if self.mpc:
      if prev_plan is None:
        batch_dims = z.shape[:-1]
        prev_plan = (
            jnp.zeros((*batch_dims, self.horizon, self.model.action_dim)),
            jnp.full(
                (*batch_dims, self.horizon, self.model.action_dim),
                self.max_plan_std
            )
        )
      action, plan = self.plan(
          z=z,
          prev_plan=prev_plan,
          train=train,
          key=action_key
      )
    else:
      action = self.model.sample_actions(
          z, self.model.policy_model.params, key=action_key
      )[0]
      plan = None

    return np.array(action), plan

  @partial(jax.jit, static_argnames=('train'))
  def plan(self,
           z: jax.Array,
           prev_plan: Tuple[jax.Array, jax.Array],
           train: bool,
           *,
           key: PRNGKeyArray,
           ) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
    batch_shape = z.shape[:-1]
    actions = jnp.zeros(
        (
            *batch_shape,
            self.population_size,
            self.horizon,
            self.model.action_dim
        )
    )

    ###########################################################
    # Policy prior samples
    ###########################################################
    if self.policy_prior_samples > 0:
      key, *prior_noise_keys = jax.random.split(key, 1+self.horizon)
      policy_actions = jnp.zeros(
          (
              *batch_shape,
              self.policy_prior_samples,
              self.horizon,
              self.model.action_dim
          )
      )
      z_t = z[..., None, :].repeat(self.policy_prior_samples, axis=-2)
      for t in range(self.horizon):
        policy_actions = policy_actions.at[..., t, :].set(
            self.model.sample_actions(
                z_t, self.model.policy_model.params, key=prior_noise_keys[t]
            )[0]
        )
        if t < self.horizon-1:  # Don't need for the last time step
          z_t = self.model.next(
              z_t,
              policy_actions[..., t, :],
              self.model.dynamics_model.params
          )

      actions = actions.at[..., :self.policy_prior_samples, :, :].set(
          policy_actions
      )

    ###########################################################
    # MPPI planning
    ###########################################################
    z_t = z[..., None, :].repeat(self.population_size, axis=-2)
    key, mppi_noise_key, *value_keys = jax.random.split(
        key, 2+self.mppi_iterations
    )
    noise = jax.random.normal(
        mppi_noise_key,
        shape=(
            *batch_shape,
            self.population_size - self.policy_prior_samples,
            self.mppi_iterations,
            self.horizon,
            self.model.action_dim
        )
    )
    # Initialize population state
    mean = jnp.zeros((*batch_shape, self.horizon, self.model.action_dim))
    std = jnp.full(
        (*batch_shape, self.horizon, self.model.action_dim),
        self.max_plan_std
    )
    if prev_plan is not None:
      mean = mean.at[..., :-1, :].set(prev_plan[0][..., 1:, :])

    for i in range(self.mppi_iterations):
      actions = actions.at[..., self.policy_prior_samples:, :, :].set(
          mean[..., None, :, :] + std[..., None, :, :] * noise[..., i, :, :]
      ).clip(-1, 1)

      # Compute elites
      values = self.estimate_value(z_t, actions, key=value_keys[i])
      elite_values, elite_inds = jax.lax.top_k(values, self.num_elites)
      elite_actions = jnp.take_along_axis(
          actions, elite_inds[..., None, None], axis=-3
      )

      # Update population distribution
      score = jax.nn.softmax(self.temperature * elite_values)
      mean = jnp.sum(score[..., None, None] * elite_actions, axis=-3)
      std = jnp.sqrt(
          jnp.sum(
              score[..., None, None] *
              (elite_actions - mean[..., None, :, :])**2,
              axis=-3
          )
      ).clip(self.min_plan_std, self.max_plan_std)

    # Select final action
    key, gumbel_key = jax.random.split(key)
    gumbels = jax.random.gumbel(gumbel_key, shape=elite_values.shape)
    gumbel_scores = jnp.log(score) + gumbels
    action_ind = jnp.argmax(gumbel_scores, axis=-1)
    action = jnp.take_along_axis(
        elite_actions, action_ind[..., None, None, None], axis=-3
    ).squeeze(-3)

    if not train:
      final_action = action[..., 0, :]
    else:
      key, final_noise_key = jax.random.split(key)
      final_action = action[..., 0, :] + std[..., 0, :] * jax.random.normal(
          final_noise_key, shape=batch_shape + (self.model.action_dim,)
      )
    return final_action.clip(-1, 1), (mean, std)

  @jax.jit
  def update(self,
             observations: PyTree,
             actions: jax.Array,
             rewards: jax.Array,
             next_observations: PyTree,
             terminated: jax.Array,
             truncated: jax.Array,
             *,
             key: PRNGKeyArray
             ) -> Tuple[TDMPC2, Dict[str, Any]]:

    world_model_key, policy_key = jax.random.split(key, 2)

    def world_model_loss_fn(encoder_params: flax.core.FrozenDict,
                            dynamics_params: flax.core.FrozenDict,
                            value_params: flax.core.FrozenDict,
                            reward_params: flax.core.FrozenDict,
                            continue_params: flax.core.FrozenDict,
                            ) -> Tuple[jax.Array, Dict[str, Any]]:
      encoder_key, value_key = jax.random.split(world_model_key, 2)

      ###########################################################
      # Encoder forward pass
      ###########################################################
      first_encoder_key, next_encoder_key = jax.random.split(encoder_key, 2)
      first_z = self.model.encode(
          jax.tree.map(lambda x: x[0], observations),
          encoder_params,
          key=first_encoder_key
      )
      next_z = sg(
          self.model.encode(
              next_observations, encoder_params, key=next_encoder_key
          )
      )

      ###########################################################
      # Latent rollout (dynamics + consistency loss)
      ###########################################################
      done = jnp.logical_or(terminated, truncated)
      finished = jnp.zeros((self.horizon+1, self.batch_size), dtype=bool)
      consistency_loss = 0
      zs = jnp.zeros((self.horizon+1, self.batch_size, next_z.shape[-1]))
      zs = zs.at[0].set(first_z)
      for t in range(self.horizon):
        z = self.model.next(zs[t], actions[t], dynamics_params)
        zs = zs.at[t+1].set(z)
        consistency_loss += self.rho**t * \
            jnp.mean((z - next_z[t])**2, where=~finished[t][:, None])
        finished = finished.at[t+1].set(jnp.logical_or(finished[t], done[t]))

      ###########################################################
      # Reward loss
      ###########################################################
      _, reward_logits = self.model.reward(zs[:-1], actions, reward_params)
      reward_loss = jnp.sum(
          self.rho**np.arange(self.horizon) * soft_crossentropy(
              reward_logits, rewards,
              self.model.symlog_min,
              self.model.symlog_max,
              self.model.num_bins
          ).mean(axis=-1, where=~finished[:-1])
      )

      ###########################################################
      # Value loss
      ###########################################################
      next_action_key, ensemble_key, Q_key = jax.random.split(value_key, 3)

      # TD targets
      next_action = self.model.sample_actions(
          next_z, self.model.policy_model.params, key=next_action_key
      )[0]
      Qs, _ = self.model.Q(
          next_z, next_action, self.model.target_value_model.params, key=Q_key
      )
      # Subsample value networks
      inds = jax.random.choice(
          ensemble_key,
          jnp.arange(0, self.model.num_value_nets),
          shape=(2, ),
          replace=False
      )
      Q = Qs[inds].min(axis=0)
      td_targets = rewards + (1 - terminated) * self.discount * Q

      _, Q_logits = self.model.Q(zs[:-1], actions, value_params, key=Q_key)
      value_loss = jnp.sum(
          self.rho**np.arange(self.horizon) * soft_crossentropy(
              Q_logits, sg(td_targets),
              self.model.symlog_min,
              self.model.symlog_max,
              self.model.num_bins
          ).mean(axis=-1, where=~finished[:-1])
      )

      ###########################################################
      # Continue loss
      ###########################################################
      if self.model.predict_continues:
        continue_logits = self.model.continue_model.apply_fn(
            {'params': continue_params}, zs[1:]
        ).squeeze(-1)
        continue_loss = optax.sigmoid_binary_cross_entropy(
            continue_logits, 1 - terminated
        ).mean()
      else:
        continue_loss = 0.0

      consistency_loss = consistency_loss / self.horizon
      reward_loss = reward_loss / self.horizon
      value_loss = value_loss / self.horizon / self.model.num_value_nets
      total_loss = (
          self.consistency_coef * consistency_loss +
          self.reward_coef * reward_loss +
          self.value_coef * value_loss +
          self.continue_coef * continue_loss
      )

      return total_loss, {
          'consistency_loss': consistency_loss,
          'reward_loss': reward_loss,
          'value_loss': value_loss,
          'continue_loss': continue_loss,
          'total_loss': total_loss,
          'zs': zs
      }

    # Update world model
    (encoder_grads, dynamics_grads, value_grads, reward_grads, continue_grads), model_info = jax.grad(
        world_model_loss_fn, argnums=(0, 1, 2, 3, 4), has_aux=True)(
            self.model.encoder.params,
            self.model.dynamics_model.params,
            self.model.value_model.params,
            self.model.reward_model.params,
            self.model.continue_model.params if self.model.predict_continues else None
    )
    zs = model_info.pop('zs')

    new_encoder = self.model.encoder.apply_gradients(grads=encoder_grads)
    new_dynamics_model = self.model.dynamics_model.apply_gradients(
        grads=dynamics_grads
    )
    new_reward_model = self.model.reward_model.apply_gradients(
        grads=reward_grads
    )
    new_value_model = self.model.value_model.apply_gradients(
        grads=value_grads)
    new_target_value_model = self.model.target_value_model.replace(
        params=optax.incremental_update(
            new_value_model.params,
            self.model.target_value_model.params,
            self.tau
        )
    )
    if self.model.predict_continues:
      new_continue_model = self.model.continue_model.apply_gradients(
          grads=continue_grads
      )
    else:
      new_continue_model = self.model.continue_model

    # Update policy
    def policy_loss_fn(actor_params: flax.core.FrozenDict,):
      action_key, Q_key = jax.random.split(policy_key, 2)
      actions, _, _, log_probs = self.model.sample_actions(
          zs, actor_params, key=action_key
      )

      # Compute Q-values
      Qs, _ = self.model.Q(zs, actions, new_value_model.params, key=Q_key)
      Q = Qs.mean(axis=0)
      # Update and apply scale
      scale = percentile_normalization(Q[0], self.scale).clip(1, None)
      Q = Q / sg(scale)

      # Compute policy objective (equation 4)
      policy_loss = (
          self.rho ** jnp.arange(self.horizon+1) *
          (self.entropy_coef * log_probs - Q).mean(axis=1)
      ).mean()
      return policy_loss, {'policy_loss': policy_loss, 'policy_scale': scale}

    policy_grads, policy_info = jax.grad(policy_loss_fn, has_aux=True)(
        self.model.policy_model.params
    )
    new_policy = self.model.policy_model.apply_gradients(grads=policy_grads)

    # Update model
    new_agent = self.replace(
        model=self.model.replace(
            encoder=new_encoder,
            dynamics_model=new_dynamics_model,
            reward_model=new_reward_model,
            value_model=new_value_model,
            policy_model=new_policy,
            target_value_model=new_target_value_model,
            continue_model=new_continue_model
        ),
        scale=policy_info['policy_scale']
    )
    info = {**model_info, **policy_info}

    return new_agent, info

  @jax.jit
  def estimate_value(self,
                     z: jax.Array,
                     actions: jax.Array,
                     key: PRNGKeyArray
                     ) -> jax.Array:
    G, discount = 0.0, 1.0
    for t in range(self.horizon):
      reward, _ = self.model.reward(
          z, actions[..., t, :], self.model.reward_model.params
      )
      z = self.model.next(
          z, actions[..., t, :], self.model.dynamics_model.params
      )
      G += discount * reward

      if self.model.predict_continues:
        continues = jax.nn.sigmoid(
            self.model.continue_model.apply_fn(
                {'params': self.model.continue_model.params}, z
            )
        ).squeeze(-1) > 0.5
      else:
        continues = 1.0

      discount *= self.discount * continues

    action_key, Q_key = jax.random.split(key, 2)
    next_action = self.model.sample_actions(
        z, self.model.policy_model.params, key=action_key
    )[0]

    Qs, _ = self.model.Q(
        z, next_action, self.model.value_model.params, key=Q_key
    )
    Q = Qs.mean(axis=0)
    return G + discount * Q

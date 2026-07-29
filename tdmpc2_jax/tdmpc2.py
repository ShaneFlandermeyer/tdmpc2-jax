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
  value_scale: jax.Array

  # Planning
  horizon: int = struct.field(pytree_node=False)
  planning_hmax: int = struct.field(pytree_node=False)
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
  consistency_loss_scale: float
  reward_loss_scale: float
  value_loss_scale: float
  continue_loss_scale: float
  entropy_coef: float
  tau: float
  grad_clip_norm: float

  @classmethod
  def create(cls,
             world_model: WorldModel,
             # Planning
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
             consistency_loss_scale: float,
             reward_loss_scale: float,
             value_loss_scale: float,
             continue_loss_scale: float,
             entropy_coef: float,
             tau: float,
             planning_hmax: Optional[int] = None,
             grad_clip_norm: float = 20.0,
             ) -> TDMPC2:

    return cls(model=world_model,
               horizon=horizon,
               planning_hmax=(int(planning_hmax) if planning_hmax is not None else int(horizon)),
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
               consistency_loss_scale=consistency_loss_scale,
               reward_loss_scale=reward_loss_scale,
               value_loss_scale=value_loss_scale,
               continue_loss_scale=continue_loss_scale,
               entropy_coef=entropy_coef,
               tau=tau,
               grad_clip_norm=grad_clip_norm,
               value_scale=jnp.array([1.0]),
               )

  def _sample_two_q_mean(self, Qs: jax.Array, key: PRNGKeyArray) -> jax.Array:
    """Match TD-MPC2's stochastic two-head value estimate."""
    inds = jax.random.choice(
        key,
        self.model.num_value_nets,
        shape=(2,),
        replace=False,
    )
    return Qs[inds].mean(axis=0)

  @partial(jax.jit, static_argnames=('mpc', 'deterministic', 'train'))
  def act(self,
          obs: PyTree,
          prev_plan: Optional[Tuple[jax.Array, jax.Array]] = None,
          mpc: bool = True,
          deterministic: bool = False,
          train: bool = False,
          horizon: Optional[int] = None,
          *,
          key: PRNGKeyArray
          ) -> Tuple[np.ndarray, Optional[Tuple[jax.Array]]]:
    encoder_key, action_key = jax.random.split(key, 2)
    z = self.model.encode(
        obs=obs,
        params=self.model.encoder.params,
        key=encoder_key
    )

    if mpc:
      chosen_horizon = self.horizon if horizon is None else horizon
      action, plan = self.plan(
          z=z,
          horizon=chosen_horizon,
          prev_plan=prev_plan,
          deterministic=deterministic,
          train=train,
          key=action_key
      )
    else:
      action, _, _, _ = self.model.sample_actions(
          z=z,
          deterministic=deterministic,
          params=self.model.policy_model.params,
          key=action_key
      )
      plan = None

    return action, plan

  @partial(jax.jit, static_argnames=('deterministic', 'train'))
  def plan(self,
           z: jax.Array,
           horizon: int,
           prev_plan: Tuple[jax.Array, jax.Array] = None,
           deterministic: bool = False,
           train: bool = False,
           *,
           key: PRNGKeyArray,
           ) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
    horizon = jnp.asarray(horizon, dtype=jnp.int32)
    batch_shape = z.shape[:-1]
    hmax = int(self.planning_hmax)
    actions = jnp.zeros(
        (
            *batch_shape,
            self.population_size,
            hmax,
            self.model.action_dim
        )
    )
    time_index = jnp.arange(hmax, dtype=jnp.int32)
    horizon_mask = time_index < horizon

    ###########################################################
    # Policy prior samples
    ###########################################################
    if self.policy_prior_samples > 0:
      key, prior_key = jax.random.split(key)
      prior_noise_keys = jax.random.split(prior_key, hmax)
      z_t = z[..., None, :].repeat(self.policy_prior_samples, axis=-2)

      def prior_step(carry, inputs):
        z_carry = carry
        t, action_key = inputs
        sampled_action = self.model.sample_actions(
            z=z_carry,
            deterministic=False,
            params=self.model.policy_model.params,
            key=action_key,
        )[0]
        active = jnp.asarray(t < horizon, dtype=jnp.bool_)
        active_mask = active.astype(sampled_action.dtype)
        while active_mask.ndim < sampled_action.ndim:
          active_mask = active_mask[..., None]
        action = sampled_action * active_mask
        next_z = self.model.next(
            z=z_carry,
            a=action,
            params=self.model.dynamics_model.params,
        )
        z_next = jnp.where(active, next_z, z_carry)
        return z_next, action

      _, policy_actions = jax.lax.scan(
          prior_step,
          z_t,
          (time_index, prior_noise_keys),
      )
      policy_actions = jnp.moveaxis(policy_actions, 0, -2)
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
            hmax,
            self.model.action_dim
        )
    )
    # Initialize population state
    mean = jnp.zeros((*batch_shape, hmax, self.model.action_dim))
    std = jnp.full(
        (*batch_shape, hmax, self.model.action_dim),
        self.max_plan_std,
        dtype=jnp.float32,
    )
    if prev_plan is not None:
      mean = mean.at[..., :-1, :].set(prev_plan[0][..., 1:, :])

    def mppi_step(carry, inputs):
      mean, std, actions = carry
      step_noise, value_key = inputs
      actions = actions.at[..., self.policy_prior_samples:, :, :].set(
          mean[..., None, :, :] + std[..., None, :, :] * step_noise
      ).clip(-1, 1)

      values = self.estimate_value(
          z=z_t, actions=actions, horizon=horizon, key=value_key
      )
      elite_values, elite_inds = jax.lax.top_k(values, self.num_elites)
      elite_actions = jnp.take_along_axis(
          actions, elite_inds[..., None, None], axis=-3
      )

      score = jax.nn.softmax(self.temperature * elite_values)
      mean = jnp.sum(score[..., None, None] * elite_actions, axis=-3)
      std = jnp.sqrt(
          jnp.sum(
              score[..., None, None] *
              (elite_actions - mean[..., None, :, :])**2,
              axis=-3
          ) + 1e-6
      ).clip(self.min_plan_std, self.max_plan_std)
      return (mean, std, actions), (elite_values, elite_actions, score)

    noise_steps = jnp.moveaxis(noise, -3, 0)
    (mean, std, actions), (elite_values_hist, elite_actions_hist, score_hist) = jax.lax.scan(
        mppi_step,
        (mean, std, actions),
        (noise_steps, jnp.asarray(value_keys)),
    )
    elite_values = elite_values_hist[-1]
    elite_actions = elite_actions_hist[-1]
    score = score_hist[-1]
    # TD-MPC2 samples from the elite distribution for evaluation too; evaluation
    # only disables the final exploration noise below.
    key, final_action_key = jax.random.split(key)
    action_ind = jax.random.categorical(
        final_action_key, logits=jnp.log(score), shape=batch_shape
    )
    action = jnp.take_along_axis(
        elite_actions, action_ind[..., None, None, None], axis=-3
    ).squeeze(-3)
    if train:
      key, final_noise_key = jax.random.split(key)
      final_action = action[..., 0, :] + std[..., 0, :] * \
          jax.random.normal(
              final_noise_key, shape=batch_shape + (self.model.action_dim,)
      )
    else:
      final_action = action[..., 0, :]

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
             key: PRNGKeyArray,
             train_horizon: Optional[int] = None,
             ) -> Tuple[TDMPC2, Dict[str, Any]]:

    world_model_key, policy_key = jax.random.split(key, 2)
    train_horizon = jnp.asarray(
        self.horizon if train_horizon is None else train_horizon,
        dtype=jnp.int32,
    )
    train_horizon_f = jnp.maximum(train_horizon.astype(jnp.float32), 1.0)

    def world_model_loss_fn(encoder_params: flax.core.FrozenDict,
                            dynamics_params: flax.core.FrozenDict,
                            value_params: flax.core.FrozenDict,
                            reward_params: flax.core.FrozenDict,
                            continue_params: flax.core.FrozenDict,
                            ) -> Tuple[jax.Array, Dict[str, Any]]:
      encoder_key, value_key = jax.random.split(world_model_key, 2)
      time_index = jnp.arange(self.horizon, dtype=jnp.int32)
      horizon_mask = time_index < train_horizon
      lam = (self.rho**time_index) / train_horizon_f
      lam = jnp.where(horizon_mask, lam, 0.0)

      ###########################################################
      # Encoder forward pass
      ###########################################################
      all_obs = jax.tree.map(
          lambda x, y: jnp.stack([x, y], axis=0),
          observations, next_observations
      )
      all_zs = self.model.encode(
          obs=all_obs, params=encoder_params, key=encoder_key
      )
      encoder_zs = jax.tree.map(lambda x: x[0], all_zs)
      next_zs = jax.tree.map(lambda x: x[1], all_zs)

      ###########################################################
      # Latent rollout (dynamics + consistency loss)
      ###########################################################
      done = jnp.logical_or(terminated, truncated)
      init_finished = jnp.zeros((self.batch_size,), dtype=bool)

      def latent_step(carry, inputs):
        z_t, finished_t = carry
        action_t, next_z_t, done_t, lam_t = inputs
        z_next = self.model.next(
            z=z_t,
            a=action_t,
            params=dynamics_params,
        )
        consistency_term = lam_t * jnp.mean(
            (z_next - sg(next_z_t))**2,
            where=~finished_t[:, None],
        )
        finished_next = jnp.logical_or(finished_t, done_t)
        return (z_next, finished_next), (z_next, finished_t, consistency_term)

      (last_z, finished_last), (latent_rollout, finished_hist, consistency_terms) = jax.lax.scan(
          latent_step,
          (encoder_zs[0], init_finished),
          (actions, next_zs, done, lam),
      )
      latent_zs = jnp.concatenate([encoder_zs[0][None], latent_rollout], axis=0)
      finished = jnp.concatenate([finished_hist, finished_last[None]], axis=0)
      consistency_loss = jnp.sum(consistency_terms)

      ###########################################################
      # Reward loss
      ###########################################################
      _, reward_logits = self.model.reward(
          z=latent_zs[:-1], a=actions, params=reward_params,
      )
      reward_loss = jnp.sum(
          lam[:, None] * soft_crossentropy(
              pred_logits=reward_logits,
              target=rewards,
              low=self.model.symlog_min,
              high=self.model.symlog_max,
              num_bins=self.model.num_bins,
          ), axis=0, where=~finished[:-1]
      ).mean()

      ###########################################################
      # Value loss
      ###########################################################
      next_action_key, value_target_key, ensemble_key, value_key = \
          jax.random.split(value_key, 4)

      # TD targets
      next_action = self.model.sample_actions(
          z=next_zs,
          deterministic=False,
          params=self.model.policy_model.params,
          key=next_action_key
      )[0]
      Qs, _ = self.model.Q(
          z=next_zs,
          a=next_action,
          params=self.model.target_value_model.params,
          key=value_target_key,
          train=False,
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

      _, Q_logits = self.model.Q(
          z=latent_zs[:-1], a=actions, params=value_params, key=value_key,
          train=True,
      )
      value_loss = jnp.sum(
          lam[:, None] * soft_crossentropy(
              pred_logits=Q_logits,
              target=sg(td_targets),
              low=self.model.symlog_min,
              high=self.model.symlog_max,
              num_bins=self.model.num_bins
          ), axis=1, where=~finished[:-1]
      ).mean()

      ###########################################################
      # Continue loss
      ###########################################################
      if self.model.predict_continues:
        continue_logits = self.model.continue_model.apply_fn(
            {'params': continue_params}, latent_zs[:-1]
        ).squeeze(-1)
        continue_loss = optax.sigmoid_binary_cross_entropy(
            continue_logits, 1 - terminated
        ).mean()
      else:
        continue_loss = 0.0

      total_loss = (
          self.consistency_loss_scale * consistency_loss +
          self.reward_loss_scale * reward_loss +
          self.value_loss_scale * value_loss +
          self.continue_loss_scale * continue_loss
      )

      return total_loss, {
          'consistency_loss': consistency_loss,
          'reward_loss': reward_loss,
          'value_loss': value_loss,
          'continue_loss': continue_loss,
          'total_loss': total_loss,
          'latent_zs': latent_zs,
          'finished': finished,
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
    if self.model.predict_continues:
      world_grads = (
          encoder_grads,
          dynamics_grads,
          value_grads,
          reward_grads,
          continue_grads,
      )
      world_grad_norm = optax.global_norm(world_grads)
      world_grad_scale = jnp.minimum(
          1.0,
          self.grad_clip_norm / (world_grad_norm + 1e-6),
      )
      (
          encoder_grads,
          dynamics_grads,
          value_grads,
          reward_grads,
          continue_grads,
      ) = jax.tree.map(lambda g: g * world_grad_scale, world_grads)
    else:
      world_grads = (
          encoder_grads,
          dynamics_grads,
          value_grads,
          reward_grads,
      )
      world_grad_norm = optax.global_norm(world_grads)
      world_grad_scale = jnp.minimum(
          1.0,
          self.grad_clip_norm / (world_grad_norm + 1e-6),
      )
      (
          encoder_grads,
          dynamics_grads,
          value_grads,
          reward_grads,
      ) = jax.tree.map(lambda g: g * world_grad_scale, world_grads)
      continue_grads = None
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
    latent_zs = model_info.pop('latent_zs')
    finished = model_info.pop('finished')

    def policy_loss_fn(actor_params: flax.core.FrozenDict):
      action_key, Q_key, Q_head_key = jax.random.split(policy_key, 3)
      actions, _, log_std, log_probs = self.model.sample_actions(
          z=latent_zs,
          deterministic=False,
          params=actor_params,
          key=action_key
      )

      # Compute policy objective (equation 4)
      policy_time_index = jnp.arange(self.horizon + 1, dtype=jnp.int32)
      policy_horizon = train_horizon + 1
      policy_lam = (self.rho**policy_time_index) / jnp.maximum(
          policy_horizon.astype(jnp.float32),
          1.0,
      )
      policy_lam = jnp.where(policy_time_index < policy_horizon, policy_lam, 0.0)
      Qs, _ = self.model.Q(
          z=latent_zs, a=actions, params=new_value_model.params, key=Q_key,
          train=True,
      )
      Q = self._sample_two_q_mean(Qs, Q_head_key)
      Q_scale = percentile_normalization(Q[0], self.value_scale).clip(1, None)
      scaled_log_probs = log_probs * self.model.action_dim
      policy_loss = jnp.sum(
          policy_lam[:, None] * (
              self.entropy_coef * scaled_log_probs - Q / sg(Q_scale)
          ),
          axis=0, where=~finished
      ).mean()
      return policy_loss, {
          'policy_loss': policy_loss,
          'policy_log_std': log_std,
          'value_scale': Q_scale
      }

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
        value_scale=policy_info['value_scale']
    )
    info = {**model_info, **policy_info}
    info['grad_norm'] = world_grad_norm

    return new_agent, info

  @jax.jit
  def estimate_value(self,
                     z: jax.Array,
                     actions: jax.Array,
                     horizon: int,
                     key: PRNGKeyArray
                     ) -> jax.Array:
    horizon = jnp.asarray(horizon, dtype=jnp.int32)
    time_index = jnp.arange(actions.shape[-2], dtype=jnp.int32)

    def value_step(carry, t):
      z_t, G, discount = carry
      reward, _ = self.model.reward(
          z=z_t,
          a=actions[..., t, :],
          params=self.model.reward_model.params,
      )
      next_z = self.model.next(
          z=z_t,
          a=actions[..., t, :],
          params=self.model.dynamics_model.params,
      )
      active = time_index[t] < horizon
      G = G + jnp.where(active, discount * reward, 0.0)
      z_next = jnp.where(active[..., None], next_z, z_t)
      next_discount = discount * self.discount
      if self.model.predict_continues:
        continues = jax.nn.sigmoid(
            self.model.continue_model.apply_fn(
                {'params': self.model.continue_model.params}, next_z
            )
        ).squeeze(-1) > 0.5
        next_discount = next_discount * continues
      discount_next = jnp.where(active, next_discount, discount)
      return (z_next, G, discount_next), None

    init_G = jnp.zeros(z.shape[:-1], dtype=jnp.float32)
    init_discount = jnp.ones(z.shape[:-1], dtype=jnp.float32)
    (z, G, discount), _ = jax.lax.scan(
        value_step,
        (z, init_G, init_discount),
        time_index,
    )

    action_key, Q_key, Q_head_key = jax.random.split(key, 3)
    next_action = self.model.sample_actions(
        z=z,
        deterministic=False,
        params=self.model.policy_model.params,
        key=action_key
    )[0]

    Qs, _ = self.model.Q(
        z=z,
        a=next_action,
        params=self.model.value_model.params,
        key=Q_key,
        train=False,
    )
    Q = self._sample_two_q_mean(Qs, Q_head_key)
    return G + discount * Q

  @jax.jit
  def update_many(self,
                  observations: PyTree,
                  actions: jax.Array,
                  rewards: jax.Array,
                  next_observations: PyTree,
                  terminated: jax.Array,
                  truncated: jax.Array,
                  *,
                  keys: PRNGKeyArray,
                  train_horizon: Optional[int] = None,
                  ) -> Tuple['TDMPC2', Dict[str, Any]]:
    def update_scan(agent, inputs):
      key, obs, act, rew, next_obs, term, trunc = inputs
      agent, info = agent.update(
          observations=obs,
          actions=act,
          rewards=rew,
          next_observations=next_obs,
          terminated=term,
          truncated=trunc,
          key=key,
          train_horizon=train_horizon,
      )
      return agent, info

    new_agent, info = jax.lax.scan(
        update_scan,
        self,
        (
            keys,
            observations,
            actions,
            rewards,
            next_observations,
            terminated,
            truncated,
        ),
    )
    return new_agent, info

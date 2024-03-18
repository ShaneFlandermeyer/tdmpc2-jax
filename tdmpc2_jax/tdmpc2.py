from __future__ import annotations
from flax import struct
import jax
from jaxtyping import PRNGKeyArray
import optax

from tdmpc2_jax.world_model import WorldModel
import jax.numpy as jnp
from tdmpc2_jax.common.loss import binary_crossentropy, mse_loss, soft_crossentropy
import numpy as np
from typing import Any, Dict, Tuple
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
             tau: float):

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
               scale=jnp.array([1])
               )

  def act(self,
          obs: np.ndarray,
          prev_plan: jax.Array = None,
          train: bool = True,
          *,
          key: PRNGKeyArray):
    z = self.model.encode(obs, self.model.encoder.params)
    z = jnp.atleast_2d(z)

    if self.mpc:
      action, plan = self.plan(z, prev_plan=prev_plan, train=train, key=key)
    else:
      action = self.model.sample_actions(
          z, self.model.policy_model.params, key=key)[0].squeeze()
      plan = None

    return np.array(action), plan

  @jax.jit
  def plan(self,
           z: jax.Array,
           prev_plan: Tuple[jax.Array, jax.Array] = None,
           train: bool = False,
           *,
           key: PRNGKeyArray) -> Tuple[jax.Array, jax.Array]:
    """
    Select next action via MPPI planner

    Parameters
    ----------
    z : jax.Array
        Enncoded environment observation
    key : PRNGKeyArray
        Jax PRNGKey
    prev_mean : jax.Array, optional
        Mean from previous planning interval. If present, MPPI is given a warm start by time-shifting this value by 1 step. If None, the MPPI mean is set to zero, by default None
    train : bool, optional
        If True, inject noise into the final selected action, by default False

    Returns
    -------
    Tuple[jax.Array, jax.Array]
        - Action output from planning
        - Final mean value (for use in warm start)
    """
    # Sample trajectories from policy prior
    key, *prior_keys = jax.random.split(key, self.horizon + 1)
    policy_actions = jnp.empty(
        (self.horizon, self.policy_prior_samples, self.model.action_dim))
    _z = z.repeat(self.policy_prior_samples, axis=0)
    for t in range(self.horizon-1):
      policy_actions = policy_actions.at[t].set(
          self.model.sample_actions(_z, self.model.policy_model.params, key=prior_keys[t])[0])
      _z = self.model.next(
          _z, policy_actions[t], self.model.dynamics_model.params)
    policy_actions = policy_actions.at[-1].set(
        self.model.sample_actions(_z, self.model.policy_model.params, key=prior_keys[-1])[0])

    # Initialize population state
    z = z.repeat(self.population_size, axis=0)
    mean = jnp.zeros((self.horizon, self.model.action_dim))
    std = self.max_plan_std * \
        jnp.ones((self.horizon, self.model.action_dim))
    # Warm start MPPI with the previous solution
    if prev_plan is not None:
      prev_mean, prev_std = prev_plan
      mean = mean.at[:-1].set(prev_mean[1:])
    #   std = std.at[:-1].set(prev_std[1:])

    actions = jnp.empty(
        (self.horizon, self.population_size, self.model.action_dim))
    actions = actions.at[:, :self.policy_prior_samples].set(policy_actions)

    # Iterate MPPI
    key, action_noise_key, *value_keys = \
        jax.random.split(key, self.mppi_iterations+1+1)
    noise = jax.random.normal(
        action_noise_key,
        shape=(
            self.mppi_iterations,
            self.horizon,
            self.population_size - self.policy_prior_samples,
            self.model.action_dim
        ))
    for i in range(self.mppi_iterations):
      # Sample actions
      actions = actions.at[:, self.policy_prior_samples:].set(
          mean[:, None, :] + std[:, None, :] * noise[i])
      actions = jnp.clip(actions, -1, 1)

      # Compute elite actions
      value = self.estimate_value(z, actions, key=value_keys[i])
      value = jnp.nan_to_num(value)  # Handle nans
      _, elite_inds = jax.lax.top_k(value, self.num_elites)
      elite_values, elite_actions = value[elite_inds], actions[:, elite_inds]

      # Update parameters
      max_value = jnp.max(elite_values)
      score = jnp.exp(self.temperature * (elite_values - max_value))
      score /= jnp.sum(score) + 1e-8

      mean = jnp.sum(score[None, :, None] * elite_actions, axis=1)
      std = jnp.sqrt(
          jnp.sum(score[None, :, None] * (elite_actions - mean[:, None, :])**2, axis=1))
      std = jnp.clip(std, self.min_plan_std, self.max_plan_std)

    # Select action based on the score
    key, *final_action_keys = jax.random.split(key, 3)
    action_ind = jax.random.choice(final_action_keys[0],
                                   a=jnp.arange(self.num_elites), p=score)
    actions = elite_actions[:, action_ind]

    action, action_std = actions[0], std[0]
    action += jnp.array(train, float) * action_std * jax.random.normal(
        final_action_keys[1], shape=action.shape)

    action = jnp.clip(action, -1, 1)
    return sg(action), (mean, std)

  @jax.jit
  def update(self,
             observations: jax.Array,
             actions: jax.Array,
             rewards: jax.Array,
             next_observations: jax.Array,
             dones: jax.Array,
             *,
             key: PRNGKeyArray
             ) -> Tuple[TDMPC2, Dict[str, Any]]:

    target_dropout, value_dropout_key1, value_dropout_key2, policy_key = \
        jax.random.split(key, 4)

    def world_model_loss_fn(encoder_params: Dict,
                            dynamics_params: Dict,
                            value_params: Dict,
                            reward_params: Dict,
                            continue_params: Dict):
      discount = jnp.ones((self.horizon+1, self.batch_size))
      horizon = jnp.zeros(self.batch_size)

      next_z = sg(self.model.encode(next_observations, encoder_params))
      td_targets = self.td_target(next_z, rewards, dones, key=target_dropout)

      # Latent rollout (compute latent dynamics + consistency loss)
      zs = jnp.empty((self.horizon+1, self.batch_size, next_z.shape[-1]))
      z = self.model.encode(observations[0], encoder_params)
      zs = zs.at[0].set(z)
      consistency_loss = jnp.zeros(self.batch_size)
      for t in range(self.horizon):
        z = self.model.next(z, actions[t], dynamics_params)
        consistency_loss += jnp.mean(
            (z - next_z[t])**2 * discount[t][:, None], -1)
        zs = zs.at[t+1].set(z)

        horizon += (discount[t] > 0)
        discount = discount.at[t+1].set(
            discount[t] * self.rho * (1 - dones[t]))

      # Get logits for loss computations
      _, q_logits = self.model.Q(
          zs[:-1], actions, value_params, value_dropout_key1)
      _, reward_logits = self.model.reward(zs[:-1], actions, reward_params)
      if self.model.predict_continues:
        continue_logits = self.model.continue_model.apply_fn(
            {'params': continue_params}, zs[1:]).squeeze(-1)

      reward_loss = jnp.zeros(self.batch_size)
      value_loss = jnp.zeros(self.batch_size)
      continue_loss = jnp.zeros(self.batch_size)
      for t in range(self.horizon):
        reward_loss += soft_crossentropy(reward_logits[t], rewards[t],
                                         self.model.symlog_min,
                                         self.model.symlog_max,
                                         self.model.num_bins) * discount[t]

        if self.model.predict_continues:
          continue_loss += optax.sigmoid_binary_cross_entropy(
              continue_logits[t], 1 - dones[t]) * discount[t]

        for q in range(self.model.num_value_nets):
          value_loss += soft_crossentropy(q_logits[q, t], td_targets[t],
                                          self.model.symlog_min,
                                          self.model.symlog_max,
                                          self.model.num_bins) * discount[t] / self.model.num_value_nets

      consistency_loss = (consistency_loss / horizon).mean()
      reward_loss = (reward_loss / horizon).mean()
      value_loss = (value_loss / horizon).mean()
      continue_loss = (continue_loss / horizon).mean()
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
          'zs': sg(zs)
      }

    # Update world model
    (encoder_grads, dynamics_grads, value_grads, reward_grads, continue_grads), model_info = jax.grad(
        world_model_loss_fn, argnums=(0, 1, 2, 3, 4), has_aux=True)(
            self.model.encoder.params,
            self.model.dynamics_model.params,
            self.model.value_model.params,
            self.model.reward_model.params,
            self.model.continue_model.params if self.model.predict_continues else None)
    zs = model_info.pop('zs')

    new_encoder = self.model.encoder.apply_gradients(grads=encoder_grads)
    new_dynamics_model = self.model.dynamics_model.apply_gradients(
        grads=dynamics_grads)
    new_reward_model = self.model.reward_model.apply_gradients(
        grads=reward_grads)
    new_value_model = self.model.value_model.apply_gradients(
        grads=value_grads)
    new_target_value_model = self.model.target_value_model.replace(
        params=optax.incremental_update(
            new_value_model.params,
            self.model.target_value_model.params,
            self.tau))
    if self.model.predict_continues:
      new_continue_model = self.model.continue_model.apply_gradients(
          grads=continue_grads)
    else:
      new_continue_model = self.model.continue_model

    # Update policy
    def policy_loss_fn(params: Dict):
      actions, _, _, log_probs = self.model.sample_actions(
          zs, params, key=policy_key)

      # Compute Q-values
      Qs, _ = self.model.Q(
          zs, actions, new_value_model.params, value_dropout_key2)
      Q = jnp.mean(Qs, axis=0)
      # Apply running scale
      scale = percentile_normalization(Q[0], self.scale)
      scale = jnp.clip(scale, 1, None)
      Q = Q / scale

      # Compute policy objective (equation 4)
      rho = self.rho ** jnp.arange(len(Q))
      policy_loss = ((self.entropy_coef * log_probs -
                     Q).mean(axis=1) * rho).mean()
      return policy_loss, {'policy_loss': policy_loss,
                           'policy_scale': scale}
    policy_grads, policy_info = jax.grad(policy_loss_fn, has_aux=True)(
        self.model.policy_model.params)
    new_policy = self.model.policy_model.apply_gradients(grads=policy_grads)

    # Update model
    new_agent = self.replace(model=self.model.replace(
        encoder=new_encoder,
        dynamics_model=new_dynamics_model,
        reward_model=new_reward_model,
        value_model=new_value_model,
        policy_model=new_policy,
        target_value_model=new_target_value_model,
        continue_model=new_continue_model),
        scale=policy_info['policy_scale'])
    info = {**model_info, **policy_info}

    return new_agent, info

  @jax.jit
  def estimate_value(self, z: jax.Array, actions: jax.Array, key: PRNGKeyArray) -> jax.Array:
    G, discount = 0, 1
    for t in range(self.horizon):
      reward, _ = self.model.reward(
          z, actions[t], self.model.reward_model.params)
      z = self.model.next(z, actions[t], self.model.dynamics_model.params)
      G += discount * reward

      if self.model.predict_continues:
        continues = jax.nn.sigmoid(self.model.continue_model.apply_fn(
            {'params': self.model.continue_model.params}, z)).squeeze(-1)
      else:
        continues = 1

      discount *= self.discount * continues

    action_key, dropout_key = jax.random.split(key, 2)
    next_action = self.model.sample_actions(
        z, self.model.policy_model.params, key=action_key)[0]

    # Sample two Q-values from the ensemble
    Qs, _ = self.model.Q(
        z, next_action, self.model.value_model.params, key=dropout_key)
    Q = jnp.mean(Qs, axis=0)
    return sg(G + discount * Q)

  @jax.jit
  def td_target(self, next_z: jax.Array, reward: jax.Array, done: jax.Array,
                key: PRNGKeyArray) -> jax.Array:
    action_key, ensemble_key, dropout_key = jax.random.split(key, 3)
    next_action = self.model.sample_actions(
        next_z, self.model.policy_model.params, key=action_key)[0]

    # Sample two Q-values from the target ensemble
    all_inds = jnp.arange(0, self.model.num_value_nets)
    inds = jax.random.choice(ensemble_key, a=all_inds,
                             shape=(2, ), replace=False)
    Qs, _ = self.model.Q(
        next_z, next_action, self.model.target_value_model.params, key=dropout_key)
    Q = jnp.min(Qs[inds], axis=0)
    return sg(reward + (1 - done) * self.discount * Q)

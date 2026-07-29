from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from tdmpc2_jax.common.loss import soft_crossentropy
from tdmpc2_jax.common.util import sg


PHASE_NAMES = ('A', 'B1', 'B2', 'B3', 'B4')
PHASE_TARGET_SIZES = (29, 18, 12, 9, 9)
EPS = 1e-6


def _tree_l2_sq(tree) -> jax.Array:
  leaves = jax.tree.leaves(tree)
  if not leaves:
    return jnp.array(0.0, dtype=jnp.float32)
  return sum(
      jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float32)))
      for leaf in leaves
      if leaf is not None
  )


def _normalise(values: np.ndarray, inverse: bool = False) -> np.ndarray:
  values = np.asarray(values, dtype=np.float32)
  if not values.size:
    return values
  vmin = float(np.min(values))
  vmax = float(np.max(values))
  if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < EPS:
    out = np.ones_like(values, dtype=np.float32)
  else:
    out = (values - vmin) / (vmax - vmin + EPS)
  if inverse:
    out = 1.0 - out
  return np.clip(out, 0.0, 1.0)


def _tolerance_linear(x: np.ndarray,
                      lower: float,
                      margin: float,
                      value_at_margin: float = 0.0) -> np.ndarray:
  x = np.asarray(x, dtype=np.float32)
  below = x < lower
  if margin <= 0:
    return np.where(below, value_at_margin, 1.0).astype(np.float32)
  delta = (lower - x) / margin
  return np.where(
      below,
      np.clip(1.0 - (1.0 - value_at_margin) * delta, value_at_margin, 1.0),
      1.0,
  ).astype(np.float32)


@struct.dataclass
class HorizonSearchState:
  horizons: jax.Array
  active_mask: jax.Array
  best_h: jax.Array
  score_sum: jax.Array
  score_sum_sq: jax.Array
  score_count: jax.Array
  gauss_mean: jax.Array
  gauss_post_std: jax.Array
  prob: jax.Array
  entropy: jax.Array
  norm_entropy: jax.Array
  phase_id: jax.Array
  phase_sample_count_A: jax.Array
  phase_sample_count_B1: jax.Array
  phase_sample_count_B2: jax.Array
  phase_sample_count_B3: jax.Array
  phase_sample_count_B4: jax.Array
  query_interval_steps: jax.Array
  start_query_step: jax.Array
  next_query_step: jax.Array
  hmax: int = struct.field(pytree_node=False)
  roughness_probe: str = struct.field(pytree_node=False)
  robust_return: str = struct.field(pytree_node=False)
  phase_min_samples_to_drop: int = struct.field(pytree_node=False)
  candidate_budget: Tuple[int, int, int, int, int] = struct.field(pytree_node=False)
  selection_return_power: float = struct.field(pytree_node=False)
  roughness_weight: float = struct.field(pytree_node=False)
  return_std_weight: float = struct.field(pytree_node=False)
  learner_proxy_enabled: bool = struct.field(pytree_node=False)
  learner_proxy_weight: float = struct.field(pytree_node=False)
  learner_proxy_mode: str = struct.field(pytree_node=False)
  local_window_radius: int = struct.field(pytree_node=False)
  max_transition_delta: int = struct.field(pytree_node=False)
  incumbent_switch_margin: float = struct.field(pytree_node=False)
  credible_transition_enabled: bool = struct.field(pytree_node=False)
  credible_transition_rule: str = struct.field(pytree_node=False)
  credible_transition_min_prob: float = struct.field(pytree_node=False)
  transition_cost_scale: float = struct.field(pytree_node=False)
  transition_risk_weight: float = struct.field(pytree_node=False)
  transition_min_expected_net: float = struct.field(pytree_node=False)
  transition_model_weight: float = struct.field(pytree_node=False)
  transition_probe_weight: float = struct.field(pytree_node=False)
  transition_planner_weight: float = struct.field(pytree_node=False)
  transition_roughness_weight: float = struct.field(pytree_node=False)
  transition_return_std_weight: float = struct.field(pytree_node=False)
  transition_uncertainty_floor: float = struct.field(pytree_node=False)

  @classmethod
  def create(cls,
             horizons: Sequence[int],
             hmax: int,
             query_interval_steps: int,
             start_query_step: Optional[int] = None,
             initial_horizon: Optional[int] = None,
             roughness_probe: str = 'projected_jvp',
             robust_return: str = 'mean_minus_std',
             phase_min_samples_to_drop: int = 3,
             candidate_budget: Optional[Mapping[str, int]] = None,
             selection_return_power: float = 1.0,
             roughness_weight: float = 1.0,
             return_std_weight: float = 1.0,
             learner_proxy_enabled: bool = False,
             learner_proxy_weight: float = 0.0,
             learner_proxy_mode: str = 'probe_mean_loss',
             local_window_radius: int = 0,
             max_transition_delta: int = 0,
             incumbent_switch_margin: float = 0.0,
             credible_transition_enabled: bool = False,
             credible_transition_rule: str = 'probability',
             credible_transition_min_prob: float = 0.0,
             transition_cost_scale: float = 0.0,
             transition_risk_weight: float = 1.0,
             transition_min_expected_net: float = 0.0,
             transition_model_weight: float = 1.0,
             transition_probe_weight: float = 1.0,
             transition_planner_weight: float = 1.0,
             transition_roughness_weight: float = 1.0,
             transition_return_std_weight: float = 1.0,
             transition_uncertainty_floor: float = 0.05) -> 'HorizonSearchState':
    horizons_arr = np.asarray(tuple(horizons), dtype=np.int32)
    if initial_horizon is None or int(initial_horizon) not in set(horizons_arr.tolist()):
      initial_horizon = int(horizons_arr[0])
    if candidate_budget is None:
      phase_budget = PHASE_TARGET_SIZES
    else:
      phase_budget = tuple(
          int(candidate_budget[name]) for name in PHASE_NAMES
      )
    nh = horizons_arr.shape[0]
    phase_budget = tuple(min(int(budget), nh) for budget in phase_budget)
    uniform = np.full(nh, 1.0 / max(nh, 1), dtype=np.float32)
    zeros = np.zeros(nh, dtype=np.float32)
    first_query_step = (
        int(query_interval_steps)
        if start_query_step is None
        else int(start_query_step)
    )
    return cls(
        horizons=jnp.asarray(horizons_arr),
        active_mask=jnp.ones(nh, dtype=bool),
        best_h=jnp.asarray(initial_horizon, dtype=jnp.int32),
        score_sum=jnp.asarray(zeros),
        score_sum_sq=jnp.asarray(zeros),
        score_count=jnp.asarray(zeros, dtype=jnp.int32),
        gauss_mean=jnp.asarray(zeros),
        gauss_post_std=jnp.ones(nh, dtype=jnp.float32),
        prob=jnp.asarray(uniform),
        entropy=jnp.asarray(math.log(max(nh, 1)), dtype=jnp.float32),
        norm_entropy=jnp.asarray(1.0, dtype=jnp.float32),
        phase_id=jnp.asarray(0, dtype=jnp.int32),
        phase_sample_count_A=jnp.zeros(nh, dtype=jnp.int32),
        phase_sample_count_B1=jnp.zeros(nh, dtype=jnp.int32),
        phase_sample_count_B2=jnp.zeros(nh, dtype=jnp.int32),
        phase_sample_count_B3=jnp.zeros(nh, dtype=jnp.int32),
        phase_sample_count_B4=jnp.zeros(nh, dtype=jnp.int32),
        query_interval_steps=jnp.asarray(query_interval_steps, dtype=jnp.int32),
        start_query_step=jnp.asarray(first_query_step, dtype=jnp.int32),
        next_query_step=jnp.asarray(first_query_step, dtype=jnp.int32),
        hmax=int(hmax),
        roughness_probe=roughness_probe,
        robust_return=robust_return,
        phase_min_samples_to_drop=int(phase_min_samples_to_drop),
        candidate_budget=phase_budget,
        selection_return_power=float(selection_return_power),
        roughness_weight=float(roughness_weight),
        return_std_weight=float(return_std_weight),
        learner_proxy_enabled=bool(learner_proxy_enabled),
        learner_proxy_weight=float(learner_proxy_weight),
        learner_proxy_mode=str(learner_proxy_mode),
        local_window_radius=int(local_window_radius),
        max_transition_delta=int(max_transition_delta),
        incumbent_switch_margin=float(incumbent_switch_margin),
        credible_transition_enabled=bool(credible_transition_enabled),
        credible_transition_rule=str(credible_transition_rule),
        credible_transition_min_prob=float(credible_transition_min_prob),
        transition_cost_scale=float(transition_cost_scale),
        transition_risk_weight=float(transition_risk_weight),
        transition_min_expected_net=float(transition_min_expected_net),
        transition_model_weight=float(transition_model_weight),
        transition_probe_weight=float(transition_probe_weight),
        transition_planner_weight=float(transition_planner_weight),
        transition_roughness_weight=float(transition_roughness_weight),
        transition_return_std_weight=float(transition_return_std_weight),
        transition_uncertainty_floor=float(transition_uncertainty_floor),
    )

  def active_horizons(self) -> np.ndarray:
    horizons = np.asarray(self.horizons)
    return horizons[np.asarray(self.active_mask)]

  def phase_name(self) -> str:
    return PHASE_NAMES[int(np.asarray(self.phase_id))]

  def should_query(self, step: int) -> bool:
    return step >= int(np.asarray(self.next_query_step))

@struct.dataclass
class DenseQueryResult:
  horizon_state: HorizonSearchState
  selected_horizon: jax.Array
  proposed_horizon: jax.Array
  candidate_horizons: jax.Array
  candidate_mask: jax.Array
  fitness: jax.Array
  deployment_score: jax.Array
  return_term: jax.Array
  roughness_term: jax.Array
  sigma_r_term: jax.Array
  learner_proxy_term: jax.Array
  transition_cost: jax.Array
  transition_adjusted_score: jax.Array
  switch_probability: jax.Array
  expected_improvement: jax.Array
  expected_loss: jax.Array
  expected_net_benefit: jax.Array
  incumbent_deployment_score: jax.Array
  proposed_deployment_score: jax.Array
  proposed_transition_cost: jax.Array
  proposed_switch_probability: jax.Array
  proposed_expected_net_benefit: jax.Array
  robust_return: jax.Array
  prefix_objectives: jax.Array
  probe_prefixes: jax.Array
  planner_prefix_returns: jax.Array
  roughness: jax.Array
  env_mean: jax.Array
  env_std: jax.Array


@struct.dataclass
class DenseQueryModelStage:
  prefix_objectives: jax.Array
  probe_prefixes: jax.Array
  planner_prefix_returns: jax.Array
  roughness: jax.Array
  candidate_horizons: jax.Array
  candidate_mask: jax.Array
  candidate_indices: jax.Array


@dataclass(frozen=True)
class DenseQueryKernelBundle:
  model_stage: Callable[..., DenseQueryModelStage]
  env_stage: Callable[..., Tuple[jax.Array, jax.Array]]
  finalize_stage: Callable[..., DenseQueryResult]


_DENSE_QUERY_KERNEL_CACHE: Dict[Tuple[int, int, int], DenseQueryKernelBundle] = {}


def _normalise_jax(values: jax.Array, inverse: bool = False) -> jax.Array:
  values = jnp.asarray(values, dtype=jnp.float32)
  vmin = jnp.min(values)
  vmax = jnp.max(values)
  scale = jnp.maximum(vmax - vmin, EPS)
  out = jnp.where(
      jnp.logical_or(~jnp.isfinite(vmin), ~jnp.isfinite(vmax)),
      jnp.ones_like(values),
      (values - vmin) / scale,
  )
  out = jnp.where(scale <= EPS, jnp.ones_like(out), out)
  if inverse:
    out = 1.0 - out
  return jnp.clip(out, 0.0, 1.0)


def _standard_normal_cdf(x: jax.Array) -> jax.Array:
  return 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))


def _standard_normal_pdf(x: jax.Array) -> jax.Array:
  return jnp.exp(-0.5 * jnp.square(x)) / jnp.sqrt(2.0 * jnp.pi)


def _normalised_shift(values: jax.Array, incumbent_idx: jax.Array) -> jax.Array:
  values = jnp.asarray(values, dtype=jnp.float32)
  shift = jnp.abs(values - values[incumbent_idx])
  scale = jnp.maximum(jnp.max(shift), EPS)
  return jnp.clip(shift / scale, 0.0, 1.0)


def _transition_cost(state: HorizonSearchState,
                     model_stage: DenseQueryModelStage,
                     env_std: jax.Array,
                     incumbent_idx: jax.Array) -> jax.Array:
  weights = jnp.asarray(
      [
          state.transition_model_weight,
          state.transition_probe_weight,
          state.transition_planner_weight,
          state.transition_roughness_weight,
          state.transition_return_std_weight,
      ],
      dtype=jnp.float32,
  )
  components = jnp.stack(
      [
          _normalised_shift(model_stage.prefix_objectives, incumbent_idx),
          _normalised_shift(model_stage.probe_prefixes, incumbent_idx),
          _normalised_shift(model_stage.planner_prefix_returns, incumbent_idx),
          _normalise_jax(model_stage.roughness),
          _normalise_jax(env_std),
      ],
      axis=0,
  )
  weight_sum = jnp.maximum(jnp.sum(weights), EPS)
  cost = state.transition_cost_scale * jnp.sum(components * weights[:, None], axis=0) / weight_sum
  cost = cost.at[incumbent_idx].set(0.0)
  return jnp.clip(cost, 0.0, 1.0)


def _phase_counts_matrix(state: HorizonSearchState) -> jax.Array:
  return jnp.stack(
      (
          state.phase_sample_count_A,
          state.phase_sample_count_B1,
          state.phase_sample_count_B2,
          state.phase_sample_count_B3,
          state.phase_sample_count_B4,
      ),
      axis=0,
  )


def _phase_counts(state: HorizonSearchState) -> np.ndarray:
  return np.asarray(_phase_counts_matrix(state)[state.phase_id])


def _replace_phase_counts(state: HorizonSearchState,
                          counts: jax.Array,
                          phase_id: Optional[jax.Array] = None) -> HorizonSearchState:
  phase_index = state.phase_id if phase_id is None else jnp.asarray(phase_id, dtype=jnp.int32)
  matrix = _phase_counts_matrix(state).at[phase_index].set(
      jnp.asarray(counts, dtype=jnp.int32)
  )
  return state.replace(
      phase_sample_count_A=matrix[0],
      phase_sample_count_B1=matrix[1],
      phase_sample_count_B2=matrix[2],
      phase_sample_count_B3=matrix[3],
      phase_sample_count_B4=matrix[4],
  )


def _phase_budget(state: HorizonSearchState) -> jax.Array:
  budgets = jnp.asarray(state.candidate_budget, dtype=jnp.int32)
  return budgets[state.phase_id]


def _rank_scores(state: HorizonSearchState) -> jax.Array:
  return jnp.where(
      state.active_mask,
      state.prob * 1e3 +
      state.gauss_mean * 1e1 -
      state.horizons.astype(jnp.float32) * 1e-3,
      -jnp.inf,
  )


def _candidate_scores(state: HorizonSearchState) -> jax.Array:
  counts = _phase_counts_matrix(state)[state.phase_id].astype(jnp.float32)
  incumbent_mask = state.horizons == state.best_h
  if state.local_window_radius > 0:
    candidate_scope = jnp.logical_and(
        state.active_mask,
        jnp.abs(state.horizons - state.best_h) <= state.local_window_radius,
    )
  else:
    candidate_scope = state.active_mask
  under_sampled_mask = jnp.logical_and(
      candidate_scope,
      counts < float(state.phase_min_samples_to_drop),
  )
  return jnp.where(
      candidate_scope,
      incumbent_mask.astype(jnp.float32) * 1e6 +
      under_sampled_mask.astype(jnp.float32) * 1e5 +
      state.prob * 1e3 +
      state.gauss_mean * 1e1 -
      counts * 1e-2 -
      state.horizons.astype(jnp.float32) * 1e-3,
      -jnp.inf,
  )


def _select_candidate_slots(state: HorizonSearchState,
                            candidate_slots: int) -> Tuple[jax.Array, jax.Array, jax.Array]:
  scores, indices = jax.lax.top_k(_candidate_scores(state), candidate_slots)
  mask = jnp.isfinite(scores)
  return state.horizons[indices], mask, indices


@jax.jit
def _maybe_advance_phase(state: HorizonSearchState) -> HorizonSearchState:
  phase_counts = _phase_counts_matrix(state)[state.phase_id]
  ready = jnp.logical_and(
      state.phase_id < len(PHASE_NAMES) - 1,
      jnp.all(jnp.where(
          state.active_mask,
          phase_counts >= state.phase_min_samples_to_drop,
          True,
      )),
  )

  def advance(st: HorizonSearchState) -> HorizonSearchState:
    next_phase_id = st.phase_id + 1
    next_budget = jnp.asarray(st.candidate_budget, dtype=jnp.int32)[next_phase_id]
    _, ranked_indices = jax.lax.top_k(_rank_scores(st), st.horizons.shape[0])
    keep_values = jnp.arange(st.horizons.shape[0], dtype=jnp.int32) < next_budget
    new_active_mask = jnp.zeros_like(st.active_mask).at[ranked_indices].set(keep_values)
    st = st.replace(
        active_mask=new_active_mask,
        phase_id=next_phase_id,
    )
    return _replace_phase_counts(
        st,
        jnp.zeros_like(phase_counts, dtype=jnp.int32),
        phase_id=next_phase_id,
    )

  return jax.lax.cond(ready, advance, lambda st: st, state)


@partial(jax.jit, static_argnames=('mc_samples',))
def _update_distribution(state: HorizonSearchState,
                         key: jax.Array,
                         selected_horizon: jax.Array,
                         mc_samples: int = 512) -> HorizonSearchState:
  counts = state.score_count.astype(jnp.float32)
  mean = state.score_sum / jnp.maximum(counts, 1.0)
  var = state.score_sum_sq / jnp.maximum(counts, 1.0) - jnp.square(mean)
  var = jnp.where(state.score_count > 1, jnp.maximum(var, EPS), 1.0)
  post_std = jnp.where(
      state.score_count > 0,
      jnp.sqrt(var / jnp.maximum(counts, 1.0)),
      1.0,
  )

  samples = mean[None, :] + post_std[None, :] * jax.random.normal(
      key,
      shape=(mc_samples, mean.shape[0]),
      dtype=jnp.float32,
  )
  samples = jnp.where(state.active_mask[None, :], samples, -jnp.inf)
  winners = jnp.argmax(samples, axis=1)
  prob = jnp.zeros((mean.shape[0],), dtype=jnp.float32).at[winners].add(
      jnp.ones((mc_samples,), dtype=jnp.float32)
  )
  prob_total = jnp.sum(prob)
  active_count = jnp.maximum(jnp.sum(state.active_mask.astype(jnp.float32)), 1.0)
  uniform = state.active_mask.astype(jnp.float32) / active_count
  prob = jnp.where(prob_total > 0.0, prob / prob_total, uniform)
  prob = jnp.where(state.active_mask, prob, 0.0)
  entropy = -jnp.sum(
      jnp.where(state.active_mask, prob * jnp.log(prob + EPS), 0.0)
  )
  norm_entropy = entropy / jnp.log(jnp.maximum(active_count, 2.0))

  state = state.replace(
      gauss_mean=mean.astype(jnp.float32),
      gauss_post_std=post_std.astype(jnp.float32),
      prob=prob.astype(jnp.float32),
      entropy=entropy.astype(jnp.float32),
      norm_entropy=norm_entropy.astype(jnp.float32),
      best_h=jnp.asarray(selected_horizon, dtype=jnp.int32),
  )
  return _maybe_advance_phase(state)


@jax.jit
def _per_step_losses(agent, batch: Mapping[str, jax.Array], key: jax.Array):
  hmax = batch['action'].shape[0]
  batch_size = batch['action'].shape[1]
  keys = jax.random.split(key, 1 + 3 * hmax)
  encoder_key = keys[0]
  sample_action_keys = keys[1:1 + hmax]
  target_q_keys = keys[1 + hmax:1 + 2 * hmax]
  value_keys = keys[1 + 2 * hmax:1 + 3 * hmax]

  all_obs = jax.tree.map(
      lambda x, y: jnp.stack([x, y], axis=0),
      batch['observation'],
      batch['next_observation'],
  )
  all_zs = agent.model.encode(
      obs=all_obs,
      params=agent.model.encoder.params,
      key=encoder_key,
  )
  encoder_zs = all_zs[0]
  next_zs = all_zs[1]
  done = jnp.logical_or(batch['terminated'], batch['truncated'])
  z0 = encoder_zs[0]

  def step_fn(carry, inputs):
    z_t, finished = carry
    action_t, reward_t, terminated_t, done_t, next_z_t, sample_key, target_q_key, value_key = inputs
    next_latent = agent.model.next(
        z=z_t,
        a=action_t,
        params=agent.model.dynamics_model.params,
    )
    cons_loss = jnp.mean((next_latent - sg(next_z_t)) ** 2, axis=-1)
    reward_logits = agent.model.reward(
        z=z_t,
        a=action_t,
        params=agent.model.reward_model.params,
    )[1]
    reward_loss = soft_crossentropy(
        pred_logits=reward_logits,
        target=reward_t,
        low=agent.model.symlog_min,
        high=agent.model.symlog_max,
        num_bins=agent.model.num_bins,
    )
    next_action = agent.model.sample_actions(
        z=next_z_t,
        deterministic=False,
        params=agent.model.policy_model.params,
        key=sample_key,
    )[0]
    target_Q, _ = agent.model.Q(
        z=next_z_t,
        a=next_action,
        params=agent.model.target_value_model.params,
        key=target_q_key,
    )
    td_target = reward_t + (1 - terminated_t) * agent.discount * target_Q.mean(axis=0)
    _, value_logits = agent.model.Q(
        z=z_t,
        a=action_t,
        params=agent.model.value_model.params,
        key=value_key,
    )
    value_loss = soft_crossentropy(
        pred_logits=value_logits,
        target=sg(td_target),
        low=agent.model.symlog_min,
        high=agent.model.symlog_max,
        num_bins=agent.model.num_bins,
    ).mean(axis=0)
    total = jnp.where(
        finished,
        0.0,
        agent.consistency_loss_scale * cons_loss +
        agent.reward_loss_scale * reward_loss +
        agent.value_loss_scale * value_loss,
    )
    probe_total = jnp.where(
        finished,
        0.0,
        agent.consistency_loss_scale * cons_loss +
        agent.reward_loss_scale * reward_loss,
    )
    finished = jnp.logical_or(finished, done_t)
    return (next_latent, finished), (jnp.mean(total), jnp.mean(probe_total), next_latent)

  (_, _), (per_step_total, per_step_probe, rollout_states) = jax.lax.scan(
      step_fn,
      (z0, jnp.zeros((batch_size,), dtype=bool)),
      (
          batch['action'],
          batch['reward'],
          batch['terminated'],
          done,
          next_zs,
          sample_action_keys,
          target_q_keys,
          value_keys,
      ),
  )
  rollout_states = jnp.concatenate([z0[None, ...], rollout_states], axis=0)
  return per_step_total, per_step_probe, rollout_states, z0


@partial(jax.jit, static_argnames=('hmax', 'population_size'))
def _planner_prefix_returns(agent,
                            z0: jax.Array,
                            hmax: int,
                            key: jax.Array,
                            population_size: Optional[int] = None) -> jax.Array:
  pop = int(population_size or agent.population_size)
  if z0.ndim > 1:
    z0 = jnp.mean(z0, axis=0)
  keys = jax.random.split(key, 1 + 2 * hmax)
  action_keys = keys[1:1 + hmax]
  q_keys = keys[1 + hmax:1 + 2 * hmax]
  z_t = z0[None, :].repeat(pop, axis=0)

  def planner_step(z_carry, action_key):
    action = agent.model.sample_actions(
        z=z_carry,
        deterministic=False,
        params=agent.model.policy_model.params,
        key=action_key,
    )[0]
    reward, _ = agent.model.reward(
        z=z_carry,
        a=action,
        params=agent.model.reward_model.params,
    )
    next_z = agent.model.next(
        z=z_carry,
        a=action,
        params=agent.model.dynamics_model.params,
    )
    return next_z, (reward, next_z)

  _, (rewards, next_states) = jax.lax.scan(planner_step, z_t, action_keys)
  rewards = jnp.swapaxes(rewards, 0, 1)
  states = jnp.swapaxes(jnp.concatenate([z_t[None, ...], next_states], axis=0), 0, 1)
  discounts = jnp.power(agent.discount, jnp.arange(hmax, dtype=jnp.float32))
  discounted_rewards = rewards * discounts[None, :]
  prefix_rewards = jnp.cumsum(discounted_rewards, axis=1)

  def bootstrap_step(inputs):
    state_h, q_key, horizon_idx = inputs
    next_action = agent.model.sample_actions(
        z=state_h,
        deterministic=False,
        params=agent.model.policy_model.params,
        key=q_key,
    )[0]
    q_values, _ = agent.model.Q(
        z=state_h,
        a=next_action,
        params=agent.model.value_model.params,
        key=q_key,
    )
    return jnp.power(agent.discount, horizon_idx + 1) * q_values.mean(axis=0)

  bootstrap_values = jax.vmap(bootstrap_step)(
      (states[:, 1:, :].transpose(1, 0, 2), q_keys, jnp.arange(hmax, dtype=jnp.int32))
  ).transpose(1, 0)
  returns = prefix_rewards + bootstrap_values
  return jnp.max(returns, axis=0)


@jax.jit
def _roughness(agent,
               batch: Mapping[str, jax.Array],
               horizons: jax.Array,
               key: jax.Array) -> jax.Array:
  hmax = batch['action'].shape[0]
  horizon_idx = jnp.clip(jnp.asarray(horizons, dtype=jnp.int32) - 1, 0, hmax - 1)

  def probe_prefix(params):
    dynamics_params, reward_params = params
    encoder_key = jax.random.split(key, 1)[0]
    all_obs = jax.tree.map(
        lambda x, y: jnp.stack([x, y], axis=0),
        batch['observation'],
        batch['next_observation'],
    )
    all_zs = agent.model.encode(
        obs=all_obs,
        params=agent.model.encoder.params,
        key=encoder_key,
    )
    next_zs = all_zs[1]
    done = jnp.logical_or(batch['terminated'], batch['truncated'])

    def step_fn(carry, inputs):
      z_t, finished = carry
      action_t, reward_t, done_t, next_z_t = inputs
      next_latent = agent.model.next(
          z=z_t,
          a=action_t,
          params=dynamics_params,
      )
      cons_loss = jnp.mean((next_latent - sg(next_z_t)) ** 2, axis=-1)
      reward_logits = agent.model.reward(
          z=z_t,
          a=action_t,
          params=reward_params,
      )[1]
      reward_loss = soft_crossentropy(
          pred_logits=reward_logits,
          target=reward_t,
          low=agent.model.symlog_min,
          high=agent.model.symlog_max,
          num_bins=agent.model.num_bins,
      )
      step_loss = jnp.where(
          finished,
          0.0,
          agent.consistency_loss_scale * cons_loss +
          agent.reward_loss_scale * reward_loss,
      )
      finished = jnp.logical_or(finished, done_t)
      return (next_latent, finished), jnp.mean(step_loss)

    (_, _), step_losses = jax.lax.scan(
        step_fn,
        (all_zs[0][0], jnp.zeros((batch['action'].shape[1],), dtype=bool)),
        (batch['action'], batch['reward'], done, next_zs),
    )
    return jnp.cumsum(step_losses)[horizon_idx]

  proj_keys = jax.random.split(key, 2)
  params = (agent.model.dynamics_model.params, agent.model.reward_model.params)

  def tangent_from_key(tangent_key, param_tree):
    leaves, treedef = jax.tree_util.tree_flatten(param_tree)
    tangent_keys = jax.random.split(tangent_key, len(leaves))
    tangent_leaves = [
        jax.random.normal(k, shape=leaf.shape, dtype=leaf.dtype)
        for k, leaf in zip(tangent_keys, leaves)
    ]
    return jax.tree_util.tree_unflatten(treedef, tangent_leaves)

  def project(proj_key):
    dyn_key, rew_key = jax.random.split(proj_key)
    tangent = (
        tangent_from_key(dyn_key, params[0]),
        tangent_from_key(rew_key, params[1]),
    )
    _, directional = jax.jvp(probe_prefix, (params,), (tangent,))
    return directional

  stacked = jax.vmap(project)(proj_keys)
  return jnp.sqrt(jnp.mean(jnp.square(stacked), axis=0) + EPS)


@jax.jit
def _state_with_score_updates(state: HorizonSearchState,
                              candidate_indices: jax.Array,
                              candidate_mask: jax.Array,
                              candidate_scores: jax.Array,
                              query_step: jax.Array,
                              selected_horizon: jax.Array,
                              key: jax.Array) -> HorizonSearchState:
  score_delta = jnp.where(candidate_mask, candidate_scores, 0.0)
  count_delta = candidate_mask.astype(state.score_count.dtype)
  score_sum = state.score_sum.at[candidate_indices].add(score_delta)
  score_sum_sq = state.score_sum_sq.at[candidate_indices].add(jnp.square(score_delta))
  score_count = state.score_count.at[candidate_indices].add(count_delta)
  phase_counts = _phase_counts_matrix(state)[state.phase_id].astype(state.score_count.dtype)
  phase_counts = phase_counts.at[candidate_indices].add(count_delta.astype(phase_counts.dtype))

  state = state.replace(
      score_sum=score_sum,
      score_sum_sq=score_sum_sq,
      score_count=score_count,
      best_h=jnp.asarray(selected_horizon, dtype=jnp.int32),
      next_query_step=jnp.asarray(
          query_step + state.query_interval_steps,
          dtype=jnp.int32,
      ),
  )
  state = _replace_phase_counts(state, phase_counts)
  return _update_distribution(state, key, selected_horizon)


def _build_dense_query_kernel(eval_state: Any,
                              candidate_slots: int,
                              env_eval_steps: Optional[int]) -> DenseQueryKernelBundle:
  supports_env_eval = hasattr(eval_state, 'evaluate_candidate_horizons_dense')
  eval_steps = int(env_eval_steps) if env_eval_steps is not None else None

  @jax.jit
  def run_model_stage(agent,
                      replay_batch: Mapping[str, jax.Array],
                      horizon_state: HorizonSearchState,
                      rng: jax.Array) -> DenseQueryModelStage:
    model_key, planner_key = jax.random.split(rng, 2)
    horizon_indices = jnp.clip(
        horizon_state.horizons - 1,
        0,
        horizon_state.hmax - 1,
    )
    per_step_total, prefix_probe_steps, _, z0 = _per_step_losses(
        agent,
        replay_batch,
        model_key,
    )
    prefix_objectives = jnp.cumsum(per_step_total)[horizon_indices]
    probe_prefixes = jnp.cumsum(prefix_probe_steps)[horizon_indices]
    planner_prefix_returns = _planner_prefix_returns(
        agent,
        z0,
        horizon_state.hmax,
        planner_key,
    )[horizon_indices]
    roughness = _roughness(
        agent,
        replay_batch,
        horizon_state.horizons,
        model_key,
    )

    candidate_horizons, candidate_mask, candidate_indices = _select_candidate_slots(
        horizon_state,
        candidate_slots,
    )
    return DenseQueryModelStage(
        prefix_objectives=prefix_objectives,
        probe_prefixes=probe_prefixes,
        planner_prefix_returns=planner_prefix_returns,
        roughness=roughness,
        candidate_horizons=candidate_horizons,
        candidate_mask=candidate_mask,
        candidate_indices=candidate_indices,
    )

  @jax.jit
  def run_env_stage(agent,
                    model_stage: DenseQueryModelStage,
                    env_key: jax.Array) -> Tuple[jax.Array, jax.Array]:
    env_mean = model_stage.planner_prefix_returns
    env_std = jnp.zeros_like(model_stage.planner_prefix_returns)
    if supports_env_eval:
      candidate_env_mean, candidate_env_std = eval_state.evaluate_candidate_horizons_dense(
          agent=agent,
          candidate_horizons=model_stage.candidate_horizons,
          candidate_mask=model_stage.candidate_mask,
          key=env_key,
          steps_per_episode=eval_steps,
      )
    else:
      candidate_env_mean = model_stage.planner_prefix_returns[model_stage.candidate_indices]
      candidate_env_std = jnp.zeros((candidate_slots,), dtype=jnp.float32)
    env_mean = env_mean.at[model_stage.candidate_indices].set(
        jnp.where(
            model_stage.candidate_mask,
            candidate_env_mean,
            env_mean[model_stage.candidate_indices],
        )
    )
    env_std = env_std.at[model_stage.candidate_indices].set(
        jnp.where(
            model_stage.candidate_mask,
            candidate_env_std,
            env_std[model_stage.candidate_indices],
        )
    )
    return env_mean, env_std

  @jax.jit
  def run_finalize_stage(horizon_state: HorizonSearchState,
                         model_stage: DenseQueryModelStage,
                         env_mean: jax.Array,
                         env_std: jax.Array,
                         query_step: jax.Array,
                         score_key: jax.Array) -> DenseQueryResult:
    if horizon_state.robust_return == 'mean_minus_std':
      robust_return = env_mean - env_std
    else:
      robust_return = env_mean
    return_term = _normalise_jax(robust_return)
    roughness_term = _normalise_jax(model_stage.roughness, inverse=True)
    sigma_r_term = _normalise_jax(env_std, inverse=True)
    horizon_lengths = jnp.maximum(horizon_state.horizons.astype(jnp.float32), 1.0)
    learner_proxy_raw = (
        model_stage.prefix_objectives / horizon_lengths
        if horizon_state.learner_proxy_mode == 'total_mean_loss'
        else model_stage.probe_prefixes / horizon_lengths
    )
    learner_proxy_term = _normalise_jax(learner_proxy_raw, inverse=True)
    learner_proxy_weight = (
        horizon_state.learner_proxy_weight
        if horizon_state.learner_proxy_enabled else 0.0
    )
    fitness = jnp.power(
        jnp.clip(return_term * roughness_term * sigma_r_term + EPS, EPS, None),
        0.25,
    )
    deployment_score = (
        jnp.power(jnp.clip(return_term, EPS, 1.0), horizon_state.selection_return_power) *
        jnp.power(jnp.clip(roughness_term, EPS, 1.0), horizon_state.roughness_weight) *
        jnp.power(jnp.clip(sigma_r_term, EPS, 1.0), horizon_state.return_std_weight) *
        jnp.power(jnp.clip(learner_proxy_term, EPS, 1.0), learner_proxy_weight)
    )
    candidate_eval_mask = jnp.zeros_like(horizon_state.active_mask).at[
        model_stage.candidate_indices
    ].set(model_stage.candidate_mask)
    deploy_mask = jnp.logical_and(horizon_state.active_mask, candidate_eval_mask)
    incumbent_idx = jnp.argmin(jnp.abs(horizon_state.horizons - horizon_state.best_h))
    transition_cost = _transition_cost(
        horizon_state,
        model_stage,
        env_std,
        incumbent_idx,
    )
    transition_adjusted_score = deployment_score - transition_cost
    proposal_score = (
        transition_adjusted_score
        if horizon_state.credible_transition_enabled
        else deployment_score
    )
    proposed_idx = jnp.argmax(jnp.where(deploy_mask, proposal_score, -jnp.inf))
    proposed_horizon = horizon_state.horizons[proposed_idx]
    incumbent_score = deployment_score[incumbent_idx]
    proposed_score = deployment_score[proposed_idx]
    delta_mean = deployment_score - incumbent_score - transition_cost
    posterior_std = jnp.maximum(
        horizon_state.gauss_post_std,
        horizon_state.transition_uncertainty_floor,
    )
    incumbent_std = jnp.maximum(
        horizon_state.gauss_post_std[incumbent_idx],
        horizon_state.transition_uncertainty_floor,
    )
    delta_std = jnp.sqrt(jnp.square(posterior_std) + jnp.square(incumbent_std))
    z_score = delta_mean / jnp.maximum(delta_std, EPS)
    switch_probability = _standard_normal_cdf(z_score)
    # Expected-improvement transition rule:
    #   X_h = score(h) - score(h_current) - transition_cost(h)
    #   EI_h = E[max(X_h, 0)], EL_h = E[max(-X_h, 0)]
    #   net_h = EI_h - risk_weight * EL_h.
    # This keeps transition decisions Bayesian but avoids the brittle hard
    # probability threshold that rejected useful h=4 switches and accepted
    # an over-confident late h=5 switch in previous runs.
    density = _standard_normal_pdf(z_score)
    expected_improvement = delta_mean * switch_probability + delta_std * density
    expected_loss = (
        -delta_mean * _standard_normal_cdf(-z_score) + delta_std * density
    )
    expected_net_benefit = (
        expected_improvement - horizon_state.transition_risk_weight * expected_loss
    )
    if (
        horizon_state.credible_transition_enabled and
        horizon_state.credible_transition_rule == 'expected_improvement'
    ):
      proposal_score = expected_net_benefit
    credible_switch = (
        expected_net_benefit[proposed_idx] > horizon_state.transition_min_expected_net
        if horizon_state.credible_transition_rule == 'expected_improvement'
        else switch_probability[proposed_idx] >= horizon_state.credible_transition_min_prob
    )
    margin_switch = proposed_score > incumbent_score + horizon_state.incumbent_switch_margin
    switch = jnp.where(
        horizon_state.credible_transition_enabled,
        jnp.logical_and(proposed_idx != incumbent_idx, credible_switch),
        margin_switch,
    )
    raw_selected_horizon = jnp.where(switch, proposed_horizon, horizon_state.best_h)
    if horizon_state.max_transition_delta > 0:
      delta = jnp.clip(
          raw_selected_horizon - horizon_state.best_h,
          -horizon_state.max_transition_delta,
          horizon_state.max_transition_delta,
      )
      target_horizon = horizon_state.best_h + delta
      selected_idx = jnp.argmin(jnp.abs(horizon_state.horizons - target_horizon))
    else:
      selected_idx = jnp.argmin(jnp.abs(horizon_state.horizons - raw_selected_horizon))
    selected_horizon = horizon_state.horizons[selected_idx]
    candidate_scores = jnp.where(
        model_stage.candidate_mask,
        fitness[model_stage.candidate_indices],
        0.0,
    )
    new_state = _state_with_score_updates(
        state=horizon_state,
        candidate_indices=model_stage.candidate_indices,
        candidate_mask=model_stage.candidate_mask,
        candidate_scores=candidate_scores,
        query_step=jnp.asarray(query_step, dtype=jnp.int32),
        selected_horizon=selected_horizon,
        key=score_key,
    )
    return DenseQueryResult(
        horizon_state=new_state,
        selected_horizon=selected_horizon,
        proposed_horizon=proposed_horizon,
        candidate_horizons=model_stage.candidate_horizons,
        candidate_mask=model_stage.candidate_mask,
        fitness=fitness,
        deployment_score=deployment_score,
        return_term=return_term,
        roughness_term=roughness_term,
        sigma_r_term=sigma_r_term,
        learner_proxy_term=learner_proxy_term,
        transition_cost=transition_cost,
        transition_adjusted_score=transition_adjusted_score,
        switch_probability=switch_probability,
        expected_improvement=expected_improvement,
        expected_loss=expected_loss,
        expected_net_benefit=expected_net_benefit,
        incumbent_deployment_score=incumbent_score,
        proposed_deployment_score=proposed_score,
        proposed_transition_cost=transition_cost[proposed_idx],
        proposed_switch_probability=switch_probability[proposed_idx],
        proposed_expected_net_benefit=expected_net_benefit[proposed_idx],
        robust_return=robust_return,
        prefix_objectives=model_stage.prefix_objectives,
        probe_prefixes=model_stage.probe_prefixes,
        planner_prefix_returns=model_stage.planner_prefix_returns,
        roughness=model_stage.roughness,
        env_mean=env_mean,
        env_std=env_std,
    )

  return DenseQueryKernelBundle(
      model_stage=run_model_stage,
      env_stage=run_env_stage,
      finalize_stage=run_finalize_stage,
  )


def _get_dense_query_kernel(eval_state: Any,
                            candidate_slots: int,
                            env_eval_steps: Optional[int]) -> DenseQueryKernelBundle:
  cache_key = (id(eval_state), int(candidate_slots), int(env_eval_steps or -1))
  if cache_key not in _DENSE_QUERY_KERNEL_CACHE:
    _DENSE_QUERY_KERNEL_CACHE[cache_key] = _build_dense_query_kernel(
        eval_state=eval_state,
        candidate_slots=int(candidate_slots),
        env_eval_steps=env_eval_steps,
    )
  return _DENSE_QUERY_KERNEL_CACHE[cache_key]

def build_dense_query_kernels(eval_state: Any,
                              env_eval_steps: Optional[int],
                              candidate_budgets: Sequence[int]) -> Dict[int, DenseQueryKernelBundle]:
  return {
      slots: _get_dense_query_kernel(
          eval_state=eval_state,
          candidate_slots=slots,
          env_eval_steps=env_eval_steps,
      )
      for slots in sorted(set(int(slot) for slot in candidate_budgets))
  }


def prewarm_dense_rhs_kernels(agent,
                              replay_batch: Mapping[str, jax.Array],
                              eval_state: Any,
                              horizon_state: HorizonSearchState,
                              rng: jax.Array,
                              env_eval_steps: Optional[int],
                              warm_all_phases: bool = False) -> Dict[int, DenseQueryKernelBundle]:
  kernels = build_dense_query_kernels(
      eval_state=eval_state,
      env_eval_steps=env_eval_steps,
      candidate_budgets=horizon_state.candidate_budget,
  )
  slots_to_prewarm = sorted(kernels)
  if not warm_all_phases:
    current_slots = int(horizon_state.candidate_budget[int(np.asarray(horizon_state.phase_id))])
    slots_to_prewarm = [current_slots]
  for offset, slots in enumerate(slots_to_prewarm):
    print(f'Prewarming Dense-RHS kernel for {slots} candidate slots...', flush=True)
    model_rng, env_rng, finalize_rng = jax.random.split(jax.random.fold_in(rng, offset), 3)
    model_stage = kernels[slots].model_stage(
        agent,
        replay_batch,
        horizon_state,
        model_rng,
    )
    jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
        model_stage,
    )
    env_mean, env_std = kernels[slots].env_stage(
        agent,
        model_stage,
        env_rng,
    )
    jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
        (env_mean, env_std),
    )
    result = kernels[slots].finalize_stage(
        horizon_state,
        model_stage,
        env_mean,
        env_std,
        jnp.asarray(0, dtype=jnp.int32),
        finalize_rng,
    )
    jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
        result,
    )
    print(f'Finished Dense-RHS kernel prewarm for {slots} candidate slots.', flush=True)
  return kernels


def dense_checkpoint_eval(agent,
                          replay_batch: Mapping[str, jax.Array],
                          eval_state: Any,
                          horizon_state: HorizonSearchState,
                          rng: jax.Array,
                          env_eval_steps: Optional[int] = None,
                          query_step: int = 0,
                          dense_query_kernels: Optional[Mapping[int, DenseQueryKernelBundle]] = None,
                          ) -> Tuple[HorizonSearchState, int, Dict[str, float]]:
  query_start = time.perf_counter()
  candidate_slots = int(horizon_state.candidate_budget[int(np.asarray(horizon_state.phase_id))])
  kernels = dense_query_kernels or build_dense_query_kernels(
      eval_state=eval_state,
      env_eval_steps=env_eval_steps,
      candidate_budgets=horizon_state.candidate_budget,
  )
  model_rng, env_rng, finalize_rng = jax.random.split(rng, 3)
  model_stage_start = time.perf_counter()
  model_stage = kernels[candidate_slots].model_stage(
      agent,
      replay_batch,
      horizon_state,
      model_rng,
  )
  jax.tree.map(
      lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
      model_stage,
  )
  model_stage_s = time.perf_counter() - model_stage_start
  env_stage_start = time.perf_counter()
  env_mean, env_std = kernels[candidate_slots].env_stage(
      agent,
      model_stage,
      env_rng,
  )
  jax.tree.map(
      lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
      (env_mean, env_std),
  )
  env_stage_s = time.perf_counter() - env_stage_start
  finalize_stage_start = time.perf_counter()
  result = kernels[candidate_slots].finalize_stage(
      horizon_state,
      model_stage,
      env_mean,
      env_std,
      jnp.asarray(query_step, dtype=jnp.int32),
      finalize_rng,
  )
  jax.tree.map(
      lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
      result,
  )
  finalize_stage_s = time.perf_counter() - finalize_stage_start
  new_state = result.horizon_state
  selected_horizon = int(np.asarray(result.selected_horizon))
  proposed_horizon = int(np.asarray(result.proposed_horizon))
  horizons = np.asarray(horizon_state.horizons, dtype=np.int32)
  fitness = np.asarray(result.fitness, dtype=np.float32)
  deployment_score = np.asarray(result.deployment_score, dtype=np.float32)
  return_term = np.asarray(result.return_term, dtype=np.float32)
  roughness_term = np.asarray(result.roughness_term, dtype=np.float32)
  sigma_r_term = np.asarray(result.sigma_r_term, dtype=np.float32)
  learner_proxy_term = np.asarray(result.learner_proxy_term, dtype=np.float32)
  transition_cost = np.asarray(result.transition_cost, dtype=np.float32)
  transition_adjusted_score = np.asarray(result.transition_adjusted_score, dtype=np.float32)
  switch_probability = np.asarray(result.switch_probability, dtype=np.float32)
  expected_improvement = np.asarray(result.expected_improvement, dtype=np.float32)
  expected_loss = np.asarray(result.expected_loss, dtype=np.float32)
  expected_net_benefit = np.asarray(result.expected_net_benefit, dtype=np.float32)
  robust_return = np.asarray(result.robust_return, dtype=np.float32)
  prefix_objectives = np.asarray(result.prefix_objectives, dtype=np.float32)
  probe_prefixes = np.asarray(result.probe_prefixes, dtype=np.float32)
  planner_prefix_returns = np.asarray(result.planner_prefix_returns, dtype=np.float32)
  roughness = np.asarray(result.roughness, dtype=np.float32)
  candidate_horizons = np.asarray(result.candidate_horizons, dtype=np.int32)
  candidate_mask = np.asarray(result.candidate_mask, dtype=bool)
  selected_idx = int(np.where(horizons == selected_horizon)[0][0])
  metrics = {
      'dense_rhs/selected_horizon': float(selected_horizon),
      'dense_rhs/proposed_horizon': float(proposed_horizon),
      'dense_rhs/phase_id': float(np.asarray(new_state.phase_id)),
      'dense_rhs/phase_name': PHASE_NAMES[int(np.asarray(new_state.phase_id))],
      'dense_rhs/num_active_horizons': float(np.sum(np.asarray(new_state.active_mask))),
      'dense_rhs/num_candidate_horizons': float(np.sum(candidate_mask)),
      'dense_rhs/entropy': float(np.asarray(new_state.entropy)),
      'dense_rhs/norm_entropy': float(np.asarray(new_state.norm_entropy)),
      'dense_rhs/best_fitness': float(fitness[selected_idx]),
      'dense_rhs/deployment_score_best': float(deployment_score[selected_idx]),
      'dense_rhs/incumbent_deployment_score': float(
          np.asarray(result.incumbent_deployment_score)
      ),
      'dense_rhs/proposed_deployment_score': float(
          np.asarray(result.proposed_deployment_score)
      ),
      'dense_rhs/proposed_transition_cost': float(
          np.asarray(result.proposed_transition_cost)
      ),
      'dense_rhs/proposed_switch_probability': float(
          np.asarray(result.proposed_switch_probability)
      ),
      'dense_rhs/proposed_expected_net_benefit': float(
          np.asarray(result.proposed_expected_net_benefit)
      ),
      'dense_rhs/transition_cost_best': float(transition_cost[selected_idx]),
      'dense_rhs/transition_adjusted_score_best': float(
          transition_adjusted_score[selected_idx]
      ),
      'dense_rhs/switch_probability_best': float(switch_probability[selected_idx]),
      'dense_rhs/expected_improvement_best': float(expected_improvement[selected_idx]),
      'dense_rhs/expected_loss_best': float(expected_loss[selected_idx]),
      'dense_rhs/expected_net_benefit_best': float(expected_net_benefit[selected_idx]),
      'dense_rhs/return_term_best': float(return_term[selected_idx]),
      'dense_rhs/roughness_term_best': float(roughness_term[selected_idx]),
      'dense_rhs/return_std_term_best': float(sigma_r_term[selected_idx]),
      'dense_rhs/learner_proxy_term_best': float(learner_proxy_term[selected_idx]),
      'dense_rhs/model_prefix_loss_best': float(prefix_objectives[selected_idx]),
      'dense_rhs/model_probe_prefix_best': float(probe_prefixes[selected_idx]),
      'dense_rhs/planner_return_best': float(planner_prefix_returns[selected_idx]),
      'dense_rhs/roughness_best': float(roughness[selected_idx]),
      'dense_rhs/robust_return_best': float(robust_return[selected_idx]),
      'timing/query_model_diag_s': float(model_stage_s),
      'timing/query_env_eval_s': float(env_stage_s),
      'timing/query_finalize_s': float(finalize_stage_s),
      'timing/query_total_s': float(time.perf_counter() - query_start),
  }
  for horizon in candidate_horizons[candidate_mask].tolist():
    idx = int(np.where(horizons == horizon)[0][0])
    metrics[f'dense_rhs/candidate_{horizon}_fitness'] = float(fitness[idx])
    metrics[f'dense_rhs/candidate_{horizon}_deployment_score'] = float(
        deployment_score[idx]
    )
    metrics[f'dense_rhs/candidate_{horizon}_transition_cost'] = float(
        transition_cost[idx]
    )
    metrics[f'dense_rhs/candidate_{horizon}_transition_adjusted_score'] = float(
        transition_adjusted_score[idx]
    )
    metrics[f'dense_rhs/candidate_{horizon}_switch_probability'] = float(
        switch_probability[idx]
    )
    metrics[f'dense_rhs/candidate_{horizon}_expected_net_benefit'] = float(
        expected_net_benefit[idx]
    )
    metrics[f'dense_rhs/candidate_{horizon}_return'] = float(robust_return[idx])
    metrics[f'dense_rhs/candidate_{horizon}_return_term'] = float(return_term[idx])
    metrics[f'dense_rhs/candidate_{horizon}_roughness_term'] = float(roughness_term[idx])
    metrics[f'dense_rhs/candidate_{horizon}_return_std_term'] = float(sigma_r_term[idx])
    metrics[f'dense_rhs/candidate_{horizon}_learner_proxy_term'] = float(
        learner_proxy_term[idx]
    )
  return new_state, selected_horizon, metrics

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from tdmpc2_jax.envs.mjx_dmc import TASK_DOMAIN, make_mjx_dmc_env


def _jsonable(value: Any) -> Any:
  if isinstance(value, np.ndarray):
    return value.tolist()
  if hasattr(value, 'shape'):
    return np.asarray(value).tolist()
  if isinstance(value, (np.floating, np.integer)):
    return value.item()
  return value


def _optional_bool(value: str | None) -> bool | None:
  if value is None:
    return None
  normalized = value.strip().lower()
  if normalized in {'1', 'true', 'yes', 'on'}:
    return True
  if normalized in {'0', 'false', 'no', 'off'}:
    return False
  raise argparse.ArgumentTypeError(f'expected boolean string, got {value!r}')


def _make_env_config(task: str,
                     num_envs: int,
                     episode_length: int,
                     reset_pool_size: int,
                     chaos: bool,
                     *,
                     enable_domain_randomization: bool | None = None,
                     enable_observation_noise: bool | None = None,
                     base_action_delay: int | None = None,
                     observation_noise_scale: float = 0.01,
                     wind_scale: float = 5.0,
                     push_scale: float = 25.0,
                     slip_scale: float = 0.15,
                     jitter_prob: float = 0.02) -> SimpleNamespace:
  if enable_domain_randomization is None:
    enable_domain_randomization = chaos
  if enable_observation_noise is None:
    enable_observation_noise = chaos
  if base_action_delay is None:
    base_action_delay = 1 if chaos else 0
  mjx_dmc = SimpleNamespace(
      task=task,
      action_repeat=2,
      episode_length=episode_length,
      observation_noise_scale=observation_noise_scale,
      enable_domain_randomization=enable_domain_randomization,
      enable_observation_noise=enable_observation_noise,
      base_action_delay=base_action_delay,
      desired_speed=5.0,
      action_repeat_dt=0.02,
      wind_scale=wind_scale,
      push_scale=push_scale,
      slip_scale=slip_scale,
      jitter_prob=jitter_prob,
      reset_pool_size=reset_pool_size,
  )
  return SimpleNamespace(mjx_dmc=mjx_dmc, num_envs=num_envs)


def _dmcontrol_reference(task: str, actions: np.ndarray) -> Dict[str, Any]:
  try:
    from dm_control import suite
  except Exception as exc:
    return {'available': False, 'error': f'{type(exc).__name__}: {exc}'}

  domain, task_name = TASK_DOMAIN[task]
  try:
    env = suite.load(domain, task_name, task_kwargs={'random': 0}, visualize_reward=False)
    env.reset()
    rewards = []
    spec = env.action_spec()
    for action in actions:
      clipped = np.clip(action, spec.minimum, spec.maximum)
      timestep = env.step(clipped)
      rewards.append(float(timestep.reward or 0.0))
    return {
        'available': True,
        'reward_sum': float(np.sum(rewards)),
        'reward_mean': float(np.mean(rewards)) if rewards else 0.0,
        'reward_finite': bool(np.all(np.isfinite(rewards))),
    }
  except Exception as exc:
    return {'available': False, 'error': f'{type(exc).__name__}: {exc}'}


def _finite_stats(value: Any) -> Dict[str, Any]:
  arr = np.asarray(value)
  if not np.issubdtype(arr.dtype, np.number):
    return {'numeric': False, 'shape': list(arr.shape), 'dtype': str(arr.dtype)}
  finite = np.isfinite(arr)
  stats: Dict[str, Any] = {
      'numeric': True,
      'shape': list(arr.shape),
      'dtype': str(arr.dtype),
      'finite': bool(np.all(finite)),
      'finite_count': int(np.sum(finite)),
      'size': int(arr.size),
  }
  if arr.size and np.any(finite):
    finite_values = arr[finite]
    stats.update({
        'min': float(np.min(finite_values)),
        'max': float(np.max(finite_values)),
        'mean': float(np.mean(finite_values)),
    })
  return stats


def _nonfinite_snapshot(env, obs: Any, reward: Any, step: int) -> Dict[str, Any] | None:
  state = env.state
  data = state.data
  fields = {
      'observation': obs,
      'reward': reward,
      'qpos': data.qpos,
      'qvel': data.qvel,
      'ctrl': data.ctrl,
      'xpos': data.xpos,
      'geom_xpos': data.geom_xpos,
      'wind_force': state.wind_force,
      'push_force': state.push_force,
      'target_pos': state.target_pos,
      'target_radius': state.target_radius,
      'last_action': state.last_action,
  }
  stats = {name: _finite_stats(value) for name, value in fields.items()}
  failing = [
      name for name, item in stats.items()
      if item.get('numeric') and not item.get('finite', True)
  ]
  if not failing:
    return None
  return {
      'step': int(step),
      'nonfinite_fields': failing,
      'field_stats': stats,
  }


def run_gate(task: str,
             seed: int,
             num_envs: int,
             steps: int,
             reset_pool_size: int,
             chaos: bool,
             *,
             enable_domain_randomization: bool | None = None,
             enable_observation_noise: bool | None = None,
             base_action_delay: int | None = None,
             observation_noise_scale: float = 0.01,
             wind_scale: float = 5.0,
             push_scale: float = 25.0,
             slip_scale: float = 0.15,
             jitter_prob: float = 0.02) -> Dict[str, Any]:
  env_config = _make_env_config(
      task=task,
      num_envs=num_envs,
      episode_length=max(steps + 2, 8),
      reset_pool_size=reset_pool_size,
      chaos=chaos,
      enable_domain_randomization=enable_domain_randomization,
      enable_observation_noise=enable_observation_noise,
      base_action_delay=base_action_delay,
      observation_noise_scale=observation_noise_scale,
      wind_scale=wind_scale,
      push_scale=push_scale,
      slip_scale=slip_scale,
      jitter_prob=jitter_prob,
  )
  env = make_mjx_dmc_env(env_config, seed=seed, num_envs=num_envs)
  obs, _ = env.reset(seed=seed)
  obs = jax.block_until_ready(obs)
  actions_for_reference: List[np.ndarray] = []
  rewards = []
  terminated_counts = []
  truncated_counts = []
  current_obs = obs
  first_nonfinite = _nonfinite_snapshot(env, current_obs, np.zeros((num_envs,), dtype=np.float32), 0)
  for step_idx in range(1, steps + 1):
    action = env.sample_actions()
    actions_for_reference.append(np.asarray(action[0]))
    current_obs, reward, terminated, truncated, _ = env.step(action)
    current_obs = jax.block_until_ready(current_obs)
    reward = jax.block_until_ready(reward)
    rewards.append(np.asarray(reward))
    terminated_counts.append(int(np.asarray(terminated).sum()))
    truncated_counts.append(int(np.asarray(truncated).sum()))
    if first_nonfinite is None:
      first_nonfinite = _nonfinite_snapshot(env, current_obs, reward, step_idx)

  rewards_arr = np.stack(rewards, axis=0) if rewards else np.zeros((0, num_envs))
  obs_arr = np.asarray(current_obs)
  action_ref = np.stack(actions_for_reference, axis=0) if actions_for_reference else np.zeros((0, env.metadata.action_dim))
  dmcontrol = _dmcontrol_reference(task, action_ref)
  reward_sum = float(np.sum(rewards_arr))
  reward_finite = bool(np.all(np.isfinite(rewards_arr)))
  obs_finite = bool(np.all(np.isfinite(obs_arr)))
  passed = (
      tuple(obs_arr.shape) == (num_envs, env.metadata.observation_dim)
      and env.single_action_space.shape == (env.metadata.action_dim,)
      and reward_finite
      and obs_finite
  )
  return {
      'task': task,
      'seed': seed,
      'num_envs': num_envs,
      'steps': steps,
      'chaos': chaos,
      'passed': bool(passed),
      'observation_shape': list(obs_arr.shape),
      'observation_dim': int(env.metadata.observation_dim),
      'action_shape': list(env.single_action_space.shape),
      'action_dim': int(env.metadata.action_dim),
      'physics_substeps_per_control': int(env.metadata.physics_substeps_per_control),
      'reward_sum': reward_sum,
      'reward_mean': float(np.mean(rewards_arr)) if rewards_arr.size else 0.0,
      'reward_min': float(np.min(rewards_arr)) if rewards_arr.size else math.nan,
      'reward_max': float(np.max(rewards_arr)) if rewards_arr.size else math.nan,
      'reward_finite': reward_finite,
      'observation_finite': obs_finite,
      'first_nonfinite': first_nonfinite,
      'terminated_counts': terminated_counts,
      'truncated_counts': truncated_counts,
      'dmcontrol_reference': dmcontrol,
      'jax_backend': jax.default_backend(),
      'chaos_params': {
          'enable_domain_randomization': bool(env_config.mjx_dmc.enable_domain_randomization),
          'enable_observation_noise': bool(env_config.mjx_dmc.enable_observation_noise),
          'base_action_delay': int(env_config.mjx_dmc.base_action_delay),
          'observation_noise_scale': float(env_config.mjx_dmc.observation_noise_scale),
          'wind_scale': float(env_config.mjx_dmc.wind_scale),
          'push_scale': float(env_config.mjx_dmc.push_scale),
          'slip_scale': float(env_config.mjx_dmc.slip_scale),
          'jitter_prob': float(env_config.mjx_dmc.jitter_prob),
      },
  }


def run_diagnostic_sweep(task: str,
                         seed: int,
                         num_envs: int,
                         steps: int,
                         reset_pool_size: int) -> Dict[str, Any]:
  cases = [
      {
          'name': 'clean_control',
          'chaos': False,
          'enable_domain_randomization': False,
          'enable_observation_noise': False,
          'base_action_delay': 0,
      },
      {
          'name': 'actuator_randomization_only',
          'chaos': True,
          'enable_domain_randomization': True,
          'enable_observation_noise': False,
          'base_action_delay': 0,
          'wind_scale': 0.0,
          'push_scale': 0.0,
          'slip_scale': 0.0,
          'jitter_prob': 0.0,
      },
      {
          'name': 'observation_noise_only',
          'chaos': True,
          'enable_domain_randomization': True,
          'enable_observation_noise': True,
          'base_action_delay': 0,
          'wind_scale': 0.0,
          'push_scale': 0.0,
          'slip_scale': 0.0,
          'jitter_prob': 0.0,
      },
      {
          'name': 'action_delay_jitter_only',
          'chaos': True,
          'enable_domain_randomization': True,
          'enable_observation_noise': False,
          'base_action_delay': 1,
          'wind_scale': 0.0,
          'push_scale': 0.0,
          'slip_scale': 0.0,
          'jitter_prob': 0.02,
      },
      {
          'name': 'wind_only',
          'chaos': True,
          'enable_domain_randomization': True,
          'enable_observation_noise': False,
          'base_action_delay': 0,
          'wind_scale': 5.0,
          'push_scale': 0.0,
          'slip_scale': 0.0,
          'jitter_prob': 0.0,
      },
      {
          'name': 'push_only',
          'chaos': True,
          'enable_domain_randomization': True,
          'enable_observation_noise': False,
          'base_action_delay': 0,
          'wind_scale': 0.0,
          'push_scale': 25.0,
          'slip_scale': 0.0,
          'jitter_prob': 0.0,
      },
      {
          'name': 'push_with_slip_decay',
          'chaos': True,
          'enable_domain_randomization': True,
          'enable_observation_noise': False,
          'base_action_delay': 0,
          'wind_scale': 0.0,
          'push_scale': 25.0,
          'slip_scale': 0.15,
          'jitter_prob': 0.0,
      },
      {
          'name': 'full_chaos',
          'chaos': True,
          'enable_domain_randomization': True,
          'enable_observation_noise': True,
          'base_action_delay': 1,
          'wind_scale': 5.0,
          'push_scale': 25.0,
          'slip_scale': 0.15,
          'jitter_prob': 0.02,
      },
  ]
  results = []
  for case in cases:
    case_kwargs = dict(case)
    case_name = case_kwargs.pop('name')
    case_chaos = bool(case_kwargs.pop('chaos'))
    try:
      summary = run_gate(
          task=task,
          seed=seed,
          num_envs=num_envs,
          steps=steps,
          reset_pool_size=reset_pool_size,
          chaos=case_chaos,
          **case_kwargs,
      )
    except Exception as exc:
      summary = {
          'task': task,
          'seed': seed,
          'num_envs': num_envs,
          'steps': steps,
          'chaos': case_chaos,
          'passed': False,
          'error': f'{type(exc).__name__}: {exc}',
          'jax_backend': jax.default_backend(),
      }
    summary['diagnostic_case'] = case_name
    results.append(summary)
  return {
      'task': task,
      'seed': seed,
      'num_envs': num_envs,
      'steps': steps,
      'reset_pool_size': reset_pool_size,
      'cases': results,
      'passed_cases': [
          item['diagnostic_case'] for item in results if item.get('passed') is True
      ],
      'failed_cases': [
          item['diagnostic_case'] for item in results if item.get('passed') is not True
      ],
  }


def main() -> None:
  parser = argparse.ArgumentParser(description='Validate TD-MPC2-JAX MJX DMC task gates.')
  parser.add_argument('--tasks', nargs='+', required=True)
  parser.add_argument('--out-dir', required=True)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--num-envs', type=int, default=4)
  parser.add_argument('--steps', type=int, default=16)
  parser.add_argument('--reset-pool-size', type=int, default=8)
  parser.add_argument('--chaos', action='store_true')
  parser.add_argument('--enable-domain-randomization', type=_optional_bool)
  parser.add_argument('--enable-observation-noise', type=_optional_bool)
  parser.add_argument('--base-action-delay', type=int)
  parser.add_argument('--observation-noise-scale', type=float, default=0.01)
  parser.add_argument('--wind-scale', type=float, default=5.0)
  parser.add_argument('--push-scale', type=float, default=25.0)
  parser.add_argument('--slip-scale', type=float, default=0.15)
  parser.add_argument('--jitter-prob', type=float, default=0.02)
  parser.add_argument('--diagnostic-sweep', action='store_true')
  args = parser.parse_args()

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  summaries = []
  for task in args.tasks:
    print(f'Running MJX gate for {task}...', flush=True)
    diagnostic = None
    try:
      if args.diagnostic_sweep:
        diagnostic = run_diagnostic_sweep(
            task=task,
            seed=args.seed,
            num_envs=args.num_envs,
            steps=args.steps,
            reset_pool_size=args.reset_pool_size,
        )
        full_chaos = [
            item for item in diagnostic['cases']
            if item.get('diagnostic_case') == 'full_chaos'
        ]
        summary = dict(full_chaos[-1] if full_chaos else diagnostic['cases'][-1])
        summary['diagnostic_sweep'] = {
            'passed_cases': diagnostic['passed_cases'],
            'failed_cases': diagnostic['failed_cases'],
            'artifact': 'diagnostic_sweep.json',
        }
      else:
        summary = run_gate(
            task=task,
            seed=args.seed,
            num_envs=args.num_envs,
            steps=args.steps,
            reset_pool_size=args.reset_pool_size,
            chaos=args.chaos,
            enable_domain_randomization=args.enable_domain_randomization,
            enable_observation_noise=args.enable_observation_noise,
            base_action_delay=args.base_action_delay,
            observation_noise_scale=args.observation_noise_scale,
            wind_scale=args.wind_scale,
            push_scale=args.push_scale,
            slip_scale=args.slip_scale,
            jitter_prob=args.jitter_prob,
        )
    except Exception as exc:
      summary = {
          'task': task,
          'seed': args.seed,
          'num_envs': args.num_envs,
          'steps': args.steps,
          'chaos': args.chaos,
          'passed': False,
          'error': f'{type(exc).__name__}: {exc}',
          'jax_backend': jax.default_backend(),
      }
    summaries.append(summary)
    task_dir = out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    if diagnostic is not None:
      (task_dir / 'diagnostic_sweep.json').write_text(
          json.dumps(diagnostic, indent=2, default=_jsonable),
          encoding='utf-8',
      )
    (task_dir / 'mjx_gate.json').write_text(
        json.dumps(summary, indent=2, default=_jsonable),
        encoding='utf-8',
    )
    print(json.dumps({
        'task': task,
        'passed': summary['passed'],
        'reward_sum': summary.get('reward_sum'),
        'obs': summary.get('observation_shape'),
        'action': summary.get('action_shape'),
        'error': summary.get('error'),
    }), flush=True)

  aggregate = {
      'passed': all(item['passed'] for item in summaries),
      'tasks': summaries,
  }
  (out_dir / 'mjx_gate_summary.json').write_text(
      json.dumps(aggregate, indent=2, default=_jsonable),
      encoding='utf-8',
  )
  if not aggregate['passed']:
    raise SystemExit(1)


if __name__ == '__main__':
  main()

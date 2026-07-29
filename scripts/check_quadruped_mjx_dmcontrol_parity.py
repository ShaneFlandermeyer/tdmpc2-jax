#!/usr/bin/env python3
"""Compare TD-MPC2-JAX MJX quadruped-run semantics against DMControl.

This is a CPU-safe smoke/parity check. It intentionally avoids learned models and
checks the environment contract that matters for paper-parity baselines:
observation shape/order, action scaling, reset statistics, and short random
rollout reward statistics.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import jax
import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import action_scale

from tdmpc2_jax.envs.mjx_quadruped import MJXQuadrupedBatchEnv, _scale_action_to_ctrl


def _flatten_dmcontrol_obs(obs: OrderedDict) -> np.ndarray:
  return np.concatenate([np.asarray(v).reshape(-1) for v in obs.values()]).astype(
      np.float32
  )


def _make_dmcontrol_env(seed: int):
  env = suite.load(
      "quadruped",
      "run",
      task_kwargs={"random": int(seed)},
      visualize_reward=False,
  )
  return action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)


def _step_dmcontrol_repeat(env, action: np.ndarray, repeats: int = 2):
  reward = 0.0
  timestep = None
  for _ in range(int(repeats)):
    timestep = env.step(action.astype(env.action_spec().dtype))
    reward += float(timestep.reward)
  assert timestep is not None
  return _flatten_dmcontrol_obs(timestep.observation), reward


def _compare_obs(seed: int):
  print("Building dm_control and MJX quadruped envs...", flush=True)
  dm_env = _make_dmcontrol_env(seed)
  dm_ts = dm_env.reset()
  dm_obs = _flatten_dmcontrol_obs(dm_ts.observation)

  mjx_env = MJXQuadrupedBatchEnv(
      num_envs=1,
      seed=seed,
      enable_domain_randomization=False,
      enable_observation_noise=False,
      base_action_delay=0,
      reset_pool_size=1,
  )
  mjx_obs, _ = mjx_env.reset(seed=seed)
  mjx_obs_np = np.asarray(mjx_obs[0], dtype=np.float32)
  print("Finished reset comparison.", flush=True)

  keys = list(dm_ts.observation.keys())
  shapes = {k: tuple(np.asarray(v).shape) for k, v in dm_ts.observation.items()}
  diff = mjx_obs_np - dm_obs if mjx_obs_np.shape == dm_obs.shape else None
  return {
      "dm_keys": keys,
      "dm_shapes": shapes,
      "dm_obs_shape": dm_obs.shape,
      "mjx_obs_shape": mjx_obs_np.shape,
      "reset_obs_max_abs_diff": None if diff is None else float(np.max(np.abs(diff))),
      "reset_obs_mean_abs_diff": None if diff is None else float(np.mean(np.abs(diff))),
  }, dm_env, mjx_env


def _compare_action_scaling(dm_env, mjx_env):
  print("Checking action scaling...", flush=True)
  raw_dm = suite.load(
      "quadruped",
      "run",
      task_kwargs={"random": 0},
      visualize_reward=False,
  )
  rng = np.random.RandomState(0)
  test_actions = np.asarray(
      [
          np.full((mjx_env.metadata.action_dim,), -1.0, dtype=np.float32),
          np.zeros((mjx_env.metadata.action_dim,), dtype=np.float32),
          np.full((mjx_env.metadata.action_dim,), 1.0, dtype=np.float32),
          rng.uniform(-1.0, 1.0, size=(mjx_env.metadata.action_dim,)).astype(np.float32),
      ]
  )
  mjx_scaled = np.asarray(
      _scale_action_to_ctrl(jnp.asarray(test_actions), mjx_env.metadata),
      dtype=np.float32,
  )
  ctrl_min = np.asarray(raw_dm.action_spec().minimum, dtype=np.float32)
  ctrl_max = np.asarray(raw_dm.action_spec().maximum, dtype=np.float32)
  expected = ctrl_min + (test_actions + 1.0) * 0.5 * (ctrl_max - ctrl_min)
  return {
      "raw_dm_action_min": ctrl_min,
      "raw_dm_action_max": ctrl_max,
      "wrapped_dm_action_min": np.asarray(dm_env.action_spec().minimum, dtype=np.float32),
      "wrapped_dm_action_max": np.asarray(dm_env.action_spec().maximum, dtype=np.float32),
      "action_scale_max_abs_diff": float(np.max(np.abs(mjx_scaled - expected))),
  }


def _compare_rollout(seed: int, steps: int, dm_env, mjx_env):
  print(f"Running fixed-action rollout for {steps} steps...", flush=True)
  rng = np.random.RandomState(seed + 123)
  actions = rng.uniform(
      -1.0,
      1.0,
      size=(int(steps), mjx_env.metadata.action_dim),
  ).astype(np.float32)
  dm_rewards = []

  for action in actions:
    _, dm_reward = _step_dmcontrol_repeat(dm_env, action, repeats=mjx_env.metadata.action_repeat)
    dm_rewards.append(dm_reward)

  def mjx_rollout(state, rollout_actions):
    def scan_step(carry, action):
      next_state, reward, _, _ = mjx_env._step_state(carry, action[None, :])
      return next_state, reward[0]

    _, rewards = jax.lax.scan(scan_step, state, rollout_actions)
    return rewards

  print("Compiling/running MJX scan rollout...", flush=True)
  mjx_rewards = np.asarray(
      jax.jit(mjx_rollout)(mjx_env.state, jnp.asarray(actions)),
      dtype=np.float32,
  )
  print("Finished MJX scan rollout.", flush=True)

  dm_rewards = np.asarray(dm_rewards, dtype=np.float32)
  mjx_rewards = np.asarray(mjx_rewards, dtype=np.float32)
  return {
      "steps": int(steps),
      "dm_return": float(dm_rewards.sum()),
      "mjx_return": float(mjx_rewards.sum()),
      "return_abs_diff": float(abs(dm_rewards.sum() - mjx_rewards.sum())),
      "dm_reward_mean": float(dm_rewards.mean()),
      "mjx_reward_mean": float(mjx_rewards.mean()),
      "reward_mean_abs_diff": float(np.mean(np.abs(dm_rewards - mjx_rewards))),
      "reward_max_abs_diff": float(np.max(np.abs(dm_rewards - mjx_rewards))),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--steps", type=int, default=100)
  parser.add_argument("--rollout-mean-tol", type=float, default=1e-2)
  parser.add_argument("--output-json", type=Path, default=None)
  args = parser.parse_args()

  obs_report, dm_env, mjx_env = _compare_obs(args.seed)
  action_report = _compare_action_scaling(dm_env, mjx_env)
  rollout_report = _compare_rollout(
      args.seed,
      args.steps,
      dm_env,
      mjx_env,
  )

  print("Observation")
  print(f"  dm_keys: {obs_report['dm_keys']}")
  print(f"  dm_shapes: {obs_report['dm_shapes']}")
  print(f"  dm_obs_shape: {obs_report['dm_obs_shape']}")
  print(f"  mjx_obs_shape: {obs_report['mjx_obs_shape']}")
  print(f"  reset_obs_max_abs_diff: {obs_report['reset_obs_max_abs_diff']}")
  print(f"  reset_obs_mean_abs_diff: {obs_report['reset_obs_mean_abs_diff']}")

  print("Action Scaling")
  print(f"  raw_dm_action_min: {action_report['raw_dm_action_min']}")
  print(f"  raw_dm_action_max: {action_report['raw_dm_action_max']}")
  print(f"  wrapped_dm_action_min: {action_report['wrapped_dm_action_min']}")
  print(f"  wrapped_dm_action_max: {action_report['wrapped_dm_action_max']}")
  print(f"  action_scale_max_abs_diff: {action_report['action_scale_max_abs_diff']}")

  print("Random Rollout")
  for key, value in rollout_report.items():
    print(f"  {key}: {value}")

  required_ok = (
      obs_report["dm_obs_shape"] == obs_report["mjx_obs_shape"]
      and action_report["action_scale_max_abs_diff"] < 1e-6
  )
  rollout_ok = rollout_report["reward_mean_abs_diff"] <= float(args.rollout_mean_tol)
  report = {
      "seed": int(args.seed),
      "steps": int(args.steps),
      "observation": obs_report,
      "action_scaling": {
          key: (value.tolist() if isinstance(value, np.ndarray) else value)
          for key, value in action_report.items()
      },
      "random_rollout": rollout_report,
      "required_ok": bool(required_ok),
      "rollout_ok": bool(rollout_ok),
      "rollout_mean_tolerance": float(args.rollout_mean_tol),
  }
  if args.output_json is not None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote JSON report: {args.output_json}", flush=True)
  if not required_ok:
    raise SystemExit(2)
  if not rollout_ok:
    raise SystemExit(3)


if __name__ == "__main__":
  main()

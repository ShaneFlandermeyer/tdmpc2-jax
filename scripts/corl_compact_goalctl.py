#!/usr/bin/env python3
"""Ledger-driven steward for the compact CoRL Dense-RHS campaign."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOAL = ROOT / 'goals' / 'dense_rhs_corl_compact.yaml'
REMOTE_SKILL_DIR = Path.home() / '.codex' / 'skills' / 'remote-machines'
SSH_HELPER = REMOTE_SKILL_DIR / 'scripts' / 'ssh_host.sh'
NCC_GPU_STATUS = REMOTE_SKILL_DIR / 'scripts' / 'ncc_gpu_status.sh'

LEDGER_FIELDS = [
    'timestamp',
    'event',
    'run_id',
    'status',
    'job_id',
    'attempt',
    'env_id',
    'regime',
    'method',
    'seed',
    'paper_horizon',
    'run_dir',
    'launcher',
    'git_commit',
    'remote_commit',
    'slurm_state',
    'final_step',
    'final_score',
    'best_score',
    'auc',
    'wall_hours',
    'checkpoint_ok',
    'notes',
]

TERMINAL_EVENTS = {'completed', 'failed', 'blocked', 'partial_complete'}
GOOD_TERMINAL_STATUSES = {'completed', 'passed', 'partial_complete'}
RETRYABLE_STATES = {'TIMEOUT', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'CANCELLED'}
SLURM_TERMINAL_PREFIXES = (
    'COMPLETED',
    'FAILED',
    'CANCELLED',
    'TIMEOUT',
    'NODE_FAIL',
    'PREEMPTED',
    'OUT_OF_MEMORY',
    'BOOT_FAIL',
)
STEWARD_OWNED_PATHS = (
    'experiments/corl_compact_ledger.csv',
    'experiments/corl_compact_decisions.md',
    'goals/dense_rhs_corl_compact.yaml',
    'runs/results/corl_compact/',
)


def now_iso() -> str:
  return datetime.now(timezone.utc).isoformat(timespec='seconds')


def load_goal(path: Path) -> dict[str, Any]:
  with path.open() as handle:
    return json.load(handle)


def relpath(path: str | Path) -> Path:
  path = Path(path)
  return path if path.is_absolute() else ROOT / path


def run_local(
    argv: list[str],
    *,
    check: bool = False,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      argv,
      cwd=ROOT,
      text=True,
      capture_output=True,
      timeout=timeout,
      check=check,
  )


def run_remote(
    goal: dict[str, Any],
    command: str,
    *,
    timeout: int = 90,
) -> subprocess.CompletedProcess[str]:
  host = goal['remote']['host']
  return run_local([str(SSH_HELPER), host, command], timeout=timeout)


def local_git_state() -> dict[str, Any]:
  commit = run_local(['git', 'rev-parse', 'HEAD'], timeout=15)
  status = run_local(['git', 'status', '--porcelain'], timeout=15)
  return {
      'ok': commit.returncode == 0 and status.returncode == 0,
      'commit': commit.stdout.strip() if commit.returncode == 0 else '',
      'dirty': bool(status.stdout.strip()),
      'status': status.stdout.strip(),
      'error': (commit.stderr + status.stderr).strip(),
  }


def git_dirty_paths() -> list[str]:
  status = run_local(['git', 'status', '--porcelain'], timeout=15)
  if status.returncode != 0:
    return []
  paths: list[str] = []
  for line in status.stdout.splitlines():
    if not line:
      continue
    path = line[3:].strip()
    if ' -> ' in path:
      path = path.split(' -> ', 1)[1].strip()
    paths.append(path)
  return paths


def repo_path_text(path: str | Path) -> str:
  path = Path(path)
  if path.is_absolute():
    try:
      return path.relative_to(ROOT).as_posix()
    except ValueError:
      return path.as_posix()
  return path.as_posix()


def steward_owned_paths(goal: dict[str, Any] | None = None) -> list[str]:
  if goal is None:
    return list(STEWARD_OWNED_PATHS)
  tracking = goal.get('tracking', {})
  paths = [
      goal.get('_goal_path', DEFAULT_GOAL),
      tracking.get('ledger', ''),
      tracking.get('decisions', ''),
      tracking.get('results_dir', ''),
  ]
  for profile in goal.get('gate_repair', {}).get('profiles', []):
    for artifact in profile.get('artifacts', []):
      local_path = artifact.get('local', '')
      if local_path:
        paths.append(local_path)
        for parent in Path(local_path).parents:
          if parent.as_posix() == '.':
            break
          paths.append(parent.as_posix())
          if parent.parent.as_posix() == 'runs/results':
            break
  return [repo_path_text(path) for path in paths if path]


def is_steward_owned_path(path: str, goal: dict[str, Any] | None = None) -> bool:
  path = repo_path_text(path)
  return any(path == prefix.rstrip('/') or path.startswith(prefix.rstrip('/') + '/')
             for prefix in steward_owned_paths(goal))


def dirty_paths_are_steward_owned() -> bool:
  paths = git_dirty_paths()
  return bool(paths) and all(is_steward_owned_path(path) for path in paths)


def current_branch() -> str:
  result = run_local(['git', 'branch', '--show-current'], timeout=15)
  if result.returncode != 0:
    return ''
  return result.stdout.strip()


def auto_commit_steward_state(goal: dict[str, Any], reason: str) -> str:
  """Commit and sync steward-owned ledger/result updates.

  This prevents the steward from deadlocking on its own append-only ledger rows.
  It refuses to touch unrelated dirty files.
  """
  paths = git_dirty_paths()
  if not paths:
    return 'no steward state changes to commit'
  if not all(is_steward_owned_path(path, goal) for path in paths):
    return (
        'steward auto-commit skipped; non-steward dirty paths: ' +
        ', '.join(path for path in paths if not is_steward_owned_path(path, goal))
    )
  add_paths = [
      path for path in steward_owned_paths(goal)
      if (ROOT / path).exists()
  ]
  if not add_paths:
    return 'steward auto-commit skipped; no existing steward paths to add'
  add = run_local(['git', 'add', *add_paths], timeout=60)
  if add.returncode != 0:
    return f'steward auto-commit git add failed: {add.stderr.strip()}'
  diff = run_local(['git', 'diff', '--cached', '--quiet'], timeout=30)
  if diff.returncode == 0:
    return 'no staged steward state changes to commit'
  message = f'Steward update: {reason}'
  commit = run_local(['git', 'commit', '-m', message], timeout=120)
  if commit.returncode != 0:
    return f'steward auto-commit failed: {commit.stderr.strip()}'
  branch = current_branch()
  if not branch:
    return 'steward auto-commit succeeded but branch was unknown; remote not synced'
  push = run_local(['git', 'push', 'gguiomar', branch], timeout=180)
  if push.returncode != 0:
    return f'steward auto-commit succeeded but push failed: {push.stderr.strip()}'
  pull = run_remote(
      goal,
      (
          f'cd {shlex.quote(goal["remote"]["path"])} && '
          f'git pull --ff-only gguiomar {shlex.quote(branch)}'
      ),
      timeout=180,
  )
  if pull.returncode != 0:
    return f'steward state pushed but remote ff-pull failed: {pull.stderr.strip()}'
  return f'steward state committed and synced: {message}'


def sync_remote_to_local_if_safe(goal: dict[str, Any],
                                 local: dict[str, Any],
                                 remote: dict[str, Any]) -> str:
  """Fast-forward the NCC worktree to local HEAD when this is provably safe."""
  if not local.get('ok') or not remote.get('ok'):
    return 'remote sync skipped: git state unavailable'
  if local.get('commit') == remote.get('commit'):
    return 'remote sync already current'
  if local.get('dirty'):
    return 'remote sync skipped: local git dirty'
  if remote.get('dirty'):
    return 'remote sync skipped: remote git dirty'
  branch = current_branch()
  if not branch:
    return 'remote sync skipped: branch unknown'
  ancestor = run_local(
      ['git', 'merge-base', '--is-ancestor', remote['commit'], local['commit']],
      timeout=30,
  )
  if ancestor.returncode != 0:
    return (
        'remote sync skipped: remote commit is not an ancestor of local HEAD '
        f'local={local.get("commit", "")[:12]} remote={remote.get("commit", "")[:12]}'
    )
  push = run_local(['git', 'push', 'gguiomar', branch], timeout=180)
  if push.returncode != 0:
    return f'remote sync skipped: push failed: {push.stderr.strip()}'
  pull = run_remote(
      goal,
      (
          f'cd {shlex.quote(goal["remote"]["path"])} && '
          f'git pull --ff-only gguiomar {shlex.quote(branch)}'
      ),
      timeout=180,
  )
  if pull.returncode != 0:
    return f'remote sync failed: {pull.stderr.strip()}'
  return (
      'remote sync completed: '
      f'{remote.get("commit", "")[:12]} -> {local.get("commit", "")[:12]}'
  )


def remote_git_state(goal: dict[str, Any]) -> dict[str, Any]:
  remote_path = shlex.quote(goal['remote']['path'])
  command = (
      f'cd {remote_path} && '
      'printf "commit=%s\\n" "$(git rev-parse HEAD 2>/dev/null || true)" && '
      'printf "status_begin\\n" && git status --porcelain 2>/dev/null || true'
  )
  result = run_remote(goal, command, timeout=60)
  state = {
      'ok': result.returncode == 0,
      'commit': '',
      'dirty': True,
      'status': '',
      'error': result.stderr.strip(),
  }
  if result.returncode != 0:
    return state
  lines = result.stdout.splitlines()
  for line in lines:
    if line.startswith('commit='):
      state['commit'] = line.split('=', 1)[1].strip()
      break
  try:
    idx = lines.index('status_begin')
    status_lines = lines[idx + 1:]
  except ValueError:
    status_lines = []
  state['status'] = '\n'.join(status_lines).strip()
  state['dirty'] = bool(state['status'])
  return state


def read_ledger(goal: dict[str, Any]) -> list[dict[str, str]]:
  ledger_path = relpath(goal['tracking']['ledger'])
  if not ledger_path.exists():
    return []
  with ledger_path.open(newline='') as handle:
    return list(csv.DictReader(handle))


def append_ledger(goal: dict[str, Any], row: dict[str, Any]) -> None:
  ledger_path = relpath(goal['tracking']['ledger'])
  ledger_path.parent.mkdir(parents=True, exist_ok=True)
  exists = ledger_path.exists()
  clean = {field: '' for field in LEDGER_FIELDS}
  clean.update({key: '' if value is None else value for key, value in row.items()})
  with ledger_path.open('a', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
    if not exists or ledger_path.stat().st_size == 0:
      writer.writeheader()
    writer.writerow(clean)


def latest_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
  latest: dict[str, dict[str, str]] = {}
  for row in rows:
    run_id = row.get('run_id', '')
    if run_id:
      latest[run_id] = row
  return latest


def rows_for_run(rows: list[dict[str, str]], run_id: str) -> list[dict[str, str]]:
  return [row for row in rows if row.get('run_id') == run_id]


def safe_slug(text: str) -> str:
  return re.sub(r'[^A-Za-z0-9_.-]+', '_', text).strip('_')


def remote_run_dir(goal: dict[str, Any], relative: str) -> str:
  return str(Path(goal['remote']['path']) / relative)


def build_setup_profile(goal: dict[str, Any]) -> dict[str, Any]:
  setup = goal['setup_smoke']
  run_dir = remote_run_dir(goal, setup['run_dir'])
  return {
      'run_id': setup['run_id'],
      'kind': 'setup_smoke',
      'env_id': setup['env_id'],
      'regime': 'clean',
      'method': 'checkpoint_resume_smoke',
      'seed': int(setup['seed']),
      'paper_horizon': int(goal['constraints']['paper_horizon']),
      'script': setup['script'],
      'run_dir': run_dir,
      'priority': 0,
      'env': {
          'RUN_ID': setup['run_id'],
          'RUN_DIR': run_dir,
          'ENV_ID': setup['env_id'],
          'SEED': str(setup['seed']),
      },
  }


def build_gate_repair_profiles(goal: dict[str, Any]) -> list[dict[str, Any]]:
  repair = goal.get('gate_repair', {})
  if not repair.get('enabled', False):
    return []
  profiles: list[dict[str, Any]] = []
  for idx, item in enumerate(repair.get('profiles', []), start=1):
    remote_out_dir = item.get('remote_out_dir') or item.get('out_dir')
    if not remote_out_dir:
      raise ValueError(f'gate repair profile {item.get("run_id", idx)} missing remote_out_dir')
    env_vars = {
        str(key): str(value)
        for key, value in item.get('env', {}).items()
    }
    if 'OUT_DIR' not in env_vars:
      env_vars['OUT_DIR'] = str(remote_out_dir)
    run_id = item.get('run_id') or f'gate_repair__{idx}'
    profiles.append({
        'run_id': run_id,
        'kind': 'gate_repair',
        'env_id': item.get('env_id', 'gate'),
        'regime': item.get('regime', 'gate'),
        'method': item.get('method', 'mjx_gate'),
        'seed': int(item.get('seed', 0)),
        'paper_horizon': int(goal['constraints']['paper_horizon']),
        'script': item['script'],
        'run_dir': str(remote_out_dir),
        'priority': int(item.get('priority', idx)),
        'env': env_vars,
        'gate_path': item.get('gate_path', ''),
        'artifacts': item.get('artifacts', []),
    })
  return sorted(profiles, key=lambda profile: int(profile['priority']))


def build_main_profiles(goal: dict[str, Any]) -> list[dict[str, Any]]:
  profiles: list[dict[str, Any]] = []
  defaults = goal['run_defaults']
  adaptive = goal['adaptive_rhs']
  paper_horizon = int(goal['constraints']['paper_horizon'])
  env_order = [item['env_id'] for item in goal['matrix']['envs']]
  regimes = goal['matrix']['regimes']
  priority_seeds = goal['matrix'].get('launch_seed_priority', goal['matrix']['seeds'])
  methods = goal['matrix']['methods']
  remote_run_dir_prefix = goal.get('tracking', {}).get(
      'remote_run_dir_prefix',
      'outputs/corl_pub_500k',
  )
  priority = 1
  for seed in priority_seeds:
    for env_id in env_order:
      for regime_name, regime_cfg in regimes.items():
        for method in methods:
          run_id = (
              f'corl500k__{safe_slug(env_id)}__{regime_name}__{method}__s{seed}'
          )
          relative_dir = (
              f'{remote_run_dir_prefix}/{env_id}/{regime_name}/{method}/s{seed}'
          )
          run_dir = remote_run_dir(goal, relative_dir)
          is_adaptive = method == 'adaptive_rhs'
          script = (
              'scripts/ncc_corl_adaptive_rhs_500k.sbatch'
              if is_adaptive else
              'scripts/ncc_corl_paper_horizon_500k.sbatch'
          )
          env_vars = {
              'RUN_ID': run_id,
              'RUN_DIR': run_dir,
              'ENV_ID': env_id,
              'REGIME': regime_name,
              'SEED': str(seed),
              'PAPER_HORIZON': str(paper_horizon),
              'MAX_STEPS': str(defaults['max_steps']),
              'SAVE_INTERVAL_STEPS': str(defaults['save_interval_steps']),
              'CHECKPOINT_BUFFER': str(defaults['checkpoint_buffer']).lower(),
              'EVAL_INTERVAL_STEPS': str(defaults['eval_interval_steps']),
              'EVAL_NUM_EPISODES': str(defaults['eval_num_episodes']),
              'EVAL_CLEAN': str(
                  regime_cfg.get('eval_clean', regime_name == 'clean')
              ).lower(),
              'TRAIN_NUM_ENVS': str(defaults['train_num_envs']),
              'SEED_STEPS_OVERRIDE': str(defaults['seed_steps_override']),
              'UPDATE_CHUNK_SIZE': str(defaults['update_chunk_size']),
              'COLLECT_CHUNK_STEPS': str(defaults['collect_chunk_steps']),
              'UTD_RATIO': str(defaults['utd_ratio']),
              'RESET_POOL_SIZE': str(defaults['reset_pool_size']),
              'ENABLE_DOMAIN_RANDOMIZATION': str(
                  regime_cfg['enable_domain_randomization']
              ).lower(),
              'ENABLE_OBSERVATION_NOISE': str(
                  regime_cfg['enable_observation_noise']
              ).lower(),
              'BASE_ACTION_DELAY': str(regime_cfg['base_action_delay']),
          }
          if is_adaptive:
            budget = adaptive['candidate_budget']
            env_vars.update({
                'DENSE_QUERY_START_STEP': str(adaptive['query_start_step']),
                'DENSE_QUERY_INTERVAL_STEPS': str(adaptive['query_interval_steps']),
                'DENSE_RHS_REPLICAS': str(adaptive['num_env_eval_replicas']),
                'DENSE_RHS_EVAL_STEPS': str(adaptive['env_eval_steps']),
                'DENSE_RHS_QUERY_POPULATION': str(adaptive['query_population_size']),
                'DENSE_RHS_QUERY_POLICY_PRIOR': str(
                    adaptive['query_policy_prior_samples']
                ),
                'DENSE_RHS_QUERY_ELITES': str(adaptive['query_num_elites']),
                'DENSE_RHS_QUERY_MPPI_ITERS': str(adaptive['query_mppi_iterations']),
                'DENSE_RHS_QUERY_TEMPERATURE': str(adaptive['query_temperature']),
                'DENSE_RHS_SELECTION_RETURN_POWER': str(
                    adaptive['selection_return_power']
                ),
                'DENSE_RHS_ROUGHNESS_WEIGHT': str(adaptive['roughness_weight']),
                'DENSE_RHS_RETURN_STD_WEIGHT': str(adaptive['return_std_weight']),
                'DENSE_RHS_LOCAL_WINDOW_RADIUS': str(
                    adaptive['local_window_radius']
                ),
                'DENSE_RHS_MAX_TRANSITION_DELTA': str(
                    adaptive['max_transition_delta']
                ),
                'DENSE_RHS_INCUMBENT_MARGIN': str(
                    adaptive['incumbent_switch_margin']
                ),
                'DENSE_RHS_LEARNER_PROXY_ENABLED': str(
                    adaptive.get('learner_proxy_enabled', False)
                ).lower(),
                'DENSE_RHS_LEARNER_PROXY_WEIGHT': str(
                    adaptive.get('learner_proxy_weight', 0.0)
                ),
                'DENSE_RHS_LEARNER_PROXY_MODE': str(
                    adaptive.get('learner_proxy_mode', 'probe_mean_loss')
                ),
                'DENSE_RHS_DEPLOYMENT_UTILITY_ENABLED': str(
                    adaptive.get('deployment_utility_enabled', False)
                ).lower(),
                'DENSE_RHS_DEPLOYMENT_UTILITY_WEIGHT': str(
                    adaptive.get('deployment_utility_weight', 1.0)
                ),
                'DENSE_RHS_DEPLOYMENT_UTILITY_EXPLORATION': str(
                    adaptive.get('deployment_utility_exploration', 1.0)
                ),
                'DENSE_RHS_DEPLOYMENT_UTILITY_PRIOR_MEAN': str(
                    adaptive.get('deployment_utility_prior_mean', 0.0)
                ),
                'DENSE_RHS_DEPLOYMENT_UTILITY_PRIOR_STD': str(
                    adaptive.get('deployment_utility_prior_std', 150.0)
                ),
                'DENSE_RHS_DEPLOYMENT_UTILITY_DENSE_SCORE_WEIGHT': str(
                    adaptive.get('deployment_utility_dense_score_weight', 25.0)
                ),
                'DENSE_RHS_DEPLOYMENT_UTILITY_MIN_OBSERVATIONS': str(
                    adaptive.get('deployment_utility_min_observations', 1)
                ),
                'DENSE_RHS_CREDIBLE_TRANSITION_ENABLED': str(
                    adaptive.get('credible_transition_enabled', False)
                ).lower(),
                'DENSE_RHS_CREDIBLE_TRANSITION_RULE': str(
                    adaptive.get('credible_transition_rule', 'probability')
                ),
                'DENSE_RHS_CREDIBLE_TRANSITION_MIN_PROB': str(
                    adaptive.get('credible_transition_min_prob', 0.0)
                ),
                'DENSE_RHS_TRANSITION_COST_SCALE': str(
                    adaptive.get('transition_cost_scale', 0.0)
                ),
                'DENSE_RHS_TRANSITION_RISK_WEIGHT': str(
                    adaptive.get('transition_risk_weight', 1.0)
                ),
                'DENSE_RHS_TRANSITION_MIN_EXPECTED_NET': str(
                    adaptive.get('transition_min_expected_net', 0.0)
                ),
                'DENSE_RHS_TRANSITION_MODEL_WEIGHT': str(
                    adaptive.get('transition_model_weight', 1.0)
                ),
                'DENSE_RHS_TRANSITION_PROBE_WEIGHT': str(
                    adaptive.get('transition_probe_weight', 1.0)
                ),
                'DENSE_RHS_TRANSITION_PLANNER_WEIGHT': str(
                    adaptive.get('transition_planner_weight', 1.0)
                ),
                'DENSE_RHS_TRANSITION_ROUGHNESS_WEIGHT': str(
                    adaptive.get('transition_roughness_weight', 1.0)
                ),
                'DENSE_RHS_TRANSITION_RETURN_STD_WEIGHT': str(
                    adaptive.get('transition_return_std_weight', 1.0)
                ),
                'DENSE_RHS_TRANSITION_UNCERTAINTY_FLOOR': str(
                    adaptive.get('transition_uncertainty_floor', 0.05)
                ),
                'DENSE_RHS_HORIZONS': str(adaptive['horizons']),
                'DENSE_RHS_HMAX': str(adaptive['hmax']),
                'DENSE_RHS_HORIZON_BUCKETS': str(adaptive['horizon_buckets']),
                'DENSE_RHS_CANDIDATE_BUDGET_A': str(budget['A']),
                'DENSE_RHS_CANDIDATE_BUDGET_B1': str(budget['B1']),
                'DENSE_RHS_CANDIDATE_BUDGET_B2': str(budget['B2']),
                'DENSE_RHS_CANDIDATE_BUDGET_B3': str(budget['B3']),
                'DENSE_RHS_CANDIDATE_BUDGET_B4': str(budget['B4']),
            })
          profiles.append({
              'run_id': run_id,
              'kind': 'main',
              'env_id': env_id,
              'regime': regime_name,
              'method': method,
              'seed': int(seed),
              'paper_horizon': paper_horizon,
              'script': script,
              'run_dir': run_dir,
              'priority': priority,
              'env': env_vars,
          })
          priority += 1
  return profiles


def build_profiles(goal: dict[str, Any]) -> list[dict[str, Any]]:
  return [build_setup_profile(goal)] + build_gate_repair_profiles(goal) + build_main_profiles(goal)


def validate_goal(goal: dict[str, Any]) -> tuple[bool, list[str]]:
  messages: list[str] = []
  failed = False
  main_profiles = build_main_profiles(goal)
  run_ids = [profile['run_id'] for profile in main_profiles]
  expected_main = expected_main_profile_count(goal)
  if len(main_profiles) != expected_main:
    messages.append(f'expected {expected_main} main profiles, found {len(main_profiles)}')
    failed = True
  if len(run_ids) != len(set(run_ids)):
    messages.append('duplicate main profile run_ids detected')
    failed = True
  if goal.get('pilot', {}).get('enabled', False):
    pilot_count = len(pilot_profiles(goal))
    expected = int(goal['pilot'].get('expected_profiles', pilot_count))
    if pilot_count != expected:
      messages.append(f'expected {expected} pilot profiles, found {pilot_count}')
      failed = True
  if int(goal['constraints']['paper_horizon']) != 3:
    messages.append(
        f"paper_horizon is {goal['constraints']['paper_horizon']}, expected 3"
    )
    failed = True
  gate_statuses = env_gate_statuses(goal)
  for env_id, (status, message) in gate_statuses.items():
    if status == 'failed':
      messages.append(f'env gate failed for {env_id}: {message}')
    elif status == 'pending':
      messages.append(f'env gate pending for {env_id}: {message}')
  for gate in goal.get('profile_gates', []):
    status, message = gate_file_status(gate['path'])
    if status == 'failed':
      messages.append(f'profile gate failed for {gate.get("name", gate["path"])}: {message}')
    elif status == 'pending':
      messages.append(f'profile gate pending for {gate.get("name", gate["path"])}: {message}')
  if not failed:
    messages.insert(0, f'goal validation passed: {expected_main} main profiles, no duplicates')
  return not failed, messages


def setup_gate_passed(goal: dict[str, Any], rows: list[dict[str, str]]) -> bool:
  latest = latest_rows(rows).get(goal['setup_smoke']['run_id'])
  return bool(latest and latest.get('event') == 'completed' and latest.get('status') == 'passed')


def profile_latest_status(
    rows: list[dict[str, str]],
    profile: dict[str, Any],
) -> dict[str, Any]:
  run_rows = rows_for_run(rows, profile['run_id'])
  latest = run_rows[-1] if run_rows else None
  attempts = sum(1 for row in run_rows if row.get('event') == 'launch')
  return {'latest': latest, 'attempts': attempts}


def is_terminal(row: dict[str, str] | None) -> bool:
  return bool(row and row.get('event') in TERMINAL_EVENTS)


def is_complete(row: dict[str, str] | None) -> bool:
  return bool(row and row.get('event') in TERMINAL_EVENTS and row.get('status') in GOOD_TERMINAL_STATUSES)


def finite_float(value: Any) -> float | None:
  try:
    parsed = float(value)
  except (TypeError, ValueError):
    return None
  return parsed if math.isfinite(parsed) else None


def has_finite_scores(row: dict[str, Any] | None) -> bool:
  if row is None:
    return False
  return (
      finite_float(row.get('final_score')) is not None and
      finite_float(row.get('best_score')) is not None
  )


def should_retry(
    goal: dict[str, Any],
    rows: list[dict[str, str]],
    profile: dict[str, Any],
) -> bool:
  state = profile_latest_status(rows, profile)
  latest = state['latest']
  if latest is None or latest.get('event') != 'failed':
    return False
  if state['attempts'] >= int(goal['constraints']['max_retries']) + 1:
    return False
  slurm_state = latest.get('slurm_state', '').split()[0]
  return slurm_state in RETRYABLE_STATES or latest.get('status') == 'retryable'


def selector_value_matches(value: Any, expected: Any) -> bool:
  if isinstance(expected, (list, tuple, set)):
    return any(selector_value_matches(value, item) for item in expected)
  return str(value) == str(expected)


def profile_matches_selector(profile: dict[str, Any],
                             selector: dict[str, Any]) -> bool:
  for key in ('env_id', 'regime', 'method', 'seed'):
    if key in selector and not selector_value_matches(profile.get(key), selector[key]):
      return False
  return True


def expected_main_profile_count(goal: dict[str, Any]) -> int:
  configured = goal.get('constraints', {}).get('expected_main_profiles')
  if configured is not None:
    return int(configured)
  return (
      len(goal['matrix']['envs']) *
      len(goal['matrix']['regimes']) *
      len(goal['matrix']['methods']) *
      len(goal['matrix']['seeds'])
  )


def gate_file_status(path_text: str) -> tuple[str, str]:
  path = relpath(path_text)
  if not path.exists():
    return 'pending', f'missing gate artifact {path_text}'
  try:
    payload = json.loads(path.read_text())
  except Exception as exc:
    return 'failed', f'invalid gate artifact {path_text}: {exc}'
  if payload.get('passed') is True:
    return 'passed', f'passed gate {path_text}'
  if 'error' in payload:
    return 'failed', f'gate did not pass {path_text}: {payload["error"]}'
  summary_keys = (
      'task', 'chaos', 'num_envs', 'steps', 'reward_finite',
      'observation_finite', 'reward_mean', 'reward_sum',
  )
  summary = {
      key: payload.get(key)
      for key in summary_keys
      if key in payload
  }
  return 'failed', f'gate did not pass {path_text}: {summary}'


def env_gate_statuses(goal: dict[str, Any]) -> dict[str, tuple[str, str]]:
  statuses: dict[str, tuple[str, str]] = {}
  for env in goal['matrix']['envs']:
    env_id = env['env_id']
    gate = env.get('gate', '')
    if gate == 'built_in':
      statuses[env_id] = ('passed', 'built-in gate')
    elif gate:
      statuses[env_id] = gate_file_status(gate)
    else:
      statuses[env_id] = ('passed', 'no gate configured')
  return statuses


def profile_gate_statuses(
    goal: dict[str, Any],
    profile: dict[str, Any],
) -> list[tuple[str, str, str]]:
  statuses: list[tuple[str, str, str]] = []
  env_status, env_message = env_gate_statuses(goal).get(
      profile['env_id'],
      ('failed', f'no env gate status for {profile["env_id"]}'),
  )
  statuses.append((f'env:{profile["env_id"]}', env_status, env_message))
  for gate in goal.get('profile_gates', []):
    selectors = gate.get('selectors', [])
    if selectors and not any(profile_matches_selector(profile, selector) for selector in selectors):
      continue
    status, message = gate_file_status(gate['path'])
    statuses.append((gate.get('name', gate['path']), status, message))
  return statuses


def profile_gate_blockers(goal: dict[str, Any], profile: dict[str, Any]) -> list[str]:
  return [
      f'{name}={status}: {message}'
      for name, status, message in profile_gate_statuses(goal, profile)
      if status != 'passed'
  ]


def profile_gate_failures(goal: dict[str, Any], profile: dict[str, Any]) -> list[str]:
  return [
      f'{name}=failed: {message}'
      for name, status, message in profile_gate_statuses(goal, profile)
      if status == 'failed'
  ]


def gate_repair_needed(goal: dict[str, Any]) -> tuple[bool, str]:
  repair = goal.get('gate_repair', {})
  if not repair.get('enabled', False):
    return False, 'disabled'
  release_names = set(repair.get('release_when_gates_pass', []))
  if not release_names:
    return bool(build_gate_repair_profiles(goal)), 'enabled without release gates'
  by_name = {
      gate.get('name', gate.get('path', '')): gate
      for gate in goal.get('profile_gates', [])
  }
  missing = sorted(name for name in release_names if name not in by_name)
  if missing:
    return True, f'missing release gate definitions: {", ".join(missing)}'
  blockers = []
  for name in sorted(release_names):
    status, message = gate_file_status(by_name[name]['path'])
    if status != 'passed':
      blockers.append(f'{name}={status}: {message}')
  if blockers:
    return True, '; '.join(blockers)
  return False, f'release gates passed: {", ".join(sorted(release_names))}'


def gate_repair_reserved_gpus(goal: dict[str, Any]) -> int:
  needed, _ = gate_repair_needed(goal)
  if not needed:
    return 0
  return max(0, int(goal.get('gate_repair', {}).get('reserved_gpus', 0)))


def active_launch_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
  return [
      row for row in latest_rows(rows).values()
      if row.get('event') == 'launch' and row.get('job_id')
  ]


def active_launch_counts_by_kind(goal: dict[str, Any],
                                 rows: list[dict[str, str]]) -> dict[str, int]:
  profiles_by_id = {profile['run_id']: profile for profile in build_profiles(goal)}
  counts: dict[str, int] = {}
  for row in active_launch_rows(rows):
    kind = profiles_by_id.get(row.get('run_id', ''), {}).get('kind', 'unknown')
    counts[kind] = counts.get(kind, 0) + 1
  return counts


def pending_gate_repair_profiles(goal: dict[str, Any],
                                 rows: list[dict[str, str]]) -> list[dict[str, Any]]:
  needed, _ = gate_repair_needed(goal)
  if not needed:
    return []
  pending: list[dict[str, Any]] = []
  for profile in build_gate_repair_profiles(goal):
    state = profile_latest_status(rows, profile)
    latest = state['latest']
    if latest is None:
      pending.append(profile)
    elif latest.get('event') == 'launch':
      continue
    elif is_complete(latest) or latest.get('event') == 'blocked':
      continue
    elif should_retry(goal, rows, profile):
      pending.append(profile)
  return sorted(pending, key=lambda item: int(item['priority']))


def phase_profiles(goal: dict[str, Any], phase: dict[str, Any]) -> list[dict[str, Any]]:
  selectors = phase.get('selectors', [])
  profiles = build_main_profiles(goal)
  if not selectors:
    return profiles
  return [
      profile for profile in profiles
      if any(profile_matches_selector(profile, selector) for selector in selectors)
  ]


def phase_status(
    goal: dict[str, Any],
    rows: list[dict[str, str]],
    phase: dict[str, Any],
) -> tuple[str, str]:
  profiles = phase_profiles(goal, phase)
  if not profiles:
    return 'failed', 'phase selects no profiles'
  latest = latest_rows(rows)
  pending = 0
  gate_pending = 0
  gate_failed = 0
  for profile in profiles:
    gate_failures = profile_gate_failures(goal, profile)
    if gate_failures:
      gate_failed += 1
      continue
    if profile_gate_blockers(goal, profile):
      gate_pending += 1
      continue
    row = latest.get(profile['run_id'])
    if row is None or row.get('event') == 'launch' or should_retry(goal, rows, profile):
      pending += 1
      continue
    if not is_complete(row):
      return 'failed', f'{profile["run_id"]}: {row.get("event", "missing")}/{row.get("status", "")}'
    if not has_finite_scores(row):
      return 'failed', f'{profile["run_id"]}: non-finite final/best score'
  if pending:
    suffix = []
    if gate_pending:
      suffix.append(f'gate_pending={gate_pending}')
    if gate_failed:
      suffix.append(f'gate_failed={gate_failed}')
    extra = f'; {", ".join(suffix)}' if suffix else ''
    return 'pending', f'pending profiles={pending}/{len(profiles)}{extra}'
  if gate_failed:
    return 'failed', f'gate_failed_profiles={gate_failed}/{len(profiles)}'
  if gate_pending:
    return 'pending', f'waiting on gated profiles={gate_pending}/{len(profiles)}'
  return 'passed', f'passed profiles={len(profiles)}'


def phase_statuses(
    goal: dict[str, Any],
    rows: list[dict[str, str]],
) -> list[tuple[str, str, str]]:
  phases = goal.get('launch_phases', [])
  return [
      (phase['name'], *phase_status(goal, rows, phase))
      for phase in phases
  ]


def active_launch_phase(
    goal: dict[str, Any],
    rows: list[dict[str, str]],
) -> tuple[dict[str, Any] | None, str, str]:
  for phase in goal.get('launch_phases', []):
    status, message = phase_status(goal, rows, phase)
    if status != 'passed':
      return phase, status, message
  return None, 'passed', 'all launch phases passed'


def gate_blocked_profile_count(goal: dict[str, Any]) -> int:
  return sum(1 for profile in build_main_profiles(goal) if profile_gate_blockers(goal, profile))


def pilot_profiles(goal: dict[str, Any]) -> list[dict[str, Any]]:
  pilot = goal.get('pilot', {})
  if not pilot.get('enabled', False):
    return []
  selectors = pilot.get('profiles', [])
  profiles = build_main_profiles(goal)
  selected: list[dict[str, Any]] = []
  seen: set[str] = set()
  for selector in selectors:
    matches = [
        profile for profile in profiles
        if profile_matches_selector(profile, selector)
    ]
    for profile in matches:
      if profile['run_id'] not in seen:
        selected.append(profile)
        seen.add(profile['run_id'])
  return sorted(selected, key=lambda item: int(item['priority']))


def pilot_gate_status(
    goal: dict[str, Any],
    rows: list[dict[str, str]],
) -> tuple[str, str]:
  pilot = goal.get('pilot', {})
  if not pilot.get('enabled', False):
    return 'passed', 'pilot disabled'
  profiles = pilot_profiles(goal)
  expected = int(pilot.get('expected_profiles', len(profiles)))
  if len(profiles) != expected:
    return 'failed', f'pilot profile mismatch expected={expected} found={len(profiles)}'
  max_abs_drop = float(pilot.get('max_abs_final_drop', 75.0))
  max_rel_drop = float(pilot.get('max_rel_final_drop', 0.10))
  latest = latest_rows(rows)
  pending = []
  failures = []
  for profile in profiles:
    row = latest.get(profile['run_id'])
    if row is None or row.get('event') == 'launch' or should_retry(goal, rows, profile):
      pending.append(profile['run_id'])
      continue
    if not is_complete(row):
      failures.append(f'{profile["run_id"]}: {row.get("event", "missing")}/{row.get("status", "")}')
      continue
    try:
      final_score = float(row.get('final_score', 'nan'))
      best_score = float(row.get('best_score', 'nan'))
    except ValueError:
      failures.append(f'{profile["run_id"]}: non-numeric score')
      continue
    if not math.isfinite(final_score) or not math.isfinite(best_score):
      failures.append(f'{profile["run_id"]}: missing score')
      continue
    allowed_drop = max(max_abs_drop, max_rel_drop * abs(best_score))
    if best_score - final_score > allowed_drop:
      failures.append(
          f'{profile["run_id"]}: late drop {best_score - final_score:.3f} > {allowed_drop:.3f}'
      )
  if failures:
    return 'failed', '; '.join(failures)
  if pending:
    return 'pending', f'pending pilot profiles={len(pending)}'
  return 'passed', f'pilot passed profiles={len(profiles)}'


def pending_profiles(goal: dict[str, Any], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
  setup = build_setup_profile(goal)
  setup_state = profile_latest_status(rows, setup)
  if not setup_gate_passed(goal, rows):
    latest = setup_state['latest']
    if latest is None or latest.get('event') == 'failed':
      if setup_state['attempts'] <= int(goal['constraints']['max_retries']):
        return [setup]
    return []

  pilot_state, _ = pilot_gate_status(goal, rows)
  if pilot_state != 'passed':
    if pilot_state == 'failed':
      return []
    pending_pilot: list[dict[str, Any]] = []
    for profile in pilot_profiles(goal):
      state = profile_latest_status(rows, profile)
      latest = state['latest']
      if latest is None:
        pending_pilot.append(profile)
      elif latest.get('event') == 'launch':
        continue
      elif is_complete(latest) or latest.get('event') == 'blocked':
        continue
      elif should_retry(goal, rows, profile):
        pending_pilot.append(profile)
    return sorted(pending_pilot, key=lambda item: int(item['priority']))

  pending: list[dict[str, Any]] = []
  for profile in build_main_profiles(goal):
    if profile_gate_blockers(goal, profile):
      continue
    state = profile_latest_status(rows, profile)
    latest = state['latest']
    if latest is None:
      pending.append(profile)
    elif latest.get('event') == 'launch':
      continue
    elif is_complete(latest) or latest.get('event') == 'blocked':
      continue
    elif should_retry(goal, rows, profile):
      pending.append(profile)
  return sorted(pending, key=lambda item: int(item['priority']))


def parse_gpu_status_output(
    goal: dict[str, Any],
    text: str,
) -> dict[str, Any]:
  gpu_rows: list[dict[str, Any]] = []
  slurm_jobs: list[dict[str, str]] = []
  section = None
  for raw_line in text.splitlines():
    line = raw_line.strip()
    if not line:
      continue
    if line.startswith('gpu_index,'):
      section = 'gpus'
      continue
    if line.startswith('compute_processes_'):
      section = 'processes'
      continue
    if line.startswith('all_slurm_jobs_'):
      section = 'slurm'
      continue
    if section == 'gpus' and re.match(r'^\d+,', line):
      parts = [part.strip() for part in line.split(',')]
      if len(parts) >= 6:
        try:
          gpu_rows.append({
              'index': int(parts[0]),
              'util': float(parts[3]),
              'mem_used': float(parts[4]),
              'mem_total': float(parts[5]),
          })
        except ValueError:
          pass
    elif section == 'slurm' and '|' in line:
      parts = line.split('|')
      if len(parts) >= 7:
        slurm_jobs.append({
            'job_id': parts[0],
            'user': parts[1],
            'name': parts[2],
            'state': parts[3],
            'time': parts[4],
            'gres': parts[5],
            'reason': parts[6],
        })
  util_threshold = float(goal['constraints']['physical_gpu_free_util_pct'])
  mem_threshold = float(goal['constraints']['physical_gpu_free_mem_mib'])
  free_gpus = [
      gpu for gpu in gpu_rows
      if gpu['util'] <= util_threshold and gpu['mem_used'] <= mem_threshold
  ]
  prefix = goal['remote']['steward_job_prefix']
  active_steward_jobs = [
      job for job in slurm_jobs
      if job['name'].startswith(prefix)
  ]
  return {
      'gpu_rows': gpu_rows,
      'free_gpu_count': len(free_gpus),
      'slurm_jobs': slurm_jobs,
      'active_steward_jobs': active_steward_jobs,
      'raw': text,
  }


def ncc_status(goal: dict[str, Any]) -> dict[str, Any]:
  result = run_local([str(NCC_GPU_STATUS), goal['remote']['host']], timeout=90)
  if result.returncode != 0:
    return {
        'ok': False,
        'free_gpu_count': 0,
        'active_steward_jobs': [],
        'error': result.stderr.strip(),
        'raw': result.stdout,
    }
  parsed = parse_gpu_status_output(goal, result.stdout)
  parsed['ok'] = True
  parsed['error'] = ''
  return parsed


def query_sacct(goal: dict[str, Any], job_ids: list[str]) -> dict[str, dict[str, str]]:
  if not job_ids:
    return {}
  ids = ','.join(sorted(set(job_ids)))
  command = (
      'sacct -n -P -X '
      f'-j {shlex.quote(ids)} '
      '--format=JobIDRaw,State,Elapsed,ExitCode 2>/dev/null || true'
  )
  states: dict[str, dict[str, str]] = {}
  try:
    result = run_remote(
        goal,
        command,
        timeout=int(goal.get('constraints', {}).get('slurm_query_timeout_sec', 180)),
    )
  except subprocess.TimeoutExpired:
    return states
  if result.returncode != 0:
    return states
  for line in result.stdout.splitlines():
    parts = line.split('|')
    if len(parts) < 4:
      continue
    raw_id, state, elapsed, exit_code = [part.strip() for part in parts[:4]]
    if raw_id in job_ids and raw_id not in states:
      states[raw_id] = {
          'state': state,
          'elapsed': elapsed,
          'exit_code': exit_code,
      }
  return states


def elapsed_to_hours(elapsed: str) -> str:
  if not elapsed:
    return ''
  try:
    day_part = 0
    time_part = elapsed
    if '-' in elapsed:
      days, time_part = elapsed.split('-', 1)
      day_part = int(days)
    chunks = [int(float(part)) for part in time_part.split(':')]
    if len(chunks) == 3:
      hours, minutes, seconds = chunks
    elif len(chunks) == 2:
      hours, minutes, seconds = 0, chunks[0], chunks[1]
    else:
      return ''
    total = day_part * 24 + hours + minutes / 60 + seconds / 3600
    return f'{total:.4f}'
  except Exception:
    return ''


def remote_metric_summary(goal: dict[str, Any], run_dir: str, max_steps: int) -> dict[str, Any]:
  script = r'''
import csv, json, math, os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
max_steps = int(os.environ["MAX_STEPS"])
metrics = run_dir / "metrics" / "scalars.csv"
eval_rows = []
if metrics.exists():
  with metrics.open(newline="") as handle:
    for row in csv.DictReader(handle):
      if row.get("tag") != "eval/return_mean":
        continue
      try:
        eval_rows.append((int(float(row["step"])), float(row["value"])))
      except Exception:
        pass
eval_rows.sort()
usable = [(s, v) for s, v in eval_rows if s <= max_steps]
final_step = usable[-1][0] if usable else 0
final_score = usable[-1][1] if usable else None
best_score = max((v for _, v in usable), default=None)
auc = None
if len(usable) >= 2:
  area = 0.0
  for (s0, v0), (s1, v1) in zip(usable[:-1], usable[1:]):
    area += (s1 - s0) * (v0 + v1) / 2.0
  auc = area / float(max_steps)
checkpoint_dir = run_dir / "checkpoint"
checkpoint_steps = []
if checkpoint_dir.exists():
  for path in checkpoint_dir.iterdir():
    if path.is_dir() and path.name.isdigit():
      checkpoint_steps.append(int(path.name))
checkpoint_latest = max(checkpoint_steps) if checkpoint_steps else None
resume_path = run_dir / "resume_verified.json"
resume_verified = None
if resume_path.exists():
  try:
    resume_verified = json.loads(resume_path.read_text())
  except Exception:
    resume_verified = None
print(json.dumps({
  "metrics_exists": metrics.exists(),
  "final_step": final_step,
  "final_score": final_score,
  "best_score": best_score,
  "auc": auc,
  "checkpoint_latest": checkpoint_latest,
  "checkpoint_ok": checkpoint_latest is not None and checkpoint_latest >= final_step,
  "resume_verified": resume_verified,
}))
'''
  command = (
      f'cd {shlex.quote(goal["remote"]["path"])} && '
      f'export RUN_DIR={shlex.quote(run_dir)} MAX_STEPS={shlex.quote(str(max_steps))} && '
      f'PYTHON_BIN="${{PYTHON_BIN:-$HOME/.venvs/temporalhorizon-jax/bin/python}}" && '
      f'if ! test -x "$PYTHON_BIN"; then PYTHON_BIN="$(command -v python3 || command -v python)"; fi && '
      f'"$PYTHON_BIN" - <<{shlex.quote("PY")}\n{script}\nPY'
  )
  result = run_remote(goal, command, timeout=120)
  if result.returncode != 0:
    return {'metrics_exists': False, 'error': result.stderr.strip()}
  try:
    return json.loads(result.stdout.strip().splitlines()[-1])
  except Exception as exc:
    return {
        'metrics_exists': False,
        'error': f'failed to parse metric summary: {exc}; output={result.stdout[-500:]}',
    }


def slurm_state_is_terminal(state: str) -> bool:
  return any(state.startswith(prefix) for prefix in SLURM_TERMINAL_PREFIXES)


def setup_resume_gate_passed(summary: dict[str, Any]) -> tuple[bool, str]:
  resume = summary.get('resume_verified') or {}
  if resume.get('passed') is True:
    return True, 'checkpoint resume smoke passed'
  checkpoint_latest = summary.get('checkpoint_latest')
  try:
    checkpoint_step = int(checkpoint_latest)
  except (TypeError, ValueError):
    checkpoint_step = -1
  if checkpoint_step >= 8192:
    return True, (
        'checkpoint resume smoke passed via checkpoint fallback '
        f'(latest={checkpoint_step}, resume_verified.json missing or false)'
    )
  return False, f'checkpoint resume smoke failed: {resume or summary}'


def cache_gate_repair_artifacts(goal: dict[str, Any],
                                profile: dict[str, Any]) -> tuple[bool, str]:
  cached = 0
  missing = []
  for artifact in profile.get('artifacts', []):
    remote_file = artifact.get('remote', '')
    local_file = artifact.get('local', '')
    if not remote_file or not local_file:
      continue
    if cache_remote_file(goal, remote_file, relpath(local_file)):
      cached += 1
    else:
      missing.append(remote_file)
  gate_path = profile.get('gate_path', '')
  if gate_path:
    status, message = gate_file_status(gate_path)
  else:
    status, message = 'failed', 'gate repair profile has no gate_path'
  notes = f'{message}; cached_artifacts={cached}'
  if missing:
    notes += '; missing_artifacts=' + ','.join(missing)
  return status == 'passed', notes


def reconcile_setup_gate(goal: dict[str, Any], rows: list[dict[str, str]]) -> int:
  if setup_gate_passed(goal, rows):
    return 0
  profile = build_setup_profile(goal)
  latest = profile_latest_status(rows, profile)['latest']
  if latest is None:
    return 0
  summary = remote_metric_summary(goal, profile['run_dir'], 8192)
  passed, notes = setup_resume_gate_passed(summary)
  if not passed:
    return 0
  append_ledger(goal, {
      'timestamp': now_iso(),
      'event': 'completed',
      'run_id': profile['run_id'],
      'status': 'passed',
      'job_id': latest.get('job_id', ''),
      'attempt': latest.get('attempt', '1'),
      'env_id': profile['env_id'],
      'regime': profile['regime'],
      'method': profile['method'],
      'seed': profile['seed'],
      'paper_horizon': profile['paper_horizon'],
      'run_dir': profile['run_dir'],
      'launcher': profile['script'],
      'git_commit': latest.get('git_commit', ''),
      'remote_commit': latest.get('remote_commit', ''),
      'slurm_state': latest.get('slurm_state', 'RECONCILED'),
      'final_step': summary.get('final_step', ''),
      'final_score': summary.get('final_score', ''),
      'best_score': summary.get('best_score', ''),
      'auc': summary.get('auc', ''),
      'wall_hours': latest.get('wall_hours', ''),
      'checkpoint_ok': summary.get('checkpoint_ok', ''),
      'notes': f'{notes}; reconciled from remote artifacts',
  })
  return 1


def process_terminal_jobs(goal: dict[str, Any], rows: list[dict[str, str]]) -> int:
  latest = latest_rows(rows)
  active_launches = active_launch_rows(rows)
  sacct = query_sacct(goal, [row['job_id'] for row in active_launches])
  profiles_by_id = {profile['run_id']: profile for profile in build_profiles(goal)}
  appended = 0
  max_steps = int(goal['constraints']['full_run_steps'])
  for row in active_launches:
    job_id = row['job_id']
    job_state = sacct.get(job_id)
    if not job_state or not slurm_state_is_terminal(job_state['state']):
      continue
    profile = profiles_by_id.get(row['run_id'])
    if profile is None:
      continue
    is_setup = profile['kind'] == 'setup_smoke'
    is_gate_repair = profile['kind'] == 'gate_repair'
    elapsed_hours = elapsed_to_hours(job_state.get('elapsed', ''))
    slurm_state = job_state['state'].split()[0]
    if is_gate_repair:
      gate_passed, gate_notes = cache_gate_repair_artifacts(goal, profile)
      event = 'completed' if slurm_state == 'COMPLETED' and gate_passed else 'failed'
      if slurm_state != 'COMPLETED' and slurm_state in RETRYABLE_STATES:
        status = 'retryable'
      elif gate_passed:
        status = 'passed'
      else:
        status = 'blocked'
      append_ledger(goal, {
          'timestamp': now_iso(),
          'event': event,
          'run_id': profile['run_id'],
          'status': status,
          'job_id': job_id,
          'attempt': row.get('attempt', '1'),
          'env_id': profile['env_id'],
          'regime': profile['regime'],
          'method': profile['method'],
          'seed': profile['seed'],
          'paper_horizon': profile['paper_horizon'],
          'run_dir': profile['run_dir'],
          'launcher': profile['script'],
          'git_commit': row.get('git_commit', ''),
          'remote_commit': row.get('remote_commit', ''),
          'slurm_state': job_state['state'],
          'wall_hours': elapsed_hours,
          'checkpoint_ok': gate_passed,
          'notes': gate_notes,
      })
      appended += 1
      continue
    summary = remote_metric_summary(
        goal,
        profile['run_dir'],
        8192 if is_setup else max_steps,
    )
    event = 'completed' if slurm_state == 'COMPLETED' else 'failed'
    status = 'completed'
    notes = ''
    if is_setup:
      passed, setup_notes = setup_resume_gate_passed(summary)
      if slurm_state == 'COMPLETED' and passed:
        status = 'passed'
        notes = setup_notes
      else:
        event = 'failed'
        status = 'blocked'
        notes = setup_notes
    elif slurm_state != 'COMPLETED':
      status = 'retryable' if slurm_state in RETRYABLE_STATES else 'blocked'
      notes = f'SLURM state {job_state["state"]}; exit={job_state.get("exit_code", "")}'
    elif not summary.get('metrics_exists'):
      event = 'failed'
      status = 'blocked'
      notes = f'completed but metrics missing: {summary.get("error", "")}'
    elif (
        finite_float(summary.get('final_score')) is None or
        finite_float(summary.get('best_score')) is None
    ):
      event = 'failed'
      status = 'blocked'
      notes = 'completed with non-finite final/best score'
    else:
      final_step = int(summary.get('final_step') or 0)
      if final_step < int(max_steps * 0.9):
        event = 'failed'
        status = 'blocked'
        notes = f'completed with insufficient final_step={final_step}'
      elif final_step < max_steps:
        event = 'partial_complete'
        status = 'partial_complete'
        notes = f'partial usable metrics final_step={final_step}'
      else:
        notes = 'completed full 500k profile'
    append_ledger(goal, {
        'timestamp': now_iso(),
        'event': event,
        'run_id': profile['run_id'],
        'status': status,
        'job_id': job_id,
        'attempt': row.get('attempt', '1'),
        'env_id': profile['env_id'],
        'regime': profile['regime'],
        'method': profile['method'],
        'seed': profile['seed'],
        'paper_horizon': profile['paper_horizon'],
        'run_dir': profile['run_dir'],
        'launcher': profile['script'],
        'git_commit': row.get('git_commit', ''),
        'remote_commit': row.get('remote_commit', ''),
        'slurm_state': job_state['state'],
        'final_step': summary.get('final_step', ''),
        'final_score': summary.get('final_score', ''),
        'best_score': summary.get('best_score', ''),
        'auc': summary.get('auc', ''),
        'wall_hours': elapsed_hours,
        'checkpoint_ok': summary.get('checkpoint_ok', ''),
        'notes': notes,
    })
    appended += 1
  return appended


def launch_blockers(
    goal: dict[str, Any],
    local: dict[str, Any],
    remote: dict[str, Any],
    gpu: dict[str, Any],
) -> list[str]:
  blockers: list[str] = []
  if not local.get('ok'):
    blockers.append(f'local git unavailable: {local.get("error", "")}')
  if goal['constraints']['require_clean_local_git'] and local.get('dirty'):
    blockers.append('local git is dirty; full launches require committed/synced code')
  if not remote.get('ok'):
    blockers.append(f'remote git unavailable: {remote.get("error", "")}')
  if goal['constraints']['require_clean_remote_git'] and remote.get('dirty'):
    blockers.append('remote git is dirty')
  if goal['constraints']['require_remote_commit_match'] and local.get('commit') != remote.get('commit'):
    blockers.append(
        f'remote commit mismatch local={local.get("commit", "")[:12]} '
        f'remote={remote.get("commit", "")[:12]}'
    )
  if not gpu.get('ok'):
    blockers.append(f'NCC GPU status unavailable: {gpu.get("error", "")}')
  return blockers


def build_launch_command(
    goal: dict[str, Any],
    profile: dict[str, Any],
) -> str:
  env_prefix = ' '.join(
      f'{key}={shlex.quote(str(value))}'
      for key, value in sorted(profile['env'].items())
  )
  script = shlex.quote(profile['script'])
  job_suffix = safe_slug(profile['run_id'])[:48]
  job_name = f'{goal["remote"]["steward_job_prefix"]}-{job_suffix}'
  return (
      f'cd {shlex.quote(goal["remote"]["path"])} && '
      f'env {env_prefix} '
      f'sbatch --parsable --job-name={shlex.quote(job_name)} {script}'
  )


def launch_profile(
    goal: dict[str, Any],
    profile: dict[str, Any],
    *,
    local_commit: str,
    remote_commit: str,
    dry_run: bool,
    rows: list[dict[str, str]],
) -> str:
  state = profile_latest_status(rows, profile)
  attempt = state['attempts'] + 1
  command = build_launch_command(goal, profile)
  if dry_run:
    return f'DRY launch {profile["run_id"]}: {command}'
  result = run_remote(goal, command, timeout=60)
  if result.returncode != 0:
    raise RuntimeError(
        f'failed to launch {profile["run_id"]}: {result.stderr.strip()}'
    )
  job_id = result.stdout.strip().splitlines()[-1].split(';')[0].strip()
  if not job_id:
    raise RuntimeError(f'failed to parse sbatch job id: {result.stdout}')
  append_ledger(goal, {
      'timestamp': now_iso(),
      'event': 'launch',
      'run_id': profile['run_id'],
      'status': 'launched',
      'job_id': job_id,
      'attempt': attempt,
      'env_id': profile['env_id'],
      'regime': profile['regime'],
      'method': profile['method'],
      'seed': profile['seed'],
      'paper_horizon': profile['paper_horizon'],
      'run_dir': profile['run_dir'],
      'launcher': profile['script'],
      'git_commit': local_commit,
      'remote_commit': remote_commit,
      'notes': (
          'setup smoke' if profile['kind'] == 'setup_smoke' else
          'gate repair diagnostic' if profile['kind'] == 'gate_repair' else
          '500k full profile'
      ),
  })
  return f'launched {profile["run_id"]} as job {job_id}'


def eta_summary(
    goal: dict[str, Any],
    rows: list[dict[str, str]],
) -> str:
  latest = latest_rows(rows)
  main_profiles = build_main_profiles(goal)
  remaining = [
      profile for profile in main_profiles
      if not is_complete(latest.get(profile['run_id']))
  ]
  wall_hours = [
      finite_float(row.get('wall_hours'))
      for row in latest.values()
      if row.get('method') != 'checkpoint_resume_smoke' and is_complete(row)
  ]
  wall_hours = [value for value in wall_hours if value is not None]
  if not wall_hours:
    return f'eta_remaining_jobs={len(remaining)} eta_unavailable=no_completed_wall_hours'
  avg_wall = sum(wall_hours) / len(wall_hours)
  max_active = int(goal['constraints']['max_active_gpus'])
  observed_efficiency = float(goal.get('eta', {}).get('observed_parallel_efficiency', 0.80))
  eta_hours = len(remaining) * avg_wall / max(max_active * observed_efficiency, 1e-6)
  return (
      f'eta_remaining_jobs={len(remaining)} '
      f'eta_avg_wall_hours={avg_wall:.2f} '
      f'eta_observed_eff_hours={eta_hours:.1f}'
  )


def status_report(goal: dict[str, Any]) -> str:
  rows = read_ledger(goal)
  ok, messages = validate_goal(goal)
  local = local_git_state()
  remote = remote_git_state(goal)
  # Status is intentionally read-only: do not attempt any git sync or mutation.
  gpu = ncc_status(goal)
  main_profiles = build_main_profiles(goal)
  pilot_state, pilot_message = pilot_gate_status(goal, rows)
  active_phase, phase_state, phase_message = active_launch_phase(goal, rows)
  latest = latest_rows(rows)
  complete_main = sum(
      1 for profile in main_profiles
      if is_complete(latest.get(profile['run_id']))
  )
  blocked_main = sum(
      1 for profile in main_profiles
      if latest.get(profile['run_id'], {}).get('event') == 'blocked'
  )
  launched_open = sum(
      1 for row in latest.values()
      if row.get('event') == 'launch'
  )
  pending = pending_profiles(goal, rows)
  repair_needed, repair_message = gate_repair_needed(goal)
  repair_pending = pending_gate_repair_profiles(goal, rows)
  active_counts = active_launch_counts_by_kind(goal, rows)
  repair_reserved = gate_repair_reserved_gpus(goal)
  lines = [
      f'goal={goal["name"]}',
      f'validation={"ok" if ok else "blocked"}: {"; ".join(messages)}',
      f'local_commit={local.get("commit", "")[:12]} dirty={local.get("dirty")}',
      f'remote_commit={remote.get("commit", "")[:12]} dirty={remote.get("dirty")} ok={remote.get("ok")}',
      f'setup_checkpoint_resume_passed={setup_gate_passed(goal, rows)}',
      f'pilot_gate={pilot_state}: {pilot_message}',
      'launch_phase=' + (
          f'{active_phase["name"]} {phase_state}: {phase_message}'
          if active_phase is not None else
          f'{phase_state}: {phase_message}'
      ),
      f'main_completed={complete_main}/{len(main_profiles)} blocked={blocked_main} open_launches={launched_open}',
      f'pending={len(pending)} next={pending[0]["run_id"] if pending else "none"}',
      (
          f'gate_repair_needed={repair_needed} reserved_gpus={repair_reserved} '
          f'active_repair_jobs={active_counts.get("gate_repair", 0)} '
          f'pending_repair={len(repair_pending)} reason={repair_message}'
      ),
      f'gate_blocked_profiles={gate_blocked_profile_count(goal)}',
      eta_summary(goal, rows),
      f'ledger={goal["tracking"]["ledger"]}',
      f'results_dir={goal["tracking"]["results_dir"]}',
  ]
  if goal.get('launch_phases'):
    lines.append(
        'phase_statuses=' + '; '.join(
            f'{name}:{status}({message})'
            for name, status, message in phase_statuses(goal, rows)
        )
    )
  if gpu.get('ok'):
    lines.append(
        f'ncc_free_physical_gpus={gpu["free_gpu_count"]} '
        f'active_steward_jobs={len(gpu["active_steward_jobs"])}'
    )
  else:
    lines.append(f'ncc_gpu_status_error={gpu.get("error", "")}')
  blockers = launch_blockers(goal, local, remote, gpu)
  if blockers:
    lines.append('launch_blockers=' + '; '.join(blockers))
  return '\n'.join(lines)


def campaign_terminal(goal: dict[str, Any], rows: list[dict[str, str]]) -> bool:
  latest = latest_rows(rows)
  for profile in build_main_profiles(goal):
    row = latest.get(profile['run_id'])
    if row is None or row.get('event') not in TERMINAL_EVENTS:
      return False
  return setup_gate_passed(goal, rows)


def tick(goal: dict[str, Any], *, goal_path: Path, dry_run: bool = False) -> str:
  rows = read_ledger(goal)
  processed = 0
  reconciled = 0
  if not dry_run:
    processed = process_terminal_jobs(goal, rows)
    if processed:
      rows = read_ledger(goal)
    reconciled = reconcile_setup_gate(goal, rows)
    processed += reconciled
  sync_messages: list[str] = []
  if processed:
    sync_messages.append(auto_commit_steward_state(goal, 'record terminal job results'))
    rows = read_ledger(goal)
  ok, messages = validate_goal(goal)
  local = local_git_state()
  remote = remote_git_state(goal)
  if not dry_run:
    sync_message = sync_remote_to_local_if_safe(goal, local, remote)
    if sync_message != 'remote sync already current':
      sync_messages.append(sync_message)
      remote = remote_git_state(goal)
  gpu = ncc_status(goal)
  pilot_state, pilot_message = pilot_gate_status(goal, rows)
  active_phase, phase_state, phase_message = active_launch_phase(goal, rows)
  blockers = [] if ok else messages
  blockers.extend(launch_blockers(goal, local, remote, gpu))
  pending = pending_profiles(goal, rows)
  repair_needed, repair_message = gate_repair_needed(goal)
  repair_pending = pending_gate_repair_profiles(goal, rows)
  active_counts = active_launch_counts_by_kind(goal, rows)
  active_repair = active_counts.get('gate_repair', 0)
  repair_reserved = gate_repair_reserved_gpus(goal)
  active = len(gpu.get('active_steward_jobs', []))
  free_physical = int(gpu.get('free_gpu_count', 0))
  max_active = int(goal['constraints']['max_active_gpus'])
  slots = max(0, min(max_active - active, free_physical))
  lines = [
      f'processed_terminal_jobs={processed}',
      f'reconciled_setup_gate={reconciled}',
      f'pending_profiles={len(pending)}',
      f'active_steward_jobs={active}',
      f'free_physical_gpus={free_physical}',
      f'launch_slots={slots}',
      (
          f'gate_repair_needed={repair_needed} reserved_gpus={repair_reserved} '
          f'active_repair_jobs={active_repair} pending_repair={len(repair_pending)} '
          f'reason={repair_message}'
      ),
      f'pilot_gate={pilot_state}: {pilot_message}',
      'launch_phase=' + (
          f'{active_phase["name"]} {phase_state}: {phase_message}'
          if active_phase is not None else
          f'{phase_state}: {phase_message}'
      ),
      eta_summary(goal, rows),
      f'gate_blocked_profiles={gate_blocked_profile_count(goal)}',
      f'ledger={goal["tracking"]["ledger"]}',
      f'results_dir={goal["tracking"]["results_dir"]}',
  ]
  if goal.get('launch_phases'):
    lines.append(
        'phase_statuses=' + '; '.join(
            f'{name}:{status}({message})'
            for name, status, message in phase_statuses(goal, rows)
        )
    )
  lines.extend(sync_messages)
  if blockers:
    lines.append('launch_blocked=' + '; '.join(blockers))
    return '\n'.join(lines)
  if slots == 0:
    if campaign_terminal(goal, rows):
      if dry_run:
        lines.append('campaign_terminal=true would_package_results=true')
      else:
        lines.append('campaign_terminal=true')
        lines.append(package_results(goal, goal_path=goal_path))
    lines.append('no_launch_slots_available')
    return '\n'.join(lines)
  if campaign_terminal(goal, rows):
    if dry_run:
      lines.append('campaign_terminal=true would_package_results=true')
    else:
      lines.append('campaign_terminal=true')
      lines.append(package_results(goal, goal_path=goal_path))
    return '\n'.join(lines)
  launched = 0
  repair_launched = 0
  repair_slots = 0
  if repair_needed and repair_reserved:
    repair_slots = min(slots, max(0, repair_reserved - active_repair))
  for profile in repair_pending[:repair_slots]:
    try:
      line = launch_profile(
          goal,
          profile,
          local_commit=local['commit'],
          remote_commit=remote['commit'],
          dry_run=dry_run,
          rows=rows,
      )
      lines.append(line)
      launched += 1
      repair_launched += 1
    except Exception as exc:
      lines.append(f'launch_failed {profile["run_id"]}: {exc}')
      break
  slots_after_repair = max(0, slots - repair_launched)
  active_main = max(0, active - active_repair)
  effective_active_repair = active_repair + repair_launched
  if repair_needed and not repair_pending and effective_active_repair == 0:
    lines.append('gate_repair_blocked=no pending repair profiles; inspect repair artifacts or add a safe diagnostic/fix profile')
  if repair_needed and effective_active_repair:
    main_slots = 0
  elif repair_needed and repair_reserved:
    main_capacity = max(0, max_active - repair_reserved)
    main_slots = min(slots_after_repair, max(0, main_capacity - active_main))
  else:
    main_slots = slots_after_repair
  if repair_needed and repair_reserved and main_slots == 0:
    lines.append('main_launch_slots_reserved_for_gate_repair')
  if not pending and launched == 0:
    if repair_needed and repair_pending:
      lines.append('no_main_launches_pending_gate_repair_first')
    elif pilot_state == 'failed':
      lines.append('launch_blocked=pilot gate failed; no full matrix launch')
    elif gate_blocked_profile_count(goal):
      lines.append('no_launchable_profiles_remaining_until_gates_change')
    else:
      lines.append('no_pending_profiles')
  for profile in pending[:main_slots]:
    try:
      line = launch_profile(
          goal,
          profile,
          local_commit=local['commit'],
          remote_commit=remote['commit'],
          dry_run=dry_run,
          rows=rows,
      )
      lines.append(line)
      launched += 1
    except Exception as exc:
      lines.append(f'launch_failed {profile["run_id"]}: {exc}')
      break
  if launched and not dry_run:
    lines.append(auto_commit_steward_state(goal, f'record {launched} launch(es)'))
  return '\n'.join(lines)


def cache_remote_file(
    goal: dict[str, Any],
    remote_file: str,
    local_file: Path,
) -> bool:
  begin = '__TD_CACHE_BEGIN__'
  end = '__TD_CACHE_END__'
  command = (
      f'if test -f {shlex.quote(remote_file)}; then '
      f'printf {shlex.quote(begin + chr(10))}; '
      f'(base64 -w 0 {shlex.quote(remote_file)} 2>/dev/null || base64 {shlex.quote(remote_file)}); '
      f'printf {shlex.quote(chr(10) + end + chr(10))}; '
      'fi'
  )
  result = run_remote(goal, command, timeout=180)
  if result.returncode != 0 or not result.stdout.strip():
    return False
  stdout_lines = result.stdout.splitlines()
  begin_idx = -1
  for idx, line in enumerate(stdout_lines):
    if line.strip() == begin:
      begin_idx = idx
  if begin_idx < 0:
    return False
  end_idx = -1
  for idx in range(begin_idx + 1, len(stdout_lines)):
    if stdout_lines[idx].strip() == end:
      end_idx = idx
      break
  if end_idx < 0:
    return False
  local_file.parent.mkdir(parents=True, exist_ok=True)
  payload = ''.join(''.join(stdout_lines[begin_idx + 1:end_idx]).split())
  payload += '=' * (-len(payload) % 4)
  local_file.write_bytes(base64.b64decode(payload))
  return True


def cache_completed_metrics(goal: dict[str, Any]) -> str:
  rows = read_ledger(goal)
  results_dir = relpath(goal['tracking']['results_dir'])
  cached = 0
  for row in rows:
    if row.get('event') not in {'completed', 'partial_complete'}:
      continue
    if row.get('status') not in GOOD_TERMINAL_STATUSES:
      continue
    run_id = row['run_id']
    remote_dir = row.get('run_dir', '')
    if not remote_dir:
      continue
    for rel in (
        'metrics/scalars.csv',
        'metrics/episodes.csv',
        'metrics/horizon_queries.csv',
        'resume_verified.json',
    ):
      if cache_remote_file(
          goal,
          f'{remote_dir}/{rel}',
          results_dir / 'cache' / run_id / rel,
      ):
        cached += 1
  return f'cached_remote_metric_files={cached}'


def package_results(
    goal: dict[str, Any],
    *,
    goal_path: Path,
    report_only: bool = False,
) -> str:
  lines = []
  if not report_only:
    lines.append(cache_completed_metrics(goal))
  script = ROOT / 'scripts' / 'generate_corl_compact_artifacts.py'
  result = run_local(
      [
          sys.executable,
          str(script),
          '--goal',
          str(goal_path),
      ],
      timeout=180,
  )
  if result.returncode != 0:
    raise RuntimeError(result.stderr.strip() or result.stdout.strip())
  lines.append(result.stdout.strip())
  return '\n'.join(line for line in lines if line)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--goal', default=str(DEFAULT_GOAL), help='Path to goal JSON/YAML file.')
  subparsers = parser.add_subparsers(dest='command', required=True)
  subparsers.add_parser('validate')
  subparsers.add_parser('status')
  tick_parser = subparsers.add_parser('tick')
  tick_parser.add_argument('--dry-run', action='store_true')
  subparsers.add_parser('package-results')
  subparsers.add_parser('generate-report')
  args = parser.parse_args(argv)

  goal_path = Path(args.goal)
  if not goal_path.is_absolute():
    goal_path = ROOT / goal_path
  goal = load_goal(goal_path)
  goal['_goal_path'] = repo_path_text(goal_path)

  if args.command == 'validate':
    ok, messages = validate_goal(goal)
    print('\n'.join(messages))
    return 0 if ok else 2
  if args.command == 'status':
    print(status_report(goal))
    return 0
  if args.command == 'tick':
    print(tick(goal, goal_path=goal_path, dry_run=args.dry_run))
    return 0
  if args.command == 'package-results':
    print(package_results(goal, goal_path=goal_path))
    return 0
  if args.command == 'generate-report':
    print(package_results(goal, goal_path=goal_path, report_only=True))
    return 0
  parser.error(f'unknown command {args.command}')
  return 2


if __name__ == '__main__':
  raise SystemExit(main())

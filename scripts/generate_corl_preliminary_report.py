#!/usr/bin/env python3
"""Generate a publication-style preliminary PDF report for a CoRL compact run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


FIXED_COLOR = '#6F2DBD'
ADAPTIVE_COLOR = '#D62728'
NEUTRAL_COLOR = '#6F6F6F'


METHOD_LABELS = {
    'paper_horizon': 'Fixed horizon',
    'adaptive_rhs': 'Adaptive RHS',
}
METHOD_COLORS = {
    'paper_horizon': FIXED_COLOR,
    'adaptive_rhs': ADAPTIVE_COLOR,
}


def pretty_env(env_id: str) -> str:
  return env_id.replace('_', '-')


def relpath(path: str | Path) -> Path:
  path = Path(path)
  return path if path.is_absolute() else ROOT / path


def read_goal(path: Path) -> dict[str, Any]:
  with path.open() as handle:
    return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
  if not path.exists():
    return []
  with path.open(newline='') as handle:
    return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow({field: row.get(field, '') for field in fields})


def tex_escape(value: Any) -> str:
  text = '' if value is None else str(value)
  return (
      text.replace('\\', r'\textbackslash{}')
      .replace('&', r'\&')
      .replace('%', r'\%')
      .replace('$', r'\$')
      .replace('#', r'\#')
      .replace('_', r'\_')
      .replace('{', r'\{')
      .replace('}', r'\}')
      .replace('~', r'\textasciitilde{}')
      .replace('^', r'\textasciicircum{}')
  )


def float_or_nan(value: Any) -> float:
  if value in (None, ''):
    return float('nan')
  try:
    parsed = float(value)
  except (TypeError, ValueError):
    return float('nan')
  return parsed if math.isfinite(parsed) else float('nan')


def int_or_none(value: Any) -> int | None:
  if value in (None, ''):
    return None
  try:
    return int(float(value))
  except (TypeError, ValueError):
    return None


def fmt(value: Any, digits: int = 2) -> str:
  if value in (None, ''):
    return ''
  try:
    parsed = float(value)
  except (TypeError, ValueError):
    return str(value)
  if not math.isfinite(parsed):
    return ''
  if abs(parsed) >= 100:
    return f'{parsed:.1f}'
  return f'{parsed:.{digits}f}'


def terminal_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
  selected: dict[str, dict[str, str]] = {}
  for row in rows:
    if row.get('method') == 'checkpoint_resume_smoke':
      continue
    if row.get('event') not in {'completed', 'partial_complete'}:
      continue
    if row.get('status') not in {'completed', 'passed', 'partial_complete'}:
      continue
    selected[row.get('run_id', '')] = row
  return sorted(selected.values(), key=lambda row: row.get('run_id', ''))


def metric_dir(results_dir: Path, run_id: str) -> Path:
  return results_dir / 'cache' / run_id / 'metrics'


def load_scalar_series(results_dir: Path, run_id: str, tag: str) -> pd.DataFrame:
  path = metric_dir(results_dir, run_id) / 'scalars.csv'
  if not path.exists():
    return pd.DataFrame(columns=['step', 'value'])
  df = pd.read_csv(path)
  if not {'step', 'tag', 'value'}.issubset(df.columns):
    return pd.DataFrame(columns=['step', 'value'])
  out = df[df['tag'] == tag][['step', 'value']].copy()
  out['step'] = pd.to_numeric(out['step'], errors='coerce')
  out['value'] = pd.to_numeric(out['value'], errors='coerce')
  return out.dropna().sort_values('step')


def load_horizon_queries(results_dir: Path, run_id: str) -> pd.DataFrame:
  path = metric_dir(results_dir, run_id) / 'horizon_queries.csv'
  if not path.exists():
    return pd.DataFrame()
  df = pd.read_csv(path)
  if 'step' not in df.columns or 'selected_horizon' not in df.columns:
    return pd.DataFrame()
  df['step'] = pd.to_numeric(df['step'], errors='coerce')
  df['selected_horizon'] = pd.to_numeric(df['selected_horizon'], errors='coerce')
  return df.dropna(subset=['step', 'selected_horizon']).sort_values('step')


def curve_final_score(results_dir: Path, run_id: str) -> tuple[float, float]:
  curve = load_scalar_series(results_dir, run_id, 'eval/return_mean')
  if curve.empty:
    return float('nan'), float('nan')
  values = curve['value'].to_numpy(dtype=float)
  return float(values[-1]), float(np.nanmax(values))


def build_rows(goal: dict[str, Any]) -> list[dict[str, Any]]:
  results_dir = relpath(goal['tracking']['results_dir'])
  ledger_rows = terminal_rows(read_csv(relpath(goal['tracking']['ledger'])))
  rows: list[dict[str, Any]] = []
  for row in ledger_rows:
    run_id = row['run_id']
    curve_final, curve_best = curve_final_score(results_dir, run_id)
    final_score = float_or_nan(row.get('final_score'))
    best_score = float_or_nan(row.get('best_score'))
    if math.isfinite(curve_final):
      final_score = curve_final
    if math.isfinite(curve_best):
      best_score = curve_best
    rows.append({
        **row,
        'final_score_float': final_score,
        'best_score_float': best_score,
        'wall_hours_float': float_or_nan(row.get('wall_hours')),
    })
  return rows


def filter_envs(
    goal: dict[str, Any],
    rows: list[dict[str, Any]],
    excluded_envs: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  if not excluded_envs:
    return goal, rows
  filtered_goal = {
      **goal,
      'matrix': {
          **goal['matrix'],
          'envs': [
              item for item in goal['matrix']['envs']
              if item['env_id'] not in excluded_envs
          ],
      },
  }
  filtered_rows = [
      row for row in rows
      if row.get('env_id') not in excluded_envs
  ]
  return filtered_goal, filtered_rows


def build_pairs(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  by_key: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
  for row in rows:
    method = row.get('method')
    if method in {'paper_horizon', 'adaptive_rhs'}:
      key = (row.get('env_id', ''), row.get('regime', ''), str(row.get('seed', '')))
      by_key[key][method] = row

  pairs: list[dict[str, Any]] = []
  excluded: list[dict[str, Any]] = []
  for (env_id, regime, seed), methods in sorted(by_key.items()):
    baseline = methods.get('paper_horizon')
    rhs = methods.get('adaptive_rhs')
    if baseline is None or rhs is None:
      excluded.append({
          'env_id': env_id,
          'regime': regime,
          'seed': seed,
          'reason': 'missing matched baseline or RHS row',
      })
      continue
    baseline_score = baseline['final_score_float']
    rhs_score = rhs['final_score_float']
    if not math.isfinite(baseline_score) or not math.isfinite(rhs_score) or baseline_score == 0:
      excluded.append({
          'env_id': env_id,
          'regime': regime,
          'seed': seed,
          'reason': 'missing, nonfinite, or zero baseline score',
      })
      continue
    pct_delta = 100.0 * (rhs_score - baseline_score) / abs(baseline_score)
    pairs.append({
        'env_id': env_id,
        'regime': regime,
        'seed': seed,
        'baseline_score': baseline_score,
        'rhs_score': rhs_score,
        'pct_delta': pct_delta,
        'rhs_won': rhs_score > baseline_score,
        'baseline_wall_hours': baseline['wall_hours_float'],
        'rhs_wall_hours': rhs['wall_hours_float'],
        'runtime_ratio': (
            rhs['wall_hours_float'] / baseline['wall_hours_float']
            if math.isfinite(rhs['wall_hours_float'])
            and math.isfinite(baseline['wall_hours_float'])
            and baseline['wall_hours_float'] > 0
            else float('nan')
        ),
        'baseline_run_dir': baseline.get('run_dir', ''),
        'rhs_run_dir': rhs.get('run_dir', ''),
    })
  return pairs, excluded


def aggregate_pairs(pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  def agg(scope: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
    pct = np.asarray([row['pct_delta'] for row in subset], dtype=float)
    wins = sum(1 for row in subset if row['rhs_won'])
    rhs_wall = np.asarray([row['rhs_wall_hours'] for row in subset], dtype=float)
    base_wall = np.asarray([row['baseline_wall_hours'] for row in subset], dtype=float)
    ratio = np.asarray([row['runtime_ratio'] for row in subset], dtype=float)
    return {
        'scope': scope,
        'n': int(pct.size),
        'mean_pct_delta': float(np.nanmean(pct)) if pct.size else float('nan'),
        'std_pct_delta': float(np.nanstd(pct, ddof=1)) if pct.size > 1 else 0.0,
        'sem_pct_delta': float(np.nanstd(pct, ddof=1) / math.sqrt(pct.size)) if pct.size > 1 else 0.0,
        'rhs_win_rate': wins / pct.size if pct.size else float('nan'),
        'mean_rhs_wall_hours': float(np.nanmean(rhs_wall)) if rhs_wall.size else float('nan'),
        'mean_baseline_wall_hours': float(np.nanmean(base_wall)) if base_wall.size else float('nan'),
        'mean_runtime_ratio': float(np.nanmean(ratio)) if ratio.size else float('nan'),
    }

  summary = [agg('all valid pairs', pairs)]
  for regime in sorted({row['regime'] for row in pairs}):
    summary.append(agg(regime, [row for row in pairs if row['regime'] == regime]))

  env_regime: list[dict[str, Any]] = []
  for env_id, regime in sorted({(row['env_id'], row['regime']) for row in pairs}):
    label = f'{env_id} | {regime}'
    item = agg(label, [row for row in pairs if row['env_id'] == env_id and row['regime'] == regime])
    item['env_id'] = env_id
    item['regime'] = regime
    env_regime.append(item)
  return summary, env_regime


def write_latex_table(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str] | tuple[str, str, bool]],
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  normalized_columns = [
      (column[0], column[1], bool(column[2]) if len(column) > 2 else False)
      for column in columns
  ]
  lines = [
      r'\begin{tabular}{' + 'l' * len(normalized_columns) + '}',
      r'\toprule',
      ' & '.join(label for _, label, _ in normalized_columns) + r' \\',
      r'\midrule',
  ]
  for row in rows:
    values = []
    for key, _, raw_latex in normalized_columns:
      value = row.get(key, '')
      if isinstance(value, float):
        value = fmt(value)
      values.append(str(value) if raw_latex else tex_escape(value))
    lines.append(' & '.join(values) + r' \\')
  lines.extend([r'\bottomrule', r'\end{tabular}', ''])
  path.write_text('\n'.join(lines))


def set_plot_style() -> None:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  plt.rcParams.update({
      'figure.dpi': 180,
      'savefig.dpi': 300,
      'font.family': 'sans-serif',
      'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
      'mathtext.fontset': 'dejavusans',
      'axes.spines.top': False,
      'axes.spines.right': False,
      'axes.linewidth': 0.8,
      'axes.titlesize': 9,
      'axes.labelsize': 8,
      'xtick.labelsize': 7,
      'ytick.labelsize': 7,
      'legend.fontsize': 7,
      'legend.frameon': False,
      'grid.color': '#D9D2C5',
      'grid.linewidth': 0.5,
      'pdf.fonttype': 42,
      'ps.fonttype': 42,
  })


def interpolate_curves(curves: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
  curves = [curve.dropna() for curve in curves if not curve.empty]
  if not curves:
    return None
  grid = np.unique(np.concatenate([curve['step'].to_numpy(dtype=float) for curve in curves]))
  values = []
  for curve in curves:
    values.append(np.interp(grid, curve['step'].to_numpy(dtype=float), curve['value'].to_numpy(dtype=float)))
  stacked = np.vstack(values)
  mean = np.nanmean(stacked, axis=0)
  sem = np.nanstd(stacked, axis=0, ddof=1) / math.sqrt(stacked.shape[0]) if stacked.shape[0] > 1 else np.zeros_like(mean)
  return grid, mean, sem


def save_fig(fig: Any, figures_dir: Path, name: str) -> None:
  figures_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(figures_dir / f'{name}.pdf', bbox_inches='tight')
  fig.savefig(figures_dir / f'{name}.png', bbox_inches='tight')


def plot_learning_curves(goal: dict[str, Any], rows: list[dict[str, Any]], results_dir: Path, figures_dir: Path, regime: str) -> None:
  import matplotlib.pyplot as plt

  envs = [item['env_id'] for item in goal['matrix']['envs']]
  fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.3), sharex=True)
  axes_flat = axes.reshape(-1)
  for axis, env_id in zip(axes_flat, envs):
    axis.set_title(pretty_env(env_id), pad=3)
    axis.grid(True, axis='y', alpha=0.45)
    for method in ('paper_horizon', 'adaptive_rhs'):
      curves = []
      for row in rows:
        if row.get('env_id') == env_id and row.get('regime') == regime and row.get('method') == method:
          curve = load_scalar_series(results_dir, row['run_id'], 'eval/return_mean')
          if not curve.empty:
            curves.append(curve)
            axis.plot(curve['step'] / 1000.0, curve['value'], color=METHOD_COLORS[method], alpha=0.16, linewidth=0.7)
      agg = interpolate_curves(curves)
      if agg is None:
        continue
      grid, mean, sem = agg
      axis.plot(grid / 1000.0, mean, color=METHOD_COLORS[method], linewidth=1.8, label=METHOD_LABELS[method])
      axis.fill_between(grid / 1000.0, mean - sem, mean + sem, color=METHOD_COLORS[method], alpha=0.16, linewidth=0)
    axis.set_xlim(left=0)
  for axis in axes_flat[len(envs):]:
    axis.axis('off')
  for axis in axes[:, 0]:
    axis.set_ylabel('Eval return')
  for axis in axes[-1, :]:
    axis.set_xlabel('Environment steps (k)')
  handles, labels = axes_flat[0].get_legend_handles_labels()
  if handles:
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.025))
  fig.suptitle(f'{regime.title()} regime learning curves', y=1.08, fontsize=10, fontweight='bold')
  save_fig(fig, figures_dir, f'fig_learning_curves_{regime}')
  plt.close(fig)


def plot_delta_summary(env_rows: list[dict[str, Any]], figures_dir: Path) -> None:
  import matplotlib.pyplot as plt

  ordered = sorted(env_rows, key=lambda row: row['mean_pct_delta'])
  labels = [row['scope'] for row in ordered]
  means = np.asarray([row['mean_pct_delta'] for row in ordered], dtype=float)
  stds = np.asarray([row['std_pct_delta'] for row in ordered], dtype=float)
  y = np.arange(len(ordered))
  colors = [ADAPTIVE_COLOR if row['mean_pct_delta'] >= 0 else FIXED_COLOR for row in ordered]
  fig, axis = plt.subplots(figsize=(7.0, 4.9))
  axis.barh(y, means, xerr=stds, color=colors, alpha=0.88, ecolor='#2B2B2B', capsize=2)
  axis.axvline(0, color=FIXED_COLOR, linewidth=0.9)
  axis.set_yticks(y, labels)
  axis.set_xlabel('Adaptive RHS final-score delta vs fixed horizon (%)')
  axis.set_title('Matched seed percent deltas by environment and regime')
  axis.grid(True, axis='x', alpha=0.45)
  save_fig(fig, figures_dir, 'fig_percent_delta_summary')
  plt.close(fig)


def pareto_frontier(points: list[tuple[float, float, int]]) -> list[int]:
  valid = sorted((x, y, i) for x, y, i in points if math.isfinite(x) and math.isfinite(y))
  frontier: list[int] = []
  best_y = -float('inf')
  for x, y, i in valid:
    if y > best_y:
      frontier.append(i)
      best_y = y
  return frontier


def plot_pareto(env_rows: list[dict[str, Any]], figures_dir: Path) -> None:
  import matplotlib.pyplot as plt

  fig, axis = plt.subplots(figsize=(5.5, 4.1))
  points = []
  markers = {'clean': 'o', 'chaos': 's'}
  for idx, row in enumerate(env_rows):
    x = row['mean_rhs_wall_hours']
    y = row['mean_pct_delta']
    points.append((x, y, idx))
    axis.scatter(
        x,
        y,
        s=62,
        color=ADAPTIVE_COLOR,
        marker=markers.get(row['regime'], 'o'),
        edgecolor='white',
        linewidth=0.7,
        zorder=3,
        label=row['regime'],
    )
    short_env = pretty_env(row['env_id']).replace('-run', '').replace('-swingup', '').replace('-hard', '')
    axis.annotate(short_env, (x, y), xytext=(3, 3), textcoords='offset points', fontsize=6)
  frontier_indices = pareto_frontier(points)
  if frontier_indices:
    frontier = sorted((env_rows[i]['mean_rhs_wall_hours'], env_rows[i]['mean_pct_delta']) for i in frontier_indices)
    axis.plot([x for x, _ in frontier], [y for _, y in frontier], color='#111111', linewidth=1.0, zorder=2)
  axis.axhline(0, color=FIXED_COLOR, linewidth=0.9)
  axis.set_xlabel('Adaptive RHS wall-clock hours (mean over seeds)')
  axis.set_ylabel('Final-score delta vs fixed horizon (%)')
  axis.set_title('Compute-performance Pareto view')
  axis.grid(True, alpha=0.45)
  handles, labels = axis.get_legend_handles_labels()
  unique = dict(zip(labels, handles))
  axis.legend(unique.values(), unique.keys(), title='Regime', loc='best')
  save_fig(fig, figures_dir, 'fig_pareto_frontier')
  plt.close(fig)


def plot_time_to_parity(pairs: list[dict[str, Any]], rows: list[dict[str, Any]], results_dir: Path, figures_dir: Path) -> None:
  import matplotlib.pyplot as plt

  row_lookup = {
      (row.get('env_id'), row.get('regime'), str(row.get('seed')), row.get('method')): row
      for row in rows
  }
  parity_rows = []
  for pair in pairs:
    rhs = row_lookup.get((pair['env_id'], pair['regime'], pair['seed'], 'adaptive_rhs'))
    if not rhs:
      continue
    curve = load_scalar_series(results_dir, rhs['run_id'], 'eval/return_mean')
    parity_step = float('nan')
    if not curve.empty:
      hit = curve[curve['value'] >= pair['baseline_score']]
      if not hit.empty:
        parity_step = float(hit.iloc[0]['step'])
    parity_rows.append({**pair, 'parity_step': parity_step})

  grouped: list[dict[str, Any]] = []
  for env_id, regime in sorted({(row['env_id'], row['regime']) for row in parity_rows}):
    subset = [row for row in parity_rows if row['env_id'] == env_id and row['regime'] == regime]
    vals = np.asarray([row['parity_step'] for row in subset], dtype=float)
    hit_rate = float(np.mean(np.isfinite(vals))) if vals.size else float('nan')
    grouped.append({
        'label': f'{env_id} | {regime}',
        'regime': regime,
        'mean_step': float(np.nanmean(vals)) / 1000.0 if np.any(np.isfinite(vals)) else float('nan'),
        'hit_rate': hit_rate,
    })
  grouped = sorted(grouped, key=lambda row: (not math.isfinite(row['mean_step']), row['mean_step']))
  fig, axis = plt.subplots(figsize=(7.0, 4.0))
  x = np.arange(len(grouped))
  vals = [row['mean_step'] if math.isfinite(row['mean_step']) else 520.0 for row in grouped]
  colors = [ADAPTIVE_COLOR if math.isfinite(row['mean_step']) else '#D0D0D0' for row in grouped]
  axis.bar(x, vals, color=colors, alpha=0.9)
  for xi, row in zip(x, grouped):
    if not math.isfinite(row['mean_step']):
      axis.text(xi, 500, 'no parity', rotation=90, ha='center', va='top', fontsize=6)
    else:
      axis.text(xi, row['mean_step'] + 8, f"{row['hit_rate']:.0%}", ha='center', va='bottom', fontsize=6)
  axis.set_xticks(x, [row['label'] for row in grouped], rotation=45, ha='right')
  axis.set_ylabel('Steps to matched fixed-horizon parity (k)')
  axis.set_title('Sample efficiency: reaching matched baseline final score')
  axis.grid(True, axis='y', alpha=0.45)
  save_fig(fig, figures_dir, 'fig_time_to_parity')
  plt.close(fig)


def plot_horizon_diagnostics(goal: dict[str, Any], rows: list[dict[str, Any]], results_dir: Path, figures_dir: Path) -> None:
  import matplotlib.pyplot as plt

  envs = [item['env_id'] for item in goal['matrix']['envs']]
  palette = plt.cm.Reds(np.linspace(0.45, 0.90, len(envs)))
  fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
  for axis, regime in zip(axes, ['clean', 'chaos']):
    axis.set_title(regime.title())
    axis.grid(True, axis='y', alpha=0.45)
    for color, env_id in zip(palette, envs):
      curves = []
      for row in rows:
        if row.get('env_id') == env_id and row.get('regime') == regime and row.get('method') == 'adaptive_rhs':
          q = load_horizon_queries(results_dir, row['run_id'])
          if not q.empty:
            curves.append(q[['step', 'selected_horizon']].rename(columns={'selected_horizon': 'value'}))
      agg = interpolate_curves(curves)
      if agg is None:
        continue
      grid, mean, sem = agg
      axis.plot(grid / 1000.0, mean, color=color, linewidth=1.35, label=env_id)
      axis.fill_between(grid / 1000.0, mean - sem, mean + sem, color=color, alpha=0.10, linewidth=0)
    axis.set_xlabel('Environment steps (k)')
  axes[0].set_ylabel('Selected RHS horizon')
  axes[1].legend(loc='center left', bbox_to_anchor=(1.01, 0.5), title='Environment')
  fig.suptitle('Adaptive RHS horizon selection', y=1.04, fontsize=10, fontweight='bold')
  save_fig(fig, figures_dir, 'fig_horizon_diagnostics')
  plt.close(fig)


def plot_loss_diagnostics(rows: list[dict[str, Any]], results_dir: Path, figures_dir: Path) -> None:
  import matplotlib.pyplot as plt

  metrics = [
      ('train/total_loss_mean', 'Normalized model loss', 'fig_loss'),
      ('train/grad_norm_mean', 'Gradient norm', 'fig_grad'),
  ]
  fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
  for axis, (tag, title, _) in zip(axes, metrics):
    axis.set_title(title)
    axis.grid(True, axis='y', alpha=0.45)
    for method in ('paper_horizon', 'adaptive_rhs'):
      curves = []
      for row in rows:
        if row.get('method') != method:
          continue
        curve = load_scalar_series(results_dir, row['run_id'], tag)
        if curve.empty:
          continue
        curve = curve.copy()
        if tag == 'train/total_loss_mean':
          first = curve['value'].replace([np.inf, -np.inf], np.nan).dropna()
          if first.empty or first.iloc[0] == 0:
            continue
          curve['value'] = curve['value'] / abs(first.iloc[0])
        curves.append(curve)
      agg = interpolate_curves(curves)
      if agg is None:
        continue
      grid, mean, sem = agg
      axis.plot(grid / 1000.0, mean, color=METHOD_COLORS[method], linewidth=1.7, label=METHOD_LABELS[method])
      axis.fill_between(grid / 1000.0, mean - sem, mean + sem, color=METHOD_COLORS[method], alpha=0.16, linewidth=0)
    axis.set_xlabel('Environment steps (k)')
  axes[0].set_ylabel('Value')
  axes[1].set_yscale('log')
  handles, labels = axes[0].get_legend_handles_labels()
  if handles:
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.04))
  fig.suptitle('Training stability diagnostics', y=1.12, fontsize=10, fontweight='bold')
  save_fig(fig, figures_dir, 'fig_loss_diagnostics')
  plt.close(fig)


def table_rows_for_tex(summary: list[dict[str, Any]], env_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  summary_tex = []
  for row in summary:
    summary_tex.append({
        'scope': row['scope'],
        'n': row['n'],
        'delta': f"{fmt(row['mean_pct_delta'])} $\\pm$ {fmt(row['std_pct_delta'])}",
        'win_rate': fmt(100.0 * row['rhs_win_rate']) + r'\%',
        'runtime_ratio': fmt(row['mean_runtime_ratio']),
    })
  env_tex = []
  for row in env_rows:
    env_tex.append({
        'env_id': row['env_id'],
        'regime': row['regime'],
        'n': row['n'],
        'delta': f"{fmt(row['mean_pct_delta'])} $\\pm$ {fmt(row['std_pct_delta'])}",
        'win_rate': fmt(100.0 * row['rhs_win_rate']) + r'\%',
        'rhs_hours': fmt(row['mean_rhs_wall_hours']),
    })
  return summary_tex, env_tex


def write_matched_pairs_latex(path: Path, pairs: list[dict[str, Any]], excluded: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  lines = [
      r'\begingroup',
      r'\scriptsize',
      r'\setlength{\tabcolsep}{3.2pt}',
      r'\begin{longtable}{lllrrrrr}',
      r'\toprule',
      r'Environment & Regime & Seed & Fixed & RHS & $\Delta$ (\%) & Fixed h & RHS h \\',
      r'\midrule',
      r'\endfirsthead',
      r'\toprule',
      r'Environment & Regime & Seed & Fixed & RHS & $\Delta$ (\%) & Fixed h & RHS h \\',
      r'\midrule',
      r'\endhead',
  ]
  for row in sorted(pairs, key=lambda item: (item['env_id'], item['regime'], int(item['seed']))):
    delta = float(row['pct_delta'])
    delta_text = f'{delta:+.2f}'
    values = [
        tex_escape(pretty_env(row['env_id'])),
        tex_escape(row['regime']),
        tex_escape(row['seed']),
        fmt(row['baseline_score']),
        fmt(row['rhs_score']),
        delta_text,
        fmt(row['baseline_wall_hours']),
        fmt(row['rhs_wall_hours']),
    ]
    lines.append(' & '.join(values) + r' \\')
  if excluded:
    lines.extend([
        r'\midrule',
        r'\multicolumn{8}{l}{Excluded matched cells:} \\',
    ])
    for row in sorted(excluded, key=lambda item: (item['env_id'], item['regime'], int(item['seed']))):
      values = [
          tex_escape(pretty_env(row['env_id'])),
          tex_escape(row['regime']),
          tex_escape(row['seed']),
          r'\multicolumn{5}{l}{' + tex_escape(row['reason']) + '}',
      ]
      lines.append(' & '.join(values) + r' \\')
  lines.extend([
      r'\bottomrule',
      r'\end{longtable}',
      r'\endgroup',
      '',
  ])
  path.write_text('\n'.join(lines))


def write_tables(out_dir: Path, pairs: list[dict[str, Any]], excluded: list[dict[str, Any]], summary: list[dict[str, Any]], env_rows: list[dict[str, Any]]) -> None:
  tables_dir = out_dir / 'tables'
  pair_fields = [
      'env_id', 'regime', 'seed', 'baseline_score', 'rhs_score', 'pct_delta',
      'rhs_won', 'baseline_wall_hours', 'rhs_wall_hours', 'runtime_ratio',
      'baseline_run_dir', 'rhs_run_dir',
  ]
  write_csv(tables_dir / 'matched_pairs.csv', pairs, pair_fields)
  write_csv(tables_dir / 'excluded_pairs.csv', excluded, ['env_id', 'regime', 'seed', 'reason'])
  write_csv(tables_dir / 'aggregate_summary.csv', summary, [
      'scope', 'n', 'mean_pct_delta', 'std_pct_delta', 'sem_pct_delta',
      'rhs_win_rate', 'mean_rhs_wall_hours', 'mean_baseline_wall_hours',
      'mean_runtime_ratio',
  ])
  write_csv(tables_dir / 'env_regime_summary.csv', env_rows, [
      'scope', 'env_id', 'regime', 'n', 'mean_pct_delta', 'std_pct_delta',
      'sem_pct_delta', 'rhs_win_rate', 'mean_rhs_wall_hours',
      'mean_baseline_wall_hours', 'mean_runtime_ratio',
  ])

  summary_tex, env_tex = table_rows_for_tex(summary, env_rows)
  write_latex_table(
      tables_dir / 'aggregate_summary.tex',
      summary_tex,
      [('scope', 'Scope'), ('n', 'n'), ('delta', r'RHS delta (\%)', True), ('win_rate', 'RHS wins', True), ('runtime_ratio', 'Runtime ratio')],
  )
  write_latex_table(
      tables_dir / 'env_regime_summary.tex',
      env_tex,
      [('env_id', 'Environment'), ('regime', 'Regime'), ('n', 'n'), ('delta', r'RHS delta (\%)', True), ('win_rate', 'RHS wins', True), ('rhs_hours', 'RHS h')],
  )
  write_matched_pairs_latex(tables_dir / 'matched_pairs_full.tex', pairs, excluded)


def write_report_tex(
    goal: dict[str, Any],
    out_dir: Path,
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    excluded_envs: set[str],
) -> Path:
  top = summary[0] if summary else {}
  goal_name = goal.get('name', 'Dense-RHS compact campaign')
  envs = ', '.join(item['env_id'] for item in goal['matrix']['envs'])
  seeds = ', '.join(str(seed) for seed in goal['matrix']['seeds'])
  excluded_env_text = ', '.join(sorted(pretty_env(env) for env in excluded_envs))
  exclusion_note = (
      f' The following environment is excluded from all results and plots: {tex_escape(excluded_env_text)}.'
      if len(excluded_envs) == 1
      else (
          f' The following environments are excluded from all results and plots: {tex_escape(excluded_env_text)}.'
          if excluded_envs else ''
      )
  )
  report = rf"""
\documentclass[10pt]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{caption}}
\usepackage{{microtype}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{longtable}}
\captionsetup{{font=small,labelfont=bf}}
\graphicspath{{{{figures/}}}}
\setlength{{\parskip}}{{0.45em}}
\setlength{{\parindent}}{{0pt}}

\title{{Preliminary Dense-RHS CoRL Compact Results}}
\author{{Autogenerated from local TD-MPC2-JAX run artifacts}}
\date{{}}

\begin{{document}}
\maketitle

\textbf{{Scope.}} This preliminary report summarizes the completed compact campaign ``{tex_escape(goal_name)}'' using local metric caches refreshed from the NCC run directories. It is intended as an internal CoRL-style evidence package, not as the final v3 matrix.{exclusion_note}

\textbf{{Headline.}} Across {len(pairs)} matched valid seed pairs, Adaptive RHS changes final score by \textbf{{{fmt(top.get('mean_pct_delta'))}\% $\pm$ {fmt(top.get('std_pct_delta'))}\%}} relative to the fixed TD-MPC2 paper horizon, with an RHS win rate of \textbf{{{fmt(100.0 * top.get('rhs_win_rate', float('nan')))}\%}}. The summary excludes {len(excluded)} missing, nonfinite, or unmatched pairs.

\section*{{Experimental Setup}}
We compare the TD-MPC2 fixed paper horizon against Adaptive RHS under matched environment, regime, and seed cells. Environments are {tex_escape(envs)}; regimes are clean and chaos; seeds are {tex_escape(seeds)}; training budget is {goal['constraints']['full_run_steps']} environment steps. All rows come from \texttt{{{tex_escape(goal['tracking']['ledger'])}}}; each row records the SLURM job id, git commit, remote run directory, score, runtime, and checkpoint status.

\section*{{Aggregate Result}}
\begin{{center}}
\input{{tables/aggregate_summary.tex}}
\end{{center}}

\section*{{Full Matched Results}}
\input{{tables/matched_pairs_full.tex}}

\section*{{Learning Curves}}
\begin{{figure}}[h]
  \centering
  \includegraphics[width=\linewidth]{{fig_learning_curves_clean.pdf}}
  \caption{{Clean-regime evaluation return. Thin traces are individual seeds; bold traces show the seed mean with standard error.}}
\end{{figure}}

\begin{{figure}}[h]
  \centering
  \includegraphics[width=\linewidth]{{fig_learning_curves_chaos.pdf}}
  \caption{{Chaos-regime evaluation return under matched randomized/noisy evaluation.}}
\end{{figure}}

\clearpage
\section*{{Matched Percent Delta}}
\begin{{figure}}[h]
  \centering
  \includegraphics[width=0.92\linewidth]{{fig_percent_delta_summary.pdf}}
  \caption{{Per-environment and per-regime Adaptive RHS percent delta against the matched fixed-horizon baseline. Error bars are across seeds.}}
\end{{figure}}

\begin{{center}}
\input{{tables/env_regime_summary.tex}}
\end{{center}}

\section*{{Efficiency and Pareto View}}
\begin{{figure}}[h]
  \centering
  \includegraphics[width=0.72\linewidth]{{fig_pareto_frontier.pdf}}
  \caption{{Compute-performance Pareto view. The x-axis is mean Adaptive RHS wall-clock hours and the y-axis is normalized final-score delta versus the matched fixed-horizon baseline; the black line highlights nondominated points.}}
\end{{figure}}

\begin{{figure}}[h]
  \centering
  \includegraphics[width=0.86\linewidth]{{fig_time_to_parity.pdf}}
  \caption{{Sample efficiency, measured as the first Adaptive RHS evaluation point that reaches the matched fixed-horizon final score. Labels above bars indicate the fraction of seeds reaching parity.}}
\end{{figure}}

\clearpage
\section*{{RHS Diagnostics}}
\begin{{figure}}[h]
  \centering
  \includegraphics[width=\linewidth]{{fig_horizon_diagnostics.pdf}}
  \caption{{Adaptive RHS selected horizon over training, averaged across seeds.}}
\end{{figure}}

\begin{{figure}}[h]
  \centering
  \includegraphics[width=0.88\linewidth]{{fig_loss_diagnostics.pdf}}
  \caption{{Training stability diagnostics pooled across runs. Total loss is normalized per run by its first finite value; gradient norms use a log axis.}}
\end{{figure}}

\section*{{Limitations}}
This is a preliminary report for the completed compact run after applying the report-level environment filter ({len(rows)} retained terminal rows). Fish-swim is intentionally omitted because it was later removed from the v3 final matrix after isolated MJX chaos-push diagnostics exposed nonfinite state/reward failures. The final v3 package should reuse the same report template with the updated environment set and six seeds.

\end{{document}}
"""
  tex_path = out_dir / 'preliminary_corl_report.tex'
  tex_path.write_text(report.strip() + '\n')
  return tex_path


def compile_pdf(tex_path: Path) -> None:
  if shutil.which('latexmk'):
    subprocess.run(
        ['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', tex_path.name],
        cwd=tex_path.parent,
        check=True,
    )
  elif shutil.which('pdflatex'):
    for _ in range(2):
      subprocess.run(
          ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex_path.name],
          cwd=tex_path.parent,
          check=True,
      )
  else:
    raise RuntimeError('Neither latexmk nor pdflatex is available')
  for suffix in ('.aux', '.fdb_latexmk', '.fls', '.log', '.out'):
    (tex_path.parent / f'{tex_path.stem}{suffix}').unlink(missing_ok=True)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--goal', default='goals/dense_rhs_corl_compact_v2.yaml')
  parser.add_argument('--out-name', default='preliminary_publication_report')
  parser.add_argument(
      '--exclude-env',
      action='append',
      default=['fish-swim'],
      help='Environment id to exclude from all report tables and plots. Defaults to fish-swim.',
  )
  parser.add_argument('--no-compile', action='store_true')
  args = parser.parse_args()

  goal_path = relpath(args.goal)
  goal = read_goal(goal_path)
  results_dir = relpath(goal['tracking']['results_dir'])
  out_dir = results_dir / args.out_name
  figures_dir = out_dir / 'figures'
  out_dir.mkdir(parents=True, exist_ok=True)

  excluded_envs = set(args.exclude_env or [])
  goal, rows = filter_envs(goal, build_rows(goal), excluded_envs)
  pairs, excluded = build_pairs(rows)
  summary, env_rows = aggregate_pairs(pairs)
  write_tables(out_dir, pairs, excluded, summary, env_rows)

  set_plot_style()
  plot_learning_curves(goal, rows, results_dir, figures_dir, 'clean')
  plot_learning_curves(goal, rows, results_dir, figures_dir, 'chaos')
  plot_delta_summary(env_rows, figures_dir)
  plot_pareto(env_rows, figures_dir)
  plot_time_to_parity(pairs, rows, results_dir, figures_dir)
  plot_horizon_diagnostics(goal, rows, results_dir, figures_dir)
  plot_loss_diagnostics(rows, results_dir, figures_dir)

  tex_path = write_report_tex(goal, out_dir, rows, pairs, excluded, summary, excluded_envs)
  if not args.no_compile:
    compile_pdf(tex_path)
  print(f'wrote preliminary report to {out_dir}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class Series:
  steps: np.ndarray
  values: np.ndarray


@dataclass(frozen=True)
class PlotArtifacts:
  png_path: Path
  pdf_path: Path


_PANEL_SPECS = (
    {
        'title': 'Episode Return',
        'tag': 'episode/return',
        'raw_style': 'scatter',
        'color': '#1f77b4',
    },
    {
        'title': 'Episode Length',
        'tag': 'episode/length',
        'raw_style': 'scatter',
        'color': '#17becf',
    },
    {
        'title': 'Eval Return',
        'tag': 'eval/return_mean',
        'band_tag': 'eval/return_std',
        'color': '#2ca02c',
    },
    {
        'title': 'Selected Horizon',
        'tag_candidates': (
            'query/selected_horizon',
            'dense_rhs/selected_horizon',
            'dense_rhs/best_h',
            'eval/selected_horizon',
        ),
        'color': '#9467bd',
    },
    {
        'title': 'Train Total Loss',
        'tag': 'train/total_loss_mean',
        'color': '#d62728',
    },
    {
        'title': 'Train Value Loss',
        'tag': 'train/value_loss_mean',
        'color': '#8c564b',
    },
    {
        'title': 'Train Policy Loss',
        'tag': 'train/policy_loss_mean',
        'color': '#e377c2',
    },
    {
        'title': 'Train Reward Loss',
        'tag': 'train/reward_loss_mean',
        'color': '#ff7f0e',
    },
    {
        'title': 'Train Consistency Loss',
        'tag': 'train/consistency_loss_mean',
        'color': '#7f7f7f',
    },
    {
        'title': 'Collect Chunk Time (s)',
        'tag': 'timing/collect_chunk_s',
        'color': '#bcbd22',
    },
    {
        'title': 'Train Chunk Time (s)',
        'tag': 'timing/train_chunk_s',
        'color': '#ff9896',
    },
    {
        'title': 'Eval Time (s)',
        'tag': 'timing/eval_s',
        'color': '#98df8a',
    },
    {
        'title': 'Query Total Time (s)',
        'tag': 'timing/query_total_s',
        'color': '#393b79',
    },
    {
        'title': 'Query Env Eval Time (s)',
        'tag': 'timing/query_env_eval_s',
        'color': '#637939',
    },
    {
        'title': 'Active Horizons',
        'tag_candidates': (
            'query/num_active_horizons',
            'dense_rhs/num_active_horizons',
        ),
        'color': '#aec7e8',
    },
    {
        'title': 'Best Horizon Prob',
        'tag_candidates': (
            'query/prob_best_h',
            'dense_rhs/prob_best_h',
        ),
        'color': '#c5b0d5',
    },
    {
        'title': 'Gaussian Mean (Best H)',
        'tag_candidates': (
            'query/gauss_mean_best_h',
            'dense_rhs/gauss_mean_best_h',
        ),
        'color': '#c49c94',
    },
    {
        'title': 'Normalized Entropy',
        'tag_candidates': (
            'query/norm_entropy',
            'dense_rhs/norm_entropy',
        ),
        'color': '#f7b6d2',
    },
    {
        'title': 'Robust Return (Best H)',
        'tag': 'dense_rhs/robust_return_best',
        'color': '#9edae5',
    },
    {
        'title': 'Dense-RHS Fitness (Best H)',
        'tag': 'dense_rhs/best_fitness',
        'color': '#dbdb8d',
    },
    {
        'title': 'Dense-RHS Deploy Score vs Fitness',
        'multi_tags': (
            'dense_rhs/deployment_score_best',
            'dense_rhs/best_fitness',
        ),
    },
    {
        'title': 'Dense-RHS Score Terms',
        'multi_tags': (
            'dense_rhs/return_term_best',
            'dense_rhs/roughness_term_best',
            'dense_rhs/return_std_term_best',
        ),
    },
    {
        'title': 'Candidate Robust Returns',
        'tag_prefix': 'dense_rhs/candidate_',
        'tag_suffix': '_return',
    },
    {
        'title': 'Query Timing Split',
        'multi_tags': (
            'timing/query_total_s',
            'timing/query_model_diag_s',
            'timing/query_env_eval_s',
            'timing/query_finalize_s',
        ),
    },
)


def _load_series_from_scalars(path: Path) -> dict[str, Series]:
  rows_by_tag: dict[str, list[tuple[int, float]]] = defaultdict(list)
  with path.open(newline='') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      try:
        step = int(float(row['step']))
        value = float(row['value'])
        tag = str(row['tag'])
      except (KeyError, TypeError, ValueError):
        continue
      rows_by_tag[tag].append((step, value))
  return {
      tag: _rows_to_series(rows)
      for tag, rows in rows_by_tag.items()
      if rows
  }


def _load_series_from_table(
    path: Path,
    *,
    skip_columns: Iterable[str],
    prefix: str,
) -> dict[str, Series]:
  rows_by_tag: dict[str, list[tuple[int, float]]] = defaultdict(list)
  skip = set(skip_columns)
  with path.open(newline='') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      try:
        step = int(float(row['step']))
      except (KeyError, TypeError, ValueError):
        continue
      for key, raw_value in row.items():
        if key in skip or raw_value in (None, ''):
          continue
        try:
          value = float(raw_value)
        except ValueError:
          continue
        rows_by_tag[f'{prefix}/{key}'].append((step, value))
  return {
      tag: _rows_to_series(rows)
      for tag, rows in rows_by_tag.items()
      if rows
  }


def _rows_to_series(rows: list[tuple[int, float]]) -> Series:
  rows.sort(key=lambda item: item[0])
  steps = np.asarray([step for step, _ in rows], dtype=np.int32)
  values = np.asarray([value for _, value in rows], dtype=np.float32)
  return Series(steps=steps, values=values)


def load_run_metrics(run_dir: str | Path) -> dict[str, Series]:
  run_path = Path(run_dir).expanduser().resolve()
  metrics_dir = run_path / 'metrics'
  scalars_path = metrics_dir / 'scalars.csv'
  if not scalars_path.exists():
    raise FileNotFoundError(f'Missing scalar metrics: {scalars_path}')

  series = _load_series_from_scalars(scalars_path)

  episodes_path = metrics_dir / 'episodes.csv'
  if episodes_path.exists():
    series.update(
        _load_series_from_table(
            episodes_path,
            skip_columns=('step', 'env_index', 'episode_index'),
            prefix='episode_table',
        )
    )
    if 'episode_table/episode_return' in series and 'episode/return' not in series:
      series['episode/return'] = series['episode_table/episode_return']
    if 'episode_table/episode_length' in series and 'episode/length' not in series:
      series['episode/length'] = series['episode_table/episode_length']

  queries_path = metrics_dir / 'horizon_queries.csv'
  if queries_path.exists():
    series.update(
        _load_series_from_table(
            queries_path,
            skip_columns=('step', 'phase_name'),
            prefix='query',
        )
    )

  return series


def generate_run_summary(
    run_dir: str | Path,
    *,
    output_stem: str = 'run_summary_grid',
) -> PlotArtifacts:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  run_path = Path(run_dir).expanduser().resolve()
  series_by_tag = load_run_metrics(run_path)

  artifacts_dir = run_path / 'artifacts'
  artifacts_dir.mkdir(parents=True, exist_ok=True)

  fig, axes = plt.subplots(6, 4, figsize=(22, 22), constrained_layout=True)
  axes_flat = axes.ravel()

  max_step = 0
  for series in series_by_tag.values():
    if series.steps.size:
      max_step = max(max_step, int(series.steps[-1]))

  for axis, spec in zip(axes_flat, _PANEL_SPECS):
    _plot_panel(axis, spec, series_by_tag)

  run_name = run_path.name
  fig.suptitle(
      f'{run_name}\nTraining / Eval Summary up to step {max_step:,}',
      fontsize=18,
      fontweight='bold',
  )

  png_path = artifacts_dir / f'{output_stem}.png'
  pdf_path = artifacts_dir / f'{output_stem}.pdf'
  fig.savefig(png_path, dpi=180)
  fig.savefig(pdf_path)
  plt.close(fig)
  return PlotArtifacts(png_path=png_path, pdf_path=pdf_path)


def _plot_panel(axis, spec: Mapping[str, object], series_by_tag: Mapping[str, Series]):
  if 'multi_tags' in spec:
    _plot_multi_panel(axis, spec, series_by_tag)
    return
  if 'tag_prefix' in spec:
    _plot_prefix_panel(axis, spec, series_by_tag)
    return

  tag = _resolve_tag(spec, series_by_tag)
  title = str(spec['title'])
  color = str(spec.get('color', '#1f77b4'))

  axis.set_title(title, fontsize=11, fontweight='bold')
  axis.grid(alpha=0.2, linestyle='--', linewidth=0.7)

  if tag is None:
    axis.text(
        0.5,
        0.5,
        'Unavailable',
        ha='center',
        va='center',
        fontsize=11,
        color='#666666',
        transform=axis.transAxes,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    return

  series = series_by_tag[tag]
  x = series.steps.astype(np.float32) / 1000.0
  y = series.values.astype(np.float32)

  if x.size == 0:
    axis.text(
        0.5,
        0.5,
        'No data',
        ha='center',
        va='center',
        fontsize=11,
        color='#666666',
        transform=axis.transAxes,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    return

  raw_style = str(spec.get('raw_style', 'line'))
  if raw_style == 'scatter':
    axis.scatter(x, y, s=10, alpha=0.25, color=color, edgecolors='none')
  else:
    axis.plot(x, y, color=color, alpha=0.25, linewidth=1.0)

  smooth = _smooth_values(y)
  axis.plot(x, smooth, color=color, linewidth=2.0)

  band_tag = spec.get('band_tag')
  if band_tag is not None:
    band_series = series_by_tag.get(str(band_tag))
    if band_series is not None and band_series.steps.size == series.steps.size:
      if np.array_equal(band_series.steps, series.steps):
        band = np.abs(band_series.values.astype(np.float32))
        axis.fill_between(
            x,
            smooth - band,
            smooth + band,
            color=color,
            alpha=0.15,
            linewidth=0.0,
        )

  axis.set_xlabel('Step (×10³)')
  axis.tick_params(axis='both', labelsize=9)


def _plot_multi_panel(axis,
                      spec: Mapping[str, object],
                      series_by_tag: Mapping[str, Series]):
  title = str(spec['title'])
  axis.set_title(title, fontsize=11, fontweight='bold')
  axis.grid(alpha=0.2, linestyle='--', linewidth=0.7)
  colors = (
      '#1f77b4', '#d62728', '#2ca02c', '#9467bd',
      '#ff7f0e', '#8c564b',
  )
  plotted = False
  for idx, tag in enumerate(spec.get('multi_tags', ())):
    tag = str(tag)
    series = series_by_tag.get(tag)
    if series is None or series.steps.size == 0:
      continue
    x = series.steps.astype(np.float32) / 1000.0
    y = series.values.astype(np.float32)
    label = tag.split('/')[-1].replace('_', ' ')
    axis.plot(
        x,
        y,
        color=colors[idx % len(colors)],
        linewidth=2.0,
        alpha=0.9,
        label=label,
    )
    plotted = True
  if not plotted:
    _mark_unavailable(axis)
    return
  axis.set_xlabel('Step (×10³)')
  axis.tick_params(axis='both', labelsize=9)
  axis.legend(fontsize=7, frameon=False)


def _plot_prefix_panel(axis,
                       spec: Mapping[str, object],
                       series_by_tag: Mapping[str, Series]):
  title = str(spec['title'])
  axis.set_title(title, fontsize=11, fontweight='bold')
  axis.grid(alpha=0.2, linestyle='--', linewidth=0.7)
  prefix = str(spec['tag_prefix'])
  suffix = str(spec.get('tag_suffix', ''))
  matching_tags = [
      tag for tag in sorted(series_by_tag)
      if tag.startswith(prefix) and tag.endswith(suffix)
  ]
  if not matching_tags:
    _mark_unavailable(axis)
    return
  cmap = None
  try:
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('viridis')
  except Exception:  # pragma: no cover - plotting fallback
    cmap = None
  for idx, tag in enumerate(matching_tags):
    series = series_by_tag[tag]
    if series.steps.size == 0:
      continue
    x = series.steps.astype(np.float32) / 1000.0
    y = series.values.astype(np.float32)
    color = cmap(idx / max(len(matching_tags) - 1, 1)) if cmap is not None else None
    label = tag[len(prefix):]
    if suffix and label.endswith(suffix):
      label = label[:-len(suffix)]
    axis.plot(x, y, linewidth=1.2, alpha=0.75, color=color, label=label)
  axis.set_xlabel('Step (×10³)')
  axis.tick_params(axis='both', labelsize=9)
  if len(matching_tags) <= 12:
    axis.legend(fontsize=7, frameon=False, ncol=2)


def _mark_unavailable(axis):
  axis.text(
      0.5,
      0.5,
      'Unavailable',
      ha='center',
      va='center',
      fontsize=11,
      color='#666666',
      transform=axis.transAxes,
  )
  axis.set_xticks([])
  axis.set_yticks([])


def _resolve_tag(spec: Mapping[str, object], series_by_tag: Mapping[str, Series]) -> str | None:
  direct_tag = spec.get('tag')
  if direct_tag is not None and str(direct_tag) in series_by_tag:
    return str(direct_tag)
  for candidate in spec.get('tag_candidates', ()):
    if str(candidate) in series_by_tag:
      return str(candidate)
  return None


def _smooth_values(values: np.ndarray) -> np.ndarray:
  values = np.asarray(values, dtype=np.float32)
  if values.size < 4:
    return values
  window = max(3, min(31, values.size // 12))
  if window % 2 == 0:
    window += 1
  kernel = np.ones(window, dtype=np.float32) / float(window)
  padded = np.pad(values, pad_width=window // 2, mode='edge')
  return np.convolve(padded, kernel, mode='valid').astype(np.float32)

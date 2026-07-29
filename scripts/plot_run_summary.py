#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SUMMARY_PATH = REPO_ROOT / 'tdmpc2_jax' / 'eval' / 'run_summary.py'
module_spec = importlib.util.spec_from_file_location(
    'tdmpc2_jax_eval_run_summary',
    RUN_SUMMARY_PATH,
)
if module_spec is None or module_spec.loader is None:
  raise ImportError(f'Could not load run summary module from {RUN_SUMMARY_PATH}')
run_summary = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = run_summary
module_spec.loader.exec_module(run_summary)
generate_run_summary = run_summary.generate_run_summary


def main():
  parser = argparse.ArgumentParser(
      description='Generate a global training/eval summary grid for one run directory.',
  )
  parser.add_argument(
      '--run-dir',
      required=True,
      help='Absolute or relative path to one tdmpc2-jax run directory.',
  )
  parser.add_argument(
      '--output-stem',
      default='run_summary_grid',
      help='Basename for the generated PNG and PDF artifacts.',
  )
  args = parser.parse_args()

  artifacts = generate_run_summary(
      run_dir=args.run_dir,
      output_stem=args.output_stem,
  )
  print(f'Wrote summary plot: {artifacts.png_path}')
  print(f'Wrote summary plot: {artifacts.pdf_path}')


if __name__ == '__main__':
  main()

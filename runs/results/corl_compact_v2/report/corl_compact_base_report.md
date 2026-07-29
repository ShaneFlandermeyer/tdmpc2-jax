# Compact CoRL Adaptive-RHS Campaign Report

## Executive Summary

- Completed main profiles: 72/72.
- Mean final fixed paper-horizon score: 680.591.
- Mean final adaptive RHS score: 702.904.
- Main claim: adaptive RHS is evaluated against the standard TD-MPC2 fixed horizon without a horizon sweep.

## Method

The baseline is TD-MPC2-JAX with Dense-RHS disabled and the paper-matched fixed horizon recorded in the ledger. Adaptive RHS uses the guarded configuration recorded in the goal file; no architecture search or per-environment horizon tuning is part of this campaign.

## Experiment Setup

Environments: quadruped-run, cheetah-run, hopper-hop, finger-turn_hard, fish-swim, cartpole-swingup.

Regimes: clean and chaos. Each profile is evaluated in its own regime: clean profiles use clean evaluation, and chaos profiles evaluate with domain randomization, observation noise, and one-step base action delay enabled.

Seeds: 1, 7, 15. Training budget: 500000 environment steps.

## Main Results

See `figures/fig1_clean_learning_curves.*`, `figures/fig2_chaos_learning_curves.*`, `tables/main_final_scores.*`, `tables/main_auc_scores.*`, and the NaN-excluded percent-delta summaries in `tables/rhs_percent_delta_summary.*`.

Clean completed rows: 36. Chaos completed rows: 36.

## Robustness

The robustness comparison should use per-method chaos-minus-clean deltas after all matched task/seed cells complete.

## Compute

See `figures/fig3_time_to_parity.*`, `figures/fig4_compute_performance.*`, `tables/parity_times.*`, and `tables/compute_runtime.*`.

## Reproducibility

All rows are sourced from `experiments/corl_compact_v2_ledger.csv`. Each row records SLURM job id, git commit, remote commit, run directory, seed, method, regime, and checkpoint status.

## Limitations

Humanoid is not included because the current MJX backend does not have a humanoid port or gate. Any blocked or failed rows must be interpreted as campaign limitations unless rerun with the same frozen method/config succeeds.

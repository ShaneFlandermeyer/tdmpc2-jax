# CoRL Compact Dense-RHS Decisions

- 2026-05-10: Created compact CoRL campaign steward for 6 MJX environments, clean/chaos regimes, paper-horizon no-RHS baseline, and frozen adaptive Dense-RHS. The main matrix contains 72 full runs. Humanoid is excluded until a separate MJX gate exists.
- 2026-05-11: Aborted v1 compact campaign before continuing the matrix. Diagnostics showed adaptive RHS could drift to longer horizons with late reward collapse on cheetah, and chaos profiles were being reported with clean evaluation. Set v1 `max_active_gpus=0`; preserve completed v1 artifacts for diagnostics only and restart in a v2 namespace with regime-matched evaluation and stricter RHS transition safeguards.

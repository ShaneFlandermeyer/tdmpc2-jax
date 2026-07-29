# MJX Port Gates

Each non-quadruped environment must pass this gate before the steward can launch clean or chaos training cells.

Gate order:

1. Inspect the existing dm_control task semantics, observation flattening, action bounds, reward, termination, and episode length.
2. Implement or wire the MJX environment backend.
3. Validate reset, observation, action, reward, done, and batch shapes on GPU.
4. Compare a fixed-action rollout against dm_control closely enough for training use.
5. Verify chaos semantics: domain randomization, observation noise, and action delay.
6. Run a short no-RHS MJX learning smoke and check for non-degenerate reward/loss/timing signals.
7. Write `gate_report.json`, `gate_summary.md`, and any parity plots under `runs/results/mjx_gates/<env_id>/`, then mark the goal gate `passed`.

The steward treats these gates as blocking tasks. CPU dm_control may be used only as a parity reference; final campaign runs must use MJX/full-GPU.

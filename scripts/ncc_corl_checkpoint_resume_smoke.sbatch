#!/usr/bin/env bash
# CoRL compact campaign setup gate: verify checkpoint-buffer resume.

#SBATCH --job-name=tdmpc2-jax-corl-smoke
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
RUN_ID="${RUN_ID:-setup_checkpoint_resume_smoke}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
ENV_ID="${ENV_ID:-cartpole-swingup}"
SEED="${SEED:-15}"

source "$VENV_PATH/bin/activate"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"

rm -rf "${RUN_DIR}"
mkdir -p "$(dirname "${RUN_DIR}")"

COMMON_ARGS=(
  hydra.run.dir="${RUN_DIR}"
  seed="${SEED}"
  save_interval_steps=2048
  log_interval_steps=1024
  update_chunk_size=16
  collect_chunk_steps=20
  seed_steps_override=512
  checkpoint_buffer=true
  env.backend=mjx_dmc
  env.env_id="${ENV_ID}"
  env.num_envs=2
  env.utd_ratio=1.0
  env.asynchronous=false
  env.mjx_dmc.task="${ENV_ID}"
  env.mjx_dmc.enable_domain_randomization=false
  env.mjx_dmc.enable_observation_noise=false
  env.mjx_dmc.base_action_delay=0
  env.mjx_dmc.reset_pool_size=8
  dense_rhs.enabled=false
  evaluation.enabled=false
  tdmpc2.horizon=3
  wandb.enabled=false
)

python -m tdmpc2_jax.train "${COMMON_ARGS[@]}" max_steps=4096
FIRST_STEP="$(find "${RUN_DIR}/checkpoint" -maxdepth 1 -type d -regex '.*/[0-9]+' -print 2>/dev/null | sed 's#.*/##' | sort -n | tail -1 || true)"
echo "FIRST_CHECKPOINT_STEP=${FIRST_STEP:-missing}"

python -m tdmpc2_jax.train "${COMMON_ARGS[@]}" max_steps=8192
SECOND_STEP="$(find "${RUN_DIR}/checkpoint" -maxdepth 1 -type d -regex '.*/[0-9]+' -print 2>/dev/null | sed 's#.*/##' | sort -n | tail -1 || true)"
echo "SECOND_CHECKPOINT_STEP=${SECOND_STEP:-missing}"

"$VENV_PATH/bin/python" - <<PY
import json
from pathlib import Path

first_raw = "${FIRST_STEP}"
second_raw = "${SECOND_STEP}"
first = int(first_raw) if first_raw else -1
second = int(second_raw) if second_raw else -1
passed = first >= 4096 and second >= 8192 and second > first
Path("${RUN_DIR}").mkdir(parents=True, exist_ok=True)
(Path("${RUN_DIR}") / "resume_verified.json").write_text(json.dumps({
    "run_id": "${RUN_ID}",
    "env_id": "${ENV_ID}",
    "first_checkpoint_step": first,
    "second_checkpoint_step": second,
    "passed": passed,
}, indent=2) + "\n")
raise SystemExit(0 if passed else 2)
PY

echo "RUN_DIR=${RUN_DIR}"

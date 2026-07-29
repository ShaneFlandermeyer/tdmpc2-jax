#!/usr/bin/env bash
# Native TD-MPC2-JAX quadruped baseline without Dense-RHS.
# Keeps the default JAX quadruped parametrization and only adds
# periodic evaluation plus W&B bookkeeping for comparison runs.
#
# Default launch:
#   sbatch scripts/ncc_tdmpc2_baseline_full.sbatch
#
# No-chaos launch:
#   CONDITION_NAME=no-chaos \
#   ENABLE_DOMAIN_RANDOMIZATION=false \
#   ENABLE_OBSERVATION_NOISE=false \
#   sbatch scripts/ncc_tdmpc2_baseline_full.sbatch

#SBATCH --job-name=tdmpc2-jax-quadruped-baseline
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
CONDITION_NAME="${CONDITION_NAME:-chaos}"
ENABLE_DOMAIN_RANDOMIZATION="${ENABLE_DOMAIN_RANDOMIZATION:-true}"
ENABLE_OBSERVATION_NOISE="${ENABLE_OBSERVATION_NOISE:-true}"
BASE_ACTION_DELAY="${BASE_ACTION_DELAY:-1}"
SEED="${SEED:-15}"
ENV_ID="${ENV_ID:-quadruped-run}"
RUN_LABEL="${RUN_LABEL:-s0_job${SLURM_JOB_ID}}"
RUN_STEM="${RUN_STEM:-$ENV_ID}"
WANDB_GROUP="${WANDB_GROUP:-tdmpc2-jax-quadruped-baseline}"
WANDB_TAGS="${WANDB_TAGS:-[ncc,h200,mjx,baseline,no-rhs,quadruped-run]}"

MAX_STEPS="${MAX_STEPS:-126000}"
SAVE_INTERVAL_STEPS="${SAVE_INTERVAL_STEPS:-10000}"
LOG_INTERVAL_STEPS="${LOG_INTERVAL_STEPS:-2000}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-4}"
SEED_STEPS_OVERRIDE="${SEED_STEPS_OVERRIDE:-}"
EVAL_INTERVAL_STEPS="${EVAL_INTERVAL_STEPS:-20000}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-10}"
EVAL_CLEAN="${EVAL_CLEAN:-true}"
EVAL_PREWARM="${EVAL_PREWARM:-false}"
PLOT_SUMMARY="${PLOT_SUMMARY:-true}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"

source "$VENV_PATH/bin/activate"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"

RUN_DIR="${PWD}/outputs/${RUN_STEM}_baseline_${CONDITION_NAME}_${RUN_LABEL}"
RUN_NAME="${RUN_STEM}_baseline_${CONDITION_NAME}_${RUN_LABEL}"

export WANDB_DIR="${RUN_DIR}/wandb"
export WANDB_CONSOLE="off"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RUN_NAME=${RUN_NAME}"
echo "ENV_ID=${ENV_ID}"
echo "CONDITION_NAME=${CONDITION_NAME}"
echo "TRAIN_NUM_ENVS=${TRAIN_NUM_ENVS}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "EVAL_INTERVAL_STEPS=${EVAL_INTERVAL_STEPS}"
echo "EVAL_NUM_EPISODES=${EVAL_NUM_EPISODES}"
echo "EVAL_CLEAN=${EVAL_CLEAN}"
echo "EVAL_PREWARM=${EVAL_PREWARM}"

PY_ARGS=(
  hydra.run.dir="${RUN_DIR}"
  seed="${SEED}"
  max_steps="${MAX_STEPS}"
  save_interval_steps="${SAVE_INTERVAL_STEPS}"
  log_interval_steps="${LOG_INTERVAL_STEPS}"
  env.env_id="${ENV_ID}"
  env.num_envs="${TRAIN_NUM_ENVS}"
  env.asynchronous=false
  env.mjx_dmc.task="${ENV_ID}"
  env.mjx_dmc.enable_domain_randomization="${ENABLE_DOMAIN_RANDOMIZATION}"
  env.mjx_dmc.enable_observation_noise="${ENABLE_OBSERVATION_NOISE}"
  env.mjx_dmc.base_action_delay="${BASE_ACTION_DELAY}"
  dense_rhs.enabled=false
  evaluation.enabled=true
  evaluation.interval_steps="${EVAL_INTERVAL_STEPS}"
  evaluation.num_episodes="${EVAL_NUM_EPISODES}"
  evaluation.clean="${EVAL_CLEAN}"
  evaluation.prewarm="${EVAL_PREWARM}"
  wandb.enabled="${WANDB_ENABLED}"
  wandb.project=ncc-runs
  wandb.entity=perpetual-policy
  wandb.group="${WANDB_GROUP}"
  wandb.name="${RUN_NAME}"
  wandb.tags="${WANDB_TAGS}"
)

if [[ -n "${SEED_STEPS_OVERRIDE}" ]]; then
  PY_ARGS+=(seed_steps_override="${SEED_STEPS_OVERRIDE}")
fi

set +e
python -m tdmpc2_jax.train "${PY_ARGS[@]}"
TRAIN_STATUS=$?
set -e

if [[ "${PLOT_SUMMARY}" == "true" ]]; then
  python scripts/plot_run_summary.py --run-dir "${RUN_DIR}" || true
fi

echo "RUN_DIR=${RUN_DIR}"
exit "${TRAIN_STATUS}"

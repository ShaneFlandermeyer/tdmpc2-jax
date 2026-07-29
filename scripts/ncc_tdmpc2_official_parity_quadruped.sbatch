#!/usr/bin/env bash
# Paper-parity TD-MPC2-JAX quadruped-run baseline.
# Matches the official single-task TD-MPC2 quadruped conditions as closely as
# the JAX/MJX implementation allows:
# - no Dense-RHS
# - no domain randomization / observation noise
# - single environment
# - seed_steps = 2500
# - temperature = 0.5
# - eval every 50k for 10 episodes
# - 4M env-step target to compare against the released quadruped-run curve
#
# Default launch:
#   sbatch scripts/ncc_tdmpc2_official_parity_quadruped.sbatch
#
# Alternate seed:
#   SEED=2 sbatch scripts/ncc_tdmpc2_official_parity_quadruped.sbatch

#SBATCH --job-name=tdmpc2-jax-quadruped-paper
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
SEED="${SEED:-1}"
RUN_STEM="${RUN_STEM:-quadruped-run}"
RUN_LABEL="${RUN_LABEL:-s${SEED}_job${SLURM_JOB_ID}}"
WANDB_GROUP="${WANDB_GROUP:-tdmpc2-jax-quadruped-paper-parity}"
WANDB_TAGS="${WANDB_TAGS:-[ncc,h200,mjx,baseline,paper-parity,no-rhs,no-chaos,quadruped-run]}"

MAX_STEPS="${MAX_STEPS:-4000000}"
SAVE_INTERVAL_STEPS="${SAVE_INTERVAL_STEPS:-50000}"
CHECKPOINT_BUFFER="${CHECKPOINT_BUFFER:-true}"
LOG_INTERVAL_STEPS="${LOG_INTERVAL_STEPS:-10000}"
UPDATE_CHUNK_SIZE="${UPDATE_CHUNK_SIZE:-32}"
SEED_STEPS_OVERRIDE="${SEED_STEPS_OVERRIDE:-2500}"
SEED_PRETRAIN_LOG_INTERVAL_UPDATES="${SEED_PRETRAIN_LOG_INTERVAL_UPDATES:-512}"
COLLECT_CHUNK_STEPS="${COLLECT_CHUNK_STEPS:-100}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-1}"
UTD_RATIO="${UTD_RATIO:-1.0}"
RESET_POOL_SIZE="${RESET_POOL_SIZE:-256}"
EVAL_INTERVAL_STEPS="${EVAL_INTERVAL_STEPS:-50000}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-10}"
EVAL_CLEAN="${EVAL_CLEAN:-true}"
EVAL_PREWARM="${EVAL_PREWARM:-false}"
PLOT_SUMMARY="${PLOT_SUMMARY:-true}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"

source "$VENV_PATH/bin/activate"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"

RUN_DIR="${PWD}/outputs/${RUN_STEM}_paper_parity_s${SEED}_job${SLURM_JOB_ID}"
RUN_NAME="${RUN_STEM}_paper_parity_s${SEED}_job${SLURM_JOB_ID}"

export WANDB_DIR="${RUN_DIR}/wandb"
export WANDB_CONSOLE="off"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RUN_NAME=${RUN_NAME}"
echo "SEED=${SEED}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "UPDATE_CHUNK_SIZE=${UPDATE_CHUNK_SIZE}"
echo "CHECKPOINT_BUFFER=${CHECKPOINT_BUFFER}"
echo "SEED_STEPS_OVERRIDE=${SEED_STEPS_OVERRIDE}"
echo "SEED_PRETRAIN_LOG_INTERVAL_UPDATES=${SEED_PRETRAIN_LOG_INTERVAL_UPDATES}"
echo "COLLECT_CHUNK_STEPS=${COLLECT_CHUNK_STEPS}"
echo "TRAIN_NUM_ENVS=${TRAIN_NUM_ENVS}"
echo "UTD_RATIO=${UTD_RATIO}"
echo "RESET_POOL_SIZE=${RESET_POOL_SIZE}"
echo "EVAL_INTERVAL_STEPS=${EVAL_INTERVAL_STEPS}"
echo "EVAL_NUM_EPISODES=${EVAL_NUM_EPISODES}"

set +e
python -m tdmpc2_jax.train \
  hydra.run.dir="${RUN_DIR}" \
  seed="${SEED}" \
  max_steps="${MAX_STEPS}" \
  save_interval_steps="${SAVE_INTERVAL_STEPS}" \
  checkpoint_buffer="${CHECKPOINT_BUFFER}" \
  log_interval_steps="${LOG_INTERVAL_STEPS}" \
  update_chunk_size="${UPDATE_CHUNK_SIZE}" \
  seed_steps_override="${SEED_STEPS_OVERRIDE}" \
  seed_pretrain_log_interval_updates="${SEED_PRETRAIN_LOG_INTERVAL_UPDATES}" \
  collect_chunk_steps="${COLLECT_CHUNK_STEPS}" \
  env.num_envs="${TRAIN_NUM_ENVS}" \
  env.utd_ratio="${UTD_RATIO}" \
  env.asynchronous=false \
  env.mjx_dmc.enable_domain_randomization=false \
  env.mjx_dmc.enable_observation_noise=false \
  env.mjx_dmc.base_action_delay=0 \
  env.mjx_dmc.reset_pool_size="${RESET_POOL_SIZE}" \
  dense_rhs.enabled=false \
  evaluation.enabled=true \
  evaluation.interval_steps="${EVAL_INTERVAL_STEPS}" \
  evaluation.num_episodes="${EVAL_NUM_EPISODES}" \
  evaluation.clean="${EVAL_CLEAN}" \
  evaluation.prewarm="${EVAL_PREWARM}" \
  tdmpc2.temperature=0.5 \
  wandb.enabled="${WANDB_ENABLED}" \
  wandb.project=ncc-runs \
  wandb.entity=perpetual-policy \
  wandb.group="${WANDB_GROUP}" \
  wandb.name="${RUN_NAME}" \
  wandb.tags="${WANDB_TAGS}"
TRAIN_STATUS=$?
set -e

if [[ "${PLOT_SUMMARY}" == "true" ]]; then
  python scripts/plot_run_summary.py --run-dir "${RUN_DIR}" || true
fi

echo "RUN_DIR=${RUN_DIR}"
exit "${TRAIN_STATUS}"

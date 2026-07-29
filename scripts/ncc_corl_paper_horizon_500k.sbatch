#!/usr/bin/env bash
# CoRL compact campaign: paper-horizon no-RHS TD-MPC2-JAX baseline.

#SBATCH --job-name=tdmpc2-jax-corl-fixed
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
ENV_ID="${ENV_ID:?ENV_ID is required}"
REGIME="${REGIME:?REGIME is required}"
SEED="${SEED:?SEED is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
PAPER_HORIZON="${PAPER_HORIZON:-3}"

MAX_STEPS="${MAX_STEPS:-500000}"
SAVE_INTERVAL_STEPS="${SAVE_INTERVAL_STEPS:-50000}"
LOG_INTERVAL_STEPS="${LOG_INTERVAL_STEPS:-10000}"
UPDATE_CHUNK_SIZE="${UPDATE_CHUNK_SIZE:-128}"
COLLECT_CHUNK_STEPS="${COLLECT_CHUNK_STEPS:-100}"
SEED_STEPS_OVERRIDE="${SEED_STEPS_OVERRIDE:-4000}"
SEED_PRETRAIN_LOG_INTERVAL_UPDATES="${SEED_PRETRAIN_LOG_INTERVAL_UPDATES:-512}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-8}"
UTD_RATIO="${UTD_RATIO:-1.0}"
RESET_POOL_SIZE="${RESET_POOL_SIZE:-64}"
CHECKPOINT_BUFFER="${CHECKPOINT_BUFFER:-true}"

EVAL_INTERVAL_STEPS="${EVAL_INTERVAL_STEPS:-50000}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-20}"
EVAL_CLEAN="${EVAL_CLEAN:-true}"
EVAL_PREWARM="${EVAL_PREWARM:-false}"

ENABLE_DOMAIN_RANDOMIZATION="${ENABLE_DOMAIN_RANDOMIZATION:-false}"
ENABLE_OBSERVATION_NOISE="${ENABLE_OBSERVATION_NOISE:-false}"
BASE_ACTION_DELAY="${BASE_ACTION_DELAY:-0}"

TRAIN_POPULATION_SIZE="${TRAIN_POPULATION_SIZE:-512}"
TRAIN_POLICY_PRIOR_SAMPLES="${TRAIN_POLICY_PRIOR_SAMPLES:-24}"
TRAIN_NUM_ELITES="${TRAIN_NUM_ELITES:-64}"
TRAIN_MPPI_ITERATIONS="${TRAIN_MPPI_ITERATIONS:-6}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-0.5}"

source "$VENV_PATH/bin/activate"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_CONSOLE=off
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p "$(dirname "${RUN_DIR}")"
RUN_NAME="${RUN_ID}"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RUN_ID=${RUN_ID}"
echo "RUN_DIR=${RUN_DIR}"
echo "ENV_ID=${ENV_ID}"
echo "REGIME=${REGIME}"
echo "SEED=${SEED}"
echo "PAPER_HORIZON=${PAPER_HORIZON}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "CHECKPOINT_BUFFER=${CHECKPOINT_BUFFER}"
echo "EVAL_CLEAN=${EVAL_CLEAN}"

set +e
python -m tdmpc2_jax.train \
  hydra.run.dir="${RUN_DIR}" \
  seed="${SEED}" \
  max_steps="${MAX_STEPS}" \
  save_interval_steps="${SAVE_INTERVAL_STEPS}" \
  log_interval_steps="${LOG_INTERVAL_STEPS}" \
  update_chunk_size="${UPDATE_CHUNK_SIZE}" \
  seed_steps_override="${SEED_STEPS_OVERRIDE}" \
  seed_pretrain_log_interval_updates="${SEED_PRETRAIN_LOG_INTERVAL_UPDATES}" \
  collect_chunk_steps="${COLLECT_CHUNK_STEPS}" \
  checkpoint_buffer="${CHECKPOINT_BUFFER}" \
  env.backend=mjx_dmc \
  env.env_id="${ENV_ID}" \
  env.num_envs="${TRAIN_NUM_ENVS}" \
  env.utd_ratio="${UTD_RATIO}" \
  env.asynchronous=false \
  env.mjx_dmc.task="${ENV_ID}" \
  env.mjx_dmc.enable_domain_randomization="${ENABLE_DOMAIN_RANDOMIZATION}" \
  env.mjx_dmc.enable_observation_noise="${ENABLE_OBSERVATION_NOISE}" \
  env.mjx_dmc.base_action_delay="${BASE_ACTION_DELAY}" \
  env.mjx_dmc.reset_pool_size="${RESET_POOL_SIZE}" \
  dense_rhs.enabled=false \
  evaluation.enabled=true \
  evaluation.interval_steps="${EVAL_INTERVAL_STEPS}" \
  evaluation.num_episodes="${EVAL_NUM_EPISODES}" \
  evaluation.clean="${EVAL_CLEAN}" \
  evaluation.prewarm="${EVAL_PREWARM}" \
  tdmpc2.horizon="${PAPER_HORIZON}" \
  tdmpc2.population_size="${TRAIN_POPULATION_SIZE}" \
  tdmpc2.policy_prior_samples="${TRAIN_POLICY_PRIOR_SAMPLES}" \
  tdmpc2.num_elites="${TRAIN_NUM_ELITES}" \
  tdmpc2.mppi_iterations="${TRAIN_MPPI_ITERATIONS}" \
  tdmpc2.temperature="${TRAIN_TEMPERATURE}" \
  wandb.enabled=false \
  wandb.group=tdmpc2-jax-corl-compact \
  wandb.name="${RUN_NAME}" \
  wandb.tags="[corl,compact,paper-horizon,${REGIME},${ENV_ID}]"
TRAIN_STATUS=$?
set -e

python scripts/plot_run_summary.py --run-dir "${RUN_DIR}" || true
echo "RUN_DIR=${RUN_DIR}"
exit "${TRAIN_STATUS}"

#!/usr/bin/env bash
# Short no-RHS TD-MPC2-JAX quadruped smoke test.
# Purpose: validate GPU-first collection/update timing before launching a full run.

#SBATCH --job-name=tdmpc2-jax-no-rhs-smoke
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
SEED="${SEED:-7}"
RUN_STEM="${RUN_STEM:-quadruped-run}"
RUN_LABEL="${RUN_LABEL:-s${SEED}_job${SLURM_JOB_ID}}"

MAX_STEPS="${MAX_STEPS:-12000}"
SAVE_INTERVAL_STEPS="${SAVE_INTERVAL_STEPS:-100000000}"
LOG_INTERVAL_STEPS="${LOG_INTERVAL_STEPS:-5000}"
UPDATE_CHUNK_SIZE="${UPDATE_CHUNK_SIZE:-128}"
SEED_STEPS_OVERRIDE="${SEED_STEPS_OVERRIDE:-1000}"
SEED_PRETRAIN_LOG_INTERVAL_UPDATES="${SEED_PRETRAIN_LOG_INTERVAL_UPDATES:-128}"
COLLECT_CHUNK_STEPS="${COLLECT_CHUNK_STEPS:-100}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-1}"
UTD_RATIO="${UTD_RATIO:-1.0}"
RESET_POOL_SIZE="${RESET_POOL_SIZE:-64}"
CHECKPOINT_BUFFER="${CHECKPOINT_BUFFER:-false}"

EVAL_ENABLED="${EVAL_ENABLED:-false}"
EVAL_INTERVAL_STEPS="${EVAL_INTERVAL_STEPS:-10000}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-2}"
EVAL_PREWARM="${EVAL_PREWARM:-false}"

TRAIN_POPULATION_SIZE="${TRAIN_POPULATION_SIZE:-512}"
TRAIN_POLICY_PRIOR_SAMPLES="${TRAIN_POLICY_PRIOR_SAMPLES:-24}"
TRAIN_NUM_ELITES="${TRAIN_NUM_ELITES:-64}"
TRAIN_MPPI_ITERATIONS="${TRAIN_MPPI_ITERATIONS:-6}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-0.5}"

source "$VENV_PATH/bin/activate"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"

RUN_DIR="${PWD}/outputs/${RUN_STEM}_no_rhs_smoke_${RUN_LABEL}"
RUN_NAME="${RUN_STEM}_no_rhs_smoke_${RUN_LABEL}"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RUN_NAME=${RUN_NAME}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "SEED_STEPS_OVERRIDE=${SEED_STEPS_OVERRIDE}"
echo "SEED_PRETRAIN_LOG_INTERVAL_UPDATES=${SEED_PRETRAIN_LOG_INTERVAL_UPDATES}"
echo "COLLECT_CHUNK_STEPS=${COLLECT_CHUNK_STEPS}"
echo "UPDATE_CHUNK_SIZE=${UPDATE_CHUNK_SIZE}"
echo "TRAIN_NUM_ENVS=${TRAIN_NUM_ENVS}"
echo "UTD_RATIO=${UTD_RATIO}"
echo "CHECKPOINT_BUFFER=${CHECKPOINT_BUFFER}"

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
  env.num_envs="${TRAIN_NUM_ENVS}" \
  env.utd_ratio="${UTD_RATIO}" \
  env.asynchronous=false \
  env.mjx_dmc.enable_domain_randomization=false \
  env.mjx_dmc.enable_observation_noise=false \
  env.mjx_dmc.base_action_delay=0 \
  env.mjx_dmc.reset_pool_size="${RESET_POOL_SIZE}" \
  dense_rhs.enabled=false \
  evaluation.enabled="${EVAL_ENABLED}" \
  evaluation.interval_steps="${EVAL_INTERVAL_STEPS}" \
  evaluation.num_episodes="${EVAL_NUM_EPISODES}" \
  evaluation.clean=true \
  evaluation.prewarm="${EVAL_PREWARM}" \
  tdmpc2.population_size="${TRAIN_POPULATION_SIZE}" \
  tdmpc2.policy_prior_samples="${TRAIN_POLICY_PRIOR_SAMPLES}" \
  tdmpc2.num_elites="${TRAIN_NUM_ELITES}" \
  tdmpc2.mppi_iterations="${TRAIN_MPPI_ITERATIONS}" \
  tdmpc2.temperature="${TRAIN_TEMPERATURE}" \
  wandb.enabled=false

python scripts/plot_run_summary.py --run-dir "${RUN_DIR}" || true
echo "RUN_DIR=${RUN_DIR}"

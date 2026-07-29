#!/usr/bin/env bash
# 30k Dense-RHS smoke on the efficient vec8 MJX pipeline.
# Usage:
#   CONDITION_NAME=clean sbatch scripts/ncc_dense_rhs_fast_smoke.sbatch
#   CONDITION_NAME=chaos sbatch scripts/ncc_dense_rhs_fast_smoke.sbatch

#SBATCH --job-name=tdmpc2-jax-dense-rhs-smoke
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
SEED="${SEED:-15}"
CONDITION_NAME="${CONDITION_NAME:-clean}"
RUN_STEM="${RUN_STEM:-quadruped-run}"
RUN_LABEL="${RUN_LABEL:-dense_rhs_fast_smoke_${CONDITION_NAME}_s${SEED}_job${SLURM_JOB_ID}}"

if [[ "${CONDITION_NAME}" == "chaos" ]]; then
  ENABLE_DOMAIN_RANDOMIZATION="${ENABLE_DOMAIN_RANDOMIZATION:-true}"
  ENABLE_OBSERVATION_NOISE="${ENABLE_OBSERVATION_NOISE:-true}"
  BASE_ACTION_DELAY="${BASE_ACTION_DELAY:-1}"
else
  ENABLE_DOMAIN_RANDOMIZATION="${ENABLE_DOMAIN_RANDOMIZATION:-false}"
  ENABLE_OBSERVATION_NOISE="${ENABLE_OBSERVATION_NOISE:-false}"
  BASE_ACTION_DELAY="${BASE_ACTION_DELAY:-0}"
fi

MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_INTERVAL_STEPS="${SAVE_INTERVAL_STEPS:-100000000}"
LOG_INTERVAL_STEPS="${LOG_INTERVAL_STEPS:-5000}"
UPDATE_CHUNK_SIZE="${UPDATE_CHUNK_SIZE:-128}"
SEED_STEPS_OVERRIDE="${SEED_STEPS_OVERRIDE:-4000}"
SEED_PRETRAIN_LOG_INTERVAL_UPDATES="${SEED_PRETRAIN_LOG_INTERVAL_UPDATES:-512}"
COLLECT_CHUNK_STEPS="${COLLECT_CHUNK_STEPS:-100}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-8}"
UTD_RATIO="${UTD_RATIO:-1.0}"
RESET_POOL_SIZE="${RESET_POOL_SIZE:-64}"
CHECKPOINT_BUFFER="${CHECKPOINT_BUFFER:-false}"

EVAL_INTERVAL_STEPS="${EVAL_INTERVAL_STEPS:-15000}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-8}"
EVAL_PREWARM="${EVAL_PREWARM:-false}"

TRAIN_POPULATION_SIZE="${TRAIN_POPULATION_SIZE:-512}"
TRAIN_POLICY_PRIOR_SAMPLES="${TRAIN_POLICY_PRIOR_SAMPLES:-24}"
TRAIN_NUM_ELITES="${TRAIN_NUM_ELITES:-64}"
TRAIN_MPPI_ITERATIONS="${TRAIN_MPPI_ITERATIONS:-6}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-0.5}"

DENSE_QUERY_INTERVAL_STEPS="${DENSE_QUERY_INTERVAL_STEPS:-10000}"
DENSE_RHS_REPLICAS="${DENSE_RHS_REPLICAS:-16}"
DENSE_RHS_EVAL_STEPS="${DENSE_RHS_EVAL_STEPS:-64}"
DENSE_RHS_QUERY_POPULATION="${DENSE_RHS_QUERY_POPULATION:-128}"
DENSE_RHS_QUERY_POLICY_PRIOR="${DENSE_RHS_QUERY_POLICY_PRIOR:-8}"
DENSE_RHS_QUERY_ELITES="${DENSE_RHS_QUERY_ELITES:-16}"
DENSE_RHS_QUERY_MPPI_ITERS="${DENSE_RHS_QUERY_MPPI_ITERS:-2}"
PREWARM_ALL_PHASES="${PREWARM_ALL_PHASES:-false}"

source "$VENV_PATH/bin/activate"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"

RUN_DIR="${PWD}/outputs/${RUN_STEM}_${RUN_LABEL}"
RUN_NAME="${RUN_STEM}_${RUN_LABEL}"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RUN_NAME=${RUN_NAME}"
echo "CONDITION_NAME=${CONDITION_NAME}"
echo "DENSE_QUERY_INTERVAL_STEPS=${DENSE_QUERY_INTERVAL_STEPS}"

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
  env.mjx_dmc.enable_domain_randomization="${ENABLE_DOMAIN_RANDOMIZATION}" \
  env.mjx_dmc.enable_observation_noise="${ENABLE_OBSERVATION_NOISE}" \
  env.mjx_dmc.base_action_delay="${BASE_ACTION_DELAY}" \
  env.mjx_dmc.reset_pool_size="${RESET_POOL_SIZE}" \
  dense_rhs.enabled=true \
  dense_rhs.initial_horizon=3 \
  dense_rhs.query_interval_steps="${DENSE_QUERY_INTERVAL_STEPS}" \
  dense_rhs.num_env_eval_replicas="${DENSE_RHS_REPLICAS}" \
  dense_rhs.env_eval_steps="${DENSE_RHS_EVAL_STEPS}" \
  dense_rhs.query_population_size="${DENSE_RHS_QUERY_POPULATION}" \
  dense_rhs.query_policy_prior_samples="${DENSE_RHS_QUERY_POLICY_PRIOR}" \
  dense_rhs.query_num_elites="${DENSE_RHS_QUERY_ELITES}" \
  dense_rhs.query_mppi_iterations="${DENSE_RHS_QUERY_MPPI_ITERS}" \
  dense_rhs.query_temperature=1.0 \
  dense_rhs.prewarm_all_phases="${PREWARM_ALL_PHASES}" \
  evaluation.enabled=true \
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

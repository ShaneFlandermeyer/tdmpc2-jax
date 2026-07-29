#!/usr/bin/env bash
# 300k Dense-RHS chaos run on the efficient vec8 MJX pipeline.

#SBATCH --job-name=tdmpc2-jax-dense-rhs-chaos
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
SEED="${SEED:-15}"
ENV_ID="${ENV_ID:-quadruped-run}"
RUN_STEM="${RUN_STEM:-$ENV_ID}"
RUN_LABEL="${RUN_LABEL:-dense_rhs_300k_vec8_s${SEED}_chaos_job${SLURM_JOB_ID}}"

MAX_STEPS="${MAX_STEPS:-300000}"
SAVE_INTERVAL_STEPS="${SAVE_INTERVAL_STEPS:-100000000}"
LOG_INTERVAL_STEPS="${LOG_INTERVAL_STEPS:-10000}"
UPDATE_CHUNK_SIZE="${UPDATE_CHUNK_SIZE:-128}"
SEED_STEPS_OVERRIDE="${SEED_STEPS_OVERRIDE:-4000}"
SEED_PRETRAIN_LOG_INTERVAL_UPDATES="${SEED_PRETRAIN_LOG_INTERVAL_UPDATES:-512}"
COLLECT_CHUNK_STEPS="${COLLECT_CHUNK_STEPS:-100}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-8}"
UTD_RATIO="${UTD_RATIO:-1.0}"
RESET_POOL_SIZE="${RESET_POOL_SIZE:-64}"
CHECKPOINT_BUFFER="${CHECKPOINT_BUFFER:-false}"

EVAL_INTERVAL_STEPS="${EVAL_INTERVAL_STEPS:-50000}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-20}"
EVAL_PREWARM="${EVAL_PREWARM:-false}"

TRAIN_POPULATION_SIZE="${TRAIN_POPULATION_SIZE:-512}"
TRAIN_POLICY_PRIOR_SAMPLES="${TRAIN_POLICY_PRIOR_SAMPLES:-24}"
TRAIN_NUM_ELITES="${TRAIN_NUM_ELITES:-64}"
TRAIN_MPPI_ITERATIONS="${TRAIN_MPPI_ITERATIONS:-6}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-0.5}"

DENSE_QUERY_INTERVAL_STEPS="${DENSE_QUERY_INTERVAL_STEPS:-20000}"
DENSE_RHS_REPLICAS="${DENSE_RHS_REPLICAS:-16}"
DENSE_RHS_EVAL_STEPS="${DENSE_RHS_EVAL_STEPS:-64}"
DENSE_RHS_QUERY_POPULATION="${DENSE_RHS_QUERY_POPULATION:-128}"
DENSE_RHS_QUERY_POLICY_PRIOR="${DENSE_RHS_QUERY_POLICY_PRIOR:-8}"
DENSE_RHS_QUERY_ELITES="${DENSE_RHS_QUERY_ELITES:-16}"
DENSE_RHS_QUERY_MPPI_ITERS="${DENSE_RHS_QUERY_MPPI_ITERS:-2}"
DENSE_RHS_QUERY_TEMPERATURE="${DENSE_RHS_QUERY_TEMPERATURE:-1.0}"
DENSE_QUERY_START_STEP="${DENSE_QUERY_START_STEP:-0}"
DENSE_RHS_SELECTION_RETURN_POWER="${DENSE_RHS_SELECTION_RETURN_POWER:-4.0}"
DENSE_RHS_ROUGHNESS_WEIGHT="${DENSE_RHS_ROUGHNESS_WEIGHT:-1.0}"
DENSE_RHS_RETURN_STD_WEIGHT="${DENSE_RHS_RETURN_STD_WEIGHT:-1.0}"
DENSE_RHS_LOCAL_WINDOW_RADIUS="${DENSE_RHS_LOCAL_WINDOW_RADIUS:-0}"
DENSE_RHS_MAX_TRANSITION_DELTA="${DENSE_RHS_MAX_TRANSITION_DELTA:-0}"
DENSE_RHS_INCUMBENT_MARGIN="${DENSE_RHS_INCUMBENT_MARGIN:-0.0}"
DENSE_RHS_CREDIBLE_TRANSITION="${DENSE_RHS_CREDIBLE_TRANSITION:-false}"
DENSE_RHS_CREDIBLE_TRANSITION_RULE="${DENSE_RHS_CREDIBLE_TRANSITION_RULE:-probability}"
DENSE_RHS_CREDIBLE_SWITCH_PROB="${DENSE_RHS_CREDIBLE_SWITCH_PROB:-0.0}"
DENSE_RHS_TRANSITION_COST_SCALE="${DENSE_RHS_TRANSITION_COST_SCALE:-0.0}"
DENSE_RHS_TRANSITION_RISK_WEIGHT="${DENSE_RHS_TRANSITION_RISK_WEIGHT:-1.0}"
DENSE_RHS_TRANSITION_MIN_EXPECTED_NET="${DENSE_RHS_TRANSITION_MIN_EXPECTED_NET:-0.0}"
DENSE_RHS_HORIZONS="${DENSE_RHS_HORIZONS:-[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]}"
DENSE_RHS_HMAX="${DENSE_RHS_HMAX:-30}"
DENSE_RHS_HORIZON_BUCKETS="${DENSE_RHS_HORIZON_BUCKETS:-[]}"
DENSE_RHS_CANDIDATE_BUDGET_A="${DENSE_RHS_CANDIDATE_BUDGET_A:-29}"
DENSE_RHS_CANDIDATE_BUDGET_B1="${DENSE_RHS_CANDIDATE_BUDGET_B1:-18}"
DENSE_RHS_CANDIDATE_BUDGET_B2="${DENSE_RHS_CANDIDATE_BUDGET_B2:-12}"
DENSE_RHS_CANDIDATE_BUDGET_B3="${DENSE_RHS_CANDIDATE_BUDGET_B3:-9}"
DENSE_RHS_CANDIDATE_BUDGET_B4="${DENSE_RHS_CANDIDATE_BUDGET_B4:-9}"
PREWARM_ALL_PHASES="${PREWARM_ALL_PHASES:-false}"

source "$VENV_PATH/bin/activate"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"

RUN_DIR="${PWD}/outputs/${RUN_STEM}_dense_rhs_chaos_${RUN_LABEL}"
RUN_NAME="${RUN_STEM}_dense_rhs_chaos_${RUN_LABEL}"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RUN_NAME=${RUN_NAME}"
echo "ENV_ID=${ENV_ID}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "DENSE_QUERY_START_STEP=${DENSE_QUERY_START_STEP}"
echo "DENSE_QUERY_INTERVAL_STEPS=${DENSE_QUERY_INTERVAL_STEPS}"
echo "DENSE_RHS_LOCAL_WINDOW_RADIUS=${DENSE_RHS_LOCAL_WINDOW_RADIUS}"
echo "DENSE_RHS_MAX_TRANSITION_DELTA=${DENSE_RHS_MAX_TRANSITION_DELTA}"
echo "DENSE_RHS_HORIZONS=${DENSE_RHS_HORIZONS}"
echo "DENSE_RHS_HMAX=${DENSE_RHS_HMAX}"
echo "DENSE_RHS_HORIZON_BUCKETS=${DENSE_RHS_HORIZON_BUCKETS}"

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
  env.env_id="${ENV_ID}" \
  env.num_envs="${TRAIN_NUM_ENVS}" \
  env.utd_ratio="${UTD_RATIO}" \
  env.asynchronous=false \
  env.mjx_dmc.task="${ENV_ID}" \
  env.mjx_dmc.enable_domain_randomization=true \
  env.mjx_dmc.enable_observation_noise=true \
  env.mjx_dmc.base_action_delay=1 \
  env.mjx_dmc.reset_pool_size="${RESET_POOL_SIZE}" \
  dense_rhs.enabled=true \
  dense_rhs.initial_horizon=3 \
  dense_rhs.horizons="${DENSE_RHS_HORIZONS}" \
  dense_rhs.hmax="${DENSE_RHS_HMAX}" \
  dense_rhs.start_query_step="${DENSE_QUERY_START_STEP}" \
  dense_rhs.query_interval_steps="${DENSE_QUERY_INTERVAL_STEPS}" \
  dense_rhs.selection_return_power="${DENSE_RHS_SELECTION_RETURN_POWER}" \
  dense_rhs.roughness_weight="${DENSE_RHS_ROUGHNESS_WEIGHT}" \
  dense_rhs.return_std_weight="${DENSE_RHS_RETURN_STD_WEIGHT}" \
  dense_rhs.local_window_radius="${DENSE_RHS_LOCAL_WINDOW_RADIUS}" \
  dense_rhs.max_transition_delta="${DENSE_RHS_MAX_TRANSITION_DELTA}" \
  dense_rhs.incumbent_switch_margin="${DENSE_RHS_INCUMBENT_MARGIN}" \
  dense_rhs.num_env_eval_replicas="${DENSE_RHS_REPLICAS}" \
  dense_rhs.env_eval_steps="${DENSE_RHS_EVAL_STEPS}" \
  dense_rhs.query_population_size="${DENSE_RHS_QUERY_POPULATION}" \
  dense_rhs.query_policy_prior_samples="${DENSE_RHS_QUERY_POLICY_PRIOR}" \
  dense_rhs.query_num_elites="${DENSE_RHS_QUERY_ELITES}" \
  dense_rhs.query_mppi_iterations="${DENSE_RHS_QUERY_MPPI_ITERS}" \
  dense_rhs.query_temperature="${DENSE_RHS_QUERY_TEMPERATURE}" \
  dense_rhs.credible_transition_enabled="${DENSE_RHS_CREDIBLE_TRANSITION}" \
  dense_rhs.credible_transition_rule="${DENSE_RHS_CREDIBLE_TRANSITION_RULE}" \
  dense_rhs.credible_transition_min_prob="${DENSE_RHS_CREDIBLE_SWITCH_PROB}" \
  dense_rhs.transition_cost_scale="${DENSE_RHS_TRANSITION_COST_SCALE}" \
  dense_rhs.transition_risk_weight="${DENSE_RHS_TRANSITION_RISK_WEIGHT}" \
  dense_rhs.transition_min_expected_net="${DENSE_RHS_TRANSITION_MIN_EXPECTED_NET}" \
  dense_rhs.horizon_buckets="${DENSE_RHS_HORIZON_BUCKETS}" \
  dense_rhs.candidate_budget.A="${DENSE_RHS_CANDIDATE_BUDGET_A}" \
  dense_rhs.candidate_budget.B1="${DENSE_RHS_CANDIDATE_BUDGET_B1}" \
  dense_rhs.candidate_budget.B2="${DENSE_RHS_CANDIDATE_BUDGET_B2}" \
  dense_rhs.candidate_budget.B3="${DENSE_RHS_CANDIDATE_BUDGET_B3}" \
  dense_rhs.candidate_budget.B4="${DENSE_RHS_CANDIDATE_BUDGET_B4}" \
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

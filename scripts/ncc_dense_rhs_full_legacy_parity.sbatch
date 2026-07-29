#!/usr/bin/env bash
# Legacy-parity launch profile for the original adaptive TD-MPC quadruped runs.
# Direct environment/training/eval equivalents are matched to the old setup.
# Dense-RHS-specific runtime knobs remain explicit because the old child-run
# search cost does not transfer 1:1 to the JAX implementation.
#
# Chaos:
#   sbatch scripts/ncc_dense_rhs_full_legacy_parity.sbatch
#
# No-chaos:
#   CONDITION_NAME=no-chaos \
#   ENABLE_DOMAIN_RANDOMIZATION=false \
#   ENABLE_OBSERVATION_NOISE=false \
#   sbatch scripts/ncc_dense_rhs_full_legacy_parity.sbatch
#
# Stricter legacy query cadence:
#   DENSE_QUERY_INTERVAL_STEPS=500 sbatch scripts/ncc_dense_rhs_full_legacy_parity.sbatch
#
# Faster but stronger-than-safe Dense-RHS search:
#   DENSE_RHS_QUERY_POPULATION=256 \
#   DENSE_RHS_QUERY_POLICY_PRIOR=12 \
#   DENSE_RHS_QUERY_ELITES=32 \
#   DENSE_RHS_QUERY_MPPI_ITERS=4 \
#   DENSE_RHS_REPLICAS=8 \
#   DENSE_RHS_EVAL_STEPS=32 \
#   sbatch scripts/ncc_dense_rhs_full_legacy_parity.sbatch

#SBATCH --job-name=tdmpc2-jax-quadruped-legacy
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
CONDITION_NAME="${CONDITION_NAME:-chaos}"
ENABLE_DOMAIN_RANDOMIZATION="${ENABLE_DOMAIN_RANDOMIZATION:-true}"
ENABLE_OBSERVATION_NOISE="${ENABLE_OBSERVATION_NOISE:-true}"
RUN_LABEL="${RUN_LABEL:-s0_job${SLURM_JOB_ID}}"
RUN_STEM="${RUN_STEM:-quadruped-run}"
WANDB_GROUP="${WANDB_GROUP:-tdmpc2-jax-quadruped-legacy-parity}"
WANDB_TAGS="${WANDB_TAGS:-[ncc,h200,mjx,dense-rhs,legacy-parity,clean-eval,quadruped-run]}"

TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-1}"
MAX_STEPS="${MAX_STEPS:-126000}"
SAVE_INTERVAL_STEPS="${SAVE_INTERVAL_STEPS:-10000}"
LOG_INTERVAL_STEPS="${LOG_INTERVAL_STEPS:-2000}"
DENSE_QUERY_INTERVAL_STEPS="${DENSE_QUERY_INTERVAL_STEPS:-5000}"
DENSE_RHS_REPLICAS="${DENSE_RHS_REPLICAS:-8}"
DENSE_RHS_EVAL_STEPS="${DENSE_RHS_EVAL_STEPS:-32}"
DENSE_RHS_QUERY_POPULATION="${DENSE_RHS_QUERY_POPULATION:-256}"
DENSE_RHS_QUERY_POLICY_PRIOR="${DENSE_RHS_QUERY_POLICY_PRIOR:-12}"
DENSE_RHS_QUERY_ELITES="${DENSE_RHS_QUERY_ELITES:-32}"
DENSE_RHS_QUERY_MPPI_ITERS="${DENSE_RHS_QUERY_MPPI_ITERS:-4}"
PREWARM_ALL_PHASES="${PREWARM_ALL_PHASES:-true}"
EVAL_INTERVAL_STEPS="${EVAL_INTERVAL_STEPS:-20000}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-10}"
EVAL_CLEAN="${EVAL_CLEAN:-true}"
EVAL_PREWARM="${EVAL_PREWARM:-true}"
PLOT_SUMMARY="${PLOT_SUMMARY:-true}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"

source "$VENV_PATH/bin/activate"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"

RUN_DIR="${PWD}/outputs/${RUN_STEM}_legacy_parity_${CONDITION_NAME}_${RUN_LABEL}"
RUN_NAME="${RUN_STEM}_legacy_parity_${CONDITION_NAME}_${RUN_LABEL}"

export WANDB_DIR="${RUN_DIR}/wandb"
export WANDB_CONSOLE="off"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "RUN_NAME=${RUN_NAME}"
echo "CONDITION_NAME=${CONDITION_NAME}"
echo "TRAIN_NUM_ENVS=${TRAIN_NUM_ENVS}"
echo "DENSE_QUERY_INTERVAL_STEPS=${DENSE_QUERY_INTERVAL_STEPS}"
echo "DENSE_RHS_REPLICAS=${DENSE_RHS_REPLICAS}"
echo "DENSE_RHS_EVAL_STEPS=${DENSE_RHS_EVAL_STEPS}"
echo "DENSE_RHS_QUERY_POPULATION=${DENSE_RHS_QUERY_POPULATION}"
echo "DENSE_RHS_QUERY_POLICY_PRIOR=${DENSE_RHS_QUERY_POLICY_PRIOR}"
echo "DENSE_RHS_QUERY_ELITES=${DENSE_RHS_QUERY_ELITES}"
echo "DENSE_RHS_QUERY_MPPI_ITERS=${DENSE_RHS_QUERY_MPPI_ITERS}"
echo "EVAL_INTERVAL_STEPS=${EVAL_INTERVAL_STEPS}"
echo "EVAL_NUM_EPISODES=${EVAL_NUM_EPISODES}"
echo "EVAL_CLEAN=${EVAL_CLEAN}"
echo "EVAL_PREWARM=${EVAL_PREWARM}"

set +e
python -m tdmpc2_jax.train \
  hydra.run.dir="${RUN_DIR}" \
  max_steps="${MAX_STEPS}" \
  save_interval_steps="${SAVE_INTERVAL_STEPS}" \
  log_interval_steps="${LOG_INTERVAL_STEPS}" \
  seed_steps_override=5000 \
  env.num_envs="${TRAIN_NUM_ENVS}" \
  env.asynchronous=false \
  env.mjx_dmc.action_repeat=4 \
  env.mjx_dmc.episode_length=250 \
  env.mjx_dmc.enable_domain_randomization="${ENABLE_DOMAIN_RANDOMIZATION}" \
  env.mjx_dmc.enable_observation_noise="${ENABLE_OBSERVATION_NOISE}" \
  tdmpc2.horizon=2 \
  tdmpc2.batch_size=512 \
  tdmpc2.population_size=512 \
  tdmpc2.policy_prior_samples=24 \
  tdmpc2.num_elites=64 \
  tdmpc2.mppi_iterations=6 \
  tdmpc2.temperature=0.5 \
  tdmpc2.reward_loss_scale=0.5 \
  tdmpc2.consistency_loss_scale=2.0 \
  dense_rhs.query_interval_steps="${DENSE_QUERY_INTERVAL_STEPS}" \
  dense_rhs.num_env_eval_replicas="${DENSE_RHS_REPLICAS}" \
  dense_rhs.env_eval_steps="${DENSE_RHS_EVAL_STEPS}" \
  dense_rhs.query_population_size="${DENSE_RHS_QUERY_POPULATION}" \
  dense_rhs.query_policy_prior_samples="${DENSE_RHS_QUERY_POLICY_PRIOR}" \
  dense_rhs.query_num_elites="${DENSE_RHS_QUERY_ELITES}" \
  dense_rhs.query_mppi_iterations="${DENSE_RHS_QUERY_MPPI_ITERS}" \
  dense_rhs.prewarm_all_phases="${PREWARM_ALL_PHASES}" \
  evaluation.enabled=true \
  evaluation.interval_steps="${EVAL_INTERVAL_STEPS}" \
  evaluation.num_episodes="${EVAL_NUM_EPISODES}" \
  evaluation.clean="${EVAL_CLEAN}" \
  evaluation.prewarm="${EVAL_PREWARM}" \
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

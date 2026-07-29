#!/usr/bin/env bash
#SBATCH --job-name=tdmpc2-jax-quadruped-full
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
CONDITION_NAME="${CONDITION_NAME:-chaos}"
ENABLE_DOMAIN_RANDOMIZATION="${ENABLE_DOMAIN_RANDOMIZATION:-true}"
RUN_LABEL="${RUN_LABEL:-s0_job${SLURM_JOB_ID}}"
RUN_STEM="${RUN_STEM:-quadruped-run}"
WANDB_TAGS="${WANDB_TAGS:-[ncc,h200,mjx,dense-rhs,full,quadruped-run]}"
PLOT_SUMMARY="${PLOT_SUMMARY:-true}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"

source "$VENV_PATH/bin/activate"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"

RUN_DIR="${PWD}/outputs/${RUN_STEM}_full_${CONDITION_NAME}_${RUN_LABEL}"
RUN_NAME="${RUN_STEM}_${CONDITION_NAME}_${RUN_LABEL}"

export WANDB_DIR="${RUN_DIR}/wandb"
export WANDB_CONSOLE="off"

set +e
python -m tdmpc2_jax.train \
  hydra.run.dir="${RUN_DIR}" \
  max_steps=126000 \
  save_interval_steps=10000 \
  log_interval_steps=2000 \
  env.num_envs=4 \
  env.asynchronous=false \
  env.mjx_dmc.enable_domain_randomization="${ENABLE_DOMAIN_RANDOMIZATION}" \
  wandb.enabled="${WANDB_ENABLED}" \
  wandb.project=ncc-runs \
  wandb.entity=perpetual-policy \
  wandb.group=tdmpc2-jax-quadruped \
  wandb.name="${RUN_NAME}" \
  wandb.tags="${WANDB_TAGS}" \
  dense_rhs.num_env_eval_replicas=4 \
  dense_rhs.env_eval_steps=16 \
  dense_rhs.query_population_size=128 \
  dense_rhs.query_policy_prior_samples=8 \
  dense_rhs.query_num_elites=16 \
  dense_rhs.query_mppi_iterations=2 \
  dense_rhs.prewarm_all_phases=true
TRAIN_STATUS=$?
set -e

if [[ "${PLOT_SUMMARY}" == "true" ]]; then
  python scripts/plot_run_summary.py --run-dir "${RUN_DIR}" || true
fi

echo "RUN_DIR=${RUN_DIR}"
exit "${TRAIN_STATUS}"

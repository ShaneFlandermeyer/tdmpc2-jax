#!/usr/bin/env bash
# Short MJX DMC environment gate validation on NCC.

#SBATCH --job-name=tdmpc2-jax-mjx-gate
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
TASKS="${TASKS:-fish-swim finger-turn_hard cheetah-run cartpole-swingup acrobot-swingup walker-run hopper-hop}"
OUT_DIR="${OUT_DIR:-runs/results/mjx_gates}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-4}"
STEPS="${STEPS:-16}"
RESET_POOL_SIZE="${RESET_POOL_SIZE:-8}"
CHAOS="${CHAOS:-false}"
ENABLE_DOMAIN_RANDOMIZATION="${ENABLE_DOMAIN_RANDOMIZATION:-}"
ENABLE_OBSERVATION_NOISE="${ENABLE_OBSERVATION_NOISE:-}"
BASE_ACTION_DELAY="${BASE_ACTION_DELAY:-}"
OBSERVATION_NOISE_SCALE="${OBSERVATION_NOISE_SCALE:-}"
WIND_SCALE="${WIND_SCALE:-}"
PUSH_SCALE="${PUSH_SCALE:-}"
SLIP_SCALE="${SLIP_SCALE:-}"
JITTER_PROB="${JITTER_PROB:-}"
DIAGNOSTIC_SWEEP="${DIAGNOSTIC_SWEEP:-false}"

source "$VENV_PATH/bin/activate"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "TASKS=${TASKS}"
echo "OUT_DIR=${OUT_DIR}"
echo "CHAOS=${CHAOS}"
echo "DIAGNOSTIC_SWEEP=${DIAGNOSTIC_SWEEP}"

ARGS=(
  --tasks ${TASKS}
  --out-dir "${OUT_DIR}"
  --seed "${SEED}"
  --num-envs "${NUM_ENVS}"
  --steps "${STEPS}"
  --reset-pool-size "${RESET_POOL_SIZE}"
)

if [[ "${CHAOS}" == "true" ]]; then
  ARGS+=(--chaos)
fi
if [[ -n "${ENABLE_DOMAIN_RANDOMIZATION}" ]]; then
  ARGS+=(--enable-domain-randomization "${ENABLE_DOMAIN_RANDOMIZATION}")
fi
if [[ -n "${ENABLE_OBSERVATION_NOISE}" ]]; then
  ARGS+=(--enable-observation-noise "${ENABLE_OBSERVATION_NOISE}")
fi
if [[ -n "${BASE_ACTION_DELAY}" ]]; then
  ARGS+=(--base-action-delay "${BASE_ACTION_DELAY}")
fi
if [[ -n "${OBSERVATION_NOISE_SCALE}" ]]; then
  ARGS+=(--observation-noise-scale "${OBSERVATION_NOISE_SCALE}")
fi
if [[ -n "${WIND_SCALE}" ]]; then
  ARGS+=(--wind-scale "${WIND_SCALE}")
fi
if [[ -n "${PUSH_SCALE}" ]]; then
  ARGS+=(--push-scale "${PUSH_SCALE}")
fi
if [[ -n "${SLIP_SCALE}" ]]; then
  ARGS+=(--slip-scale "${SLIP_SCALE}")
fi
if [[ -n "${JITTER_PROB}" ]]; then
  ARGS+=(--jitter-prob "${JITTER_PROB}")
fi
if [[ "${DIAGNOSTIC_SWEEP}" == "true" ]]; then
  ARGS+=(--diagnostic-sweep)
fi

python scripts/check_mjx_dmc_gate.py "${ARGS[@]}"

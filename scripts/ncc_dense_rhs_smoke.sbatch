#!/usr/bin/env bash
#SBATCH --job-name=tdmpc2-dense-rhs-smoke
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.log

set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
source "$VENV_PATH/bin/activate"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"

python -m tdmpc2_jax.train \
  max_steps=12 \
  seed_steps_override=8 \
  buffer_size=512 \
  save_interval_steps=4 \
  log_interval_steps=4 \
  env.num_envs=2 \
  env.asynchronous=false \
  env.mjx_dmc.episode_length=8 \
  env.mjx_dmc.enable_domain_randomization=false \
  env.mjx_dmc.enable_observation_noise=false \
  dense_rhs.horizons='[2,3,4]' \
  dense_rhs.hmax=4 \
  dense_rhs.num_env_eval_replicas=1 \
  dense_rhs.env_eval_steps=8 \
  dense_rhs.prewarm_all_phases=false \
  dense_rhs.query_population_size=32 \
  dense_rhs.query_policy_prior_samples=4 \
  dense_rhs.query_num_elites=4 \
  dense_rhs.query_mppi_iterations=1 \
  dense_rhs.query_interval_steps=4 \
  tdmpc2.population_size=32 \
  tdmpc2.policy_prior_samples=4 \
  tdmpc2.num_elites=4 \
  tdmpc2.mppi_iterations=1 \
  tdmpc2.batch_size=8

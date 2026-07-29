#!/usr/bin/env bash
#SBATCH --job-name=tdmpc2-mjx-smoke
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
bash scripts/ncc_mjx_smoke.sh

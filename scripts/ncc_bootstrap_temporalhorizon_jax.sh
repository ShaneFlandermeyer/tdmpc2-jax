#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"

python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -U "jax[cuda12]" mujoco mujoco-mjx
python -m pip install -U flax optax orbax-checkpoint hydra-core tensorboard einops jaxtyping gymnasium[mujoco] dm_control
python -m pip install -e .

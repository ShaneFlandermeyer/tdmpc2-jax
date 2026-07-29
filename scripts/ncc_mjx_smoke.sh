#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/.venvs/temporalhorizon-jax}"
source "$VENV_PATH/bin/activate"

python - <<'PY'
import jax
import mujoco
from mujoco import mjx
from dm_control.suite import common, quadruped

xml = quadruped.make_model(
    floor_size=quadruped._DEFAULT_TIME_LIMIT * quadruped._RUN_SPEED
)
model = mujoco.MjModel.from_xml_string(xml, common.ASSETS)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)

@jax.jit
def run_step(d):
    return mjx.step(mjx_model, d)

devices = jax.devices()
print('devices', devices)
print('default_backend', jax.default_backend())
print('qpos_shape', mjx_data.qpos.shape)
print('jit_step_qpos_shape', run_step(mjx_data).qpos.shape)
PY

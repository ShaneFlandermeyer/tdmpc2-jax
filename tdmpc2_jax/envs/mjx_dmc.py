from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Dict, Mapping, Optional, Sequence, Tuple

from flax import struct
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from tdmpc2_jax.envs.mjx_quadruped import MJXQuadrupedBatchEnv


try:
  import mujoco
  from mujoco import mjx
except Exception:  # pragma: no cover - runtime dependency
  mujoco = None
  mjx = None

try:
  from dm_control import suite
except Exception:  # pragma: no cover - runtime dependency
  suite = None


EPS = 1e-6
DEFAULT_VALUE_AT_MARGIN = 0.1


TASK_DOMAIN = {
    'cup-catch': ('ball_in_cup', 'catch'),
    'cheetah-run': ('cheetah', 'run'),
    'walker-run': ('walker', 'run'),
    'hopper-hop': ('hopper', 'hop'),
    'pendulum-swingup': ('pendulum', 'swingup'),
    'cartpole-swingup': ('cartpole', 'swingup'),
    'acrobot-swingup': ('acrobot', 'swingup'),
    'reacher-hard': ('reacher', 'hard'),
    'finger-turn_hard': ('finger', 'turn_hard'),
    'fish-swim': ('fish', 'swim'),
}

CONTROL_TIMESTEP = {
    'cup-catch': 0.02,
    'cheetah-run': 0.02,
    'walker-run': 0.025,
    'hopper-hop': 0.02,
    'pendulum-swingup': 0.02,
    'cartpole-swingup': 0.01,
    'acrobot-swingup': 0.01,
    'reacher-hard': 0.02,
    'finger-turn_hard': 0.02,
    'fish-swim': 0.04,
}


@dataclass(frozen=True)
class DMCMJXMetadata:
  task: str
  domain: str
  ctrl_min: np.ndarray
  ctrl_max: np.ndarray
  action_dim: int
  observation_dim: int
  episode_length: int
  action_repeat: int
  action_repeat_dt: float
  physics_substeps_per_control: int
  force_body_id: int
  body_torso_id: int
  body_pole_id: int
  body_ball_id: int
  body_foot_id: int
  body_upper_arm_id: int
  body_lower_arm_id: int
  geom_mouth_id: int
  geom_ball_id: int
  geom_finger_id: int
  geom_target_id: int
  site_tip_id: int
  site_target_id: int
  sensor_torso_velocity_slice: slice
  sensor_touch_slice: slice
  sensor_proximal_id: int
  sensor_distal_id: int
  sensor_proximal_velocity_id: int
  sensor_distal_velocity_id: int
  sensor_hinge_velocity_id: int
  sensor_tip_slice: slice
  sensor_spinner_slice: slice
  fish_joint_qpos_adr: np.ndarray
  target_default_pos: np.ndarray
  target_default_size_xz: np.ndarray
  target_default_radius: float
  mouth_radius: float
  ball_radius: float


@struct.dataclass
class MJXDMCBatchState:
  model: any = struct.field(pytree_node=False)
  data: any
  rng: jax.Array
  episode_step: jax.Array
  return_so_far: jax.Array
  length_so_far: jax.Array
  delayed_actions: jax.Array
  last_action: jax.Array
  obs_noise_scale: jax.Array
  actuator_strength: jax.Array
  wind_force: jax.Array
  push_force: jax.Array
  jitter_mask: jax.Array
  target_pos: jax.Array
  target_radius: jax.Array
  done: jax.Array


def _parse_task(task: str) -> Tuple[str, str]:
  if task in TASK_DOMAIN:
    return TASK_DOMAIN[task]
  normalized = task.replace('-', '_')
  try:
    domain, task_name = normalized.split('_', 1)
  except ValueError as exc:
    raise ValueError(f'Invalid DMC task name {task!r}.') from exc
  return domain, task_name


def _safe_name_id(model, obj_type, name: str) -> int:
  if mujoco is None:
    return -1
  try:
    return int(mujoco.mj_name2id(model, obj_type, name))
  except Exception:
    return -1


def _sensor_id(model, name: str) -> int:
  return _safe_name_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)


def _sensor_slice(model, name: str) -> slice:
  sensor_id = _sensor_id(model, name)
  if sensor_id < 0:
    return slice(0, 0)
  start = int(model.sensor_adr[sensor_id])
  return slice(start, start + int(model.sensor_dim[sensor_id]))


def _sensor_scalar_adr(model, name: str) -> int:
  sensor_id = _sensor_id(model, name)
  if sensor_id < 0:
    return -1
  return int(model.sensor_adr[sensor_id])


def _joint_qpos_adr(model, names: Sequence[str]) -> np.ndarray:
  values = []
  for name in names:
    joint_id = _safe_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id >= 0:
      values.append(int(model.jnt_qposadr[joint_id]))
  return np.asarray(values, dtype=np.int32)


def _target_defaults(model, task: str) -> Tuple[np.ndarray, float]:
  target_pos = np.zeros((3,), dtype=np.float32)
  target_radius = 0.0
  if task == 'fish-swim':
    geom_id = _safe_name_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'target')
    if geom_id >= 0:
      target_pos = np.asarray(model.geom_pos[geom_id], dtype=np.float32)
      target_radius = float(model.geom_size[geom_id, 0])
  elif task == 'finger-turn_hard':
    site_id = _safe_name_id(model, mujoco.mjtObj.mjOBJ_SITE, 'target')
    if site_id >= 0:
      target_pos = np.asarray(model.site_pos[site_id], dtype=np.float32)
      target_radius = float(model.site_size[site_id, 0])
  elif task == 'acrobot-swingup':
    site_id = _safe_name_id(model, mujoco.mjtObj.mjOBJ_SITE, 'target')
    if site_id >= 0:
      target_pos = np.asarray(model.site_pos[site_id], dtype=np.float32)
      target_radius = float(model.site_size[site_id, 0])
  elif task == 'reacher-hard':
    target_geom_id = _safe_name_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'target')
    finger_geom_id = _safe_name_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'finger')
    if target_geom_id >= 0:
      target_pos = np.asarray(model.geom_pos[target_geom_id], dtype=np.float32)
      target_radius = float(model.geom_size[target_geom_id, 0])
    if finger_geom_id >= 0:
      target_radius += float(model.geom_size[finger_geom_id, 0])
  return target_pos, target_radius


def _observation_dim(model, task: str) -> int:
  if task == 'cup-catch':
    return int(model.nq + model.nv)
  if task == 'cheetah-run':
    return int(model.nq - 1 + model.nv)
  if task == 'walker-run':
    return int((model.nbody - 1) * 2 + 1 + model.nv)
  if task == 'hopper-hop':
    return int(model.nq - 1 + model.nv + 2)
  if task == 'pendulum-swingup':
    return int(2 + model.nv)
  if task == 'cartpole-swingup':
    return int(1 + (model.nbody - 2) * 2 + model.nv)
  if task == 'acrobot-swingup':
    return int(4 + model.nv)
  if task == 'reacher-hard':
    return int(model.nq + 2 + model.nv)
  if task == 'finger-turn_hard':
    return 12
  if task == 'fish-swim':
    return int(7 + 1 + 3 + model.nv)
  raise ValueError(f'Unsupported MJX DMC task: {task}')


def _metadata(model,
              task: str,
              action_repeat: int,
              episode_length: int,
              action_repeat_dt: Optional[float]) -> DMCMJXMetadata:
  domain, _ = _parse_task(task)
  if action_repeat_dt is None or (
      task != 'quadruped-run' and abs(float(action_repeat_dt) - 0.02) < 1e-9
  ):
    action_repeat_dt = CONTROL_TIMESTEP.get(task, float(action_repeat_dt or 0.02))
  ctrl_range = np.asarray(model.actuator_ctrlrange, dtype=np.float32)
  physics_timestep = float(model.opt.timestep)
  physics_substeps_per_control = max(
      1,
      int(round(float(action_repeat_dt) / physics_timestep)),
  )
  target_pos, target_radius = _target_defaults(model, task)
  mouth_geom_id = _safe_name_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'mouth')
  mouth_radius = float(model.geom_size[mouth_geom_id, 0]) if mouth_geom_id >= 0 else 0.0
  target_site_id = _safe_name_id(model, mujoco.mjtObj.mjOBJ_SITE, 'target')
  target_default_size_xz = np.zeros((2,), dtype=np.float32)
  if target_site_id >= 0:
    target_default_size_xz = np.asarray(model.site_size[target_site_id, [0, 2]], dtype=np.float32)
  ball_geom_id = _safe_name_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'ball')
  ball_radius = float(model.geom_size[ball_geom_id, 0]) if ball_geom_id >= 0 else 0.0
  if task == 'cup-catch':
    # The ball is the reward object; applying chaos xfrc to it makes catch
    # dynamics degenerate. Disturb the controlled cup body instead.
    force_body_id = _safe_name_id(model, mujoco.mjtObj.mjOBJ_BODY, 'cup')
  else:
    force_body_id = _safe_name_id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    if force_body_id < 0:
      force_body_id = _safe_name_id(model, mujoco.mjtObj.mjOBJ_BODY, 'spinner')
    if force_body_id < 0:
      force_body_id = _safe_name_id(model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
    if force_body_id < 0:
      force_body_id = _safe_name_id(model, mujoco.mjtObj.mjOBJ_BODY, 'finger')
    if force_body_id < 0:
      force_body_id = _safe_name_id(model, mujoco.mjtObj.mjOBJ_BODY, 'pole')
  return DMCMJXMetadata(
      task=task,
      domain=domain,
      ctrl_min=ctrl_range[:, 0],
      ctrl_max=ctrl_range[:, 1],
      action_dim=int(model.nu),
      observation_dim=_observation_dim(model, task),
      episode_length=int(episode_length),
      action_repeat=int(action_repeat),
      action_repeat_dt=float(action_repeat_dt),
      physics_substeps_per_control=int(physics_substeps_per_control),
      force_body_id=int(force_body_id),
      body_torso_id=_safe_name_id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso'),
      body_pole_id=_safe_name_id(model, mujoco.mjtObj.mjOBJ_BODY, 'pole'),
      body_ball_id=_safe_name_id(model, mujoco.mjtObj.mjOBJ_BODY, 'ball'),
      body_foot_id=_safe_name_id(model, mujoco.mjtObj.mjOBJ_BODY, 'foot'),
      body_upper_arm_id=_safe_name_id(model, mujoco.mjtObj.mjOBJ_BODY, 'upper_arm'),
      body_lower_arm_id=_safe_name_id(model, mujoco.mjtObj.mjOBJ_BODY, 'lower_arm'),
      geom_mouth_id=mouth_geom_id,
      geom_ball_id=ball_geom_id,
      geom_finger_id=_safe_name_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'finger'),
      geom_target_id=_safe_name_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'target'),
      site_tip_id=_safe_name_id(model, mujoco.mjtObj.mjOBJ_SITE, 'tip'),
      site_target_id=target_site_id,
      sensor_torso_velocity_slice=_sensor_slice(model, 'torso_subtreelinvel'),
      sensor_touch_slice=slice(
          _sensor_scalar_adr(model, 'touch_toe'),
          _sensor_scalar_adr(model, 'touch_heel') + 1
          if _sensor_scalar_adr(model, 'touch_heel') >= 0 else 0,
      ),
      sensor_proximal_id=_sensor_scalar_adr(model, 'proximal'),
      sensor_distal_id=_sensor_scalar_adr(model, 'distal'),
      sensor_proximal_velocity_id=_sensor_scalar_adr(model, 'proximal_velocity'),
      sensor_distal_velocity_id=_sensor_scalar_adr(model, 'distal_velocity'),
      sensor_hinge_velocity_id=_sensor_scalar_adr(model, 'hinge_velocity'),
      sensor_tip_slice=_sensor_slice(model, 'tip'),
      sensor_spinner_slice=_sensor_slice(model, 'spinner'),
      fish_joint_qpos_adr=_joint_qpos_adr(
          model,
          (
              'tail1',
              'tail_twist',
              'tail2',
              'finright_roll',
              'finright_pitch',
              'finleft_roll',
              'finleft_pitch',
          ),
      ),
      target_default_pos=target_pos,
      target_default_size_xz=target_default_size_xz,
      target_default_radius=float(target_radius),
      mouth_radius=mouth_radius,
      ball_radius=ball_radius,
  )


def _load_model(domain: str):
  module = importlib.import_module(f'dm_control.suite.{domain}')
  xml_string, assets = module.get_model_and_assets()
  model = mujoco.MjModel.from_xml_string(xml_string, assets)
  # Some DMC XMLs enable MuJoCo diagnostics such as energy accounting that MJX
  # does not implement. They do not affect task dynamics or rewards.
  model.opt.enableflags = 0
  # Fish swims in free space and the task reward is geometric distance to a
  # target. DMC's XML contains ellipsoid-vs-box contact pairs that this MJX
  # version cannot lower; contacts are not part of the swim objective.
  if domain == 'fish':
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
  elif domain == 'finger':
    # Finger rewards are geometric and observations use the distal touch sites.
    # Keep only fingertip-vs-spinner cap contacts that can feed those touch
    # sites; decoration, proximal, and ground contacts are not task signals and
    # make MJX first-step compilation impractically heavy.
    keep_contact_geoms = {'fingertip', 'cap1', 'cap2'}
    for geom_id in range(model.ngeom):
      geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
      if geom_name not in keep_contact_geoms:
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
  return model


def _copy_physics_data(model, physics):
  data = mujoco.MjData(model)
  np.copyto(data.qpos, np.asarray(physics.data.qpos))
  np.copyto(data.qvel, np.asarray(physics.data.qvel))
  if model.na:
    np.copyto(data.act, np.asarray(physics.data.act))
  if model.nu:
    data.ctrl[:] = 0.0
  if model.nmocap:
    np.copyto(data.mocap_pos, np.asarray(physics.data.mocap_pos))
    np.copyto(data.mocap_quat, np.asarray(physics.data.mocap_quat))
  mujoco.mj_forward(model, data)
  return data


def _extract_target_from_physics(physics, metadata: DMCMJXMetadata) -> Tuple[np.ndarray, float]:
  target_pos = np.asarray(metadata.target_default_pos, dtype=np.float32).copy()
  target_radius = float(metadata.target_default_radius)
  if metadata.task == 'fish-swim':
    try:
      target_pos = np.asarray(physics.named.model.geom_pos['target'], dtype=np.float32)
      target_radius = float(physics.named.model.geom_size['target', 0])
    except Exception:
      pass
  elif metadata.task == 'finger-turn_hard':
    try:
      target_pos = np.asarray(physics.named.data.sensordata['target'], dtype=np.float32)
      target_radius = float(physics.named.model.site_size['target', 0])
    except Exception:
      pass
  elif metadata.task == 'acrobot-swingup':
    try:
      target_pos = np.asarray(physics.named.data.site_xpos['target'], dtype=np.float32)
      target_radius = float(physics.named.model.site_size['target', 0])
    except Exception:
      pass
  elif metadata.task == 'reacher-hard':
    try:
      target_pos = np.asarray(physics.named.model.geom_pos['target'], dtype=np.float32)
      target_radius = float(
          physics.named.model.geom_size[['target', 'finger'], 0].sum()
      )
    except Exception:
      pass
  return target_pos.astype(np.float32), target_radius


def _scale_action_to_ctrl(action: jax.Array, metadata: DMCMJXMetadata) -> jax.Array:
  ctrl_min = jnp.asarray(metadata.ctrl_min, dtype=action.dtype)
  ctrl_max = jnp.asarray(metadata.ctrl_max, dtype=action.dtype)
  normalized = jnp.clip(action, -1.0, 1.0)
  return ctrl_min + (normalized + 1.0) * 0.5 * (ctrl_max - ctrl_min)


def _sigmoid(x: jax.Array,
             value_at_1: float = DEFAULT_VALUE_AT_MARGIN,
             kind: str = 'gaussian') -> jax.Array:
  if kind == 'linear':
    scale = 1.0 - value_at_1
    scaled = x * scale
    return jnp.where(jnp.abs(scaled) < 1.0, 1.0 - scaled, 0.0)
  if kind == 'quadratic':
    scale = jnp.sqrt(1.0 - value_at_1)
    scaled = x * scale
    return jnp.where(jnp.abs(scaled) < 1.0, 1.0 - scaled ** 2, 0.0)
  scale = jnp.sqrt(-2.0 * jnp.log(value_at_1))
  return jnp.exp(-0.5 * (x * scale) ** 2)


def _tolerance(x: jax.Array,
               bounds: Tuple[float, float] = (0.0, 0.0),
               margin: float = 0.0,
               value_at_margin: float = DEFAULT_VALUE_AT_MARGIN,
               sigmoid: str = 'gaussian') -> jax.Array:
  lower, upper = bounds
  in_bounds = jnp.logical_and(lower <= x, x <= upper)
  margin = jnp.asarray(margin, dtype=jnp.result_type(x))
  zero_margin_value = jnp.where(in_bounds, 1.0, 0.0)
  safe_margin = jnp.maximum(margin, EPS)
  d = jnp.where(x < lower, lower - x, x - upper) / safe_margin
  margin_value = jnp.where(in_bounds, 1.0, _sigmoid(d, value_at_margin, sigmoid))
  return jnp.where(margin == 0, zero_margin_value, margin_value)


def _body_xmat(data, body_id: int) -> jax.Array:
  return data.xmat[..., body_id, :, :]


def _finger_tip_position(data, metadata: DMCMJXMetadata) -> jax.Array:
  tip = data.sensordata[..., metadata.sensor_tip_slice]
  spinner = data.sensordata[..., metadata.sensor_spinner_slice]
  return tip[..., [0, 2]] - spinner[..., [0, 2]]


def _finger_target_position(data,
                            metadata: DMCMJXMetadata,
                            target_pos: jax.Array) -> jax.Array:
  spinner = data.sensordata[..., metadata.sensor_spinner_slice]
  return target_pos[..., [0, 2]] - spinner[..., [0, 2]]


def _finger_dist_to_target(data,
                           metadata: DMCMJXMetadata,
                           target_pos: jax.Array,
                           target_radius: jax.Array) -> jax.Array:
  tip = _finger_tip_position(data, metadata)
  target = _finger_target_position(data, metadata, target_pos)
  return jnp.linalg.norm(target - tip, axis=-1) - target_radius


def _fish_mouth_to_target(data,
                          metadata: DMCMJXMetadata,
                          target_pos: jax.Array) -> jax.Array:
  mouth_pos = data.geom_xpos[..., metadata.geom_mouth_id, :]
  mouth_xmat = data.geom_xmat[..., metadata.geom_mouth_id, :, :]
  mouth_to_target_global = target_pos - mouth_pos
  return jnp.einsum('...i,...ij->...j', mouth_to_target_global, mouth_xmat)


def _compute_observation(data,
                         metadata: DMCMJXMetadata,
                         target_pos: jax.Array,
                         target_radius: jax.Array) -> jax.Array:
  task = metadata.task
  if task == 'cup-catch':
    return jnp.concatenate([data.qpos, data.qvel], axis=-1).astype(jnp.float32)
  if task == 'cheetah-run':
    return jnp.concatenate([data.qpos[..., 1:], data.qvel], axis=-1).astype(jnp.float32)
  if task == 'walker-run':
    xmat = data.xmat[..., 1:, :, :]
    orientations = jnp.stack([xmat[..., 0, 0], xmat[..., 0, 2]], axis=-1)
    orientations = orientations.reshape(data.qpos.shape[:-1] + (-1,))
    height = data.xpos[..., metadata.body_torso_id, 2:3]
    return jnp.concatenate([orientations, height, data.qvel], axis=-1).astype(jnp.float32)
  if task == 'hopper-hop':
    touch = jnp.log1p(data.sensordata[..., metadata.sensor_touch_slice])
    return jnp.concatenate([data.qpos[..., 1:], data.qvel, touch], axis=-1).astype(jnp.float32)
  if task == 'pendulum-swingup':
    pole_xmat = data.xmat[..., metadata.body_pole_id, :, :]
    orientation = jnp.stack([pole_xmat[..., 2, 2], pole_xmat[..., 0, 2]], axis=-1)
    return jnp.concatenate([orientation, data.qvel], axis=-1).astype(jnp.float32)
  if task == 'cartpole-swingup':
    pole_xmat = data.xmat[..., 2:, :, :]
    pole_orientation = jnp.stack([pole_xmat[..., 2, 2], pole_xmat[..., 0, 2]], axis=-1)
    pole_orientation = pole_orientation.reshape(data.qpos.shape[:-1] + (-1,))
    cart_position = data.qpos[..., 0:1]
    return jnp.concatenate([cart_position, pole_orientation, data.qvel], axis=-1).astype(jnp.float32)
  if task == 'acrobot-swingup':
    arms = data.xmat[..., [metadata.body_upper_arm_id, metadata.body_lower_arm_id], :, :]
    orientations = jnp.concatenate([arms[..., 0, 2], arms[..., 2, 2]], axis=-1)
    return jnp.concatenate([orientations, data.qvel], axis=-1).astype(jnp.float32)
  if task == 'reacher-hard':
    to_target = target_pos[..., :2] - data.geom_xpos[..., metadata.geom_finger_id, :2]
    return jnp.concatenate([data.qpos, to_target, data.qvel], axis=-1).astype(jnp.float32)
  if task == 'finger-turn_hard':
    proximal = data.sensordata[..., metadata.sensor_proximal_id:metadata.sensor_proximal_id + 1]
    distal = data.sensordata[..., metadata.sensor_distal_id:metadata.sensor_distal_id + 1]
    position = jnp.concatenate([proximal, distal, _finger_tip_position(data, metadata)], axis=-1)
    velocity = jnp.stack([
        data.sensordata[..., metadata.sensor_proximal_velocity_id],
        data.sensordata[..., metadata.sensor_distal_velocity_id],
        data.sensordata[..., metadata.sensor_hinge_velocity_id],
    ], axis=-1)
    touch = jnp.log1p(data.sensordata[..., 14:16])
    target = _finger_target_position(data, metadata, target_pos)
    dist = _finger_dist_to_target(data, metadata, target_pos, target_radius)[..., None]
    return jnp.concatenate([position, velocity, touch, target, dist], axis=-1).astype(jnp.float32)
  if task == 'fish-swim':
    joint_angles = data.qpos[..., metadata.fish_joint_qpos_adr]
    upright = data.xmat[..., metadata.body_torso_id, 2, 2:3]
    target = _fish_mouth_to_target(data, metadata, target_pos)
    return jnp.concatenate([joint_angles, upright, target, data.qvel], axis=-1).astype(jnp.float32)
  raise ValueError(f'Unsupported MJX DMC task: {task}')


def _compute_reward(data,
                    metadata: DMCMJXMetadata,
                    target_pos: jax.Array,
                    target_radius: jax.Array) -> jax.Array:
  task = metadata.task
  if task == 'cup-catch':
    ball_to_target = (
        data.site_xpos[..., metadata.site_target_id, [0, 2]] -
        data.xpos[..., metadata.body_ball_id, [0, 2]]
    )
    target_margin = jnp.asarray(metadata.target_default_size_xz) - metadata.ball_radius
    return jnp.all(jnp.abs(ball_to_target) < target_margin, axis=-1).astype(jnp.float32)
  if task == 'cheetah-run':
    speed = data.sensordata[..., metadata.sensor_torso_velocity_slice.start]
    return _tolerance(speed, bounds=(10.0, jnp.inf), margin=10.0, value_at_margin=0.0, sigmoid='linear').astype(jnp.float32)
  if task == 'walker-run':
    torso_height = data.xpos[..., metadata.body_torso_id, 2]
    torso_upright = data.xmat[..., metadata.body_torso_id, 2, 2]
    horizontal_velocity = data.sensordata[..., metadata.sensor_torso_velocity_slice.start]
    standing = _tolerance(torso_height, bounds=(1.2, jnp.inf), margin=0.6)
    upright = (1.0 + torso_upright) / 2.0
    stand_reward = (3.0 * standing + upright) / 4.0
    move_reward = _tolerance(
        horizontal_velocity,
        bounds=(8.0, jnp.inf),
        margin=4.0,
        value_at_margin=0.5,
        sigmoid='linear',
    )
    return (stand_reward * (5.0 * move_reward + 1.0) / 6.0).astype(jnp.float32)
  if task == 'hopper-hop':
    height = data.xipos[..., metadata.body_torso_id, 2] - data.xipos[..., metadata.body_foot_id, 2]
    speed = data.sensordata[..., metadata.sensor_torso_velocity_slice.start]
    standing = _tolerance(height, bounds=(0.6, 2.0), margin=0.0)
    hopping = _tolerance(
        speed,
        bounds=(2.0, jnp.inf),
        margin=1.0,
        value_at_margin=0.5,
        sigmoid='linear',
    )
    return (standing * hopping).astype(jnp.float32)
  if task == 'pendulum-swingup':
    pole_vertical = data.xmat[..., metadata.body_pole_id, 2, 2]
    return _tolerance(
        pole_vertical,
        bounds=(0.9902680687415704, 1.0),
    ).astype(jnp.float32)
  if task == 'cartpole-swingup':
    cart_position = data.qpos[..., 0]
    pole_cos = data.xmat[..., 2:, 2, 2]
    upright = (pole_cos + 1.0) / 2.0
    centered = (_tolerance(cart_position, margin=2.0) + 1.0) / 2.0
    small_control = _tolerance(
        data.ctrl[..., 0],
        margin=1.0,
        value_at_margin=0.0,
        sigmoid='quadratic',
    )
    small_control = (4.0 + small_control) / 5.0
    small_velocity = _tolerance(data.qvel[..., 1:], margin=5.0)
    small_velocity = (1.0 + jnp.min(small_velocity, axis=-1)) / 2.0
    return (jnp.mean(upright, axis=-1) * small_control * small_velocity * centered).astype(jnp.float32)
  if task == 'acrobot-swingup':
    tip_to_target = target_pos - data.site_xpos[..., metadata.site_tip_id, :]
    dist = jnp.linalg.norm(tip_to_target, axis=-1)
    return _tolerance(dist, bounds=(0.0, target_radius), margin=1.0).astype(jnp.float32)
  if task == 'reacher-hard':
    finger_to_target = target_pos[..., :2] - data.geom_xpos[..., metadata.geom_finger_id, :2]
    dist = jnp.linalg.norm(finger_to_target, axis=-1)
    return _tolerance(dist, bounds=(0.0, target_radius)).astype(jnp.float32)
  if task == 'finger-turn_hard':
    return (_finger_dist_to_target(data, metadata, target_pos, target_radius) <= 0.0).astype(jnp.float32)
  if task == 'fish-swim':
    mouth_to_target = _fish_mouth_to_target(data, metadata, target_pos)
    radii = metadata.mouth_radius + target_radius
    dist = jnp.linalg.norm(mouth_to_target, axis=-1)
    in_target = _tolerance(dist, bounds=(0.0, radii), margin=2.0 * radii)
    is_upright = 0.5 * (data.xmat[..., metadata.body_torso_id, 2, 2] + 1.0)
    return ((7.0 * in_target + is_upright) / 8.0).astype(jnp.float32)
  raise ValueError(f'Unsupported MJX DMC task: {task}')


class MJXDMCBatchEnv:
  def __init__(self,
               num_envs: int,
               seed: int,
               task: str,
               action_repeat: int = 2,
               episode_length: int = 500,
               observation_noise_scale: float = 0.01,
               enable_domain_randomization: bool = True,
               enable_observation_noise: bool = True,
               base_action_delay: int = 1,
               action_repeat_dt: Optional[float] = None,
               wind_scale: float = 5.0,
               push_scale: float = 25.0,
               slip_scale: float = 0.15,
               jitter_prob: float = 0.02,
               reset_pool_size: int = 256):
    if mujoco is None or mjx is None or suite is None:
      raise ImportError('mjx_dmc backend requires mujoco, mujoco.mjx, and dm_control.')
    if task not in TASK_DOMAIN:
      raise ValueError(
          f'MJX DMC backend does not yet support {task!r}. '
          f'Supported tasks: {sorted(TASK_DOMAIN)}'
      )
    self.num_envs = int(num_envs)
    self.seed = int(seed)
    self.enable_domain_randomization = bool(enable_domain_randomization)
    self.enable_observation_noise = bool(enable_observation_noise)
    self.observation_noise_scale = float(observation_noise_scale)
    self.base_action_delay = int(base_action_delay)
    self.wind_scale = float(wind_scale)
    self.push_scale = float(push_scale)
    self.slip_scale = float(slip_scale)
    self.jitter_prob = float(jitter_prob)
    self.reset_pool_size = int(reset_pool_size)

    domain, _ = _parse_task(task)
    self._mj_model = _load_model(domain)
    self._metadata = _metadata(
        self._mj_model,
        task=task,
        action_repeat=action_repeat,
        episode_length=episode_length,
        action_repeat_dt=action_repeat_dt,
    )
    self._mjx_model = mjx.put_model(self._mj_model)
    base_data = mujoco.MjData(self._mj_model)
    mujoco.mj_resetData(self._mj_model, base_data)
    mujoco.mj_forward(self._mj_model, base_data)
    self._base_data = mjx.put_data(self._mj_model, base_data)
    self._reset_pool = self._build_reset_pool(seed=self.seed)
    self.action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(self._metadata.action_dim,),
        dtype=np.float32,
    )
    self.single_action_space = self.action_space
    self.observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(self._metadata.observation_dim,),
        dtype=np.float32,
    )
    self.single_observation_space = self.observation_space
    self._step_fn = jax.jit(jax.vmap(self._step_single, in_axes=(0, 0, 0)))
    self.state = None

  @property
  def metadata(self):
    return self._metadata

  def _build_reset_pool(self, seed: int):
    domain, task_name = _parse_task(self._metadata.task)
    task_random = np.random.RandomState(int(seed))
    env = suite.load(
        domain,
        task_name,
        task_kwargs={'random': task_random},
        visualize_reward=False,
    )
    reset_data = []
    target_pos = []
    target_radius = []
    for _ in range(max(self.reset_pool_size, 1)):
      env.reset()
      data = _copy_physics_data(self._mj_model, env.physics)
      pos, radius = _extract_target_from_physics(env.physics, self._metadata)
      reset_data.append(mjx.put_data(self._mj_model, data))
      target_pos.append(pos)
      target_radius.append(radius)
    return {
        'data': jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *reset_data),
        'target_pos': jnp.asarray(np.stack(target_pos, axis=0), dtype=jnp.float32),
        'target_radius': jnp.asarray(np.asarray(target_radius), dtype=jnp.float32),
    }

  def _sample_reset_params_jax(self,
                               key: jax.Array,
                               leading_shape: Tuple[int, ...]) -> Dict[str, jax.Array]:
    if not self.enable_domain_randomization:
      return {
          'actuator_strength': jnp.ones(leading_shape, dtype=jnp.float32),
          'obs_noise_scale': jnp.zeros(leading_shape, dtype=jnp.float32),
          'wind_force': jnp.zeros(leading_shape + (3,), dtype=jnp.float32),
          'push_force': jnp.zeros(leading_shape + (3,), dtype=jnp.float32),
          'jitter_mask': jnp.zeros(leading_shape, dtype=bool),
      }
    actuator_key, obs_noise_key, wind_key, push_key, jitter_key = jax.random.split(key, 5)
    return {
        'actuator_strength': jax.random.uniform(
            actuator_key, shape=leading_shape, minval=0.9, maxval=1.1
        ).astype(jnp.float32),
        'obs_noise_scale': (
            jax.random.uniform(obs_noise_key, shape=leading_shape).astype(jnp.float32) *
            (self.observation_noise_scale if self.enable_observation_noise else 0.0)
        ),
        'wind_force': jax.random.normal(
            wind_key, shape=leading_shape + (3,)
        ).astype(jnp.float32) * self.wind_scale,
        'push_force': jax.random.normal(
            push_key, shape=leading_shape + (3,)
        ).astype(jnp.float32) * self.push_scale,
        'jitter_mask': jax.random.uniform(jitter_key, shape=leading_shape) < self.jitter_prob,
    }

  def _sample_reset(self, key: jax.Array, leading_shape: Tuple[int, ...]):
    indices = jax.random.randint(
        key,
        shape=leading_shape,
        minval=0,
        maxval=max(self.reset_pool_size, 1),
    )
    return {
        'data': jax.tree.map(lambda x: jnp.take(x, indices, axis=0), self._reset_pool['data']),
        'target_pos': jnp.take(self._reset_pool['target_pos'], indices, axis=0),
        'target_radius': jnp.take(self._reset_pool['target_radius'], indices, axis=0),
    }

  def _make_state(self, key: jax.Array, leading_shape: Tuple[int, ...]) -> MJXDMCBatchState:
    reset_key, data_key, noise_key = jax.random.split(key, 3)
    reset_params = self._sample_reset_params_jax(reset_key, leading_shape)
    reset = self._sample_reset(data_key, leading_shape)
    zeros_action = jnp.zeros(leading_shape + (self._metadata.action_dim,), dtype=jnp.float32)
    return MJXDMCBatchState(
        model=self._mjx_model,
        data=reset['data'],
        rng=noise_key,
        episode_step=jnp.zeros(leading_shape, dtype=jnp.int32),
        return_so_far=jnp.zeros(leading_shape, dtype=jnp.float32),
        length_so_far=jnp.zeros(leading_shape, dtype=jnp.int32),
        delayed_actions=jnp.zeros(
            leading_shape + (max(self.base_action_delay, 1), self._metadata.action_dim),
            dtype=jnp.float32,
        ),
        last_action=zeros_action,
        obs_noise_scale=reset_params['obs_noise_scale'],
        actuator_strength=reset_params['actuator_strength'],
        wind_force=reset_params['wind_force'],
        push_force=reset_params['push_force'],
        jitter_mask=reset_params['jitter_mask'],
        target_pos=reset['target_pos'],
        target_radius=reset['target_radius'],
        done=jnp.zeros(leading_shape, dtype=bool),
    )

  @staticmethod
  def _broadcast_mask(mask: jax.Array, target_ndim: int) -> jax.Array:
    return jnp.reshape(mask, mask.shape + (1,) * (target_ndim - mask.ndim))

  def _masked_replace(self,
                      old_state: MJXDMCBatchState,
                      new_state: MJXDMCBatchState,
                      mask: jax.Array) -> MJXDMCBatchState:
    def choose(old_value, new_value):
      expanded_mask = self._broadcast_mask(mask, old_value.ndim)
      return jnp.where(expanded_mask, new_value, old_value)

    return MJXDMCBatchState(
        model=old_state.model,
        data=jax.tree.map(choose, old_state.data, new_state.data),
        rng=new_state.rng,
        episode_step=choose(old_state.episode_step, new_state.episode_step),
        return_so_far=choose(old_state.return_so_far, new_state.return_so_far),
        length_so_far=choose(old_state.length_so_far, new_state.length_so_far),
        delayed_actions=choose(old_state.delayed_actions, new_state.delayed_actions),
        last_action=choose(old_state.last_action, new_state.last_action),
        obs_noise_scale=choose(old_state.obs_noise_scale, new_state.obs_noise_scale),
        actuator_strength=choose(old_state.actuator_strength, new_state.actuator_strength),
        wind_force=choose(old_state.wind_force, new_state.wind_force),
        push_force=choose(old_state.push_force, new_state.push_force),
        jitter_mask=choose(old_state.jitter_mask, new_state.jitter_mask),
        target_pos=choose(old_state.target_pos, new_state.target_pos),
        target_radius=choose(old_state.target_radius, new_state.target_radius),
        done=choose(old_state.done, new_state.done),
    )

  def reset(self, seed: Optional[int] = None):
    rng_key = jax.random.PRNGKey(self.seed if seed is None else int(seed))
    self.state = self._make_state(rng_key, (self.num_envs,))
    obs_key, next_rng = jax.random.split(self.state.rng)
    self.state = self.state.replace(rng=next_rng)
    obs = self._observation(self.state, key=obs_key)
    return obs, {}

  def _observation(self,
                   state: MJXDMCBatchState,
                   key: Optional[jax.Array] = None) -> jax.Array:
    obs = _compute_observation(
        state.data,
        self._metadata,
        state.target_pos,
        state.target_radius,
    )
    if not self.enable_observation_noise:
      return obs
    noise_scale = jnp.reshape(state.obs_noise_scale, state.obs_noise_scale.shape + (1,))
    if key is None:
      key = state.rng
    noise = jax.random.normal(key, shape=obs.shape) * noise_scale
    return obs + noise

  def sample_actions(self) -> jax.Array:
    if self.state is None:
      key = jax.random.PRNGKey(self.seed)
    else:
      key, next_rng = jax.random.split(self.state.rng)
      self.state = self.state.replace(rng=next_rng)
    return jax.random.uniform(
        key,
        shape=(self.num_envs, self._metadata.action_dim),
        minval=-1.0,
        maxval=1.0,
    ).astype(jnp.float32)

  def _step_single(self,
                   data,
                   env_state: Dict[str, jax.Array],
                   action: jax.Array):
    def done_branch(_):
      return data, env_state, jnp.array(0.0, dtype=jnp.float32), jnp.array(False), jnp.array(True)

    def active_branch(_):
      delayed_actions = env_state['delayed_actions']
      if self.base_action_delay > 0:
        action_to_apply = delayed_actions[0]
        delayed_actions = jnp.concatenate([delayed_actions[1:], action[None, :]], axis=0)
      else:
        action_to_apply = action
      raw_action_to_apply = jnp.where(env_state['jitter_mask'], env_state['last_action'], action_to_apply)
      ctrl_to_apply = _scale_action_to_ctrl(
          raw_action_to_apply * env_state['actuator_strength'],
          self._metadata,
      )
      repeat_data = data.replace(ctrl=ctrl_to_apply)
      if self._metadata.force_body_id >= 0:
        body_force = jnp.concatenate(
            [env_state['wind_force'] + env_state['push_force'], jnp.zeros((3,), dtype=jnp.float32)]
        )
        repeat_data = repeat_data.replace(
            xfrc_applied=repeat_data.xfrc_applied.at[self._metadata.force_body_id].set(body_force)
        )

      def control_step(carry, _):
        control_data, reward_total = carry

        def physics_step(physics_data, _):
          return mjx.step(self._mjx_model, physics_data), None

        control_data, _ = jax.lax.scan(
            physics_step,
            control_data,
            xs=None,
            length=self._metadata.physics_substeps_per_control,
        )
        reward_total = reward_total + _compute_reward(
            control_data,
            self._metadata,
            env_state['target_pos'],
            env_state['target_radius'],
        )
        return (control_data, reward_total), None

      (repeat_data, reward), _ = jax.lax.scan(
          control_step,
          (repeat_data, jnp.array(0.0, dtype=jnp.float32)),
          xs=None,
          length=self._metadata.action_repeat,
      )
      step = env_state['episode_step'] + 1
      truncated = step >= self._metadata.episode_length
      next_env_state = {
          'episode_step': step,
          'return_so_far': env_state['return_so_far'] + reward,
          'length_so_far': env_state['length_so_far'] + 1,
          'delayed_actions': delayed_actions,
          'last_action': raw_action_to_apply,
          'obs_noise_scale': env_state['obs_noise_scale'],
          'actuator_strength': env_state['actuator_strength'],
          'wind_force': env_state['wind_force'],
          'push_force': env_state['push_force'] * (1.0 - self.slip_scale),
          'jitter_mask': env_state['jitter_mask'],
          'target_pos': env_state['target_pos'],
          'target_radius': env_state['target_radius'],
          'done': truncated,
      }
      return repeat_data, next_env_state, reward, jnp.array(False), truncated

    return jax.lax.cond(env_state['done'], done_branch, active_branch, operand=None)

  def _step_state(self, state: MJXDMCBatchState, action: jax.Array):
    leading_shape = action.shape[:-1]
    flat_size = int(np.prod(leading_shape))
    flat_data = jax.tree.map(lambda x: x.reshape((flat_size,) + x.shape[len(leading_shape):]), state.data)
    flat_env_state = {
        'episode_step': state.episode_step.reshape((flat_size,)),
        'return_so_far': state.return_so_far.reshape((flat_size,)),
        'length_so_far': state.length_so_far.reshape((flat_size,)),
        'delayed_actions': state.delayed_actions.reshape((flat_size,) + state.delayed_actions.shape[len(leading_shape):]),
        'last_action': state.last_action.reshape((flat_size,) + state.last_action.shape[len(leading_shape):]),
        'obs_noise_scale': state.obs_noise_scale.reshape((flat_size,)),
        'actuator_strength': state.actuator_strength.reshape((flat_size,)),
        'wind_force': state.wind_force.reshape((flat_size, 3)),
        'push_force': state.push_force.reshape((flat_size, 3)),
        'jitter_mask': state.jitter_mask.reshape((flat_size,)),
        'target_pos': state.target_pos.reshape((flat_size, 3)),
        'target_radius': state.target_radius.reshape((flat_size,)),
        'done': state.done.reshape((flat_size,)),
    }
    flat_action = action.reshape((flat_size, action.shape[-1]))
    data, next_state, reward, terminated, truncated = self._step_fn(
        flat_data,
        flat_env_state,
        flat_action,
    )
    new_state = MJXDMCBatchState(
        model=state.model,
        data=jax.tree.map(lambda x: x.reshape(leading_shape + x.shape[1:]), data),
        rng=state.rng,
        episode_step=next_state['episode_step'].reshape(leading_shape),
        return_so_far=next_state['return_so_far'].reshape(leading_shape),
        length_so_far=next_state['length_so_far'].reshape(leading_shape),
        delayed_actions=next_state['delayed_actions'].reshape(leading_shape + next_state['delayed_actions'].shape[1:]),
        last_action=next_state['last_action'].reshape(leading_shape + next_state['last_action'].shape[1:]),
        obs_noise_scale=next_state['obs_noise_scale'].reshape(leading_shape),
        actuator_strength=next_state['actuator_strength'].reshape(leading_shape),
        wind_force=next_state['wind_force'].reshape(leading_shape + (3,)),
        push_force=next_state['push_force'].reshape(leading_shape + (3,)),
        jitter_mask=next_state['jitter_mask'].reshape(leading_shape),
        target_pos=next_state['target_pos'].reshape(leading_shape + (3,)),
        target_radius=next_state['target_radius'].reshape(leading_shape),
        done=next_state['done'].reshape(leading_shape),
    )
    return (
        new_state,
        reward.reshape(leading_shape),
        terminated.reshape(leading_shape),
        truncated.reshape(leading_shape),
    )

  def _reset_done_envs_jax(self,
                           state: MJXDMCBatchState,
                           done_mask: jax.Array,
                           reset_key: jax.Array) -> MJXDMCBatchState:
    def do_reset(_):
      reset_state = self._make_state(reset_key, tuple(done_mask.shape))
      return self._masked_replace(state, reset_state, done_mask)

    return jax.lax.cond(jnp.any(done_mask), do_reset, lambda _: state, operand=None)

  def _step_autoreset_state(self, state: MJXDMCBatchState, action: jax.Array):
    next_state, reward, terminated, truncated = self._step_state(state, action)
    done = jnp.logical_or(terminated, truncated)
    next_rng, transition_obs_key, reset_key, reset_obs_key = jax.random.split(state.rng, 4)
    transition_next_observation = self._observation(
        next_state.replace(rng=transition_obs_key),
        key=transition_obs_key,
    )
    reset_state = self._reset_done_envs_jax(
        next_state.replace(rng=next_rng),
        done,
        reset_key,
    )
    observation_after_reset = self._observation(
        reset_state.replace(rng=reset_obs_key),
        key=reset_obs_key,
    )
    final_state = reset_state.replace(rng=next_rng)
    return (
        final_state,
        transition_next_observation,
        observation_after_reset,
        reward,
        terminated,
        truncated,
        next_state.return_so_far,
        next_state.length_so_far,
        done,
    )

  def step(self, action: np.ndarray):
    if self.state is None:
      raise RuntimeError('Call reset() before step().')
    action = jnp.asarray(action, dtype=jnp.float32)
    (
        self.state,
        _transition_next_observation,
        observation,
        reward,
        terminated,
        truncated,
        episode_return,
        episode_length,
        _done,
    ) = self._step_autoreset_state(self.state, action)
    info = {'episode': {'r': episode_return, 'l': episode_length}}
    return observation, reward, terminated, truncated, info

  @partial(jax.jit, static_argnums=(0, 4))
  def _candidate_rollout(self,
                         agent,
                         candidate_horizons: jax.Array,
                         key: jax.Array,
                         steps_per_episode: int):
    num_candidates = int(candidate_horizons.shape[0])
    leading_shape = (num_candidates, self.num_envs)
    init_state = self._make_state(key, leading_shape)
    init_plan = (
        jnp.zeros(
            leading_shape + (agent.planning_hmax, self._metadata.action_dim),
            dtype=jnp.float32,
        ),
        jnp.full(
            leading_shape + (agent.planning_hmax, self._metadata.action_dim),
            agent.max_plan_std,
            dtype=jnp.float32,
        ),
    )

    def act_one_candidate(obs, plan, horizon_len, act_key):
      return agent.act(
          obs,
          prev_plan=plan,
          mpc=True,
          deterministic=True,
          train=False,
          horizon=horizon_len,
          key=act_key,
      )

    def rollout_step(carry, step_idx):
      del step_idx
      state, plan, rng = carry
      rng, obs_key, action_key = jax.random.split(rng, 3)
      obs = self._observation(state.replace(rng=obs_key))
      candidate_keys = jax.random.split(action_key, num_candidates)
      actions, next_plan = jax.vmap(act_one_candidate, in_axes=(0, 0, 0, 0))(
          obs,
          plan,
          candidate_horizons,
          candidate_keys,
      )
      next_state, reward, _, _ = self._step_state(state, actions)
      return (next_state, next_plan, rng), reward

    (_, _, _), rewards = jax.lax.scan(
        rollout_step,
        (init_state, init_plan, key),
        jnp.arange(steps_per_episode, dtype=jnp.int32),
    )
    returns = jnp.sum(rewards, axis=0)
    return returns

  def evaluate_candidate_horizons(self,
                                  agent,
                                  candidate_horizons: Sequence[int],
                                  key: jax.Array,
                                  steps_per_episode: Optional[int] = None) -> Dict[int, Dict[str, float]]:
    steps = int(steps_per_episode or self._metadata.episode_length)
    candidate_horizons = jnp.asarray(candidate_horizons, dtype=jnp.int32)
    return_mean, return_std = self.evaluate_candidate_horizons_dense(
        agent=agent,
        candidate_horizons=candidate_horizons,
        candidate_mask=jnp.ones_like(candidate_horizons, dtype=bool),
        key=key,
        steps_per_episode=steps,
    )
    return {
        int(horizon): {
            'return_mean': float(return_mean[idx]),
            'return_std': float(return_std[idx]),
        }
        for idx, horizon in enumerate(np.asarray(candidate_horizons))
    }

  def evaluate_candidate_horizons_dense(self,
                                        agent,
                                        candidate_horizons: jax.Array,
                                        candidate_mask: jax.Array,
                                        key: jax.Array,
                                        steps_per_episode: Optional[int] = None):
    steps = int(steps_per_episode or self._metadata.episode_length)
    candidate_returns = self._candidate_rollout(
        agent=agent,
        candidate_horizons=jnp.asarray(candidate_horizons, dtype=jnp.int32),
        key=key,
        steps_per_episode=steps,
    )
    return_mean = jnp.mean(candidate_returns, axis=-1)
    return_std = jnp.std(candidate_returns, axis=-1)
    return (
        jnp.where(candidate_mask, return_mean, 0.0),
        jnp.where(candidate_mask, return_std, 0.0),
    )

  def run_eval_chunk(self,
                     agent,
                     horizon: int,
                     key: jax.Array,
                     steps_per_episode: Optional[int] = None) -> jax.Array:
    returns = self._candidate_rollout(
        agent=agent,
        candidate_horizons=jnp.asarray([horizon], dtype=jnp.int32),
        key=key,
        steps_per_episode=int(steps_per_episode or self._metadata.episode_length),
    )
    return returns[0]


def make_mjx_dmc_env(env_config, seed: int, num_envs: Optional[int] = None):
  cfg = env_config.mjx_dmc
  task = str(cfg.task)
  if task == 'quadruped-run':
    return MJXQuadrupedBatchEnv(
        num_envs=int(num_envs or env_config.num_envs),
        seed=int(seed),
        task=task,
        action_repeat=int(cfg.action_repeat),
        episode_length=int(cfg.episode_length),
        observation_noise_scale=float(cfg.observation_noise_scale),
        enable_domain_randomization=bool(cfg.enable_domain_randomization),
        enable_observation_noise=bool(cfg.enable_observation_noise),
        base_action_delay=int(cfg.base_action_delay),
        desired_speed=float(cfg.desired_speed),
        action_repeat_dt=float(cfg.action_repeat_dt),
        wind_scale=float(cfg.wind_scale),
        push_scale=float(cfg.push_scale),
        slip_scale=float(cfg.slip_scale),
        jitter_prob=float(cfg.jitter_prob),
        reset_pool_size=int(getattr(cfg, 'reset_pool_size', 256)),
    )
  return MJXDMCBatchEnv(
      num_envs=int(num_envs or env_config.num_envs),
      seed=int(seed),
      task=task,
      action_repeat=int(cfg.action_repeat),
      episode_length=int(cfg.episode_length),
      observation_noise_scale=float(cfg.observation_noise_scale),
      enable_domain_randomization=bool(cfg.enable_domain_randomization),
      enable_observation_noise=bool(cfg.enable_observation_noise),
      base_action_delay=int(cfg.base_action_delay),
      action_repeat_dt=getattr(cfg, 'action_repeat_dt', None),
      wind_scale=float(cfg.wind_scale),
      push_scale=float(cfg.push_scale),
      slip_scale=float(cfg.slip_scale),
      jitter_prob=float(cfg.jitter_prob),
      reset_pool_size=int(getattr(cfg, 'reset_pool_size', 256)),
  )

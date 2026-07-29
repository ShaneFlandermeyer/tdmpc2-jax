from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from flax import struct
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


try:
  import mujoco
  from mujoco import mjx
except Exception:  # pragma: no cover - runtime dependency
  mujoco = None
  mjx = None

try:
  from dm_control.suite import common, quadruped
except Exception:  # pragma: no cover - runtime dependency
  common = None
  quadruped = None


EPS = 1e-6


@dataclass(frozen=True)
class QuadrupedMetadata:
  root_qpos_start: int
  torso_body_id: int
  hinge_qpos_adr: np.ndarray
  hinge_dof_adr: np.ndarray
  velocimeter_slice: slice
  imu_slice: slice
  force_torque_slice: slice
  ctrl_min: np.ndarray
  ctrl_max: np.ndarray
  action_dim: int
  observation_dim: int
  episode_length: int
  action_repeat: int
  action_repeat_dt: float
  physics_substeps_per_control: int
  desired_speed: float


@struct.dataclass
class MJXQuadrupedBatchState:
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
  done: jax.Array


def _sensor_slice(model, sensor_names: Sequence[str]) -> slice:
  adr_min = None
  adr_max = None
  for sensor_name in sensor_names:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id < 0:
      continue
    adr = int(model.sensor_adr[sensor_id])
    dim = int(model.sensor_dim[sensor_id])
    adr_min = adr if adr_min is None else min(adr_min, adr)
    adr_max = adr + dim if adr_max is None else max(adr_max, adr + dim)
  if adr_min is None or adr_max is None:
    return slice(0, 0)
  return slice(adr_min, adr_max)


def _sensor_type_slice(model, sensor_types: Sequence[int]) -> slice:
  sensor_ids = np.where(np.isin(model.sensor_type, np.asarray(sensor_types)))[0]
  if sensor_ids.size == 0:
    return slice(0, 0)
  adr_min = int(np.min(model.sensor_adr[sensor_ids]))
  adr_max = int(
      np.max(model.sensor_adr[sensor_ids] + model.sensor_dim[sensor_ids])
  )
  return slice(adr_min, adr_max)


def _hinge_addresses(model) -> Tuple[np.ndarray, np.ndarray]:
  hinge_ids = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]
  qpos = np.asarray(model.jnt_qposadr[hinge_ids], dtype=np.int32)
  qvel = np.asarray(model.jnt_dofadr[hinge_ids], dtype=np.int32)
  return qpos, qvel


def _quadruped_metadata(model,
                        action_repeat: int,
                        episode_length: int,
                        desired_speed: float,
                        action_repeat_dt: float) -> QuadrupedMetadata:
  root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'root')
  hinge_qpos_adr, hinge_dof_adr = _hinge_addresses(model)
  velocimeter_slice = _sensor_type_slice(model, (mujoco.mjtSensor.mjSENS_VELOCIMETER,))
  imu_slice = _sensor_type_slice(
      model,
      (mujoco.mjtSensor.mjSENS_ACCELEROMETER, mujoco.mjtSensor.mjSENS_GYRO),
  )
  force_torque_slice = _sensor_type_slice(
      model,
      (mujoco.mjtSensor.mjSENS_FORCE, mujoco.mjtSensor.mjSENS_TORQUE),
  )
  torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
  ctrl_range = np.asarray(model.actuator_ctrlrange, dtype=np.float32)
  physics_timestep = float(model.opt.timestep)
  physics_substeps_per_control = max(1, int(round(float(action_repeat_dt) / physics_timestep)))
  observation_dim = (
      hinge_qpos_adr.shape[0] +
      hinge_dof_adr.shape[0] +
      int(model.na) +
      (velocimeter_slice.stop - velocimeter_slice.start) +
      1 +
      (imu_slice.stop - imu_slice.start) +
      (force_torque_slice.stop - force_torque_slice.start)
  )
  return QuadrupedMetadata(
      root_qpos_start=int(model.jnt_qposadr[root_joint_id]),
      torso_body_id=int(torso_body_id),
      hinge_qpos_adr=hinge_qpos_adr,
      hinge_dof_adr=hinge_dof_adr,
      velocimeter_slice=velocimeter_slice,
      imu_slice=imu_slice,
      force_torque_slice=force_torque_slice,
      ctrl_min=ctrl_range[:, 0],
      ctrl_max=ctrl_range[:, 1],
      action_dim=int(model.nu),
      observation_dim=int(observation_dim),
      episode_length=int(episode_length),
      action_repeat=int(action_repeat),
      action_repeat_dt=float(action_repeat_dt),
      physics_substeps_per_control=int(physics_substeps_per_control),
      desired_speed=float(desired_speed),
  )


def _sample_unit_quaternion(random_state: np.random.RandomState) -> np.ndarray:
  orientation = random_state.randn(4)
  orientation /= np.linalg.norm(orientation)
  return orientation.astype(np.float32)


def _find_non_contacting_height(model,
                                data,
                                metadata: QuadrupedMetadata,
                                orientation: np.ndarray,
                                x_pos: float = 0.0,
                                y_pos: float = 0.0) -> None:
  z_pos = 0.0
  attempts = 0
  while True:
    try:
      mujoco.mj_resetData(model, data)
      root_qpos = data.qpos
      root_qpos[metadata.root_qpos_start:metadata.root_qpos_start+3] = (
          x_pos,
          y_pos,
          z_pos,
      )
      root_qpos[metadata.root_qpos_start+3:metadata.root_qpos_start+7] = orientation
      mujoco.mj_forward(model, data)
      if int(data.ncon) == 0:
        return
    except Exception:
      pass
    z_pos += 0.01
    attempts += 1
    if attempts > 10_000:
      raise RuntimeError('Failed to find a non-contacting quadruped configuration.')


def _linear_tolerance(x: jax.Array,
                      lower: float,
                      margin: float,
                      value_at_margin: float = 0.0) -> jax.Array:
  delta = (lower - x) / jnp.maximum(margin, EPS)
  return jnp.where(
      x >= lower,
      1.0,
      jnp.clip(1.0 - (1.0 - value_at_margin) * delta, value_at_margin, 1.0),
  )


def _torso_upright_component(xmat: jax.Array, metadata: QuadrupedMetadata) -> jax.Array:
  torso_xmat = xmat[..., metadata.torso_body_id, :, :]
  return torso_xmat[..., 2, 2]


def _compute_observation(data, metadata: QuadrupedMetadata) -> jax.Array:
  qpos = data.qpos[..., metadata.hinge_qpos_adr]
  qvel = data.qvel[..., metadata.hinge_dof_adr]
  act = data.act if hasattr(data, 'act') else jnp.zeros(qpos.shape[:-1] + (0,))
  sensordata = data.sensordata
  torso_velocity = sensordata[..., metadata.velocimeter_slice]
  torso_upright = _torso_upright_component(data.xmat, metadata)[..., None]
  imu = sensordata[..., metadata.imu_slice]
  force_torque = jnp.arcsinh(sensordata[..., metadata.force_torque_slice])
  return jnp.concatenate(
      [qpos, qvel, act, torso_velocity, torso_upright, imu, force_torque],
      axis=-1,
  ).astype(jnp.float32)


def _compute_reward(data, metadata: QuadrupedMetadata) -> jax.Array:
  torso_velocity_x = data.sensordata[..., metadata.velocimeter_slice.start]
  torso_upright = _torso_upright_component(data.xmat, metadata)
  move_reward = _linear_tolerance(
      torso_velocity_x,
      lower=metadata.desired_speed,
      margin=metadata.desired_speed,
      value_at_margin=0.5,
  )
  upright_reward = _linear_tolerance(
      torso_upright,
      lower=1.0,
      margin=2.0,
      value_at_margin=0.0,
  )
  return (upright_reward * move_reward).astype(jnp.float32)


def _scale_action_to_ctrl(action: jax.Array,
                          metadata: QuadrupedMetadata) -> jax.Array:
  """Matches dm_control action_scale.Wrapper(minimum=-1, maximum=1)."""
  ctrl_min = jnp.asarray(metadata.ctrl_min, dtype=action.dtype)
  ctrl_max = jnp.asarray(metadata.ctrl_max, dtype=action.dtype)
  normalized = jnp.clip(action, -1.0, 1.0)
  return ctrl_min + (normalized + 1.0) * 0.5 * (ctrl_max - ctrl_min)


class MJXQuadrupedBatchEnv:
  def __init__(self,
               num_envs: int,
               seed: int,
               task: str = 'quadruped-run',
               action_repeat: int = 2,
               episode_length: int = 500,
               observation_noise_scale: float = 0.01,
               enable_domain_randomization: bool = True,
               enable_observation_noise: bool = True,
               base_action_delay: int = 1,
               desired_speed: float = 5.0,
               action_repeat_dt: float = 0.02,
               wind_scale: float = 5.0,
               push_scale: float = 25.0,
               slip_scale: float = 0.15,
               jitter_prob: float = 0.02,
               reset_pool_size: int = 256):
    if task != 'quadruped-run':
      raise ValueError(f'MJX quadruped backend only supports quadruped-run, got {task}.')
    if mujoco is None or mjx is None or quadruped is None or common is None:
      raise ImportError(
          'mjx_dmc backend requires mujoco, mujoco.mjx, and dm_control.suite.quadruped.'
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

    xml_string = quadruped.make_model(
        floor_size=quadruped._DEFAULT_TIME_LIMIT * quadruped._RUN_SPEED
    )
    self._mj_model = mujoco.MjModel.from_xml_string(xml_string, common.ASSETS)
    self._metadata = _quadruped_metadata(
        self._mj_model,
        action_repeat=action_repeat,
        episode_length=episode_length,
        desired_speed=desired_speed,
        action_repeat_dt=action_repeat_dt,
    )
    self._mjx_model = mjx.put_model(self._mj_model)
    base_data = mujoco.MjData(self._mj_model)
    mujoco.mj_resetData(self._mj_model, base_data)
    mujoco.mj_forward(self._mj_model, base_data)
    self._base_data = mjx.put_data(self._mj_model, base_data)
    self._reset_pool_data = self._build_reset_pool(seed=self.seed)
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

  def _build_reset_pool(self, seed: int):
    random_state = np.random.RandomState(int(seed))
    reset_data = []
    for _ in range(max(self.reset_pool_size, 1)):
      data = mujoco.MjData(self._mj_model)
      orientation = _sample_unit_quaternion(random_state)
      _find_non_contacting_height(
          self._mj_model,
          data,
          self._metadata,
          orientation=orientation,
      )
      reset_data.append(mjx.put_data(self._mj_model, data))
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *reset_data)

  def _broadcast_base_data(self, leading_shape: Tuple[int, ...]):
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x, leading_shape + x.shape),
        self._base_data,
    )

  def _sample_reset_data(self,
                         key: jax.Array,
                         leading_shape: Tuple[int, ...]):
    indices = jax.random.randint(
        key,
        shape=leading_shape,
        minval=0,
        maxval=max(self.reset_pool_size, 1),
    )
    return jax.tree.map(lambda x: jnp.take(x, indices, axis=0), self._reset_pool_data)

  def _make_state(self,
                  key: jax.Array,
                  leading_shape: Tuple[int, ...]) -> MJXQuadrupedBatchState:
    reset_key, data_key, noise_key = jax.random.split(key, 3)
    reset_params = self._sample_reset_params_jax(reset_key, leading_shape)
    zeros_action = jnp.zeros(leading_shape + (self._metadata.action_dim,), dtype=jnp.float32)
    return MJXQuadrupedBatchState(
        model=self._mjx_model,
        data=self._sample_reset_data(data_key, leading_shape),
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
        done=jnp.zeros(leading_shape, dtype=bool),
    )

  @staticmethod
  def _broadcast_mask(mask: jax.Array, target_ndim: int) -> jax.Array:
    return jnp.reshape(mask, mask.shape + (1,) * (target_ndim - mask.ndim))

  def _masked_replace(self,
                      old_state: MJXQuadrupedBatchState,
                      new_state: MJXQuadrupedBatchState,
                      mask: jax.Array) -> MJXQuadrupedBatchState:
    def choose(old_value, new_value):
      expanded_mask = self._broadcast_mask(mask, old_value.ndim)
      return jnp.where(expanded_mask, new_value, old_value)

    return MJXQuadrupedBatchState(
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
                   state: MJXQuadrupedBatchState,
                   key: Optional[jax.Array] = None) -> jax.Array:
    obs = _compute_observation(state.data, self._metadata)
    if not self.enable_observation_noise:
      return obs
    noise_scale = jnp.reshape(state.obs_noise_scale, state.obs_noise_scale.shape + (1,))
    if key is None:
      key = state.rng
    noise = jax.random.normal(key, shape=obs.shape) * noise_scale
    return obs + noise

  def sample_actions(self) -> np.ndarray:
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
        delayed_actions = jnp.concatenate(
            [delayed_actions[1:], action[None, :]],
            axis=0,
        )
      else:
        action_to_apply = action
      raw_action_to_apply = jnp.where(env_state['jitter_mask'], env_state['last_action'], action_to_apply)
      ctrl_to_apply = _scale_action_to_ctrl(
          raw_action_to_apply * env_state['actuator_strength'],
          self._metadata,
      )
      torso_force = jnp.concatenate([env_state['wind_force'] + env_state['push_force'], jnp.zeros((3,), dtype=jnp.float32)])
      repeat_data = data.replace(
          ctrl=ctrl_to_apply,
          xfrc_applied=data.xfrc_applied.at[self._metadata.torso_body_id].set(torso_force),
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
        reward_total = reward_total + _compute_reward(control_data, self._metadata)
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
          'done': truncated,
      }
      return repeat_data, next_env_state, reward, jnp.array(False), truncated

    return jax.lax.cond(env_state['done'], done_branch, active_branch, operand=None)

  def _step_state(self,
                  state: MJXQuadrupedBatchState,
                  action: jax.Array):
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
        'done': state.done.reshape((flat_size,)),
    }
    flat_action = action.reshape((flat_size, action.shape[-1]))
    data, next_state, reward, terminated, truncated = self._step_fn(
        flat_data,
        flat_env_state,
        flat_action,
    )
    new_state = MJXQuadrupedBatchState(
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
        done=next_state['done'].reshape(leading_shape),
    )
    return (
        new_state,
        reward.reshape(leading_shape),
        terminated.reshape(leading_shape),
        truncated.reshape(leading_shape),
    )

  def _reset_done_envs_jax(self,
                           state: MJXQuadrupedBatchState,
                           done_mask: jax.Array,
                           reset_key: jax.Array) -> MJXQuadrupedBatchState:
    def do_reset(_):
      reset_state = self._make_state(reset_key, tuple(done_mask.shape))
      return self._masked_replace(state, reset_state, done_mask)

    return jax.lax.cond(
        jnp.any(done_mask),
        do_reset,
        lambda _: state,
        operand=None,
    )

  def _reset_done_envs(self,
                       state: MJXQuadrupedBatchState,
                       done_mask: jax.Array) -> MJXQuadrupedBatchState:
    if not bool(np.any(np.asarray(done_mask))):
      return state
    reset_key, next_rng = jax.random.split(state.rng)
    state = self._reset_done_envs_jax(state, done_mask, reset_key)
    return state.replace(rng=next_rng)

  def _step_autoreset_state(self,
                            state: MJXQuadrupedBatchState,
                            action: jax.Array):
    next_state, reward, terminated, truncated = self._step_state(state, action)
    done = jnp.logical_or(terminated, truncated)

    next_rng, transition_obs_key, reset_key, reset_obs_key = jax.random.split(
        state.rng, 4
    )
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
      next_state, reward, _, truncated = self._step_state(state, actions)
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
  return MJXQuadrupedBatchEnv(
      num_envs=int(num_envs or env_config.num_envs),
      seed=int(seed),
      task=str(cfg.task),
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

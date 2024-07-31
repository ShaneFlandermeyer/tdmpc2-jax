from copy import deepcopy
import warnings

import gymnasium as gym

from tdmpc2_jax.envs.wrappers.pixels import PixelWrapper

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
	from tdmpc2_jax.envs.dmcontrol import make_env as make_dm_control_env
except:
	make_dm_control_env = missing_dependencies

warnings.filterwarnings('ignore', category=DeprecationWarning)

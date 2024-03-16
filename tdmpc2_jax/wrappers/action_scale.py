import gymnasium as gym


class RescaleActions(gym.Wrapper):
  """
  Rescale actions from the range [-1, 1] to the environment action space ranges.
  """

  def __init__(self, env: gym.Env):
    super().__init__(env)
    self.action_scale = (env.action_space.high - env.action_space.low) / 2
    self.action_bias = (env.action_space.high + env.action_space.low) / 2

  def step(self, action):
    action = action * self.action_scale + self.action_bias
    return self.env.step(action)

import gymnasium as gym


class RepeatAction(gym.Wrapper):
  def __init__(self, env, repeat=1):
    super().__init__(env)
    self.repeat = repeat

  def step(self, action):
    reward = 0
    for _ in range(self.repeat):
      observation, r, terminated, truncated, info = self.env.step(action)
      reward += r
      if terminated or truncated:
        break
    return observation, reward, terminated, truncated, info

from __future__ import annotations

from typing import Dict

import numpy as np


def summarize_episode_statistics(returns: np.ndarray,
                                 lengths: np.ndarray) -> Dict[str, float]:
  returns = np.asarray(returns, dtype=np.float32)
  lengths = np.asarray(lengths, dtype=np.int32)
  return {
      'return_mean': float(np.mean(returns)) if returns.size else 0.0,
      'return_std': float(np.std(returns)) if returns.size else 0.0,
      'return_min': float(np.min(returns)) if returns.size else 0.0,
      'return_max': float(np.max(returns)) if returns.size else 0.0,
      'length_mean': float(np.mean(lengths)) if lengths.size else 0.0,
      'length_std': float(np.std(lengths)) if lengths.size else 0.0,
  }


def compare_rollout_statistics(reference_rewards: np.ndarray,
                               candidate_rewards: np.ndarray) -> Dict[str, float]:
  reference_rewards = np.asarray(reference_rewards, dtype=np.float32)
  candidate_rewards = np.asarray(candidate_rewards, dtype=np.float32)
  delta = candidate_rewards - reference_rewards
  return {
      'reward_mean_abs_error': float(np.mean(np.abs(delta))) if delta.size else 0.0,
      'reward_mean_error': float(np.mean(delta)) if delta.size else 0.0,
      'reward_std_error': float(np.std(candidate_rewards) - np.std(reference_rewards))
      if delta.size else 0.0,
  }

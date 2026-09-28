"""NC84 reset mixture; all gameplay and observations remain TheGameEnv's."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from src.synthetic.curriculum import validate_distribution
from src.natural.bank import NaturalBank, restore_checkpoint
from src.env import TheGameEnv

from src.natural.curriculum import NC_SYNTHETIC_TARGETS, NC_SYNTHETIC_WEIGHTS
CONTROL_WEIGHTS = [.05, .10, .10, .15, .25, .35]


class NaturalConditionedEnv(TheGameEnv):
    def __init__(self, *, bank: NaturalBank, nc_probability=.30, split='train',
                 synthetic_targets=None, synthetic_probs=None, backward_teacher_prob=.30):
        if split != 'train':
            raise ValueError('training resets may use train split only')
        if not np.isfinite(nc_probability) or not 0 <= nc_probability <= 1:
            raise ValueError('nc_probability must be in [0,1]')
        if synthetic_targets is None:
            synthetic_targets = NC_SYNTHETIC_TARGETS
        if synthetic_probs is None:
            synthetic_probs = np.asarray(NC_SYNTHETIC_WEIGHTS) / .70
        targets, probs = validate_distribution(synthetic_targets, synthetic_probs)
        super().__init__(reset_source='reverse', target_remaining=84,
                         curriculum_targets=targets, curriculum_probs=probs,
                         backward_teacher_prob=backward_teacher_prob)
        self.nc_probability = float(nc_probability)
        self._nc_states = [cp for _, cp in bank.checkpoints(split, 84)]
        if self.nc_probability and not self._nc_states:
            raise ValueError('NC84 train bucket is empty')

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if self.np_random.random() < self.nc_probability:
            checkpoint = self._nc_states[int(self.np_random.integers(len(self._nc_states)))]
            restore_checkpoint(self, checkpoint)
        else:
            # Do not reseed here: source and target sampling share the env RNG.
            return super().reset(options=options)
        return self._get_obs(), self._get_info()


from src.common.source_usage import SourceUsageWrapper

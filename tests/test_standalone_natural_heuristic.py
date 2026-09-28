import unittest

import numpy as np

from src.natural.bank import build_record, natural_start, validate_record
from test_natural_bank import fixture


class StandaloneNaturalHeuristicTests(unittest.TestCase):
    def test_policy_never_reads_future_deck_order(self):
        from src.natural.heuristics import StandalonePolicy

        class NoDeckAccess:
            def __init__(self, env):
                self._env = env

            def __getattr__(self, name):
                if name == 'deck':
                    raise AssertionError('policy read future deck order')
                return getattr(self._env, name)

        env = natural_start(fixture()[0])
        for strategy in ('exponential', 'combined'):
            with self.subTest(strategy=strategy):
                policy = StandalonePolicy(strategy)
                action = policy.choose(NoDeckAccess(env))
                self.assertTrue(env.action_masks()[action])

    def test_winning_rollout_records_the_first_nc84_boundary(self):
        from src.natural.heuristics import play_natural_deck

        # The deterministic fixture is a real natural deck and demonstrates
        # witness construction independently of random-deck win probability.
        result = play_natural_deck(fixture()[0], 'combined_depth2')
        self.assertTrue(result.won)
        self.assertIsNotNone(result.nc84_checkpoint)
        self.assertEqual(result.nc84_checkpoint['q'], 0)
        self.assertTrue(82 <= result.nc84_checkpoint['remaining_count'] <= 86)
        record = build_record(fixture()[0], result.actions,
                              prefix_action_offset=result.nc84_checkpoint['action_offset'])
        validate_record(record)

    def test_depth_modes_choose_legal_actions_without_peeking(self):
        from src.natural.heuristics import StandalonePolicy

        env = natural_start(fixture()[0])
        for name in ('exponential', 'combined', 'combined_depth1', 'combined_depth2'):
            with self.subTest(name=name):
                action = StandalonePolicy(name).choose(env)
                self.assertTrue(env.action_masks()[action])

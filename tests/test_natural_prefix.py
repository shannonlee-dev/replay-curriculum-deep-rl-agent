import copy
import unittest
from unittest.mock import patch

import numpy as np

from src.natural.bank import build_record, natural_start, validate_record, restore_checkpoint
from test_natural_bank import fixture
from src.env import TheGameEnv


class TwoCardPolicy:
    """Hand-authored observable-only policy, not a learned model."""
    def __init__(self, turn_size=2):
        self.turn_size, self.modes, self.observations = turn_size, [], []

    def predict(self, obs, *, deterministic, action_masks):
        self.modes.append(deterministic)
        self.observations.append(obs.copy())
        if round(float(obs[-2])*8) >= self.turn_size and action_masks[392]:
            return 392, None
        return int(np.flatnonzero(action_masks)[0]), None


class PrefixTests(unittest.TestCase):
    def test_rollout_stops_at_first_nc84_boundary(self):
        from archive.privileged_teacher.prefix import rollout_prefix
        for deterministic in (True, False):
            policy = TwoCardPolicy()
            result = rollout_prefix(fixture()[0], policy, deterministic=deterministic)
            self.assertEqual(result.status, 'prefix_reached')
            self.assertEqual(result.checkpoint['remaining_count'], 86)
            self.assertEqual(result.checkpoint['q'], 0)
            self.assertEqual(len(result.actions), 18)  # 12 plays + 6 ENDs
            self.assertEqual(policy.modes, [deterministic]*18)
            self.assertTrue(all(obs.shape == (300,) for obs in policy.observations))
            env = natural_start(fixture()[0])
            for action in result.actions:
                env.step(action)
            restored = TheGameEnv()
            restore_checkpoint(restored, result.checkpoint)
            np.testing.assert_array_equal(env._get_obs(), restored._get_obs())
            np.testing.assert_array_equal(env.action_masks(), restored.action_masks())

    def test_prefix_overshoot_and_loss_are_not_search_failures(self):
        from archive.privileged_teacher.prefix import rollout_prefix
        # Three 6-card turns skip from remaining 86? use 5: 98,93,88,83 succeeds.
        # Seven-card turns: 98,91,84 succeeds. Crafted 8 then 7 then 4: 98,90,83 succeeds.
        # 8, then 8, gives 82 and is valid; 4 then 5 then 8 gives 98,94,89,81.
        class Overshoot(TwoCardPolicy):
            def predict(self, obs, **kwargs):
                played_total = int(obs[196:294].sum())
                q = round(float(obs[-2])*8)
                turn_start = played_total-q
                self.turn_size = 4 if turn_start == 0 else (5 if turn_start == 4 else 8)
                return super().predict(obs, **kwargs)
        result = rollout_prefix(fixture()[0], Overshoot())
        self.assertEqual(result.status, 'policy_failed_to_reach_nc84')
        self.assertEqual(result.reason, 'overshot_target_window')
        self.assertIsNone(result.checkpoint)
        class IllegalPolicy:
            def predict(self, obs, **kwargs):
                return 392, None
        result = rollout_prefix(fixture()[0], IllegalPolicy())
        self.assertEqual(result.status, 'policy_failed_to_reach_nc84')
        self.assertEqual(result.reason, 'policy_loss')

    def test_suffix_search_starts_at_prefix_and_preserves_it(self):
        from archive.privileged_teacher.prefix import rollout_prefix
        from archive.privileged_teacher.teacher import search_checkpoint, TeacherConfig
        prefix = rollout_prefix(fixture()[0], TwoCardPolicy())
        before = copy.deepcopy(prefix.checkpoint)
        seen_remaining = []
        def prior(env):
            seen_remaining.append(env._remaining_cards())
            return -np.arange(393, dtype=float), 0.
        config = TeacherConfig(beam_width=1, candidate_turns_per_state=1,
                               policy_top_k_actions=1, stochastic_candidates=0,
                               max_nodes_per_deck=5000, timeout_seconds_per_deck=10)
        result = search_checkpoint(prefix.checkpoint, prior, config)
        self.assertEqual(seen_remaining[0], 86)
        self.assertLessEqual(max(seen_remaining), 86)
        self.assertEqual(prefix.checkpoint, before)
        self.assertEqual(result.status, 'win_found')
        self.assertEqual(result.max_cards_played, 86)
        self.assertEqual(result.deepest_remaining, 0)
        record = build_record(fixture()[0], prefix.actions+result.actions,
                              prefix_action_offset=len(prefix.actions))
        self.assertEqual(record['schema_version'], 2)
        self.assertEqual(record['checkpoints'], [prefix.checkpoint])
        validate_record(record)
        record['prefix_action_offset'] += 1
        with self.assertRaises(ValueError):
            validate_record(record)

    def test_timeout_reports_progress_beyond_search_root(self):
        from archive.privileged_teacher.prefix import rollout_prefix
        from archive.privileged_teacher.teacher import search_checkpoint, TeacherConfig
        prefix = rollout_prefix(fixture()[0], TwoCardPolicy())
        clock = [0.]
        calls = [0]
        def prior(env):
            calls[0] += 1
            if calls[0] == 3:
                clock[0] = 30.
            return -np.arange(393, dtype=float), 0.
        config = TeacherConfig(beam_width=1, candidate_turns_per_state=1,
                               policy_top_k_actions=1, stochastic_candidates=0)
        with patch('archive.privileged_teacher.teacher.time.monotonic', side_effect=lambda: clock[0]):
            result = search_checkpoint(prefix.checkpoint, prior, config)
        self.assertEqual(result.status, 'teacher_failed_timeout')
        self.assertEqual(result.deepest_remaining, 84)
        self.assertEqual(result.max_cards_played, 2)
        self.assertEqual(result.deepest_turn, 0)
        self.assertEqual(result.frontier_size, 1)

    def test_attempt_separates_reach_failure_and_suffix_timeout(self):
        from archive.privileged_teacher.build_bank import generate_attempt
        from archive.privileged_teacher.teacher import TeacherConfig
        teacher = dict(model_sha256='0'*64, config={}, code_sha256={'fixture': '1'*64})
        deck = fixture()[0]
        class IllegalPolicy:
            def predict(self, obs, **kwargs):
                return 392, None
        with patch('archive.privileged_teacher.build_bank.search_checkpoint', side_effect=AssertionError('teacher must not run')):
            failed = generate_attempt(deck, IllegalPolicy(), None, TeacherConfig(), index=0,
                                      deterministic=True, prefix_seed=7, search_seed=8, teacher=teacher)
        self.assertEqual(failed['status'], 'policy_failed_to_reach_nc84')
        self.assertEqual(failed['prefix']['reason'], 'policy_loss')
        self.assertEqual(failed['nodes'], 0)
        self.assertIsNone(failed['deepest_remaining'])
        timeout = generate_attempt(deck, TwoCardPolicy(), lambda env: (-np.arange(393), 0.),
                                   TeacherConfig(timeout_seconds_per_deck=1e-12), index=0,
                                   deterministic=False, prefix_seed=7, search_seed=8, teacher=teacher)
        self.assertEqual(timeout['status'], 'teacher_failed_timeout')
        self.assertEqual(timeout['prefix']['status'], 'prefix_reached')
        self.assertEqual(timeout['deepest_remaining'], 86)
        self.assertEqual(timeout['max_cards_played'], 0)
        self.assertEqual(timeout['deepest_turn'], 0)
        self.assertEqual(timeout['frontier_size'], 1)
        self.assertIsNone(timeout['record'])

    def test_winning_attempt_saves_only_the_policy_checkpoint(self):
        from archive.privileged_teacher.build_bank import generate_attempt, BankJournal
        from src.natural.bank import NaturalBank
        from archive.privileged_teacher.teacher import TeacherConfig
        from pathlib import Path
        import tempfile
        teacher = dict(model_sha256='0'*64, config={}, code_sha256={'fixture': '1'*64})
        config = TeacherConfig(beam_width=1, candidate_turns_per_state=1,
                               policy_top_k_actions=1, stochastic_candidates=0)
        row = generate_attempt(fixture()[0], TwoCardPolicy(), lambda env: (-np.arange(393), 0.),
                               config, index=0, deterministic=True, prefix_seed=7, search_seed=8,
                               teacher=teacher)
        self.assertEqual(row['status'], 'win_found')
        self.assertEqual([c['bucket'] for c in row['record']['checkpoints']], [84])
        self.assertEqual(row['record']['prefix_action_offset'], len(row['prefix']['actions']))
        validate_record(row['record'])
        with tempfile.TemporaryDirectory() as td:
            with BankJournal(td, {'schema_version': 2}, resume=False) as journal:
                journal.append(row)
                summary = journal.export()
            bank = NaturalBank(Path(td)/'bank.jsonl')
            self.assertEqual(len(bank.deck_ids), 1)
            self.assertEqual(summary['policy_reached_nc84'], 1)
            self.assertEqual(summary['searched_decks'], 1)
            self.assertEqual(summary['pilot']['status'], 'incomplete_pilot')
            bad = copy.deepcopy(row)
            bad.pop('frontier_size')
            with tempfile.TemporaryDirectory() as bad_dir:
                with BankJournal(bad_dir, {'schema_version': 2}, resume=False) as journal:
                    with self.assertRaises(ValueError):
                        journal.append(bad)

    def test_pilot_gate_and_estimates(self):
        from archive.privileged_teacher.build_bank import pilot_report
        def rows(wins):
            return [dict(status='win_found' if i < wins else 'teacher_failed_timeout',
                         attempt_seconds=25.) for i in range(100)]
        self.assertEqual(pilot_report(rows(0))['status'], 'teacher_redesign_required')
        self.assertFalse(pilot_report(rows(2))['minimum_witnesses_met'])
        self.assertFalse(pilot_report(rows(3))['scale_estimate_ready'])
        report = pilot_report(rows(10))
        self.assertTrue(report['minimum_witnesses_met'])
        self.assertTrue(report['scale_estimate_ready'])
        self.assertEqual(report['estimated_decks_for_5000'], 50000)
        self.assertEqual(report['estimated_search_hours_for_5000'], 50000*25/3600)
        self.assertEqual(pilot_report(rows(20))['estimated_decks_for_5000'], 25000)
        self.assertEqual(pilot_report(rows(20)[:99])['status'], 'incomplete_pilot')


if __name__ == '__main__':
    unittest.main()

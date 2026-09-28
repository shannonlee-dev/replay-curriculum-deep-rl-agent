import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.natural.bank import NaturalBank, natural_start, checkpoint_from_env, validate_record
from archive.privileged_teacher.teacher import TeacherConfig
from test_natural_bank import fixture
from test_natural_prefix import TwoCardPolicy


class HandoffTests(unittest.TestCase):
    def test_continuation_keeps_real_deck_and_public_observations(self):
        from archive.handoff_v3.build_bank import rollout_trajectory
        policy = TwoCardPolicy()
        result = rollout_trajectory(fixture()[0], policy)
        self.assertEqual(result['prefix']['remaining'], 86)
        self.assertEqual(result['stop_reason'], 'depth_reached')
        self.assertEqual(result['remaining'], 48)
        self.assertEqual([c['target'] for c in result['checkpoints']], list(range(80, 47, -4)))
        env = natural_start(fixture()[0])
        by_offset = {c['checkpoint']['action_offset']: c['checkpoint'] for c in result['checkpoints']}
        for i, action in enumerate(result['actions']):
            np.testing.assert_array_equal(policy.observations[i], env._get_obs())
            self.assertEqual(policy.observations[i].shape, (300,))
            env.step(action)
            if i+1 in by_offset:
                self.assertEqual(checkpoint_from_env(env, i+1), by_offset[i+1])
        self.assertEqual([c['checkpoint']['remaining_count'] for c in result['checkpoints']], list(range(80, 47, -4)))

    def test_skipped_boundaries_are_not_counted_as_reached(self):
        from archive.handoff_v3.build_bank import rollout_trajectory
        result = rollout_trajectory(fixture()[0], TwoCardPolicy(turn_size=8))
        self.assertEqual(result['prefix']['remaining'], 82)
        self.assertEqual(result['stop_reason'], 'depth_reached')
        self.assertEqual(result['remaining'], 42)
        self.assertEqual([c['target'] for c in result['checkpoints']], [72, 64, 56, 48])
        self.assertEqual([c['checkpoint']['remaining_count'] for c in result['checkpoints']], [74, 66, 58, 50])

    def test_policy_death_rewinds_to_live_handoff_and_replays_nc84_win(self):
        from archive.handoff_v3.build_bank import generate_attempt, V3Journal, handoff_report
        class DiesAfter60(TwoCardPolicy):
            def predict(self, obs, **kwargs):
                if int(obs[196:294].sum()) >= 38 and obs[-2] == 0:
                    return 392, None
                return super().predict(obs, **kwargs)
        row = self.attempt(DiesAfter60())
        self.assertEqual(row['trajectory']['stop_reason'], 'policy_loss')
        self.assertEqual(row['selected_handoff'], 60)
        self.assertEqual(row['winning_handoff'], 60)
        self.assertEqual(row['status'], 'win_found')
        record = row['record']
        validate_record(record)
        self.assertEqual(record['checkpoints'][0]['remaining_count'], 86)
        offset = row['handoffs'][0]['action_offset']
        self.assertEqual(record['actions'][:offset], row['trajectory']['actions'][:offset])
        self.assertNotEqual(record['actions'][offset], 392)  # dying action was discarded
        with tempfile.TemporaryDirectory() as td:
            with V3Journal(td, {'schema_version': 3}, resume=False) as journal:
                journal.append(row)
                summary = journal.export()
            self.assertEqual(len(NaturalBank(Path(td)/'bank.jsonl').deck_ids), 1)
            self.assertEqual(summary['handoff']['selected_distribution']['60']['count'], 1)
            with V3Journal(td, {'schema_version': 3}, resume=True) as journal:
                self.assertEqual(journal.next_index, 1)
        bad = copy.deepcopy(row)
        bad['trajectory']['checkpoints'][-1]['checkpoint']['future_deck_order'].reverse()
        with tempfile.TemporaryDirectory() as td:
            with V3Journal(td, {'schema_version': 3}, resume=False) as journal:
                with self.assertRaises(ValueError):
                    journal.append(bad)

    @staticmethod
    def attempt(policy=None, config=None):
        from archive.handoff_v3.build_bank import generate_attempt
        return generate_attempt(fixture()[0], policy or TwoCardPolicy(),
            lambda env: (-np.arange(393, dtype=float), 0.),
            config or TeacherConfig(beam_width=1, candidate_turns_per_state=1,
                policy_top_k_actions=1, stochastic_candidates=0, timeout_seconds_per_deck=10),
            index=0, deterministic=True, prefix_seed=7, search_seed=8,
            teacher=dict(model_sha256='0'*64, config={}, code_sha256={'fixture': '1'*64}))

    def test_budget_timeout_and_node_limit_do_not_fabricate_witness(self):
        from archive.handoff_v3.build_bank import handoff_report
        for config, status in [(TeacherConfig(timeout_seconds_per_deck=1e-12), 'teacher_failed_timeout'),
                               (TeacherConfig(max_nodes_per_deck=1), 'teacher_failed_node_limit')]:
            row = self.attempt(config=config)
            self.assertEqual(row['status'], status)
            self.assertIsNone(row['record'])
            self.assertEqual(row['selected_handoff'], 48)
            self.assertEqual(len(row['handoffs']), 1)
            self.assertLessEqual(row['nodes'], config.max_nodes_per_deck)
            report = handoff_report([row])
            self.assertEqual(report['by_handoff']['48'][status], 1)
            self.assertEqual(report['reach']['80']['rate_all_decks'], 1.)
            self.assertEqual(report['nc84_witness_rate'], 0.)

    def test_exhausted_falls_back_with_remaining_budget_and_rebuilds_prefix(self):
        from unittest.mock import patch
        from archive.privileged_teacher.teacher import SearchResult, search_checkpoint
        calls = []
        def first_exhausts(cp, prior, config, seed):
            calls.append((cp['remaining_count'], config.max_nodes_per_deck))
            if len(calls) == 1:
                return SearchResult('teacher_failed_exhausted', [], 17, 0., 48, 0, 0, 0)
            return search_checkpoint(cp, prior, config, seed)
        with patch('archive.handoff_v3.build_bank.search_checkpoint', side_effect=first_exhausts):
            row = self.attempt(config=TeacherConfig(beam_width=1, candidate_turns_per_state=1,
                policy_top_k_actions=1, stochastic_candidates=0, max_nodes_per_deck=5000))
        self.assertEqual(calls, [(48, 5000), (52, 4983)])
        self.assertEqual(row['selected_handoff'], 48)
        self.assertEqual(row['winning_handoff'], 52)
        self.assertEqual([h['status'] for h in row['handoffs']], ['teacher_failed_exhausted', 'win_found'])
        validate_record(row['record'])
        self.assertEqual(row['record']['handoff']['checkpoint']['remaining_count'], 52)
        offset = row['record']['handoff']['policy_action_offset']
        self.assertEqual(row['record']['actions'][:offset], row['trajectory']['actions'][:offset])

    def test_future_order_does_not_leak_before_draw(self):
        from archive.handoff_v3.build_bank import rollout_trajectory
        a, b = fixture()[0], fixture()[0]
        b[60:] = reversed(b[60:])
        policies = [TwoCardPolicy(), TwoCardPolicy()]
        trajectories = [rollout_trajectory(deck, policy) for deck, policy in zip((a, b), policies)]
        self.assertEqual(trajectories[0]['actions'], trajectories[1]['actions'])
        for first, second in zip(policies[0].observations, policies[1].observations):
            np.testing.assert_array_equal(first, second)
        self.assertNotEqual(trajectories[0]['checkpoints'][-1]['checkpoint']['future_deck_order'],
                            trajectories[1]['checkpoints'][-1]['checkpoint']['future_deck_order'])

    def test_nc84_reached_but_no_handoff_is_distinct(self):
        class DiesAtNC84(TwoCardPolicy):
            def predict(self, obs, **kwargs):
                if int(obs[196:294].sum()) >= 12 and obs[-2] == 0:
                    return 392, None
                return super().predict(obs, **kwargs)
        row = self.attempt(DiesAtNC84())
        self.assertEqual(row['prefix']['status'], 'prefix_reached')
        self.assertEqual(row['status'], 'policy_failed_to_reach_handoff')
        self.assertEqual(row['handoffs'], [])
        self.assertIsNone(row['record'])

    def test_prefix_failure_never_runs_teacher(self):
        class IllegalPolicy:
            def predict(self, obs, **kwargs):
                return 392, None
        row = self.attempt(IllegalPolicy())
        self.assertEqual(row['status'], 'policy_failed_to_reach_nc84')
        self.assertEqual(row['handoffs'], [])
        self.assertEqual(row['nodes'], 0)


if __name__ == '__main__':
    unittest.main()

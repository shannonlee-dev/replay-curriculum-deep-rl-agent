import unittest

import numpy as np

from src.natural.bank import build_record, natural_start, validate_record
from test_natural_bank import fixture


class TeacherTests(unittest.TestCase):
    def test_winning_search_with_hand_authored_prior(self):
        from archive.privileged_teacher.teacher import TeacherConfig, search
        # A deterministic test prior, not a neural model or a production bank run.
        def prior(env):
            scores = np.full(393, -1000.)
            for action in np.flatnonzero(env.action_masks()):
                scores[action] = -float(action)
            return scores, 0.
        deck, _ = fixture()
        result = search(deck, prior, TeacherConfig(beam_width=1, candidate_turns_per_state=1,
                        policy_top_k_actions=1, stochastic_candidates=0,
                        max_nodes_per_deck=5000, timeout_seconds_per_deck=10), seed=0)
        self.assertEqual(result.status, 'win_found')
        validate_record(build_record(deck, result.actions))
        self.assertGreater(result.nodes, 98)

    def test_search_limits_do_not_claim_loss_or_unwinnable(self):
        from archive.privileged_teacher.teacher import TeacherConfig, search
        deck, _ = fixture()
        prior = lambda env: (np.zeros(393), 0.)
        for config, reason in [(TeacherConfig(max_nodes_per_deck=1), 'teacher_failed_node_limit'),
                               (TeacherConfig(timeout_seconds_per_deck=1e-12), 'teacher_failed_timeout')]:
            result = search(deck, prior, config)
            self.assertEqual(result.status, reason)
            self.assertFalse(result.actions)
            self.assertLessEqual(result.nodes, config.max_nodes_per_deck)

    def test_symmetry_key_keeps_hand_and_draw_position(self):
        from archive.privileged_teacher.teacher import state_key, clone_env
        a = natural_start(fixture()[0])
        a.piles[:] = [5, 9, 91, 95]
        b = clone_env(a)
        b.piles[:] = [9, 5, 95, 91]
        self.assertEqual(state_key(a), state_key(b))
        b.hand.remove(2)
        self.assertNotEqual(state_key(a), state_key(b))
        self.assertIn(2, a.hand)
        self.assertEqual(a.piles.tolist(), [5,9,91,95])

    def test_invalid_search_config(self):
        from archive.privileged_teacher.teacher import TeacherConfig
        for kwargs in [dict(beam_width=0), dict(stochastic_candidates=-1),
                       dict(timeout_seconds_per_deck=float('nan')), dict(policy_top_k_actions=True)]:
            with self.assertRaises(ValueError):
                TeacherConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()

import math
import unittest

from src.natural.bank import natural_start, build_record, validate_record
from test_natural_bank import fixture
import archive.privileged_teacher.teacher as natural_teacher
class DietrichTests(unittest.TestCase):
    def test_exponential_counts_all_unplayed_including_backward(self):
        from archive.handoff_v3.teacher import penalty_score
        env = natural_start(fixture()[0])
        env.piles[:] = [50, 60, 40, 30]
        env.hand = {40}
        env.deck = [50, 55]
        env.played = set(range(2, 100)) - {40, 50, 55}
        self.assertAlmostEqual(penalty_score(env), -(math.exp(-1.5)+2*math.exp(-3)))
        env.deck.reverse()
        self.assertAlmostEqual(penalty_score(env), -(math.exp(-1.5)+2*math.exp(-3)))

    def test_recovery_softens_near_blocked_cards(self):
        from archive.handoff_v3.teacher import pile_penalty
        self.assertEqual(pile_penalty(40, 50, True, recovery=True), 1)
        self.assertLess(pile_penalty(49, 50, True, recovery=True),
                        pile_penalty(20, 50, True, recovery=True))
        self.assertGreaterEqual(pile_penalty(49, 50, True, recovery=True), 1)
        self.assertEqual(pile_penalty(60, 50, False, recovery=True), 1)

    def test_rollout_witness_and_budget(self):
        from archive.handoff_v3.teacher import solve
        from src.natural.bank import checkpoint_from_env
        deck, _ = fixture()
        cp = checkpoint_from_env(natural_start(deck), 0)
        config = natural_teacher.TeacherConfig(timeout_seconds_per_deck=10)
        result = solve(cp, None, config, mode='rollout')
        self.assertEqual(result.status, 'win_found')
        validate_record(build_record(deck, result.actions))
        tiny = natural_teacher.TeacherConfig(max_nodes_per_deck=1)
        result = solve(cp, None, tiny, mode='search')
        self.assertEqual(result.status, 'teacher_failed_node_limit')
        self.assertEqual(result.nodes, 1)

class FixedCohortTests(unittest.TestCase):
    def test_frozen_seed_rejects_different_permutation(self):
        import numpy as np
        from archive.handoff_v3.evaluate import frozen_seeds
        rng = np.random.default_rng(np.random.SeedSequence([730000, 0]))
        deck = rng.permutation(np.arange(2, 100)).tolist()
        prefix, search = (int(rng.integers(2**32)) for _ in range(2))
        row = dict(index=0, full_permutation=deck, prefix=dict(seed=prefix))
        self.assertEqual(frozen_seeds(row, 730000), (prefix, search))
        row['full_permutation'] = list(reversed(deck))
        with self.assertRaises(ValueError):
            frozen_seeds(row, 730000)

    def test_frozen_trajectory_bypasses_policy_and_replays_win(self):
        from archive.handoff_v3.build_bank import generate_attempt, rollout_trajectory
        from test_natural_prefix import TwoCardPolicy
        from archive.handoff_v3.teacher import solve
        from functools import partial
        import copy
        deck, _ = fixture()
        trajectory = rollout_trajectory(deck, TwoCardPolicy())
        before = copy.deepcopy(trajectory)
        result = generate_attempt(deck, None, None, natural_teacher.TeacherConfig(),
            index=0, deterministic=True, prefix_seed=0, search_seed=0,
            teacher=dict(model_sha256='0'*64, config={}, code_sha256={'fixture':'1'*64}),
            trajectory=trajectory, search_fn=partial(solve, mode='rollout'))
        self.assertEqual(result['status'], 'win_found')
        validate_record(result['record'])
        self.assertEqual(trajectory, before)
        self.assertEqual(result['record']['actions'][:trajectory['checkpoints'][-1]['checkpoint']['action_offset']],
                         trajectory['actions'])

import unittest
import numpy as np
from src.env import TheGameEnv


class ReplayEnvironmentTests(unittest.TestCase):
    def test_sampling_reproducible_and_episode_info_stable(self):
        kwargs = dict(reset_source='reverse', curriculum_targets=[12, 20, 32],
                      curriculum_probs=[.1, .3, .6])
        a, b = TheGameEnv(**kwargs), TheGameEnv(**kwargs)
        samples = []
        for i in range(120):
            oa, ia = a.reset(seed=123 if i == 0 else None)
            ob, ib = b.reset(seed=123 if i == 0 else None)
            np.testing.assert_array_equal(oa, ob)
            self.assertEqual(ia, ib)
            t = ia['sampled_target_remaining']
            samples.append(t)
            self.assertEqual(a._remaining_cards(), t)
            self.assertEqual(a.played_this_turn, 0)
            _, _, _, _, info = a.step(int(np.flatnonzero(a.action_masks())[0]))
            b.step(int(np.flatnonzero(b.action_masks())[0]))
            self.assertEqual(info['sampled_target_remaining'], t)
        self.assertEqual(set(samples), {12, 20, 32})
        self.assertGreater(samples.count(32), samples.count(12))

    def test_invalid_distributions(self):
        for targets, probs in [([13], [1]), ([20, 32], [1]), ([20], [-1]),
                               ([20], [float('nan')]), ([20, 20], [.5, .5]),
                               ([20], [0]), ([], [])]:
            with self.subTest(targets=targets, probs=probs), self.assertRaises(ValueError):
                TheGameEnv(curriculum_targets=targets, curriculum_probs=probs)

    def test_natural_bridge_endpoints(self):
        for p, source, target in [(0., 'reverse', 20), (1., 'natural', 98)]:
            env = TheGameEnv(reset_source='mixed', mixed_natural_prob=p,
                            curriculum_targets=[20], curriculum_probs=[1.])
            _, info = env.reset(seed=42)
            self.assertEqual(info['reset_source'], source)
            self.assertEqual(info['sampled_target_remaining'], target)
            self.assertEqual(env._remaining_cards(), target)

    def test_no_future_order_in_observation(self):
        env = TheGameEnv(reset_source='reverse', target_remaining=32)
        obs, _ = env.reset(seed=7)
        env.deck.reverse()
        np.testing.assert_array_equal(obs, env._get_obs())

    def test_winning_continuation_every_even_target(self):
        # Same generated plan and seed let the test reconstruct the private witness.
        for target in range(2, 99, 2):
            for seed in (3, 19):
                witness = TheGameEnv()
                witness.reset(seed=seed)
                witness.np_random = np.random.default_rng(seed)
                plan = witness._generate_winning_plan()
                env = TheGameEnv(reset_source='reverse', target_remaining=target)
                env.reset(seed=seed)
                for card, pile in plan[98-target:]:
                    action = env.encode_action(card, pile)
                    self.assertTrue(env.action_masks()[action])
                    _, reward, done, _, info = env.step(action)
                    self.assertEqual(reward, float(done and info['won']))
                    if not done and env.played_this_turn == 2:
                        _, reward, done, _, info = env.step(env.END_TURN)
                        self.assertEqual(reward, 0.)
                self.assertTrue(info['won'])
                self.assertEqual(reward, 1.)

    def test_loss_reward_and_turn_rules(self):
        env = TheGameEnv()
        env.reset(seed=8)
        self.assertFalse(env.action_masks()[env.END_TURN])
        _, reward, done, _, info = env.step(env.END_TURN)
        self.assertTrue(done)
        self.assertFalse(info['won'])
        self.assertEqual(reward, 0.)
        env.reset(seed=8)
        env.piles[:] = [44, 44, 34, 34]
        self.assertTrue(env._legal_on_pile(34, 0))
        self.assertTrue(env._legal_on_pile(44, 2))
        self.assertFalse(env._legal_on_pile(33, 0))
        env.deck = []
        self.assertEqual(env._minimum_turn_play(), 1)


class CurriculumTests(unittest.TestCase):
    def test_distribution_and_overrides(self):
        from src.synthetic.curriculum import replay_distribution
        targets, probs = replay_distribution(32, {})
        self.assertAlmostEqual(sum(probs), 1.)
        self.assertIn(12, targets)
        self.assertIn(20, targets)
        self.assertGreater(probs[targets.index(32)], .25)
        self.assertEqual(replay_distribution(32, {'distributions': {'32': {'targets': [20,32], 'probs': [.4,.6]}}}), ([20,32], [.4,.6]))
        self.assertIn(96, replay_distribution(98, {})[0])

    def test_promotion_requires_all_win_rate_gates(self):
        from src.synthetic.curriculum import promotion_checks
        config = dict(current_min_win_rate=.6, previous_min_win_rate=.7, older_floor_win_rate=.5)
        rates = {12: .8, 20: .82, 24: .77, 28: .688, 32: .612}
        checks = promotion_checks(32, rates, [20,24,28,32], config)
        self.assertFalse(all(checks.values()))
        rates[28] = .71
        self.assertTrue(all(promotion_checks(32, rates, [20,24,28,32], config).values()))
        rates[20] = .49
        self.assertFalse(all(promotion_checks(32, rates, [20,24,28,32], config).values()))


if __name__ == '__main__':
    unittest.main()

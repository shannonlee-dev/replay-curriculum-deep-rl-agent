"""Replay records must reproduce the environment, including turn boundaries."""
import json
import unittest

import numpy as np

from src.tools.html_model import build_html, record_game
from src.env import TheGameEnv


class LegalPolicy:
    def __init__(self):
        self.rng = np.random.default_rng(15)

    def predict(self, obs, *, deterministic, action_masks):
        return int(self.rng.choice(np.flatnonzero(action_masks))), None


class HTMLReplayTests(unittest.TestCase):
    def test_records_reproduce_states_and_draws(self):
        for source, target in [('natural', 98), ('reverse', 20)]:
            with self.subTest(source=source):
                game = record_game(LegalPolicy(), source=source, target=target, seed=73)
                env = TheGameEnv(reset_source=source, target_remaining=target)
                try:
                    env.reset(seed=73)
                    turn = 1
                    for frame in game['frames']:
                        move = frame['move']
                        if move:
                            before = set(env.hand)
                            self.assertTrue(env.action_masks()[move['action']])
                            if move['kind'] == 'play':
                                card, pile = env.decode_action(move['action'])
                                self.assertEqual(move['before'], int(env.piles[pile]))
                                self.assertEqual(move['backward'], card == env.piles[pile] + (-10 if pile < 2 else 10))
                            _, _, done, _, _ = env.step(move['action'])
                            if move['kind'] == 'end' and not done:
                                turn += 1
                            self.assertEqual(move['drawn'], sorted(env.hand - before))
                        state = frame['state']
                        self.assertEqual(state['piles'], env.piles.tolist())
                        self.assertEqual(state['hand'], sorted(env.hand))
                        self.assertEqual(state['deck'], len(env.deck))
                        self.assertEqual(state['turn'], turn)
                        self.assertEqual(state['turn_played'], env.played_this_turn)
                        self.assertEqual(state['played'], env.agent_cards_played)
                        self.assertEqual(state['legal'], np.flatnonzero(env.action_masks()).tolist())
                        self.assertEqual(state['done'], env.terminated)
                    self.assertTrue(game['frames'][-1]['state']['done'])
                    self.assertEqual(game['won'], env.won)
                finally:
                    env.close()

    def test_data_is_embedded_without_script_injection(self):
        payload = {'model': '</script><script>alert(1)</script>&', 'games': []}
        html = build_html(payload)
        embedded = html.split('<script id="replay-data" type="application/json">')[1].split('</script>')[0]
        self.assertEqual(json.loads(embedded), payload)
        self.assertNotIn('<', embedded)
        self.assertNotIn('__REPLAY_DATA__', html)
        self.assertNotIn('/* REPLAY_SCRIPT */', html)
        self.assertNotIn('/* REPLAY_STYLE */', html)


if __name__ == '__main__':
    unittest.main()

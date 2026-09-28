import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.env import TheGameEnv


def fixture(turn_size=2, seed=0):
    """Hand-authored winning fixture, NOT a sampled production natural bank."""
    deck = list(range(2, 100))
    first = deck[:8]
    np.random.default_rng(seed).shuffle(first)
    deck[:8] = first
    actions = []
    for i, card in enumerate(range(2, 100), 1):
        actions.append(TheGameEnv.encode_action(card, 0))
        if i % turn_size == 0 and i != 98:
            actions.append(TheGameEnv.END_TURN)
    return deck, actions


def witness_record(permutation, actions):
    from src.natural.bank import build_record
    record = build_record(permutation, actions)
    record['teacher'] = dict(model_sha256=hashlib.sha256(b'no model').hexdigest(),
                             config={'beam_width': 1}, code_sha256={'fixture': '0'*64})
    record['search'] = dict(nodes=123, elapsed_seconds=1.)
    return record


class BankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import src.natural.bank as natural_bank
        cls.bank = natural_bank

    def test_full_replay_and_continuations(self):
        record = witness_record(*fixture())
        self.bank.validate_record(record)
        self.assertEqual([c['bucket'] for c in record['checkpoints']], [84, 88, 92, 96, 98])
        for checkpoint in record['checkpoints']:
            env = TheGameEnv()
            self.bank.restore_checkpoint(env, checkpoint)
            self.assertEqual(env.played_this_turn, 0)
            self.assertEqual(env._remaining_cards(), checkpoint['remaining_count'])
            for action in record['actions'][checkpoint['action_offset']:]:
                self.assertTrue(env.action_masks()[action])
                _, reward, done, _, info = env.step(action)
                self.assertEqual(reward, float(done and info['won']))
            self.assertTrue(env.won)
            self.assertEqual(reward, 1.)

    def test_odd_remaining_and_exact_nc98(self):
        record = witness_record(*fixture(turn_size=3))
        self.bank.validate_record(record)
        self.assertTrue(any(c['remaining_count'] % 2 for c in record['checkpoints']))
        self.assertEqual(self.bank.bucket_for(86), 84)
        self.assertEqual(self.bank.bucket_for(97), 96)
        self.assertEqual(self.bank.bucket_for(98), 98)
        self.assertIsNone(self.bank.bucket_for(81))
        full = next(c for c in record['checkpoints'] if c['bucket'] == 98)
        self.assertEqual((full['remaining_count'], full['action_offset']), (98, 0))

    def test_corrupt_prefix_suffix_metadata_and_split_rejected(self):
        original = witness_record(*fixture())
        mutations = [lambda r: r['checkpoints'][0]['pile_tops'].__setitem__(0, 50),
                     lambda r: r['actions'].__setitem__(-1, TheGameEnv.END_TURN),
                     lambda r: r.__setitem__('split', 'wrong'),
                     lambda r: r['checkpoints'][0].__setitem__('q', 1),
                     lambda r: r['checkpoints'].append(copy.deepcopy(r['checkpoints'][0])),
                     lambda r: r['full_permutation'].__setitem__(0, r['full_permutation'][1])]
        for mutate in mutations:
            record = copy.deepcopy(original)
            mutate(record)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.bank.validate_record(record)

    def test_observation_and_mask_hide_future_and_metadata(self):
        checkpoint = witness_record(*fixture())['checkpoints'][0]
        a, b = TheGameEnv(), TheGameEnv()
        self.bank.restore_checkpoint(a, checkpoint)
        self.bank.restore_checkpoint(b, checkpoint)
        b.deck.reverse()
        b.actual_reset_source = 'some_other_source'
        np.testing.assert_array_equal(a._get_obs(), b._get_obs())
        np.testing.assert_array_equal(a.action_masks(), b.action_masks())
        self.assertEqual(a._get_obs().shape, (300,))
        self.assertEqual(a.action_space.n, 393)
        _, reward, done, _, info = a.step(a.END_TURN)
        self.assertEqual((reward, done, info['won']), (0., True, False))

    def test_deck_split_deterministic_and_bank_duplicates_rejected(self):
        records = [witness_record(*fixture(seed=i)) for i in range(50)]
        ids = {s: {r['deck_id'] for r in records if r['split'] == s}
               for s in ['train', 'validation', 'test']}
        self.assertTrue(all(ids.values()))
        self.assertFalse(ids['train'] & ids['validation'] | ids['train'] & ids['test'])
        for record in records:
            self.assertEqual(record['split'], self.bank.split_for(record['deck_id']))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'bank.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
            bank = self.bank.NaturalBank(path)
            self.assertEqual(len(bank.deck_ids), 50)
            self.assertTrue(bank.checkpoints('train', 84))
            with path.open('a') as f:
                f.write(json.dumps(records[0])+'\n')
            with self.assertRaises(ValueError):
                self.bank.NaturalBank(path)

    def test_restored_state_matches_natural_prefix_masks_and_transitions(self):
        record = witness_record(*fixture(turn_size=3))
        for checkpoint in record['checkpoints']:
            original = self.bank.natural_start(record['full_permutation'])
            for action in record['actions'][:checkpoint['action_offset']]:
                original.step(action)
            restored = TheGameEnv()
            self.bank.restore_checkpoint(restored, checkpoint)
            for action in record['actions'][checkpoint['action_offset']:]:
                np.testing.assert_array_equal(original._get_obs(), restored._get_obs())
                np.testing.assert_array_equal(original.action_masks(), restored.action_masks())
                left, right = original.step(action), restored.step(action)
                self.assertEqual(left[1:4], right[1:4])
                self.assertEqual(left[4]['won'], right[4]['won'])

    def test_source_mixture_and_transition_counters(self):
        from archive.nc84_direct_mix.env import NaturalConditionedEnv, SourceUsageWrapper
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'bank.jsonl'
            records = [witness_record(*fixture(seed=i)) for i in range(10)]
            path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
            env = SourceUsageWrapper(NaturalConditionedEnv(bank=self.bank.NaturalBank(path)))
            counts = {}
            for i in range(1600):
                _, info = env.reset(seed=123 if i == 0 else None)
                key = 'NC84' if info['reset_source'] == 'natural_conditioned' else f"R{info['sampled_target_remaining']}"
                counts[key] = counts.get(key, 0)+1
                mask = env.unwrapped.action_masks()
                env.step(int(np.flatnonzero(mask)[0]))
            expected = dict(NC84=.30, R84=.25, R80=.15, R76=.09, R72=.06, R20=.10, R12=.05)
            self.assertEqual(set(counts), set(expected))
            for key, probability in expected.items():
                self.assertAlmostEqual(counts[key]/1600, probability, delta=.045)
            self.assertEqual(sum(env.source_episodes.values()), 1600)
            self.assertEqual(env.source_transitions, env.source_episodes)

    def test_missing_or_mixed_teacher_provenance_rejected(self):
        first, second = [witness_record(*fixture(seed=i)) for i in (0, 1)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'bank.jsonl'
            second['teacher']['model_sha256'] = 'a'*64
            path.write_text('\n'.join(json.dumps(r) for r in (first, second))+'\n')
            with self.assertRaises(ValueError):
                self.bank.NaturalBank(path)
            first.pop('teacher')
            path.write_text(json.dumps(first)+'\n')
            with self.assertRaises(ValueError):
                self.bank.NaturalBank(path)

    def test_reset_reproducible_independent_and_transition_equivalence(self):
        from archive.nc84_direct_mix.env import NaturalConditionedEnv
        records = [witness_record(*fixture(seed=i)) for i in range(10)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'bank.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
            bank = self.bank.NaturalBank(path)
            a = NaturalConditionedEnv(bank=bank, nc_probability=1.)
            b = NaturalConditionedEnv(bank=bank, nc_probability=1.)
            oa, ia = a.reset(seed=42)
            ob, ib = b.reset(seed=42)
            np.testing.assert_array_equal(oa, ob)
            self.assertEqual(ia, ib)
            self.assertNotIn('deck_id', ia)
            self.assertEqual(ia['reset_source'], 'natural_conditioned')
            for _ in range(3):
                action = min(np.flatnonzero(a.action_masks()))
                ta, tb = a.step(action), b.step(action)
                np.testing.assert_array_equal(ta[0], tb[0])
                self.assertEqual(ta[1:], tb[1:])
            a.deck.clear()
            self.assertTrue(b.deck)
            with self.assertRaises(ValueError):
                NaturalConditionedEnv(bank=bank, split='test')


if __name__ == '__main__':
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

STAGES = (12, 20, 32, 48, 64, 72, 80, 84)
REHEARSAL = dict(current_weight=.7, older_weight=.3)

from src.natural.bank import natural_start
from test_natural_bank import fixture
from test_nc_direct_mix_audit import make_inputs


class NaturalReverseTests(unittest.TestCase):
    def test_checkpoints_are_real_states_on_same_winning_deck_and_same_split(self):
        from src.natural.trainer import NaturalReverseBank
        with tempfile.TemporaryDirectory() as tmp:
            _, path, record = make_inputs(Path(tmp))
            bank = NaturalReverseBank(path, STAGES)
            for target in STAGES:
                states = bank.checkpoints('train', target)
                self.assertEqual(len(states), 1)
                identity, cp = states[0]
                self.assertEqual(identity, record['deck_id'])
                self.assertLessEqual(abs(cp['remaining_count']-target), 2)
                original = natural_start(record['full_permutation'])
                for action in record['actions'][:cp['action_offset']]:
                    original.step(action)
                self.assertEqual(sorted(original.hand), cp['hand'])
                self.assertEqual(original.deck, cp['future_deck_order'])
                self.assertEqual(original.played_this_turn, 0)
                for action in record['actions'][cp['action_offset']:]:
                    self.assertTrue(original.action_masks()[action])
                    original.step(action)
                self.assertTrue(original.won)
                self.assertEqual(bank.checkpoints('validation', target), [])

    def test_natural_reset_uses_easy_stage_not_nc84(self):
        from src.natural.trainer import NaturalReverseBank, NaturalReverseEnv
        with tempfile.TemporaryDirectory() as tmp:
            _, path, _ = make_inputs(Path(tmp))
            env = NaturalReverseEnv(bank=NaturalReverseBank(path, STAGES), target=12, nc_episode_probability=1., rehearsal=REHEARSAL)
            obs, info = env.reset(seed=42)
            self.assertEqual(info['reset_source'], 'natural_conditioned')
            self.assertTrue(10 <= info['remaining_cards'] <= 14)
            self.assertEqual(obs.shape, (300,))
            self.assertEqual(env.action_space.n, 393)
            self.assertEqual(info['sampled_target_remaining'], info['remaining_cards'])
            env.close()

    def test_episode_mix_and_natural_previous_stage_replay(self):
        from src.natural.trainer import NaturalReverseBank, NaturalReverseEnv
        with tempfile.TemporaryDirectory() as tmp:
            _, path, _ = make_inputs(Path(tmp))
            env = NaturalReverseEnv(bank=NaturalReverseBank(path, STAGES), target=20, nc_episode_probability=.3, rehearsal=REHEARSAL)
            for probability in (.3, .6):
                env.nc_episode_probability = probability
                counts = {'natural_conditioned': 0, 'reverse': 0}
                targets = set()
                for index in range(1000):
                    _, info = env.reset(seed=index)
                    counts[info['reset_source']] += 1
                    if info['reset_source'] == 'natural_conditioned':
                        targets.add(info['remaining_cards'])
                self.assertAlmostEqual(counts['natural_conditioned']/1000, probability, delta=.05)
                self.assertTrue(any(abs(t-12) <= 2 for t in targets))
                self.assertTrue(any(abs(t-20) <= 2 for t in targets))
            env.close()

    def test_current_stage_win_counter_counts_real_terminal_reward(self):
        from src.natural.trainer import NaturalReverseBank, NaturalReverseEnv, WinUsageWrapper
        with tempfile.TemporaryDirectory() as tmp:
            _, path, record = make_inputs(Path(tmp))
            bank = NaturalReverseBank(path, STAGES)
            wrapped = WinUsageWrapper(NaturalReverseEnv(bank=bank, target=12, nc_episode_probability=1., rehearsal=REHEARSAL))
            wrapped.reset(seed=42)
            cp = bank.checkpoints('train', 12)[0][1]
            for action in record['actions'][cp['action_offset']:]:
                result = wrapped.step(action)
            self.assertTrue(result[2])
            self.assertEqual(result[1], 1.)
            self.assertEqual(wrapped.source_completed, {'natural_conditioned': 1})
            self.assertEqual(wrapped.source_wins, {'natural_conditioned': 1})
            self.assertEqual(wrapped.source_completed_transitions,
                             {'natural_conditioned': len(record['actions'])-cp['action_offset']})
            wrapped.close()

    def test_stage_transition_and_stall_with_fake_learner_no_model(self):
        from unittest.mock import patch
        from test_natural_bank import witness_record
        from src.natural.trainer import NaturalReverseBank, run
        from test_natural_pipeline import training_config as load_config

        class FakeLearner:
            num_timesteps = 100
            def learn(self, *, total_timesteps, reset_num_timesteps, callback):
                self.num_timesteps += total_timesteps
            def set_env(self, env):
                self.env = env
            def save(self, path):
                Path(str(path)+'.zip').write_bytes(b'fixture, not a model')

        def evaluate(*args, bucket, **kwargs):
            return dict(wins=0 if bucket == 32 else 4, games=4,
                        win_rate=0. if bucket == 32 else 1.)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank_path = root/'bank.jsonl'
            bank_path.write_text(''.join(json.dumps(witness_record(*fixture(seed=i)))+'\n'
                                         for i in range(50)))
            bank = NaturalReverseBank(bank_path, STAGES)
            model = root/'r80.zip'
            model.write_bytes(b'fixture')
            model.with_suffix('.json').write_text(json.dumps(dict(current_stage='R80',
                status='promoted', global_steps=100)))
            c = load_config()
            c.update(num_envs=2, min_steps_per_stage=64, max_steps_per_stage=64, eval_interval=64)
            c['ppo'].update(n_steps=32, batch_size=64, n_epochs=1)
            c['natural_curriculum']['evaluation'].update(nc_games=4, natural_large_games=4)
            out = root/'experiment'
            with patch('src.common.checkpoints.load_or_create', return_value=FakeLearner()) as load, \
                 patch('src.evaluation.natural.evaluate_nc', side_effect=evaluate), \
                 patch('src.evaluation.natural.evaluate_natural', return_value={'wins': 0, 'games': 4}) as natural, \
                 patch('src.evaluation.natural.evaluate_reverse_panel', return_value={}):
                result = run(c, model, bank, out, stop_at=20)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(result['current_stage'], 'NC20')
            self.assertEqual(result['status'], 'stalled')
            self.assertEqual(result['total_added_steps'], 128)
            self.assertTrue((out/'stage_NC12_complete.zip').exists())
            self.assertTrue((out/'stage_NC20_stalled.zip').exists())
            self.assertGreaterEqual(natural.call_count, 3)
            self.assertEqual(natural.call_args.kwargs['forbidden_decks'], bank.deck_ids)
            self.assertEqual(natural.call_args.kwargs['games'], 4)
            final = json.loads((out/'final_held_out.json').read_text())
            self.assertEqual(final['results']['natural_full'], {'wins': 0, 'games': 4})
            self.assertEqual(final['evaluation_protocol']['final_objective_diagnostic'],
                             'unconditional_natural_full_game_terminal_win_rate')
            self.assertIn('NC20_test', final['results'])

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from src.natural.bank import NaturalBank
from test_natural_bank import fixture, witness_record as build_record


class LegalTestPolicy:
    """No learned parameters or ML inference. Only a deterministic legal action."""
    def predict(self, obs, *, deterministic, action_masks):
        return int(np.flatnonzero(action_masks)[0]), None


class NCTrainingTests(unittest.TestCase):
    def bank(self, directory):
        path = Path(directory)/'bank.jsonl'
        records = [build_record(*fixture(seed=i)) for i in range(50)]
        path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
        return NaturalBank(path)

    def test_nc_evaluation_unique_decks_and_sparse_rewards(self):
        from src.evaluation.natural import evaluate_nc
        with tempfile.TemporaryDirectory() as td:
            bank = self.bank(td)
            result = evaluate_nc(LegalTestPolicy(), bank, split='test', bucket=84, games=1000)
            self.assertEqual(result['games'], len(bank.checkpoints('test', 84)))
            self.assertEqual(result['wins'], result['games'])
            self.assertEqual(result['unique_decks'], result['games'])
            self.assertGreater(result['games'], 0)

    def test_natural_eval_rejects_bank_overlap_before_policy(self):
        from src.evaluation.natural import evaluate_natural, natural_permutation
        from src.natural.bank import deck_id
        identity = deck_id(natural_permutation(73))
        with self.assertRaises(ValueError):
            evaluate_natural(LegalTestPolicy(), games=1, seed=73, forbidden_decks={identity})

    def test_schedule_relative_steps_and_final_eval(self):
        from archive.nc84_direct_mix.trainer import natural_eval_games
        self.assertEqual(natural_eval_games(0, 0, False, False), 1000)
        self.assertEqual(natural_eval_games(1048576, 0, False, False), 10000)
        self.assertEqual(natural_eval_games(1310720, 1048576, False, False), 1000)
        self.assertEqual(natural_eval_games(2097152, 1048576, False, False), 10000)
        self.assertEqual(natural_eval_games(262144, 0, True, False), 10000)
        self.assertEqual(natural_eval_games(2998272, 2097152, False, True), 10000)

    def test_configuration_keeps_gate_and_source_probabilities(self):
        from archive.nc84_direct_mix.trainer import load_config, validate_experiment, PANEL
        from archive.nc84_direct_mix.env import NC_SYNTHETIC_WEIGHTS, CONTROL_WEIGHTS
        c = load_config()
        validate_experiment(c)
        self.assertEqual(c['promotion'], dict(current_min_win_rate=.6, previous_min_win_rate=.6,
                                              older_floor_win_rate=.5))
        self.assertAlmostEqual(sum(NC_SYNTHETIC_WEIGHTS)+.30, 1.)
        self.assertAlmostEqual(sum(CONTROL_WEIGHTS), 1.)
        self.assertTrue({64, 68, 72, 76, 80, 84, 98}.issubset(PANEL))
        c['experiment']['nc_probability'] = .8
        with self.assertRaises(ValueError):
            validate_experiment(c)

    def test_training_control_flow_with_fake_learner_no_model_execution(self):
        from archive.nc84_direct_mix.trainer import run, load_config
        class FakeLearner(LegalTestPolicy):
            num_timesteps = 19398656
            def learn(self, *, total_timesteps, reset_num_timesteps):
                self.num_timesteps += total_timesteps
            def save(self, path):
                Path(str(path)+'.zip').write_bytes(b'test double, NOT a model')
        def reverse_panel(*args, **kwargs):
            from archive.nc84_direct_mix.trainer import PANEL
            return {f'R{t}': dict(wins=49 if t == 64 else 90, games=100, win_rate=.49 if t == 64 else .9) for t in PANEL}
        def natural(*args, **kwargs):
            return dict(wins=0, games=kwargs['games'], win_rate=0.)
        with tempfile.TemporaryDirectory() as td:
            bank = self.bank(td)
            checkpoint = Path(td)/'fixture.zip'
            checkpoint.write_bytes(b'no model')
            checkpoint.with_suffix('.json').write_text(json.dumps(dict(current_stage='R80', status='promoted',
                                                                       global_steps=19398656)))
            c = load_config()
            c.update(num_envs=2, max_steps_per_stage=64, min_steps_per_stage=64, eval_interval=64)
            c['ppo'].update(n_steps=32, batch_size=64, n_epochs=1)
            c['experiment']['minimum_winning_decks'] = 1
            c['output_dir'] = str(Path(td)/'run')
            with patch('archive.nc84_direct_mix.trainer.load_or_create', side_effect=lambda *args: FakeLearner()), \
                 patch('archive.nc84_direct_mix.trainer.evaluate_reverse_panel', side_effect=reverse_panel), \
                 patch('archive.nc84_direct_mix.trainer.evaluate_natural', side_effect=natural):
                result = run(c, checkpoint, bank, arm='nc')
                self.assertEqual(result['status'], 'stalled')
                self.assertEqual(result['stage_steps'], 64)
                self.assertFalse(result['checks']['older_floor'])
                self.assertTrue(result['checks']['current'])
                self.assertTrue(result['checks']['previous'])
                out = Path(c['output_dir'])
                rows = [json.loads(s) for s in (out/'evaluations.jsonl').read_text().splitlines()]
                self.assertEqual([r['stage_steps'] for r in rows], [0, 64])
                self.assertEqual([r['results']['natural_full']['games'] for r in rows], [1000, 10000])
                self.assertTrue((out/'final_held_out.json').exists())
                with self.assertRaises(FileExistsError):
                    run(c, checkpoint, bank, arm='nc')
                c['output_dir'] = str(Path(td)/'control')
                control = run(c, checkpoint, bank, arm='control')
                self.assertEqual(control['stage_steps'], 64)
                manifest = json.loads((Path(c['output_dir'])/'manifest.json').read_text())
                self.assertEqual(manifest['reset_probabilities']['R84'], .35)
                self.assertNotIn('NC84', manifest['reset_probabilities'])


if __name__ == '__main__':
    unittest.main()

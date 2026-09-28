import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

STAGES = (12, 20, 32, 48, 64, 72, 80, 84)
REHEARSAL = dict(current_weight=.7, older_weight=.3)

from src.natural.bank import file_sha256


class FakeLearner:
    def __init__(self, steps=100, marker='selected'):
        self.num_timesteps = steps
        self.marker = marker
        self.learned = []
    def learn(self, *, total_timesteps, reset_num_timesteps, callback):
        self.learned.append(total_timesteps)
        self.num_timesteps += total_timesteps
    def set_env(self, env):
        pass
    def save(self, path):
        Path(str(path)+'.zip').write_text(json.dumps({'steps': self.num_timesteps, 'marker': self.marker}))


class NaturalResumeTests(unittest.TestCase):
    def test_multiple_step_snapshots_are_preserved_and_selected_explicitly(self):
        from src.natural.trainer import save_checkpoint, read_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            learner = FakeLearner(164, 'old')
            metadata = dict(checkpoint_kind='natural_reverse', current_stage='NC20',
                status='checkpoint', global_steps=164, stage_steps=64, total_added_steps=64,
                origin_global_steps=100, bank_sha256='a'*64, config={})
            first = save_checkpoint(learner, out, metadata)
            original = first.read_bytes()
            learner.num_timesteps = 228
            learner.marker = 'new'
            second = save_checkpoint(learner, out, dict(metadata, global_steps=228,
                                     stage_steps=128, total_added_steps=128))
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), original)
            self.assertEqual(read_checkpoint(first)['global_steps'], 164)
            self.assertEqual(read_checkpoint(out/'latest.zip')['global_steps'], 228)
            first.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash'):
                read_checkpoint(first)

    def test_resume_preserves_stage_step_and_budget_from_older_checkpoint(self):
        from src.natural.trainer import NaturalReverseBank, run, save_checkpoint
        from test_natural_pipeline import training_config as load_config
        from test_natural_bank import fixture, witness_record
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank_file = root/'bank.jsonl'
            bank_file.write_text(''.join(json.dumps(witness_record(*fixture(seed=i)))+'\n' for i in range(50)))
            bank = NaturalReverseBank(bank_file, STAGES)
            c = load_config()
            c.update(num_envs=2, min_steps_per_stage=64, max_steps_per_stage=192, eval_interval=64)
            c['ppo'].update(n_steps=32, batch_size=64, n_epochs=1)
            c['natural_curriculum']['evaluation'].update(nc_games=4, natural_large_games=4)
            old_run = root/'old'
            old_run.mkdir()
            meta = dict(checkpoint_kind='natural_reverse', current_stage='NC20', status='checkpoint',
                global_steps=164, stage_steps=64, total_added_steps=64, origin_global_steps=100,
                bank_sha256=bank.sha256, config=c)
            selected = save_checkpoint(FakeLearner(164, 'old'), old_run, meta)
            save_checkpoint(FakeLearner(228, 'new'), old_run,
                            dict(meta, global_steps=228, stage_steps=128, total_added_steps=128))
            loaded = FakeLearner(164, 'old')
            def evaluate(*args, bucket, **kwargs):
                return dict(wins=0, games=4, win_rate=0.)
            with patch('src.common.checkpoints.load_or_create', return_value=loaded) as load, \
                 patch('src.evaluation.natural.evaluate_nc', side_effect=evaluate), \
                 patch('src.evaluation.natural.evaluate_natural', return_value={'wins': 0, 'games': 4}), \
                 patch('src.evaluation.natural.evaluate_reverse_panel', return_value={}):
                result = run(c, selected, bank, root/'branch', stop_at=20, checkpoint_every=64)
            self.assertEqual(Path(load.call_args.args[2]), selected)
            self.assertEqual(sum(loaded.learned), 128)
            self.assertEqual(result['current_stage'], 'NC20')
            self.assertEqual(result['stage_steps'], 192)
            self.assertEqual(result['total_added_steps'], 192)
            self.assertEqual(json.loads((root/'branch'/'latest.zip').read_text())['marker'], 'old')
            self.assertTrue((old_run/'latest.zip').exists())

    def test_requested_pause_saves_completed_update_and_skips_final_evaluation(self):
        from src.natural.trainer import NaturalReverseBank, run, read_checkpoint
        from test_natural_pipeline import training_config as load_config
        from test_natural_bank import fixture, witness_record
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank_file = root/'bank.jsonl'
            bank_file.write_text(''.join(json.dumps(witness_record(*fixture(seed=i)))+'\n' for i in range(50)))
            bank = NaturalReverseBank(bank_file, STAGES)
            model = root/'r80.zip'
            model.write_bytes(b'fixture')
            model.with_suffix('.json').write_text(json.dumps(dict(current_stage='R80', status='promoted', global_steps=100)))
            c = load_config()
            c.update(num_envs=2, min_steps_per_stage=64, max_steps_per_stage=192, eval_interval=128)
            c['ppo'].update(n_steps=32, batch_size=64, n_epochs=1)
            c['natural_curriculum']['evaluation'].update(nc_games=4, natural_large_games=4)
            learner = FakeLearner()
            with patch('src.common.checkpoints.load_or_create', return_value=learner), \
                 patch('src.evaluation.natural.evaluate_nc', return_value={'wins': 0, 'games': 4, 'win_rate': 0.}), \
                 patch('src.evaluation.natural.evaluate_natural', return_value={'wins': 0, 'games': 4}) as final:
                result = run(c, model, bank, root/'paused', stop_at=12,
                             stop_requested=lambda: learner.num_timesteps >= 164)
            self.assertEqual(result['status'], 'paused')
            self.assertEqual(result['stage_steps'], 64)
            self.assertEqual(read_checkpoint(root/'paused'/'latest.zip')['global_steps'], 164)
            final.assert_called_once()
            self.assertFalse((root/'paused'/'final_held_out.json').exists())

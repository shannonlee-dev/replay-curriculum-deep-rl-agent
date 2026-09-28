"""Run-directory continuation without executing a learned model."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_natural_pipeline import training_config, result
from test_natural_resume import FakeLearner
from test_natural_bank import fixture, witness_record
from src.natural.bank import NaturalReverseBank


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        bank_file = self.root/'bank.jsonl'
        bank_file.write_text(''.join(json.dumps(witness_record(*fixture(seed=i)))+'\n' for i in range(50)))
        self.c = training_config()
        self.c.update(num_envs=2, max_steps_per_stage=192, eval_interval=128)
        self.c['ppo'].update(n_steps=32, batch_size=64, n_epochs=1)
        self.c['natural_curriculum']['stages'] = [12, 16, 20, 84]
        self.c['natural_curriculum']['evaluation'].update(natural_large_games=8, natural_large_interval=100)
        self.bank = NaturalReverseBank(bank_file, [12, 16, 20, 84])
        self.initial = self.root/'r80.zip'
        self.initial.write_bytes(b'fixture')
        self.initial.with_suffix('.json').write_text(json.dumps(dict(current_stage='R80', status='promoted', global_steps=100)))
        self.out = self.root/'run'
        self.learner = FakeLearner()
        self.nc_calls = []
        self.natural_calls = []
        def nc(*args, bucket, **kwargs):
            self.nc_calls.append((self.learner.num_timesteps, bucket, kwargs['split']))
            return result(0, 4)
        def natural(*args, games, **kwargs):
            self.natural_calls.append((self.learner.num_timesteps, games))
            return result(0, games)
        for target, replacement in [('src.common.checkpoints.load_or_create', lambda *a: self.learner),
                                     ('src.evaluation.natural.evaluate_nc', nc),
                                     ('src.evaluation.natural.evaluate_natural', natural),
                                     ('src.evaluation.natural.evaluate_reverse_panel', lambda *a, **k: {})]:
            mock = patch(target, side_effect=replacement)
            mock.start(); self.addCleanup(mock.stop)

    def start(self, **kwargs):
        from src.natural.trainer import run
        return run(self.c, self.initial, self.bank, self.out, **kwargs)

    def resume(self, **kwargs):
        from src.natural.trainer import run
        return run(self.c, self.out/'latest.zip', self.bank, self.out, resume_run=True, **kwargs)

    def test_nc12_pause_resume_appends_without_repeating_baseline_or_large_eval(self):
        stopped = self.start(stop_requested=lambda: self.learner.num_timesteps >= 164)
        self.assertEqual(stopped['stage_steps'], 64)
        old_jsonl = (self.out/'evaluations.jsonl').read_bytes()
        old_csv = (self.out/'history.csv').read_bytes()
        self.nc_calls.clear(); self.natural_calls.clear()
        resumed = self.resume()
        self.assertEqual(resumed['current_stage'], 'NC12')
        self.assertEqual(resumed['stage_steps'], 192)
        self.assertEqual(resumed['total_added_steps'], 192)
        self.assertTrue((self.out/'evaluations.jsonl').read_bytes().startswith(old_jsonl))
        self.assertTrue((self.out/'history.csv').read_bytes().startswith(old_csv))
        self.assertFalse(any(steps == 164 for steps, _, _ in self.nc_calls))
        rows = [json.loads(line) for line in (self.out/'evaluations.jsonl').read_text().splitlines()]
        self.assertEqual([r['total_added_steps'] for r in rows], [0, 128, 192])
        self.assertEqual([r['total_added_steps'] for r in rows if r['natural_full_large']], [128])
        self.assertEqual(sum(1 for steps, games in self.natural_calls if steps == 228 and games == 8), 1)

    def test_pause_during_large_eval_keeps_result_and_does_not_repeat_it(self):
        stop = {'value': False}
        def natural(*args, games, **kwargs):
            self.natural_calls.append((self.learner.num_timesteps, games))
            if games == 8:
                stop['value'] = True
            return result(0, games)
        with patch('src.evaluation.natural.evaluate_natural', side_effect=natural):
            self.start(stop_requested=lambda: stop['value'])
        self.assertEqual(self.learner.num_timesteps, 228)
        self.natural_calls.clear()
        self.resume()
        self.assertNotIn((228, 8), self.natural_calls)
        rows = [json.loads(line) for line in (self.out/'evaluations.jsonl').read_text().splitlines()]
        self.assertEqual([r['total_added_steps'] for r in rows if r['natural_full_large']], [128])

    def test_promotion_boundary_resume_advances_to_nc16_once(self):
        self.c['eval_interval'] = 64
        signal = {'value': False}
        def nc(*args, bucket, **kwargs):
            if self.learner.num_timesteps >= 164 and bucket == 20:
                signal['value'] = True
            return result(4 if bucket in (12, 16) else 0, 4)
        with patch('src.evaluation.natural.evaluate_nc', side_effect=nc):
            stopped = self.start(stop_requested=lambda: signal['value'])
        self.assertEqual(stopped['current_stage'], 'NC16')
        self.assertEqual(stopped['global_steps'], 164)
        prefix = (self.out/'evaluations.jsonl').read_bytes()
        resumed = self.resume()
        self.assertEqual(resumed['current_stage'], 'NC16')
        self.assertEqual(resumed['stage_steps'], 192)
        tail = (self.out/'evaluations.jsonl').read_bytes()[len(prefix):].decode()
        self.assertNotIn('"current_stage": "NC12"', tail)
        self.assertEqual(len(resumed['promotion_history']), 1)

    def test_resume_rejects_config_bank_model_and_history_mismatch(self):
        self.start(stop_requested=lambda: self.learner.num_timesteps >= 164)
        before = (self.out/'evaluations.jsonl').read_bytes()
        self.c['natural_curriculum']['nc_episode_probability'] = .6
        with self.assertRaisesRegex(ValueError, 'config'):
            self.resume()
        self.c['natural_curriculum']['nc_episode_probability'] = .3
        original_hash = self.bank.sha256
        self.bank.sha256 = 'a'*64
        with self.assertRaisesRegex(ValueError, 'bank'):
            self.resume()
        self.bank.sha256 = original_hash
        with (self.out/'evaluations.jsonl').open('ab') as f:
            f.write(b'{}\n')
        with self.assertRaisesRegex(ValueError, 'log|history'):
            self.resume()
        (self.out/'evaluations.jsonl').write_bytes(before)
        (self.out/'latest.zip').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.resume()

    def test_identical_scan_provenance_reuses_cache_but_changed_model_does_not(self):
        from src.common.checkpoints import cached_scan
        cache = self.root/'cache'
        calls = []
        def compute():
            calls.append(1); return {12: {'validation': result(1, 4)}}
        provenance = {'model': 'a', 'bank': 'b', 'seed': 1}
        first, reused = cached_scan(cache, provenance, compute)
        self.assertFalse(reused)
        second, reused = cached_scan(cache, provenance, compute)
        self.assertTrue(reused)
        self.assertEqual(first, second)
        cached_scan(cache, dict(provenance, model='c'), compute)
        self.assertEqual(len(calls), 2)

    def test_resume_immediately_before_promotion_does_not_repeat_evaluation(self):
        stop = {'value': False}
        def natural(*args, games, **kwargs):
            if self.learner.num_timesteps == 228 and games == 4:
                stop['value'] = True
            return result(0, games)
        with patch('src.evaluation.natural.evaluate_nc', return_value=result(4, 4)), \
             patch('src.evaluation.natural.evaluate_natural', side_effect=natural):
            paused = self.start(stop_requested=lambda: stop['value'])
        self.assertEqual(paused['current_stage'], 'NC12')
        self.assertEqual(paused['stage_steps'], 128)
        self.assertEqual(paused['promotion_history'], [])
        self.nc_calls.clear()
        resumed = self.resume()
        self.assertEqual(resumed['current_stage'], 'NC16')
        self.assertEqual(len(resumed['promotion_history']), 1)
        rows = [json.loads(line) for line in (self.out/'evaluations.jsonl').read_text().splitlines()]
        self.assertEqual(sum(r['status'] == 'promoted' and r['current_stage'] == 'NC12' for r in rows), 1)

    def test_partial_log_append_is_recovered_once_without_overwrite(self):
        from src.common.checkpoints import reconcile_logs, event_bytes
        from src.natural.trainer import read_checkpoint
        from unittest.mock import patch
        original = reconcile_logs
        def crash(out, metadata, *, write=False):
            if write and metadata.get('pending_evaluation'):
                event = copy.deepcopy(metadata['pending_evaluation'])
                event['model_sha256'] = metadata['model_sha256']
                with (out/'evaluations.jsonl').open('ab') as stream:
                    stream.write(event_bytes(event)['evaluations.jsonl'])
                raise RuntimeError('simulated crash between JSONL and CSV')
            return original(out, metadata, write=write)
        with patch('src.common.checkpoints.reconcile_logs', side_effect=crash):
            with self.assertRaisesRegex(RuntimeError, 'simulated crash'):
                self.start()
        prefix = (self.out/'evaluations.jsonl').read_bytes()
        self.resume()
        self.assertTrue((self.out/'evaluations.jsonl').read_bytes().startswith(prefix))
        rows = [json.loads(line) for line in (self.out/'evaluations.jsonl').read_text().splitlines()]
        self.assertEqual(sum(r['total_added_steps'] == 0 for r in rows), 1)
        self.assertEqual(len((self.out/'history.csv').read_text().splitlines()), len(rows)+1)

    def test_resume_rejects_changed_manifest_scan_and_resolved_config(self):
        self.start(stop_requested=lambda: self.learner.num_timesteps >= 164)
        for name in ('manifest.json', 'difficulty_scan.json', 'config.resolved.yaml'):
            path = self.out/name
            original = path.read_bytes()
            if name.endswith('yaml'):
                path.write_bytes(original.replace(b'seed: 42', b'seed: 43'))
            else:
                path.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError, 'hash|provenance'):
                self.resume()
            path.write_bytes(original)

    def test_resume_rejects_tampered_continuation_state(self):
        self.start(stop_requested=lambda: self.learner.num_timesteps >= 164)
        path = self.out/'latest.json'
        metadata = json.loads(path.read_text())
        metadata['continuation']['target'] = 16
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'metadata hash'):
            self.resume()

    def test_completed_resume_does_not_repeat_held_out_or_append_evaluations(self):
        finished = self.start()
        jsonl = (self.out/'evaluations.jsonl').read_bytes()
        held = (self.out/'final_held_out.json').read_bytes()
        self.nc_calls.clear(); self.natural_calls.clear()
        resumed = self.resume()
        self.assertEqual(resumed['stage_steps'], finished['stage_steps'])
        self.assertEqual(self.nc_calls, [])
        self.assertEqual(self.natural_calls, [])
        self.assertEqual((self.out/'evaluations.jsonl').read_bytes(), jsonl)
        self.assertEqual((self.out/'final_held_out.json').read_bytes(), held)
        self.assertEqual(json.loads(held)['model_sha256'], resumed['model_sha256'])

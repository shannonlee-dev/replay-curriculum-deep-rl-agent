import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
import zipfile

import numpy as np
import torch
import yaml
from sb3_contrib import MaskablePPO

from src.synthetic.curriculum import replay_distribution
from src.evaluation.synthetic import evaluate_model, wilson_interval
from src.env import TheGameEnv
from src.synthetic.trainer import ROOT, load_or_create, make_env, run, validate_config


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, out):
        c = yaml.safe_load((ROOT/'configs/synthetic.yaml').read_text())
        c.update(num_envs=2, min_steps_per_stage=64, max_steps_per_stage=64,
                 eval_interval=64, output_dir=str(out), device='cpu')
        c['ppo'].update(n_steps=32, batch_size=64, n_epochs=1)
        c['evaluation'].update(games=1, balanced_targets=[12,20,98])
        c['bridge']['max_steps_per_phase'] = 64
        return c

    def test_default_cli_preserves_existing_run_and_uses_new_directory(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)/'models'
            base.mkdir()
            existing = base/'history.csv'
            existing.write_text('existing results\n')
            c = self.config(base)
            config = Path(td)/'config.yaml'
            config.write_text(yaml.safe_dump(c))
            command = [sys.executable, '-m', 'src.synthetic.trainer',
                       '--config', str(config), '--from-scratch', '--stop-at', '20']
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(existing.read_text(), 'existing results\n')
            runs = list(base.glob('run_*'))
            self.assertEqual(len(runs), 1)
            self.assertTrue((runs[0]/'latest.zip').is_file())
            self.assertIn(str(runs[0]), result.stdout)
            explicit = subprocess.run(command + ['--output-dir', str(base)],
                                      capture_output=True, text=True, timeout=60)
            self.assertNotEqual(explicit.returncode, 0)
            self.assertIn('already contains a run', explicit.stderr)
            self.assertEqual(existing.read_text(), 'existing results\n')

    def test_checkpoint_load_keeps_every_policy_parameter(self):
        checkpoint = ROOT/'models/1.0.0/reverse_r20.zip'
        if not checkpoint.exists():
            self.skipTest('Original checkpoint not supplied')
        original = MaskablePPO.load(checkpoint, device='cpu')
        with tempfile.TemporaryDirectory() as td:
            c = self.config(td)
            env = make_env(c, 20, None, Path(td)/'episodes', 42)
            loaded = load_or_create(c, env, checkpoint)
            for key, value in original.policy.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.policy.state_dict()[key]), key)
            self.assertEqual(original.observation_space, loaded.observation_space)
            self.assertEqual(original.action_space, loaded.action_space)
            env.close()

    def test_vector_envs_sample_independently(self):
        with tempfile.TemporaryDirectory() as td:
            c = self.config(td)
            c['num_envs'] = 4
            env = make_env(c, 32, None, Path(td)/'episodes', 55)
            rows = []
            for _ in range(12):
                env.reset()
                rows.append([i['sampled_target_remaining'] for i in env.reset_infos])
            self.assertTrue(any(len(set(row)) > 1 for row in rows))
            env.close()

    def test_original_fixed_environment_semantics_unchanged(self):
        archive = ROOT/'the_game_rl_reverse_curriculum_linux.zip'
        if not archive.exists():
            self.skipTest('Original source archive not supplied')
        with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(archive) as z:
            source = Path(td)/'original_env.py'
            source.write_bytes(z.read('the_game_rl_reverse_curriculum_linux/the_game_env.py'))
            spec = importlib.util.spec_from_file_location('original_env', source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            rng = np.random.default_rng(19)
            for target in [4,12,20,32,64,98]:
                for seed in [7,21]:
                    old = module.TheGameEnv(reset_source='reverse', target_remaining=target)
                    new = TheGameEnv(reset_source='reverse', target_remaining=target)
                    a, _ = old.reset(seed=seed)
                    b, _ = new.reset(seed=seed)
                    np.testing.assert_array_equal(a,b)
                    for _ in range(200):
                        np.testing.assert_array_equal(old.action_masks(), new.action_masks())
                        action = int(rng.choice(np.flatnonzero(old.action_masks())))
                        a, ar, ad, at, ai = old.step(action)
                        b, br, bd, bt, bi = new.step(action)
                        np.testing.assert_array_equal(a,b)
                        self.assertEqual((ar,ad,at), (br,bd,bt))
                        self.assertEqual(ai, {k:v for k,v in bi.items() if k != 'sampled_target_remaining'})
                        if ad:
                            break

    def test_all_stage_distributions_and_config_rejections(self):
        with tempfile.TemporaryDirectory() as td:
            c = self.config(td)
            validate_config(c)
            for t in c['stages']:
                targets, probs = replay_distribution(t, {**c['replay'], 'stages':c['stages']})
                self.assertAlmostEqual(sum(probs),1.)
                self.assertTrue(any(x < t for x in targets))
            bad = copy.deepcopy(c)
            bad['ppo']['batch_size'] = 100
            with self.assertRaises(ValueError):
                validate_config(bad)
            bad = copy.deepcopy(c)
            bad['bridge']['natural_probs'][-1] = .9
            with self.assertRaises(ValueError):
                validate_config(bad)

    def test_evaluation_fixed_seeds_and_ci(self):
        a = evaluate_model(None, source='reverse', target_remaining=20, games=8, seed=2)
        b = evaluate_model(None, source='reverse', target_remaining=20, games=8, seed=2)
        self.assertEqual(a,b)
        self.assertAlmostEqual(wilson_interval(0,100)[1], .0369948, places=6)
        self.assertAlmostEqual(wilson_interval(100,100)[0], .9630052, places=6)
        with self.assertRaises(ValueError):
            evaluate_model(None, games=0)

    def test_promotion_and_all_five_bridge_phases(self):
        # Zero thresholds deliberately exercise control flow, not research success.
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/'run'
            c = self.config(out)
            c['start_at'] = 96
            c['promotion'] = {k:0. for k in c['promotion']}
            c['bridge']['natural_min_win_rates'] = [0.]*5
            c['bridge']['reverse_floor_win_rate'] = 0.
            self.assertEqual(run(c, None), 'complete')
            for name in ['R96','R98'] + [f'bridge_{i}' for i in range(1,6)]:
                self.assertTrue((out/f'stage_{name}_complete.zip').exists())
            with (out/'history.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows),14)  # baseline plus one trained evaluation per phase
            self.assertEqual(int(rows[-1]['global_steps']),7*64)
            final = json.loads((out/'latest.json').read_text())
            self.assertEqual(final['current_stage'], 'bridge_5')
            self.assertTrue((out/'win_rate_by_target.png').stat().st_size > 0)
            records = [json.loads(line) for line in (out/'evaluations.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),14)
            selected = json.loads((out/'best_balanced.json').read_text())
            self.assertEqual(selected['balanced_score'], max(r['balanced_score'] for r in records))
            with (out/'episodes/bridge_5/0.monitor.csv').open() as f:
                next(f)
                episodes = list(csv.DictReader(f))
            self.assertTrue(episodes)
            self.assertTrue(all(e['reset_source'] == 'natural' for e in episodes))
            with self.assertRaises(FileExistsError):
                run(c, None)

    def test_stalled_advance_and_stop_at_do_not_run_bridge(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/'run'
            c = self.config(out)
            c['start_at'] = 96
            c['on_max_steps'] = 'advance'
            c['promotion'] = {k:1. for k in c['promotion']}
            self.assertEqual(run(c, None, stop_at=98), 'stalled_advanced')
            summary = json.loads((out/'run_summary.json').read_text())
            self.assertEqual([s['stage'] for s in summary['stages']], ['R96', 'R98'])
            self.assertTrue((out/'stage_R96_stalled_advanced.zip').exists())
            self.assertTrue((out/'stage_R98_stalled_advanced.zip').exists())
            self.assertFalse((out/'stage_R98_complete.zip').exists())
            self.assertFalse((out/'episodes/bridge_1').exists())

    def test_stalled_stops_without_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/'run'
            c = self.config(out)
            c['start_at'] = 98
            c['promotion'] = {k:1. for k in c['promotion']}
            self.assertEqual(run(c, None), 'stalled')
            self.assertFalse((out/'stage_R98_complete.zip').exists())
            self.assertTrue((out/'stage_R98_stalled.zip').exists())
            self.assertFalse((out/'episodes/bridge_1').exists())


if __name__ == '__main__':
    unittest.main()

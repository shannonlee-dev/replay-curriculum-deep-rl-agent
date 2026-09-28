import copy
import io
import unittest


def settings():
    return dict(stages=[4, 8, 12, 84], start_stage='auto', start_stage_reason='fixture control-flow choice', nc_episode_probability=.3,
        minimum_winning_decks=1, minimum_stage_decks=dict(train=1, validation=1, test=1),
        rehearsal=dict(current_weight=.7, older_weight=.3),
        auto_start=dict(min_wins=2, min_win_rate=.1),
        promotion=dict(mode='frontier', min_stage_steps=10, next_stage_min_wins=2,
                       next_stage_min_win_rate=.1, retention_relative_floor=.5,
                       retention_absolute_floor=.05, final_stage_rule='budget_and_retention'),
        evaluation=dict(nc_games=4, nc_seed=200000, natural_games=4, natural_seed=900000,
                        natural_large_games=8, natural_large_interval=100,
                        synthetic_games=4, synthetic_seed=100000))


def result(wins, games=10):
    return dict(wins=wins, games=games, win_rate=wins/games)


class PipelineTests(unittest.TestCase):
    def test_old_config_and_unset_thresholds_fail_closed(self):
        from src.natural.curriculum import validate_natural_config
        with self.assertRaisesRegex(ValueError, 'natural_curriculum'):
            validate_natural_config({})
        n = settings()
        n['auto_start']['min_wins'] = None
        validate_natural_config({'natural_curriculum': n}, require_criteria=False)
        with self.assertRaisesRegex(ValueError, 'min_wins'):
            validate_natural_config({'natural_curriculum': n})

    def test_auto_start_uses_hardest_with_signal_and_all_splits(self):
        from src.natural.curriculum import select_start
        rows = {4: dict(available=dict(train=3, validation=3, test=3), validation=result(9)),
                8: dict(available=dict(train=3, validation=3, test=3), validation=result(2)),
                12: dict(available=dict(train=3, validation=3, test=3), validation=result(1))}
        self.assertEqual(select_start(settings(), rows)['stage'], 8)
        rows[8]['available']['test'] = 0
        self.assertEqual(select_start(settings(), rows)['stage'], 4)
        rows[4]['validation'] = result(0)
        with self.assertRaisesRegex(ValueError, 'no eligible'):
            select_start(settings(), rows)

    def test_frontier_retention_and_final_rule(self):
        from src.natural.curriculum import frontier_checks
        n = settings()
        n['stages'] = [4, 8, 12]
        results = {4: result(4), 8: result(0), 12: result(2)}
        checks = frontier_checks(n, 8, results, {4: .8}, 10, 20)
        self.assertTrue(all(checks.values()))  # Current mastery is not required.
        results[4] = result(3)
        self.assertFalse(all(frontier_checks(n, 8, results, {4: .8}, 10, 20).values()))
        results[4] = result(4)
        self.assertFalse(all(frontier_checks(n, 12, results, {4: .8, 8: 0}, 19, 20).values()))
        results[8] = result(1)
        self.assertTrue(all(frontier_checks(n, 12, results, {4: .8, 8: 0}, 20, 20).values()))

    def test_completed_shares_window_and_inflight_transitions(self):
        from src.common.source_usage import usage_summary, subtract_usage
        before = dict(completed={'natural_conditioned': 1, 'reverse': 1},
                      completed_transitions={'natural_conditioned': 2, 'reverse': 18},
                      wins={'natural_conditioned': 1}, transitions={'natural_conditioned': 3, 'reverse': 19})
        summary = usage_summary(before)
        self.assertEqual(summary['episode_shares']['natural_conditioned'], .5)
        self.assertEqual(summary['transition_shares']['natural_conditioned'], .1)
        self.assertEqual(summary['sources']['natural_conditioned']['win_rate'], 1)
        self.assertEqual(summary['sources']['natural_conditioned']['transitions'], 3)
        self.assertEqual(summary['sources']['natural_conditioned']['completed_transitions'], 2)
        self.assertEqual(usage_summary(subtract_usage(before, before))['sources'], {})

    def test_natural_schedule_is_global_not_per_stage(self):
        from src.evaluation.natural import natural_eval_games, next_large_step
        self.assertEqual(next_large_step(99, 100), 100)
        self.assertEqual(natural_eval_games(100, 0, False, False, regular_games=4, large_games=8, interval=100), 8)
        self.assertEqual(natural_eval_games(101, 100, False, False, regular_games=4, large_games=8, interval=100), 4)

    def test_non_tty_dashboard_emits_only_evaluation_lines(self):
        from src.common.terminal_ui import Dashboard
        from rich.console import Console
        stream = io.StringIO()
        with Dashboard(console=Console(file=stream, force_terminal=False)) as ui:
            for _ in range(3):
                ui.update(dict(current_stage='NC4', global_steps=10, stage_steps=10, status='training'))
            ui.update(dict(current_stage='NC4', global_steps=10, stage_steps=10, status='training',
                           natural_full=result(1)), history=True)
        self.assertEqual(len(stream.getvalue().splitlines()), 1)
        self.assertIn('Natural full', stream.getvalue())
        self.assertNotIn('\x1b', stream.getvalue())


def training_config():
    from src.natural.curriculum import load_config
    from pathlib import Path
    c = load_config(Path(__file__).resolve().parents[1]/'configs/natural_replay.yaml')
    n = settings()
    n.update(stages=[12, 20, 32, 48, 64, 72, 80, 84], start_stage=12)
    n['promotion']['min_stage_steps'] = 64
    n['evaluation']['natural_large_interval'] = 1000000
    n['evaluation']['natural_large_games'] = 4
    c['natural_curriculum'] = n
    return c


class AdditionalPipelineTests(unittest.TestCase):
    def test_scan_real_fixture_and_public_observation_only(self):
        import tempfile
        import json
        from pathlib import Path
        import numpy as np
        from src.natural.bank import NaturalReverseBank, restore_checkpoint
        from src.natural.difficulty_scan import difficulty_scan
        from test_natural_bank import fixture, witness_record
        from test_natural_training import LegalTestPolicy
        from src.env import TheGameEnv
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bank.jsonl'
            path.write_text(''.join(json.dumps(witness_record(*fixture(seed=i)))+'\n' for i in range(50)))
            bank = NaturalReverseBank(path, [4, 8, 12, 84])
            scan = difficulty_scan(LegalTestPolicy(), bank, settings()['evaluation'])
            self.assertEqual(list(scan), [4, 8, 12, 84])
            self.assertGreater(scan[4]['validation']['games'], 0)
            self.assertEqual(scan[4]['validation']['win_rate'], 1)
            self.assertEqual(len(scan[4]['validation']['wilson_ci_95']), 2)
            checkpoint = bank.checkpoints('train', 84)[0][1]
            a, b = TheGameEnv(), TheGameEnv()
            restore_checkpoint(a, checkpoint)
            other = copy.deepcopy(checkpoint)
            other['future_deck_order'].reverse()
            restore_checkpoint(b, other)
            np.testing.assert_array_equal(a._get_obs(), b._get_obs())
            np.testing.assert_array_equal(a.action_masks(), b.action_masks())
            a.close(); b.close()

    def test_config_override_and_nonfinite_values(self):
        from src.natural.curriculum import load_config, validate_natural_config
        from pathlib import Path
        path = Path(__file__).resolve().parents[1]/'configs/natural_replay.yaml'
        c = load_config(path, ['natural_curriculum.nc_episode_probability=0.60'])
        self.assertEqual(c['natural_curriculum']['nc_episode_probability'], .6)
        with self.assertRaisesRegex(ValueError, 'unknown override'):
            load_config(path, ['natural_curriculum.typo=0.6'])
        for bad in (float('nan'), True, -1, 1.1):
            c['natural_curriculum']['nc_episode_probability'] = bad
            with self.assertRaises(ValueError):
                validate_natural_config(c, require_criteria=False)

    def test_audit_features_use_only_exact_public_observation(self):
        from src.evaluation.distribution_audit import summarize, collect_rollin
        from test_natural_training import LegalTestPolicy
        samples = collect_rollin(LegalTestPolicy(), stages=[84], games=3, seed=2, heuristic=False)
        summary = summarize(samples[84])
        self.assertEqual(len(summary['feature_mean']), 300)
        self.assertEqual(summary['sample_count'], 3)

    def test_large_evaluation_crosses_stage_boundary_and_scan_never_learns(self):
        import tempfile
        import json
        from pathlib import Path
        from unittest.mock import patch
        from src.natural.bank import NaturalReverseBank
        from src.natural.trainer import run
        from test_natural_resume import FakeLearner
        from test_natural_bank import fixture, witness_record
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'bank.jsonl'
            path.write_text(''.join(json.dumps(witness_record(*fixture(seed=i)))+'\n' for i in range(50)))
            c = training_config()
            c.update(num_envs=2, max_steps_per_stage=64, eval_interval=64)
            c['ppo'].update(n_steps=32, batch_size=64, n_epochs=1)
            n = c['natural_curriculum']
            n['stages'] = [12, 20, 84]
            n['evaluation'].update(natural_large_interval=100, natural_games=4, natural_large_games=8)
            bank = NaturalReverseBank(path, n['stages'])
            model_path = root/'r80.zip'
            model_path.write_bytes(b'fixture')
            model_path.with_suffix('.json').write_text(json.dumps(dict(current_stage='R80', status='promoted', global_steps=100)))
            learner = FakeLearner()
            def nc(*args, bucket, **kwargs):
                return result(0 if bucket == 84 else 4, 4)
            with patch('src.common.checkpoints.load_or_create', return_value=learner), \
                 patch('src.evaluation.natural.evaluate_nc', side_effect=nc), \
                 patch('src.evaluation.natural.evaluate_natural', side_effect=lambda *a, games, **k: result(0, games)), \
                 patch('src.evaluation.natural.evaluate_reverse_panel', return_value={}):
                run(c, model_path, bank, root/'scan', scan_only=True)
                self.assertEqual(learner.learned, [])
                self.assertFalse((root/'scan'/'latest.zip').exists())
                run(c, model_path, bank, root/'train', stop_at=20)
            rows = [json.loads(line) for line in (root/'train'/'evaluations.jsonl').read_text().splitlines()]
            self.assertTrue(all(row['natural_full']['games'] == 4 for row in rows))
            manifest = json.loads((root/'train'/'manifest.json').read_text())
            self.assertEqual(manifest['promotion'], n['promotion'])
            self.assertEqual(manifest['start_selection']['reason'], n['start_stage_reason'])
            for row in rows:
                for check in row['promotion_checks'].values():
                    self.assertIn('actual', check)
                    self.assertIn('required', check)
                    self.assertIn(check['status'], ('PASS', 'FAIL'))
            large = [row for row in rows if row['natural_full_large']]
            self.assertEqual(len(large), 1)
            self.assertEqual(large[0]['total_added_steps'], 128)
            self.assertEqual(large[0]['current_stage'], 'NC20')
            self.assertEqual(large[0]['natural_full_large']['games'], 8)

    def test_natural_trainer_has_no_legacy_mastery_threshold(self):
        import ast
        from pathlib import Path
        tree = ast.parse((Path(__file__).resolve().parents[1]/'src/natural/trainer.py').read_text())
        self.assertFalse(any(isinstance(node, ast.Name) and node.id == 'STAGES' for node in ast.walk(tree)))
        self.assertFalse(any(isinstance(node, ast.Constant) and type(node.value) is float and node.value in (.6, .5)
                             for node in ast.walk(tree)))

    def test_explicit_start_does_not_require_unused_auto_thresholds(self):
        from src.natural.curriculum import validate_natural_config
        n = settings()
        n.update(start_stage=12, start_stage_reason='User selected NC12 based on prior held-out learnability')
        n['auto_start'] = dict(min_wins=None, min_win_rate=None)
        self.assertEqual(validate_natural_config({'natural_curriculum': n})['start_stage'], 12)
        n['start_stage_reason'] = ''
        with self.assertRaisesRegex(ValueError, 'start_stage_reason'):
            validate_natural_config({'natural_curriculum': n})

    def test_operational_frontier_logs_actual_values_and_pass_fail(self):
        from src.natural.curriculum import frontier_assessment
        n = settings()
        n['promotion'].update(next_stage_min_wins=25, next_stage_min_win_rate=.05,
                              retention_relative_floor=.70, retention_absolute_floor=.05)
        values = {4: result(7, 100), 8: result(1, 100), 12: result(25, 500)}
        checks = frontier_assessment(n, 8, values, {4: .1}, 10, 20)
        self.assertEqual(checks['next_stage_wins']['actual'], 25)
        self.assertEqual(checks['next_stage_wins']['required'], 25)
        self.assertEqual(checks['next_stage_wins']['status'], 'PASS')
        self.assertEqual(checks['next_stage_win_rate']['actual'], .05)
        self.assertEqual(checks['NC4_retention_relative']['reference'], .1)
        self.assertAlmostEqual(checks['NC4_retention_relative']['required'], .07)
        self.assertTrue(all(row['passed'] for row in checks.values()))
        values[12] = result(24, 400)  # Rate alone cannot replace minimum wins.
        checks = frontier_assessment(n, 8, values, {4: .1}, 10, 20)
        self.assertEqual(checks['next_stage_wins']['status'], 'FAIL')
        self.assertEqual(checks['next_stage_win_rate']['status'], 'PASS')

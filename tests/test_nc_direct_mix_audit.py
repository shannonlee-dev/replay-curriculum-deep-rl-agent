import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.natural.bank import build_record, file_sha256
from test_natural_bank import fixture


def make_inputs(root):
    run = root/'run'
    (run/'episodes').mkdir(parents=True)
    record = build_record(*fixture(), prefix_action_offset=21)
    record['teacher'] = dict(model_sha256='a'*64, config={'policy_kind': 'heuristic'},
                             code_sha256={'fixture': 'b'*64})
    record['search'] = dict(nodes=0, elapsed_seconds=0., deepest_remaining=0,
                            max_cards_played=84, deepest_turn=0, frontier_size=1)
    record['prefix_policy'] = dict(model_sha256='a'*64, deterministic=True, seed=0)
    bank = root/'bank.jsonl'
    bank.write_text(json.dumps(record)+'\n')
    manifest = dict(arm='nc', bank_sha256=file_sha256(bank), code_sha256={},
                    config={'num_envs': 1, 'experiment': {'nc_probability': .3,
                                                        'minimum_winning_decks': 1}})
    source_root = Path(__file__).resolve().parents[1]
    for name in ('src/env.py', 'src/natural/bank.py', 'archive/nc84_direct_mix/env.py',
                 'archive/nc84_direct_mix/trainer.py'):
        manifest['code_sha256'][name] = file_sha256(source_root/name)
    (run/'manifest.json').write_text(json.dumps(manifest))
    write_monitor(run, [(0, 'False', 'natural_conditioned', 84),
                        (1, 'True', 'reverse', 20)])
    return run, bank, record


def write_monitor(run, rows):
    with (run/'episodes'/'0.monitor.csv').open('w', newline='') as stream:
        stream.write('#{"t_start": 0}\n')
        writer = csv.writer(stream)
        writer.writerow(['r', 'l', 't', 'won', 'sampled_target_remaining', 'reset_source'])
        for index, (reward, won, source, target) in enumerate(rows):
            writer.writerow([reward, 20, index+1, won, target, source])


class DirectMixAuditTests(unittest.TestCase):
    def test_counts_actual_nc_wins_not_truthiness_of_false_string(self):
        from archive.nc84_direct_mix.audit import audit_training_logs
        with tempfile.TemporaryDirectory() as tmp:
            run, _, _ = make_inputs(Path(tmp))
            report = audit_training_logs(run, expected_envs=1)
            self.assertEqual(report['sources']['natural_conditioned']['wins'], 0)
            self.assertEqual(report['sources']['reverse']['wins'], 1)
            write_monitor(run, [(1, 'True', 'natural_conditioned', 83)])
            self.assertEqual(audit_training_logs(run, expected_envs=1)
                             ['sources']['natural_conditioned']['wins'], 1)

    def test_reward_win_mismatch_and_partial_row_are_rejected(self):
        from archive.nc84_direct_mix.audit import audit_training_logs
        with tempfile.TemporaryDirectory() as tmp:
            run, _, _ = make_inputs(Path(tmp))
            write_monitor(run, [(1, 'False', 'natural_conditioned', 84)])
            with self.assertRaisesRegex(ValueError, 'reward/WIN'):
                audit_training_logs(run, expected_envs=1)
            write_monitor(run, [(0, 'False', 'natural_conditioned', 84)])
            with (run/'episodes'/'0.monitor.csv').open('a') as stream:
                stream.write('0,20,2,False,84')
            with self.assertRaisesRegex(ValueError, 'row'):
                audit_training_logs(run, expected_envs=1)

    def test_fresh_replay_and_train_reset_path_pass_before_zero_win_verdict(self):
        from archive.nc84_direct_mix.audit import audit
        with tempfile.TemporaryDirectory() as tmp:
            run, bank, record = make_inputs(Path(tmp))
            self.assertEqual(record['split'], 'train')
            report = audit(run, bank, reset_samples=2)
            self.assertTrue(report['checks_passed'])
            self.assertEqual(report['bank']['verified_unique_decks'], 1)
            self.assertEqual(report['training_reset_replay']['wins'], 2)
            self.assertEqual(report['verdict'], 'direct_mix_failed_for_observed_run')

    def test_corrupt_witness_cannot_be_misdiagnosed_as_design_failure(self):
        from archive.nc84_direct_mix.audit import audit
        with tempfile.TemporaryDirectory() as tmp:
            run, bank, record = make_inputs(Path(tmp))
            record['actions'][-1] = 392
            bank.write_text(json.dumps(record)+'\n')
            # Even with an updated manifest hash, replay must reject the bad witness.
            manifest = json.loads((run/'manifest.json').read_text())
            manifest['bank_sha256'] = file_sha256(bank)
            (run/'manifest.json').write_text(json.dumps(manifest))
            report = audit(run, bank, reset_samples=2)
            self.assertFalse(report['checks_passed'])
            self.assertEqual(report['verdict'], 'integrity_failure_investigate_first')

    def test_changed_bank_and_missing_source_are_not_zero_win_success_cases(self):
        from archive.nc84_direct_mix.audit import audit
        with tempfile.TemporaryDirectory() as tmp:
            run, bank, _ = make_inputs(Path(tmp))
            write_monitor(run, [(1, 'True', 'reverse', 20)])
            report = audit(run, bank, reset_samples=2)
            self.assertEqual(report['verdict'], 'no_completed_nc_training_episodes')
            self.assertFalse(report['checks_passed'])
            bank.write_text(bank.read_text()+'\n')
            self.assertEqual(audit(run, bank, reset_samples=2)['verdict'],
                             'integrity_failure_investigate_first')

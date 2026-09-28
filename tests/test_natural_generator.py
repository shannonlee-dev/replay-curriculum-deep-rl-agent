import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.natural.bank import NaturalBank
from test_natural_bank import fixture, witness_record as build_record


class GeneratorTests(unittest.TestCase):
    def test_journal_recovery_and_statistics(self):
        from archive.privileged_teacher.build_bank import BankJournal
        manifest = {'schema_version': 1, 'seed': 8, 'teacher': {'beam_width': 1}}
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/'bank'
            with BankJournal(out, manifest, resume=False) as journal:
                record = build_record(*fixture())
                journal.append(dict(index=0, deck_id=record['deck_id'], status='win_found',
                                    nodes=123, elapsed_seconds=1., record=record))
                journal.append(dict(index=1, deck_id='f'*64, status='teacher_failed_timeout',
                                    nodes=8, elapsed_seconds=10., record=None))
                summary = journal.export()
                self.assertEqual(summary['wins_found'], 1)
                self.assertEqual(summary['sampled_decks'], 2)
                self.assertEqual(summary['success_rate'], .5)
                self.assertEqual(summary['mean_search_nodes'], 65.5)
            # bank.jsonl is a derived artifact: a crash during export is recoverable.
            (out/'bank.jsonl').write_text('broken export')
            with BankJournal(out, manifest, resume=True) as journal:
                self.assertEqual(journal.next_index, 2)
                journal.export()
            self.assertEqual(len(NaturalBank(out/'bank.jsonl').deck_ids), 1)
            with self.assertRaises(FileExistsError):
                with BankJournal(out, manifest, resume=False):
                    pass
            changed = copy.deepcopy(manifest)
            changed['seed'] = 9
            with self.assertRaises(ValueError):
                with BankJournal(out, changed, resume=True):
                    pass

    def test_resume_refuses_missing_authoritative_journal(self):
        from archive.privileged_teacher.build_bank import BankJournal
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/'bank'
            with BankJournal(out, {}, resume=False) as journal:
                record = build_record(*fixture())
                journal.append(dict(index=0, deck_id=record['deck_id'], status='win_found',
                                    nodes=100, elapsed_seconds=1., record=record))
                journal.export()
            (out/'attempts.jsonl').unlink()
            before = (out/'bank.jsonl').read_bytes()
            with self.assertRaises(ValueError):
                with BankJournal(out, {}, resume=True):
                    pass
            self.assertEqual((out/'bank.jsonl').read_bytes(), before)

    def test_partial_journal_and_invalid_winner_fail_closed(self):
        from archive.privileged_teacher.build_bank import BankJournal
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/'bank'
            with BankJournal(out, {}, resume=False) as journal:
                with self.assertRaises(ValueError):
                    journal.append(dict(index=0, deck_id='f'*64, status='win_found',
                                        nodes=1, elapsed_seconds=1., record=None))
            (out/'attempts.jsonl').write_text('{"index":')
            with self.assertRaises(ValueError):
                with BankJournal(out, {}, resume=True):
                    pass


if __name__ == '__main__':
    unittest.main()

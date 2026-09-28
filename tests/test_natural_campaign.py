import tempfile
import unittest
from pathlib import Path
import src.evaluation.heuristic as evaluation


class CampaignTests(unittest.TestCase):
    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'exponential'
            directory.mkdir()
            sentinel = directory/'bank.jsonl'
            sentinel.write_text('preserve me')
            with self.assertRaises(FileExistsError):
                evaluation.evaluate('exponential', seed=1, decks=1, workers=1,
                                    output=Path(tmp), code_hash={})
            self.assertEqual(sentinel.read_text(), 'preserve me')

    def test_efficiency_selection_uses_wins_per_second(self):
        rows = [dict(strategy='exponential', wins=30, runtime_seconds=10),
                dict(strategy='combined', wins=40, runtime_seconds=20)]
        self.assertEqual(evaluation.select_strategy(rows), 'exponential')

    def test_bank_collection_deduplicates_deck_ids_and_caps_target(self):
        import json
        from archive.nc84_direct_mix.campaign import collect_records
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bank.jsonl'
            path.write_text(''.join(json.dumps(dict(deck_id=identity))+'\n'
                                    for identity in ('a', 'a', 'b', 'c')))
            self.assertEqual([r['deck_id'] for r in collect_records([path], 2)], ['a', 'b'])

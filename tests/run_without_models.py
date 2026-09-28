"""Run repository tests without constructing, loading or running a learned PPO model.

Usage: .venv/bin/python tests/run_without_models.py
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# These existing tests intentionally execute real PPO or a CLI that executes PPO.
MODEL_TESTS = {
    'test_default_cli_preserves_existing_run_and_uses_new_directory',
    'test_checkpoint_load_keeps_every_policy_parameter',
    'test_promotion_and_all_five_bridge_phases',
    'test_stalled_advance_and_stop_at_do_not_run_bridge',
    'test_stalled_stops_without_complete_checkpoint',
}


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


def main():
    from sb3_contrib import MaskablePPO
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_*.py')
    for case in cases(suite):
        if case.__class__.__name__ == 'TrainingTests' and case._testMethodName in MODEL_TESTS:
            method = getattr(case, case._testMethodName)
            setattr(case, case._testMethodName, unittest.skip('real PPO execution deferred to user')(method))
    forbidden = AssertionError('real model execution is prohibited in this test run')
    with patch.object(MaskablePPO, '__init__', side_effect=forbidden), \
         patch.object(MaskablePPO, 'load', side_effect=forbidden), \
         patch.object(MaskablePPO, 'learn', side_effect=forbidden), \
         patch.object(MaskablePPO, 'predict', side_effect=forbidden):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()

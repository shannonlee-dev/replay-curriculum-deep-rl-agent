"""Package boundaries, provenance migration and executable entry points."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from src.common.paths import ROOT
from src.common.provenance import source_hashes, validate_code_hashes


class ProjectLayoutTests(unittest.TestCase):
    def test_all_modules_import_and_active_r80_validation_never_imports_archive(self):
        code = '''
import importlib, json, pathlib, pkgutil, sys, tempfile
import src
for module in pkgutil.walk_packages(src.__path__, 'src.'):
    importlib.import_module(module.name)
from src.common.checkpoints import read_checkpoint
with tempfile.TemporaryDirectory() as tmp:
    path = pathlib.Path(tmp)/'r80.zip'
    path.write_bytes(b'fixture')
    path.with_suffix('.json').write_text(json.dumps(dict(current_stage='R80', status='promoted', global_steps=64)))
    assert read_checkpoint(path)['current_stage'] == 'R80'
assert not any(m == 'archive' or m.startswith('archive.') for m in sys.modules)
import archive
for module in pkgutil.walk_packages(archive.__path__, 'archive.'):
    importlib.import_module(module.name)
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        for path in (ROOT/'src').rglob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('archive'), str(path))
                elif isinstance(node, ast.Import):
                    self.assertFalse(any(n.name.startswith('archive') for n in node.names), str(path))

    def test_shell_entrypoints_from_other_directory_and_no_pythonpath(self):
        with tempfile.TemporaryDirectory() as tmp:
            for script in ('run_train.sh', 'run_natural_train.sh'):
                source = (ROOT/script).read_text()
                self.assertNotIn('PYTHONPATH', source)
                self.assertIn(' -m src.', source)
                result = subprocess.run([str(ROOT/script), '--help'], cwd=tmp,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertIn('--resume', result.stdout)

    def test_game_and_bank_algorithms_unchanged_by_move(self):
        def semantics(source):
            tree = ast.parse(source)
            tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
            return ast.dump(tree, include_attributes=False)
        with zipfile.ZipFile(ROOT/'archive/pre_package/source.zip') as source:
            for old, new in [('the_game_env.py', 'src/env.py'), ('natural_bank.py', 'src/natural/bank.py')]:
                self.assertEqual(semantics(source.read(old)), semantics((ROOT/new).read_text()))

    def test_migration_accepts_exact_sources_and_rejects_source_changes(self):
        receipt = json.loads((ROOT/'archive/pre_package/migration.json').read_text())
        current = source_hashes()
        self.assertEqual(current, receipt['new_code_sha256'])
        validate_code_hashes(current, current)
        validate_code_hashes(receipt['old_code_sha256'], current)
        changed = dict(current)
        changed['src/natural/trainer.py'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'provenance'):
            validate_code_hashes(receipt['old_code_sha256'], changed)
        old = dict(receipt['old_code_sha256'])
        old['train_natural_reverse.py'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'provenance'):
            validate_code_hashes(old, current)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'archive/pre_package'
            directory.mkdir(parents=True)
            (directory/'migration.json').write_text(json.dumps(receipt))
            with zipfile.ZipFile(directory/'source.zip', 'w') as damaged:
                for name in receipt['old_code_sha256']:
                    damaged.writestr(name, b'changed source')
            with self.assertRaisesRegex(ValueError, 'archived source'):
                validate_code_hashes(receipt['old_code_sha256'], current, root=Path(tmp))

    def test_shell_preflight_preserves_inputs_and_config_resolution(self):
        from test_natural_bank import fixture, witness_record
        from test_natural_pipeline import training_config
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = root/'bank.jsonl'
            bank.write_text(''.join(json.dumps(witness_record(*fixture(seed=i)))+'\n' for i in range(50)))
            checkpoint = root/'r80.zip'
            checkpoint.write_bytes(b'preflight must not execute this fake model')
            checkpoint.with_suffix('.json').write_text(json.dumps(dict(current_stage='R80', status='promoted', global_steps=64)))
            c = training_config()
            c.update(resume=str(checkpoint), num_envs=2, max_steps_per_stage=64)
            c['ppo'].update(n_steps=32, batch_size=64, n_epochs=1)
            c['natural_curriculum']['stages'] = [12, 84]
            config = root/'config.yaml'
            config.write_text(yaml.safe_dump(c))
            inputs = {p: p.read_bytes() for p in (bank, checkpoint, checkpoint.with_suffix('.json'), config)}
            result = subprocess.run([str(ROOT/'run_natural_train.sh'), '--check', '--config', str(config),
                                     '--bank', str(bank), '--output-dir', str(root/'check')],
                                    cwd=tmp, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads((root/'check/preflight.json').read_text())
            self.assertFalse(report['training_started'])
            self.assertFalse(report['model_executed'])
            self.assertEqual(report['bank_sha256'], hashlib.sha256(inputs[bank]).hexdigest())
            self.assertEqual(inputs, {p: p.read_bytes() for p in inputs})
            self.assertEqual(ROOT, Path(__file__).resolve().parents[1])

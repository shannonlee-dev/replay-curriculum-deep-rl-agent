"""Strict source provenance, including one audited package-layout migration.

A legacy run is accepted only for the exact archived source set and the exact
packaged replacement recorded in migration.json. Unknown revisions fail closed.
The run's manifest, config, model, bank and scan files are never rewritten.
"""
import hashlib
import json
import zipfile

from src.common.paths import ROOT
from src.natural.bank import file_sha256


def source_hashes(root=ROOT):
    return {p.relative_to(root).as_posix(): file_sha256(p)
            for p in sorted((root/'src').rglob('*.py'))}


def validate_code_hashes(saved, current, *, root=ROOT):
    if saved == current:
        return
    directory = root/'archive/pre_package'
    receipt = json.loads((directory/'migration.json').read_text())
    if saved != receipt['old_code_sha256'] or current != receipt.get('new_code_sha256'):
        raise ValueError('run code provenance hash mismatch (unknown source revision)')
    with zipfile.ZipFile(directory/'source.zip') as source:
        archived = {name: hashlib.sha256(source.read(name)).hexdigest() for name in saved}
    if archived != saved:
        raise ValueError('archived source provenance hash mismatch')

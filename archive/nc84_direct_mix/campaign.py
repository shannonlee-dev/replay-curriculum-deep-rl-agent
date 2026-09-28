"""Order-blind 10k benchmark -> 5000 unique NC84 witnesses -> R80 PPO A/B."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

from src.evaluation.heuristic import evaluate, select_strategy
from src.natural.bank import NaturalBank, file_sha256

from src.common.paths import ROOT


def collect_records(paths, limit):
    records = {}
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                record = json.loads(line)
                records.setdefault(record['deck_id'], record)
                if len(records) == limit:
                    return list(records.values())
    return list(records.values())


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/natural_campaign'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    model = ROOT/'models/restart_r64_from_line5/stage_R80_complete.zip'
    names = ('src/env.py', 'src/natural/bank.py', 'src/natural/heuristics.py',
             'src/evaluation/heuristic.py', 'archive/nc84_direct_mix/campaign.py')
    hashes = {name: file_sha256(ROOT/name) for name in names}
    write_json(out/'manifest.json', dict(code_sha256=hashes, model_sha256=file_sha256(model),
        benchmark_decks=10000, benchmark_seed=730000, production_seed_start=740000,
        target_unique_winning_decks=5000, workers=args.workers, selection='wins_per_second',
        strategies=['exponential', 'combined'], privileged_suffix=False,
        training_config_sha256=file_sha256(ROOT/'archive/nc84_direct_mix/config.yaml'),
        arms={'nc': 'NC84 30% + synthetic/replay 70%', 'control': 'synthetic/replay 100%'}))
    started = time.monotonic()
    try:
        summaries = []
        for strategy in ('exponential', 'combined'):
            write_json(out/'status.json', dict(phase='benchmark', strategy=strategy))
            summaries.append(evaluate(strategy, seed=730000, decks=10000, workers=args.workers,
                                      output=out/'benchmark', code_hash=hashes))
        strategy = select_strategy(summaries)
        write_json(out/'benchmark'/'summary.json', dict(strategies=summaries, selected=strategy))
        paths = [out/'benchmark'/strategy/'bank.jsonl']
        records = collect_records(paths, 5000)
        batch = 0
        while len(records) < 5000:
            write_json(out/'status.json', dict(phase='production', strategy=strategy,
                       unique_winning_decks=len(records), target=5000, batch=batch))
            directory = out/'production'/f'batch_{batch:04d}'
            evaluate(strategy, seed=740000+batch, decks=1000, workers=args.workers,
                     output=directory, code_hash=hashes)
            paths.append(directory/strategy/'bank.jsonl')
            records = collect_records(paths, 5000)
            batch += 1
        bank_path = out/'bank.jsonl'
        with bank_path.open('x') as stream:
            for record in records:
                stream.write(json.dumps(record, separators=(',', ':'))+'\n')
        write_json(out/'status.json', dict(phase='full_bank_validation', unique_winning_decks=len(records)))
        bank = NaturalBank(bank_path)  # Full start-to-WIN and independent NC84-to-WIN replay.
        if len(bank.deck_ids) != 5000:
            raise ValueError('expected exactly 5000 unique winning decks')
        write_json(out/'bank_summary.json', dict(unique_winning_decks=len(bank.deck_ids),
            split_counts=dict(Counter(record['split'] for record in records)),
            checkpoint_counts=bank.counts(), bank_sha256=bank.sha256, strategy=strategy,
            full_replay_verified=True, nc84_continuation_verified=True,
            generation_runtime_seconds=time.monotonic()-started))
        # Both arms use the existing fixed seed, budget, evaluation and mixture config.
        write_json(out/'status.json', dict(phase='ppo_ab', arms=['nc', 'control']))
        processes, streams = [], []
        try:
            for arm in ('nc', 'control'):
                log = (out/f'{arm}.log').open('x')
                streams.append(log)
                command = [sys.executable, '-u', '-m', 'archive.nc84_direct_mix.trainer',
                           '--resume', str(model), '--bank', str(bank_path), '--arm', arm,
                           '--output-dir', str(out/arm)]
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                processes.append((arm, process))
            write_json(out/'training_pids.json', {arm: process.pid for arm, process in processes})
            returncodes = {arm: process.wait() for arm, process in processes}
            if any(returncodes.values()):
                raise RuntimeError(f'PPO arms failed: {returncodes}; see arm logs')
        finally:
            for stream in streams:
                stream.close()
        results = {arm: json.loads((out/arm/'final_held_out.json').read_text()) for arm in ('nc', 'control')}
        write_json(out/'comparison.json', results)
        write_json(out/'status.json', dict(phase='complete', runtime_seconds=time.monotonic()-started))
    except BaseException as exc:
        write_json(out/'status.json', dict(phase='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()

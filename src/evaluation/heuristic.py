"""Evaluate order-blind standalone natural policies and export NC84 WIN witnesses."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
from src.common.paths import ROOT
import time

import numpy as np

from src.natural.bank import build_record, file_sha256, validate_provenance, validate_record
from src.natural.heuristics import STRATEGIES, play_natural_deck, exponential_score, combined_score, _distance_score


def _deck(seed, index):
    return np.random.default_rng(np.random.SeedSequence([seed, index])).permutation(np.arange(2, 100)).tolist()


def _run(item):
    strategy, seed, index = item
    deck = _deck(seed, index)
    exponential_score.cache_clear()
    combined_score.cache_clear()
    _distance_score.cache_clear()
    result = play_natural_deck(deck, strategy)
    return index, deck, result


def _record(deck, result, teacher):
    if not result.won or result.nc84_checkpoint is None:
        return None
    record = build_record(deck, result.actions,
                          prefix_action_offset=result.nc84_checkpoint['action_offset'])
    if record['checkpoints'] != [result.nc84_checkpoint]:
        raise ValueError('NC84 replay checkpoint differs from live rollout')
    remaining = result.nc84_checkpoint['remaining_count']
    record['teacher'] = teacher
    record['search'] = dict(nodes=0, elapsed_seconds=0., deepest_remaining=0,
                            max_cards_played=remaining, deepest_turn=0, frontier_size=1)
    record['prefix_policy'] = dict(model_sha256=teacher['model_sha256'], deterministic=True, seed=0)
    validate_record(record)
    validate_provenance(record)
    return record


def select_strategy(rows):
    return max(rows, key=lambda row: (row['wins']/row['runtime_seconds'], row['wins']))['strategy']


def evaluate(strategy, *, seed, decks, workers, output, code_hash):
    directory = output/strategy
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    digest = hashlib.sha256(('standalone-natural:'+strategy).encode()).hexdigest()
    teacher = dict(model_sha256=digest, config=dict(strategy=strategy, future_deck_order='unread',
        lookahead='hand_only', policy_kind='heuristic', privileged_suffix=False), code_sha256=code_hash)
    wins = witnesses = 0
    with (directory/'attempts.jsonl').open('x') as journal, (directory/'bank.jsonl').open('x') as bank:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            # Bound outstanding work and memory; preserve deterministic deck order.
            for start in range(0, decks, 128):
                items = ((strategy, seed, i) for i in range(start, min(start+128, decks)))
                for index, deck, result in pool.map(_run, items):
                    record = _record(deck, result, teacher)
                    wins += int(result.won)
                    row = dict(index=index, seed=seed, deck_id=hashlib.sha256(bytes(deck)).hexdigest(),
                               won=result.won, nc84_reached=result.nc84_checkpoint is not None)
                    if result.won:
                        # Even a WIN that misses NC84 must be replay-validated and retained.
                        if record is None:
                            validate_record(build_record(deck, result.actions))
                        row.update(full_permutation=deck, actions=result.actions)
                    journal.write(json.dumps(row, separators=(',', ':'))+'\n')
                    if record is not None:
                        witnesses += 1
                        bank.write(json.dumps(record, separators=(',', ':'))+'\n')
                journal.flush()
                bank.flush()
                print(json.dumps(dict(strategy=strategy, processed=min(start+128, decks),
                                      decks=decks, wins=wins)), flush=True)
    runtime = time.monotonic()-started
    summary = dict(strategy=strategy, seed=seed, decks=decks, workers=workers, wins=wins,
                   win_rate=wins/decks, wins_per_second=wins/runtime, runtime_seconds=runtime,
                   nc84_winning_witnesses=witnesses)
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/standalone_natural_1000'))
    parser.add_argument('--seed', type=int, default=730000)
    parser.add_argument('--decks', type=int, default=10000)
    parser.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=['exponential', 'combined'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.decks <= 0 or args.workers <= 0:
        parser.error('decks and workers must be positive')
    code_hash = {name: file_sha256(ROOT/name) for name in ('src/env.py', 'src/natural/bank.py',
        'src/natural/heuristics.py', 'src/evaluation/heuristic.py')}
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(seed=args.seed, decks=args.decks, strategies=[], code_sha256=code_hash)
    for strategy in args.strategies:
        result = evaluate(strategy, seed=args.seed, decks=args.decks, workers=args.workers,
                          output=args.output, code_hash=code_hash)
        summary['strategies'].append(result)
        print(json.dumps(result), flush=True)
    summary['best_strategy'] = select_strategy(summary['strategies'])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()

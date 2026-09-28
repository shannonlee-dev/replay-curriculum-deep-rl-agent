"""Fixed-100-deck ablation of Dietrich suffix heuristics against frozen v3."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import multiprocessing
from pathlib import Path
from src.common.paths import ROOT

import numpy as np

from archive.privileged_teacher.build_bank import atomic_json, code_hashes
from archive.handoff_v3.build_bank import V3Journal, generate_attempt, handoff_report
from archive.handoff_v3.teacher import solve
from src.natural.bank import file_sha256
from archive.privileged_teacher.teacher import TeacherConfig


def frozen_seeds(row, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, row['index']]))
    if rng.permutation(np.arange(2, 100)).tolist() != row['full_permutation']:
        raise ValueError('frozen permutation differs from seed')
    prefix_seed, search_seed = (int(rng.integers(2**32)) for _ in range(2))
    if prefix_seed != row['prefix']['seed']:
        raise ValueError('frozen prefix seed differs')
    return prefix_seed, search_seed


def run_one(args):
    row, manifest, mode, recovery, teacher = args
    prefix_seed, search_seed = frozen_seeds(row, manifest['seed'])
    result = generate_attempt(row['full_permutation'], None, None,
        TeacherConfig(**manifest['teacher']), index=row['index'],
        deterministic=row['prefix']['deterministic'], prefix_seed=prefix_seed,
        search_seed=search_seed, teacher=teacher, trajectory=row['trajectory'],
        search_fn=partial(solve, mode=mode, recovery=recovery))
    result['search_seed'] = search_seed
    return result


def report(rows):
    summary = handoff_report(rows)
    for target, stats in summary['by_handoff'].items():
        attempts = [h for r in rows for h in r['handoffs'] if h['target'] == int(target)]
        depths = [h['deepest_remaining'] for h in attempts]
        stats.update(exhausted_rate=stats['teacher_failed_exhausted']/len(attempts) if attempts else None,
                     deepest_remaining_min=min(depths) if depths else None,
                     deepest_remaining_mean=sum(depths)/len(depths) if depths else None,
                     witness_rate_all_decks=stats['win_found']/len(rows) if rows else None)
    summary['decision'] = ('stopped_zero_wins' if len(rows) == 100 and not summary['nc84_witnesses']
                           else 'complete_witnesses_found' if len(rows) == 100 else 'incomplete')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('runs/nc84_bank_pilot_v3'))
    parser.add_argument('--output', type=Path, default=Path('runs/nc84_dietrich_fixed100'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.baseline/'manifest.json').read_text())
    with V3Journal(args.baseline, manifest, resume=True) as journal:
        rows = list(journal.rows)
    if len(rows) != 100 or any(r['record'] is not None for r in rows):
        raise ValueError('requires completed 0/100 v3 baseline')
    for row in rows:
        frozen_seeds(row, manifest['seed'])
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = dict(code_hashes(), **{p: file_sha256(ROOT/p) for p in
        ('archive/handoff_v3/build_bank.py', 'archive/handoff_v3/teacher.py', 'archive/handoff_v3/evaluate.py')})
    for p in ('src/env.py', 'src/natural/bank.py', 'archive/privileged_teacher/prefix.py'):
        if hashes[p] != manifest['code_sha256'][p]:
            raise ValueError(f'baseline invariant changed: {p}')
    frozen = dict(baseline_attempts_sha256=file_sha256(args.baseline/'attempts.jsonl'),
                  baseline_manifest_sha256=file_sha256(args.baseline/'manifest.json'),
                  deck_ids=[r['deck_id'] for r in rows],
                  seeds=[dict(index=r['index'], prefix=frozen_seeds(r, manifest['seed'])[0],
                              search=frozen_seeds(r, manifest['seed'])[1]) for r in rows],
                  baseline_decision='stopped_zero_wins', deck_count=100)
    atomic_json(args.output/'frozen_cohort.json', frozen)
    comparison = {'v3': report(rows)}
    for recovery in (False, True):
        for mode in ('rollout', 'search'):
            name = ('distance' if recovery else 'exponential')+'_'+mode
            current = dict(manifest, generation_mode='frozen_v3_dietrich', variant=name,
                           code_sha256=hashes, frozen=frozen,
                           heuristic=dict(alpha=1.5, recovery_a=3.5, recovery_b=1., recovery_g=.03,
                               recovery=recovery, distance_exponent_sign='negative', neural_prior=False),
                           rollout_definition='greedy complete turn; v3 local enumeration; no backtracking',
                           effective_beam_width=1 if mode == 'rollout' else manifest['teacher']['beam_width'])
            teacher = dict(model_sha256=manifest['model_sha256'], config=current['teacher'],
                           code_sha256=hashes, variant=name)
            with V3Journal(args.output/name, current, resume=args.resume) as journal:
                tasks = [(r, manifest, mode, recovery, teacher) for r in rows[journal.next_index:]]
                with ProcessPoolExecutor(max_workers=manifest['workers'],
                        mp_context=multiprocessing.get_context('spawn')) as pool:
                    for row in pool.map(run_one, tasks):
                        journal.append(row)
                        print(json.dumps(dict(variant=name, index=row['index'], status=row['status'],
                                              nodes=row['nodes'])), flush=True)
                journal.export()
                comparison[name] = report(journal.rows)
            atomic_json(args.output/'comparison.json', comparison)
    atomic_json(args.output/'comparison.json', comparison)


if __name__ == '__main__':
    main()

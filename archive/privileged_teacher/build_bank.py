"""Offline natural bank CLI. Importing this module never loads/runs a model."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from src.natural.bank import BUCKETS, SPLITS, build_record, deck_id, file_sha256, validate_record, validate_provenance
from archive.privileged_teacher.teacher import PPOPrior, TeacherConfig, search_checkpoint
from archive.privileged_teacher.prefix import rollout_prefix

from src.common.paths import ROOT
STATUSES = {'win_found', 'teacher_failed_timeout', 'teacher_failed_node_limit',
            'teacher_failed_exhausted', 'duplicate_deck', 'policy_failed_to_reach_nc84'}
SEARCH_STATUSES = {s for s in STATUSES if s.startswith('teacher_failed_')} | {'win_found'}
PROGRESS_FIELDS = ('deepest_remaining', 'max_cards_played', 'deepest_turn', 'frontier_size')


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def code_hashes():
    return {name: file_sha256(ROOT/name) for name in
            ('src/env.py', 'src/natural/bank.py', 'archive/privileged_teacher/teacher.py', 'archive/privileged_teacher/prefix.py', 'archive/privileged_teacher/build_bank.py')}


def pilot_report(rows):
    """Predeclared first-100-deck assessment, never a test of NC learning efficacy."""
    from src.evaluation.synthetic import wilson_interval
    attempts = [r for r in rows if r['status'] != 'duplicate_deck'][:100]
    n = len(attempts)
    wins = sum(r['status'] == 'win_found' for r in attempts)
    complete = n == 100
    status = ('incomplete_pilot' if not complete else
              'teacher_redesign_required' if wins == 0 else
              'insufficient_witnesses' if wins < 3 else
              'witnesses_found_more_tuning_needed' if wins < 10 else
              'scale_estimate_available')
    ready = complete and wins >= 10
    low, high = wilson_interval(wins, n)
    mean_seconds = sum(r.get('attempt_seconds', r.get('elapsed_seconds', 0.)) for r in attempts)/n if n else 0.
    estimate = 5000/(wins/n) if ready else None
    return dict(decks=n, wins=wins, status=status, complete=complete,
                minimum_witnesses_met=complete and wins >= 3, scale_estimate_ready=ready,
                win_rate=wins/n if n else None, wilson_ci_95=[low, high],
                estimated_decks_for_5000=estimate,
                estimated_deck_range_for_5000=[5000/high, 5000/low] if ready else None,
                estimated_search_hours_for_5000=estimate*mean_seconds/3600 if ready else None,
                estimate_scope='prefix + suffix search + witness construction; excludes model load/export IO')


def generate_attempt(permutation, policy, prior, config, *, index, deterministic,
                     prefix_seed, search_seed, teacher):
    """Public-observation policy prefix, THEN privileged suffix search. No fallback."""
    started = time.monotonic()
    prefix = rollout_prefix(permutation, policy, deterministic=deterministic)
    prefix_info = asdict(prefix)
    prefix_info.pop('checkpoint')
    prefix_info.update(deterministic=deterministic, seed=prefix_seed)
    row = dict(index=index, deck_id=deck_id(permutation), prefix=prefix_info,
               status=prefix.status, failure_reason=prefix.reason, nodes=0,
               elapsed_seconds=0., record=None, **{k: None for k in PROGRESS_FIELDS})
    if prefix.checkpoint is not None:
        result = search_checkpoint(prefix.checkpoint, prior, config, seed=search_seed)
        stats = {k: getattr(result, k) for k in ('nodes', 'elapsed_seconds', *PROGRESS_FIELDS)}
        row.update(status=result.status, failure_reason=None if result.status == 'win_found' else result.status,
                   **stats)
        if result.status == 'win_found':
            record = build_record(permutation, prefix.actions+result.actions,
                                  prefix_action_offset=len(prefix.actions))
            if record['checkpoints'] != [prefix.checkpoint]:
                raise ValueError('stored checkpoint differs from policy-reached state')
            record['teacher'] = teacher
            record['search'] = stats
            record['prefix_policy'] = dict(model_sha256=teacher['model_sha256'],
                                           deterministic=deterministic, seed=prefix_seed)
            validate_record(record)
            validate_provenance(record)
            row['record'] = record
    row['attempt_seconds'] = time.monotonic()-started
    return row


class BankJournal:
    """One fsynced attempt per line is authoritative; bank/summary are derived."""
    def __init__(self, output, manifest, *, resume):
        self.output = Path(output)
        self.manifest, self.resume = manifest, resume
        self.rows, self.seen = [], set()

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output/'.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest_path = self.output/'manifest.json'
            if self.resume:
                if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != self.manifest:
                    raise ValueError('resume manifest differs (seed/model/code/config); use a new bank directory')
                path = self.output/'attempts.jsonl'
                if not path.is_file():
                    raise ValueError('authoritative attempts.jsonl is missing; cannot resume safely')
                if path.exists():
                    with path.open() as stream:
                        for line in stream:
                            if not line.endswith('\n'):
                                raise ValueError('partial journal tail; preserve it and repair before resuming')
                            self._accept(json.loads(line))
            else:
                if any(p.name != '.lock' for p in self.output.iterdir()):
                    raise FileExistsError('bank output is not empty; use --resume or a new directory')
                atomic_json(manifest_path, self.manifest)
            self.stream = (self.output/'attempts.jsonl').open('a')
            return self
        except BaseException:
            self.lock.close()
            raise

    def __exit__(self, *args):
        self.stream.close()
        self.lock.close()

    @property
    def next_index(self):
        return len(self.rows)

    @property
    def wins(self):
        return sum(row['status'] == 'win_found' for row in self.rows)

    def _validate(self, row):
        if row.get('index') != len(self.rows) or row.get('status') not in STATUSES:
            raise ValueError('invalid attempt index/status')
        if (type(row.get('nodes')) is not int or row['nodes'] < 0
                or not isinstance(row.get('elapsed_seconds'), (int, float))
                or not math.isfinite(row['elapsed_seconds']) or row['elapsed_seconds'] < 0):
            raise ValueError('invalid attempt statistics')
        if self.manifest.get('schema_version') == 2 and row['status'] != 'duplicate_deck':
            if (not isinstance(row.get('attempt_seconds'), (int, float))
                    or not math.isfinite(row['attempt_seconds']) or row['attempt_seconds'] < 0):
                raise ValueError('invalid attempt duration')
            prefix = row.get('prefix', {})
            if row['status'] in SEARCH_STATUSES:
                if prefix.get('status') != 'prefix_reached':
                    raise ValueError('teacher search requires a reached policy prefix')
                if any(type(row.get(k)) is not int or row[k] < 0 for k in PROGRESS_FIELDS):
                    raise ValueError('missing/invalid teacher progress diagnostics')
                remaining = prefix.get('remaining')
                if (type(remaining) is not int or not 82 <= remaining <= 86
                        or row['deepest_remaining'] + row['max_cards_played'] != remaining):
                    raise ValueError('inconsistent teacher depth metrics')
            elif (prefix.get('status') != 'policy_failed_to_reach_nc84'
                  or not prefix.get('reason') or row['nodes'] != 0
                  or any(row.get(k, 'missing') is not None for k in PROGRESS_FIELDS)):
                raise ValueError('prefix reach failure must be distinct from a search failure')
        identity = row.get('deck_id')
        if not isinstance(identity, str) or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
            raise ValueError('invalid attempt deck_id')
        if (identity in self.seen) != (row['status'] == 'duplicate_deck'):
            raise ValueError('duplicate deck status mismatch')
        if row['status'] == 'win_found':
            if not isinstance(row.get('record'), dict):
                raise ValueError('winner must contain a validated witness')
            validate_record(row['record'])
            validate_provenance(row['record'])
            if self.manifest.get('schema_version') == 2:
                if row['record']['schema_version'] != 2:
                    raise ValueError('on-policy generator cannot journal a legacy full-search record')
                if any(row[k] != row['record']['search'][k] for k in ('nodes', 'elapsed_seconds', *PROGRESS_FIELDS)):
                    raise ValueError('record search diagnostics differ from attempt')
            if row['record']['deck_id'] != identity:
                raise ValueError('winner deck_id mismatch')
        elif row.get('record') is not None:
            raise ValueError('failed attempt must not contain a bank record')

    def _accept(self, row):
        self._validate(row)
        self.rows.append(row)
        self.seen.add(row['deck_id'])

    def append(self, row):
        self._validate(row)
        self.stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.rows.append(row)
        self.seen.add(row['deck_id'])

    def export(self):
        counts = {s: {b: 0 for b in BUCKETS} for s in SPLITS}
        decks = {s: 0 for s in SPLITS}
        temp = self.output/'bank.jsonl.tmp'
        with temp.open('w') as stream:
            for row in self.rows:
                if row['status'] == 'win_found':
                    record = row['record']
                    stream.write(json.dumps(record, separators=(',', ':'))+'\n')
                    decks[record['split']] += 1
                    for cp in record['checkpoints']:
                        counts[record['split']][cp['bucket']] += 1
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(self.output/'bank.jsonl')
        searched = [row for row in self.rows if row['status'] in SEARCH_STATUSES]
        policy_reached = sum(row.get('prefix', {}).get('status') == 'prefix_reached' for row in self.rows)
        summary = dict(sampled_decks=len(self.rows), unique_sampled_decks=len(self.seen),
                       searched_decks=len(searched), wins_found=self.wins,
                       success_rate=self.wins/len(self.seen) if self.seen else 0.,
                       policy_reached_nc84=policy_reached,
                       prefix_reach_rate=policy_reached/len(self.seen) if self.seen else 0.,
                       teacher_success_rate=self.wins/len(searched) if searched else 0.,
                       policy_failure_reasons=dict(Counter(r['prefix']['reason'] for r in self.rows
                           if r['status'] == 'policy_failed_to_reach_nc84')),
                       pilot=pilot_report(self.rows),
                       reasons=dict(Counter(row['status'] for row in self.rows)),
                       mean_search_nodes=sum(r['nodes'] for r in searched)/len(searched) if searched else 0.,
                       mean_search_seconds=sum(r['elapsed_seconds'] for r in searched)/len(searched) if searched else 0.,
                       split_decks=decks, checkpoint_counts=counts,
                       bank_sha256=file_sha256(self.output/'bank.jsonl'))
        atomic_json(self.output/'summary.json', summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description='On-policy natural-prefix NC84 bank; default: 100-deck pilot.')
    parser.add_argument('--config', default=str(ROOT/'archive/nc84_direct_mix/config.yaml'))
    parser.add_argument('--model', required=True, help='R80 stage-complete checkpoint with .json metadata')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--winning-decks', type=int, help='production target; requires >=10 witnesses in first 100 decks')
    parser.add_argument('--max-sampled-decks', type=int, help='cumulative limit: default 100 for pilot, 100000 for production')
    parser.add_argument('--prefix-mode', choices=['deterministic', 'stochastic', 'mixed'], default='mixed',
                        help='mixed alternates deterministic/stochastic by deck index')
    parser.add_argument('--seed', type=int, default=730000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    pilot_only = args.winning_decks is None
    limit = args.max_sampled_decks if args.max_sampled_decks is not None else (100 if pilot_only else 100000)
    if limit <= 0 or args.torch_threads <= 0 or args.seed < 0 or (args.winning_decks is not None and args.winning_decks <= 0):
        parser.error('budgets/threads must be positive; seed must be nonnegative')
    if not pilot_only and limit < 100:
        parser.error('production requires at least 100 sampled decks for the pilot gate')
    if pilot_only and limit != 100:
        parser.error('the pilot uses exactly 100 decks; omit --max-sampled-decks or use 100')
    import yaml
    config = TeacherConfig(**yaml.safe_load(Path(args.config).read_text())['teacher'])
    model_path = Path(args.model).resolve()
    metadata = json.loads(model_path.with_suffix('.json').read_text())
    if metadata.get('current_stage') != 'R80' or metadata.get('status') != 'promoted':
        parser.error('--model must be an R80 stage-complete checkpoint')
    manifest = dict(schema_version=2, generation_mode='on_policy_natural_prefix',
                    prefix_mode=args.prefix_mode, target_remaining=84, target_tolerance=2,
                    seed=args.seed, teacher=asdict(config), model_sha256=file_sha256(model_path),
                    model_metadata_sha256=file_sha256(model_path.with_suffix('.json')),
                    code_sha256=code_hashes(), device=args.device, torch_threads=args.torch_threads,
                    permutation_rng='numpy.default_rng(SeedSequence([seed,index]))')
    teacher = dict(model_sha256=manifest['model_sha256'], config=manifest['teacher'],
                   code_sha256=manifest['code_sha256'], prefix_mode=args.prefix_mode)
    with BankJournal(args.output_dir, manifest, resume=args.resume) as journal:
        assessment = pilot_report(journal.rows)
        if assessment['complete'] and not assessment['scale_estimate_ready']:
            journal.export()
            raise SystemExit(f"Pilot {assessment['wins']}/100: {assessment['status']}; redesign/tune in a new bank directory.")
        # Prefix alone samples stochastic actions. Seed it per deck, independent of
        # previous deck timeouts; search has its own NumPy RNG and no policy sampling.
        import torch
        from sb3_contrib import MaskablePPO
        torch.set_num_threads(args.torch_threads)
        policy = MaskablePPO.load(model_path, device=args.device)
        if policy.num_timesteps != metadata.get('global_steps'):
            raise ValueError('R80 model timestep differs from checkpoint metadata')
        prior = PPOPrior(policy)
        try:
            while journal.next_index < limit and (
                    not pilot_report(journal.rows)['complete']
                    or (not pilot_only and journal.wins < args.winning_decks)):
                index = journal.next_index
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, index]))
                permutation = rng.permutation(np.arange(2, 100)).tolist()
                identity = deck_id(permutation)
                prefix_seed, search_seed = (int(rng.integers(2**32)) for _ in range(2))
                deterministic = args.prefix_mode == 'deterministic' or (args.prefix_mode == 'mixed' and index % 2 == 0)
                if identity in journal.seen:
                    row = dict(index=index, deck_id=identity, status='duplicate_deck', nodes=0,
                               elapsed_seconds=0., attempt_seconds=0., record=None)
                else:
                    torch.manual_seed(prefix_seed)
                    row = generate_attempt(permutation, policy, prior, config, index=index,
                                           deterministic=deterministic, prefix_seed=prefix_seed,
                                           search_seed=search_seed, teacher=teacher)
                journal.append(row)
                print(json.dumps({k: row.get(k) for k in
                      ('index', 'status', 'failure_reason', 'nodes', 'elapsed_seconds', *PROGRESS_FIELDS)}), flush=True)
                if journal.next_index % 10 == 0 or row['status'] == 'win_found':
                    summary = journal.export()
                    print(json.dumps(summary), flush=True)
                assessment = pilot_report(journal.rows)
                if assessment['complete'] and not assessment['scale_estimate_ready']:
                    break
        finally:
            summary = journal.export()
            summary['target_winning_decks'] = args.winning_decks
            summary['target_reached'] = (summary['pilot']['scale_estimate_ready'] if pilot_only
                                         else journal.wins >= args.winning_decks and summary['pilot']['scale_estimate_ready'])
            atomic_json(Path(args.output_dir)/'summary.json', summary)
        print(json.dumps(summary, indent=2))
        if not summary['target_reached']:
            raise SystemExit(f"Pilot/production target not reached: {summary['pilot']['status']}. "
                             'Prefix failures and search failures are recorded separately; neither proves NC ineffective.')


if __name__ == '__main__':
    main()

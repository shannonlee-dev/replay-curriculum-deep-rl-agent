"""Manual, model-free audit of stopped NC84 direct-mix training and its WIN bank."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from src.natural.bank import NaturalBank, file_sha256
from archive.nc84_direct_mix.env import NaturalConditionedEnv

from src.common.paths import ROOT
SOURCE_CODE = ('src/env.py', 'src/natural/bank.py', 'archive/nc84_direct_mix/env.py',
               'archive/nc84_direct_mix/trainer.py')


def audit_training_logs(run, *, expected_envs):
    """Count completed training episodes, validating both terminal WIN and return."""
    paths = sorted((Path(run)/'episodes').glob('*.monitor.csv'))
    if type(expected_envs) is not int or expected_envs < 1 or len(paths) != expected_envs:
        raise ValueError(f'expected {expected_envs} monitor files, found {len(paths)}')
    sources = {s: dict(completed_episodes=0, wins=0, positive_return_episodes=0,
                      completed_episode_transitions=0)
               for s in ('natural_conditioned', 'reverse')}
    fingerprints = {str(p): file_sha256(p) for p in paths}
    required = {'r', 'l', 't', 'won', 'sampled_target_remaining', 'reset_source'}
    for path in paths:
        with path.open(newline='') as stream:
            header = stream.readline()
            if not header.startswith('#') or not isinstance(json.loads(header[1:]), dict):
                raise ValueError(f'{path}: invalid Monitor metadata')
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or ()) != required:
                raise ValueError(f'{path}: unexpected Monitor columns')
            previous_time = -1.
            for row in reader:
                location = f'{path}:{reader.line_num+1}'
                if set(row) != required or any(row[k] in (None, '') for k in required):
                    raise ValueError(f'{location}: malformed or partial row')
                source = row['reset_source']
                if source not in sources or row['won'] not in ('True', 'False'):
                    raise ValueError(f'{location}: invalid source/WIN row')
                reward, length = float(row['r']), int(row['l'])
                elapsed, remaining = float(row['t']), int(row['sampled_target_remaining'])
                won = row['won'] == 'True'
                if not math.isfinite(reward) or reward != float(won):
                    raise ValueError(f'{location}: sparse reward/WIN mismatch')
                if not 1 <= length <= 196 or not math.isfinite(elapsed) or elapsed < previous_time:
                    raise ValueError(f'{location}: invalid length/time row')
                if source == 'natural_conditioned' and not 82 <= remaining <= 86:
                    raise ValueError(f'{location}: NC source is not an NC84 reset')
                if source == 'reverse' and remaining not in (12, 20, 72, 76, 80, 84):
                    raise ValueError(f'{location}: unexpected synthetic reset')
                previous_time = elapsed
                stats = sources[source]
                stats['completed_episodes'] += 1
                stats['wins'] += int(won)
                stats['positive_return_episodes'] += int(reward == 1.)
                stats['completed_episode_transitions'] += length
    for path in paths:
        if file_sha256(path) != fingerprints[str(path)]:
            raise ValueError(f'{path}: log changed during audit; stop the run first')
    episodes = sum(s['completed_episodes'] for s in sources.values())
    transitions = sum(s['completed_episode_transitions'] for s in sources.values())
    for stats in sources.values():
        n = stats['completed_episodes']
        stats.update(win_rate=stats['wins']/n if n else None,
                     completed_episode_share=n/episodes if episodes else None,
                     completed_transition_share=(stats['completed_episode_transitions']/transitions
                                                 if transitions else None))
    return dict(sources=sources, monitor_files=len(paths), file_sha256=fingerprints,
                reward_win_consistency=True,
                scope='Completed training rollout episodes; evaluation games and unfinished episodes excluded. '
                      'Interrupted rollouts may not have been applied in an optimizer update.')


def _state_key(hand, played, piles, deck):
    return (tuple(sorted(hand)), tuple(sorted(played)), tuple(piles), tuple(deck))


def replay_training_resets(bank, samples):
    """Exercise the actual NC reset path with saved continuations, without a model."""
    if type(samples) is not int or samples < 1:
        raise ValueError('reset_samples must be a positive integer')
    witnesses = {}
    with bank.path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record['split'] != 'train':
                continue
            cp = record['checkpoints'][0]
            key = _state_key(cp['hand'], cp['played'], cp['pile_tops'], cp['future_deck_order'])
            witnesses[key] = (record['deck_id'], record['actions'][cp['action_offset']:])
    env = NaturalConditionedEnv(bank=bank, nc_probability=1.)
    seen = set()
    try:
        for index in range(samples):
            obs, info = env.reset(seed=810000+index)
            key = _state_key(env.hand, env.played, env.piles, env.deck)
            if key not in witnesses or info['reset_source'] != 'natural_conditioned':
                raise ValueError('training reset did not restore a train-split checkpoint')
            if obs.shape != (300,) or env.action_space.n != 393 or env.played_this_turn != 0:
                raise ValueError('NC training observation/action/boundary contract mismatch')
            identity, actions = witnesses[key]
            seen.add(identity)
            for action in actions:
                if env.terminated or not env.action_masks()[action]:
                    raise ValueError('training reset witness contains an illegal action')
                _, reward, done, truncated, info = env.step(action)
                if truncated or reward != float(done and info['won']):
                    raise ValueError('training reset replay violates sparse reward contract')
            if not env.won or not env.terminated:
                raise ValueError('training reset continuation did not WIN')
    finally:
        env.close()
    return dict(samples=samples, unique_train_decks=len(seen), wins=samples,
                method='Saved witness replay through NaturalConditionedEnv.reset; not PPO wins')


def audit(run, bank_path, *, reset_samples=64, progress=None):
    run, bank_path = Path(run).resolve(), Path(bank_path).resolve()
    report = dict(run=str(run), bank_path=str(bank_path), checks_passed=False,
                  started_at=datetime.now(timezone.utc).isoformat())
    started = time.monotonic()
    notify = progress or (lambda message: None)
    try:
        manifest_path = run/'manifest.json'
        input_hashes = {str(p): file_sha256(p) for p in (manifest_path, bank_path)}
        manifest = json.loads(manifest_path.read_text())
        if (manifest['arm'] != 'nc'
                or manifest['config']['experiment']['nc_probability'] != .30):
            raise ValueError('expected the NC84 30% direct-mix arm')
        if input_hashes[str(bank_path)] != manifest['bank_sha256']:
            raise ValueError('bank hash differs from the bank used for training')
        code_hashes = {name: file_sha256(ROOT/name) for name in SOURCE_CODE}
        if any(manifest['code_sha256'].get(n) != h for n, h in code_hashes.items()):
            raise ValueError('training/replay source code changed since this run')
        report['code_sha256'] = code_hashes
        notify('1/3: Counting completed NC training episodes and checking reward/WIN consistency...')
        report['training'] = audit_training_logs(run, expected_envs=manifest['config']['num_envs'])
        input_hashes.update(report['training']['file_sha256'])
        notify('2/3: Replaying EVERY bank record from natural start and independently from NC84; '
               'this can take several minutes. No model is loaded.')
        bank = NaturalBank(bank_path)
        if bank.schema_version != 2:
            raise ValueError('expected schema-v2 NC84 bank')
        minimum = manifest['config']['experiment']['minimum_winning_decks']
        if len(bank.deck_ids) < minimum:
            raise ValueError(f'bank has fewer than {minimum} required unique winning decks')
        counts = bank.counts()
        if sum(buckets[84] for buckets in counts.values()) != len(bank.deck_ids):
            raise ValueError('expected exactly one NC84 checkpoint per unique deck')
        if 'bank_counts' in manifest:
            saved_counts = {s: {int(k): v for k, v in buckets.items()}
                            for s, buckets in manifest['bank_counts'].items()}
            if counts != saved_counts:
                raise ValueError('bank split counts differ from training manifest')
        report['bank'] = dict(verified_unique_decks=len(bank.deck_ids), sha256=bank.sha256,
                             checkpoint_counts=counts, full_start_to_win_replay=True,
                             independent_nc84_to_win_replay=True, split_and_unique_ids_verified=True)
        notify(f'3/3: Replaying {reset_samples} train resets through the actual NC environment...')
        report['training_reset_replay'] = replay_training_resets(bank, reset_samples)
        evaluation_path = run/'evaluations.jsonl'
        if evaluation_path.exists():
            input_hashes[str(evaluation_path)] = file_sha256(evaluation_path)
            last = None
            with evaluation_path.open() as stream:
                for line in stream:
                    last = json.loads(line)
            if last is not None:
                report['latest_saved_evaluation'] = dict(stage_steps=last['stage_steps'],
                    nc84_validation=last['results']['NC84_validation'], source_usage=last['source_usage'])
        for path, digest in input_hashes.items():
            if file_sha256(path) != digest:
                raise ValueError(f'{path}: audit input changed; stop the run first')
        if {name: file_sha256(ROOT/name) for name in SOURCE_CODE} != code_hashes:
            raise ValueError('source code changed during audit')
        report['input_sha256'] = input_hashes
        nc = report['training']['sources']['natural_conditioned']
        report['checks_passed'] = nc['completed_episodes'] > 0
        if not nc['completed_episodes']:
            report['verdict'] = 'no_completed_nc_training_episodes'
        elif nc['wins'] == 0:
            report['verdict'] = 'direct_mix_failed_for_observed_run'
            report['next_step'] = 'Prepare natural reverse curriculum from the same train-split winning trajectories; do not auto-launch.'
        else:
            report['verdict'] = 'nc_training_wins_present'
            report['next_step'] = 'Compare training WIN frequency and held-out validation before changing the design.'
        report['limitations'] = [
            'This diagnoses the observed run and budget, not every possible direct-mix configuration.',
            'Historical Monitor rows contain no deck_id; individual episode split membership cannot be reconstructed. '
            'Code hashes, bank splits and sampled train-reset replays are checked instead.',
        ]
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        report.update(checks_passed=False, verdict='integrity_failure_investigate_first',
                      error=f'{type(exc).__name__}: {exc}')
    report['runtime_seconds'] = time.monotonic()-started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Stopped NC arm directory')
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON report; never overwritten')
    parser.add_argument('--reset-samples', type=int, default=64)
    args = parser.parse_args()
    if args.reset_samples < 1:
        parser.error('--reset-samples must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Claim the output before expensive replay; preserve all existing reports.
    with args.output.open('x') as stream:
        report = audit(args.run, args.bank, reset_samples=args.reset_samples,
                       progress=lambda message: print(message, flush=True))
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    nc = report.get('training', {}).get('sources', {}).get('natural_conditioned')
    if nc:
        print(f"NC training WIN: {nc['wins']}/{nc['completed_episodes']} completed episodes", flush=True)
    print(f"Verdict: {report['verdict']}\nReport: {args.output.resolve()}", flush=True)
    if 'error' in report:
        print(report['error'], flush=True)
    raise SystemExit(0 if report['checks_passed'] else 2)


if __name__ == '__main__':
    main()

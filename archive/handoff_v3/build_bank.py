"""V3: public natural rollout, deepest live handoff, replay-verified NC84 witness.

The R84-start generator/search and banks remain untouched. Record schema stays
v2 for training compatibility; the v3 journal contains private rollout evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from archive.privileged_teacher.build_bank import BankJournal, ROOT, atomic_json, code_hashes, pilot_report
from src.natural.bank import (build_record, checkpoint_from_env, deck_id, file_sha256,
                          natural_start, validate_record, validate_provenance)
from archive.privileged_teacher.teacher import PPOPrior, TeacherConfig, search_checkpoint

HANDOFFS = tuple(range(80, 47, -4))


def rollout_trajectory(permutation, policy, *, deterministic=True):
    """One real deck, no resampling. Only obs/mask cross the policy boundary.

    NC84 uses the existing first boundary in [82,86]. Later boundaries use
    nearest handoff +/-2 (ties go to the smaller target), one closest state per
    target. Stop at death or the first live boundary with remaining <=48.
    """
    started = time.monotonic()
    env = natural_start(permutation)
    actions, checkpoints, prefix_cp = [], {}, None
    prefix, reason, turns = None, 'action_limit', 0
    for _ in range(196):
        action, _ = policy.predict(env._get_obs(), deterministic=deterministic,
                                   action_masks=env.action_masks())
        action = int(action)
        _, reward, done, truncated, info = env.step(action)
        actions.append(action)
        if reward != float(done and info['won']) or truncated:
            raise RuntimeError('policy rollout violates reward/termination contract')
        if done:
            reason = 'policy_win' if info['won'] else 'policy_loss'
            break
        if env.played_this_turn == 0:
            turns += 1
            remaining = env._remaining_cards()
            if prefix_cp is None and 82 <= remaining <= 86:
                prefix_cp = checkpoint_from_env(env, len(actions))
                prefix = dict(status='prefix_reached', reason=None, actions=list(actions),
                              remaining=remaining, cards_played=len(env.played), turns=turns,
                              elapsed_seconds=time.monotonic()-started)
            elif prefix_cp is not None:
                candidates = [k for k in HANDOFFS if abs(remaining-k) <= 2]
                if candidates:
                    target = min(candidates, key=lambda k: (abs(remaining-k), k))
                    previous = checkpoints.get(target)
                    if previous is None or abs(remaining-target) < abs(previous['remaining_count']-target):
                        checkpoints[target] = checkpoint_from_env(env, len(actions))
                if remaining <= 48:
                    reason = 'depth_reached'
                    break
        if prefix_cp is None and env._remaining_cards() < 82:
            reason = 'overshot_target_window'
            break
    if prefix is None:
        prefix = dict(status='policy_failed_to_reach_nc84', reason=reason, actions=list(actions),
                      remaining=env._remaining_cards(), cards_played=len(env.played), turns=turns,
                      elapsed_seconds=time.monotonic()-started)
    return dict(prefix=prefix, prefix_checkpoint=prefix_cp, actions=actions,
                checkpoints=[dict(target=k, checkpoint=checkpoints[k]) for k in HANDOFFS if k in checkpoints],
                stop_reason=reason, remaining=env._remaining_cards(), turns=turns,
                elapsed_seconds=time.monotonic()-started)


def generate_attempt(permutation, policy, prior, config, *, index, deterministic,
                     prefix_seed, search_seed, teacher, trajectory=None, search_fn=None):
    started = time.monotonic()
    if trajectory is None:
        trajectory = rollout_trajectory(permutation, policy, deterministic=deterministic)
    prefix = dict(trajectory['prefix'], deterministic=deterministic, seed=prefix_seed)
    candidates = sorted(trajectory['checkpoints'], key=lambda c: c['checkpoint']['remaining_count'])
    row = dict(index=index, deck_id=deck_id(permutation), full_permutation=list(permutation),
               prefix=prefix, trajectory=trajectory, status=prefix['status'], record=None,
               failure_reason=prefix['reason'], selected_handoff=candidates[0]['target'] if candidates else None,
               winning_handoff=None, handoffs=[], nodes=0, elapsed_seconds=0.,
               deepest_remaining=None, max_cards_played=None, deepest_turn=None, frontier_size=None)
    if trajectory['prefix_checkpoint'] is not None and not candidates:
        row.update(status='policy_failed_to_reach_handoff', failure_reason=trajectory['stop_reason'])
    search_started = time.monotonic()
    for candidate in candidates:
        # A single deck budget across all fallback roots; no hidden 9x budget.
        seconds_left = config.timeout_seconds_per_deck-(time.monotonic()-search_started)
        nodes_left = config.max_nodes_per_deck-row['nodes']
        if row['handoffs'] and (seconds_left <= 0 or nodes_left <= 0):
            row['status'] = 'teacher_failed_timeout' if seconds_left <= 0 else 'teacher_failed_node_limit'
            break
        cp, target = candidate['checkpoint'], candidate['target']
        budget = replace(config, timeout_seconds_per_deck=max(seconds_left, 1e-12),
                         max_nodes_per_deck=nodes_left)
        result = (search_fn or search_checkpoint)(cp, prior, budget,
                                   seed=np.random.SeedSequence([search_seed, target]))
        stats = asdict(result)
        stats.pop('actions')
        row['handoffs'].append(dict(target=target, remaining=cp['remaining_count'],
                                    action_offset=cp['action_offset'], **stats))
        row['nodes'] += result.nodes
        row['elapsed_seconds'] += result.elapsed_seconds
        row.update(status=result.status, deepest_remaining=min(
            result.deepest_remaining, row['deepest_remaining'] if row['deepest_remaining'] is not None else 98),
            frontier_size=result.frontier_size)
        row['max_cards_played'] = prefix['remaining']-row['deepest_remaining']
        prefix_offset = len(prefix['actions'])
        policy_turns = trajectory['actions'][prefix_offset:cp['action_offset']].count(392)
        row['deepest_turn'] = max(row['deepest_turn'] or 0, policy_turns+result.deepest_turn)
        if result.status == 'win_found':
            actions = trajectory['actions'][:cp['action_offset']]+result.actions
            record = build_record(permutation, actions, prefix_action_offset=prefix_offset)
            if record['checkpoints'] != [trajectory['prefix_checkpoint']]:
                raise ValueError('NC84 checkpoint differs from saved policy state')
            # Legacy v2 progress is relative to NC84; handoffs retain raw Rk stats.
            record['teacher'] = teacher
            record['search'] = {k: row[k] for k in ('nodes', 'elapsed_seconds', 'deepest_remaining',
                                                   'max_cards_played', 'deepest_turn', 'frontier_size')}
            record['prefix_policy'] = dict(model_sha256=teacher['model_sha256'],
                                           deterministic=deterministic, seed=prefix_seed)
            record['handoff'] = dict(generator_version=3, target=target, checkpoint=cp,
                                     policy_action_offset=cp['action_offset'], search=stats)
            validate_record(record)
            validate_provenance(record)
            row.update(record=record, winning_handoff=target)
            break
        if result.status != 'teacher_failed_exhausted':
            break
    row['failure_reason'] = None if row['status'] == 'win_found' else row['failure_reason'] or row['status']
    tried = {h['target'] for h in row['handoffs']}
    row['unattempted_handoffs'] = [c['target'] for c in candidates if c['target'] not in tried]
    row['attempt_seconds'] = time.monotonic()-started
    return row


def handoff_report(rows):
    rows = [r for r in rows if r['status'] != 'duplicate_deck']
    n = len(rows)
    reached = sum(r['prefix']['status'] == 'prefix_reached' for r in rows)
    wins = sum(r['status'] == 'win_found' for r in rows)
    selected = Counter(r['selected_handoff'] for r in rows if r['selected_handoff'] is not None)
    total_selected = sum(selected.values())
    reach, distribution, outcomes = {}, {}, {}
    for target in HANDOFFS:
        key = str(target)
        count = sum(any(c['target'] == target for c in r['trajectory']['checkpoints']) for r in rows)
        reach[key] = dict(count=count, rate_all_decks=count/n if n else None,
                          rate_given_nc84=count/reached if reached else None)
        distribution[key] = dict(count=selected[target], fraction=selected[target]/total_selected if total_selected else None)
        attempts = [h for r in rows for h in r['handoffs'] if h['target'] == target]
        counts = Counter(h['status'] for h in attempts)
        outcomes[key] = dict(attempts=len(attempts), **{s: counts[s] for s in (
            'win_found', 'teacher_failed_timeout', 'teacher_failed_exhausted', 'teacher_failed_node_limit')},
            success_rate=counts['win_found']/len(attempts) if attempts else None,
            nodes=sum(h['nodes'] for h in attempts), seconds=sum(h['elapsed_seconds'] for h in attempts))
    return dict(decks=n, prefix_nc84_reached=reached, prefix_nc84_reach_rate=reached/n if n else None,
                reach=reach, selected_distribution=distribution, selected_decks=total_selected,
                by_handoff=outcomes, nc84_witnesses=wins, nc84_witness_rate=wins/n if n else None,
                nc84_witness_rate_given_prefix=wins/reached if reached else None,
                no_handoff_after_nc84=sum(r['status'] == 'policy_failed_to_reach_handoff' for r in rows),
                policy_stop_reasons=dict(Counter(r['trajectory']['stop_reason'] for r in rows)))


class V3Journal(BankJournal):
    def _validate(self, row):
        # Reuse generic journal identity/statistics/witness validation, without
        # changing the v1/v2 status vocabulary or its R84-relative validation.
        generic = row
        if row.get('status') == 'policy_failed_to_reach_handoff':
            generic = dict(row, status='teacher_failed_exhausted')
        super()._validate(generic)
        if row['status'] == 'duplicate_deck':
            return
        trajectory = row['trajectory']
        if deck_id(row['full_permutation']) != row['deck_id']:
            raise ValueError('trajectory deck identity mismatch')
        env = natural_start(row['full_permutation'])
        checkpoints = {c['checkpoint']['action_offset']: c['checkpoint'] for c in trajectory['checkpoints']}
        prefix_cp = trajectory['prefix_checkpoint']
        if prefix_cp is not None:
            checkpoints[prefix_cp['action_offset']] = prefix_cp
        observed = set()
        for offset, action in enumerate(trajectory['actions'], 1):
            if env.terminated:
                raise ValueError('policy trajectory continues after termination')
            env.step(action)
            if offset in checkpoints:
                if checkpoint_from_env(env, offset) != checkpoints[offset]:
                    raise ValueError('saved trajectory checkpoint differs from replay')
                observed.add(offset)
        if observed != set(checkpoints) or env._remaining_cards() != trajectory['remaining']:
            raise ValueError('incomplete trajectory checkpoint replay')
        labels = [c['target'] for c in trajectory['checkpoints']]
        if (len(labels) != len(set(labels)) or any(k not in HANDOFFS for k in labels)
                or any(abs(c['target']-c['checkpoint']['remaining_count']) > 2
                       for c in trajectory['checkpoints'])):
            raise ValueError('invalid handoff target/checkpoint assignment')
        if prefix_cp is not None:
            offset = prefix_cp['action_offset']
            if (row['prefix']['status'] != 'prefix_reached'
                    or not 82 <= prefix_cp['remaining_count'] <= 86
                    or row['prefix']['remaining'] != prefix_cp['remaining_count']
                    or row['prefix']['actions'] != trajectory['actions'][:offset]
                    or any(c['checkpoint']['action_offset'] <= offset for c in trajectory['checkpoints'])):
                raise ValueError('policy prefix differs from trajectory')
        elif row['status'] != 'policy_failed_to_reach_nc84' or trajectory['checkpoints']:
            raise ValueError('handoff requires an NC84 prefix')
        ordered = sorted(trajectory['checkpoints'], key=lambda c: c['checkpoint']['remaining_count'])
        if row['selected_handoff'] != (ordered[0]['target'] if ordered else None):
            raise ValueError('selected handoff is not deepest live checkpoint')
        if len(row['handoffs']) > len(ordered):
            raise ValueError('too many handoff searches')
        if any(h['status'] != 'teacher_failed_exhausted' for h in row['handoffs'][:-1]):
            raise ValueError('fallback requires exhausted search')
        if row['unattempted_handoffs'] != [c['target'] for c in ordered[len(row['handoffs']):]]:
            raise ValueError('unattempted handoff list mismatch')
        for h, c in zip(row['handoffs'], ordered):
            cp = c['checkpoint']
            if (h['target'] != c['target'] or h['action_offset'] != cp['action_offset']
                    or h['remaining'] != cp['remaining_count']
                    or h['deepest_remaining']+h['max_cards_played'] != h['remaining']):
                raise ValueError('handoff diagnostics mismatch')
        if row['nodes'] != sum(h['nodes'] for h in row['handoffs']):
            raise ValueError('handoff nodes mismatch')
        if row['status'] == 'win_found':
            record = row['record']
            if (not row['handoffs'] or row['handoffs'][-1]['status'] != 'win_found'
                    or row['winning_handoff'] != row['handoffs'][-1]['target']
                    or record['handoff']['target'] != row['winning_handoff']):
                raise ValueError('winning handoff mismatch')
            offset = record['handoff']['policy_action_offset']
            if (record['actions'][:offset] != trajectory['actions'][:offset]
                    or record['checkpoints'] != [prefix_cp]
                    or record['handoff']['checkpoint'] != ordered[len(row['handoffs'])-1]['checkpoint']):
                raise ValueError('winning witness differs from policy trajectory')

    def export(self):
        summary = super().export()
        summary['generator_version'] = 3
        summary['handoff'] = handoff_report(self.rows)
        atomic_json(self.output/'summary.json', summary)
        return summary


_WORKER = None


def _init_worker(model_path, device, threads, config, teacher, seed, mode, expected_steps):
    global _WORKER
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(threads)
    policy = MaskablePPO.load(model_path, device=device)
    if policy.num_timesteps != expected_steps:
        raise ValueError('R80 model timestep differs from metadata')
    _WORKER = (policy, PPOPrior(policy), config, teacher, seed, mode)


def _run_deck(index):
    import torch
    policy, prior, config, teacher, seed, mode = _WORKER
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    permutation = rng.permutation(np.arange(2, 100)).tolist()
    prefix_seed, search_seed = (int(rng.integers(2**32)) for _ in range(2))
    deterministic = mode == 'deterministic' or (mode == 'mixed' and index % 2 == 0)
    torch.manual_seed(prefix_seed)
    return generate_attempt(permutation, policy, prior, config, index=index,
                            deterministic=deterministic, prefix_seed=prefix_seed,
                            search_seed=search_seed, teacher=teacher)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output-dir', default=str(ROOT/'runs/nc84_bank_pilot_v3'))
    parser.add_argument('--config', default=str(ROOT/'archive/nc84_direct_mix/config.yaml'))
    parser.add_argument('--prefix-mode', choices=['mixed', 'deterministic', 'stochastic'], default='mixed')
    parser.add_argument('--seed', type=int, default=730000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.torch_threads < 1 or args.seed < 0:
        parser.error('workers/threads must be positive and seed nonnegative')
    import yaml
    config = TeacherConfig(**yaml.safe_load(Path(args.config).read_text())['teacher'])
    model_path = Path(args.model).resolve()
    metadata = json.loads(model_path.with_suffix('.json').read_text())
    if metadata.get('current_stage') != 'R80' or metadata.get('status') != 'promoted':
        parser.error('model must be an R80 stage-complete checkpoint')
    hashes = dict(code_hashes(), **{'archive/handoff_v3/build_bank.py': file_sha256(__file__)})
    manifest = dict(schema_version=3, record_schema_version=2,
        generation_mode='natural_policy_deepest_handoff', seed=args.seed,
        prefix_mode=args.prefix_mode, target_remaining=84, target_tolerance=2,
        handoff_targets=list(HANDOFFS), handoff_tolerance=2, stop_remaining=48,
        handoff_selection='nearest +/-2, ties lower, closest boundary per target; deepest first',
        fallback='shallower saved checkpoints after exhausted, shared per-deck time/node budget',
        teacher=asdict(config), model_sha256=file_sha256(model_path),
        model_metadata_sha256=file_sha256(model_path.with_suffix('.json')),
        code_sha256=hashes, device=args.device, workers=args.workers, torch_threads=args.torch_threads,
        pilot_decks=100, permutation_rng='numpy.default_rng(SeedSequence([seed,index]))')
    teacher = dict(model_sha256=manifest['model_sha256'], config=manifest['teacher'],
                   code_sha256=hashes, prefix_mode=args.prefix_mode, generator_version=3)
    with V3Journal(args.output_dir, manifest, resume=args.resume) as journal:
        if journal.next_index > 100:
            raise ValueError('v3 pilot journal exceeds 100 decks')
        if journal.next_index < 100:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=_init_worker, initargs=(str(model_path), args.device, args.torch_threads,
                    config, teacher, args.seed, args.prefix_mode, metadata['global_steps'])) as pool:
                try:
                    for row in pool.map(_run_deck, range(journal.next_index, 100)):
                        journal.append(row)
                        journal.export()
                        print(json.dumps({k: row[k] for k in ('index', 'status', 'selected_handoff',
                            'winning_handoff', 'nodes', 'attempt_seconds')}, allow_nan=False), flush=True)
                finally:
                    journal.export()
        summary = journal.export()
        print(json.dumps(summary['handoff'], indent=2))
        # Completing a pilot is not a claim that production scaling is ready.
        print(json.dumps(pilot_report(journal.rows), indent=2))


if __name__ == '__main__':
    main()

"""Config-driven natural WIN-conditioned frontier; objective: unconditional natural WIN."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import signal
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.natural.bank import NaturalReverseBank, file_sha256, restore_checkpoint
from src.natural.curriculum import NC_SYNTHETIC_TARGETS, NC_SYNTHETIC_WEIGHTS
from src.common.source_usage import SourceUsageWrapper
from src.natural.curriculum import DISTRIBUTION, load_config, validate_natural_config, select_start, frontier_assessment
from src.common.source_usage import source_usage, add_usage, subtract_usage, usage_summary
from src.common.terminal_ui import Dashboard
from src.env import TheGameEnv

from src.common.paths import ROOT
from src.common.provenance import source_hashes
EVALUATION_PROTOCOL = dict(
    promotion='next_nc_terminal_win_signal_with_anchor_retention',
    nc_distribution=DISTRIBUTION,
    final_objective_diagnostic='unconditional_natural_full_game_terminal_win_rate',
    remaining_card_metric_used=False,
)


class NaturalReverseEnv(TheGameEnv):
    def __init__(self, *, bank, target, nc_episode_probability, rehearsal, backward_teacher_prob=.30):
        if target not in bank.stages or not np.isfinite(nc_episode_probability) or not 0 <= nc_episode_probability <= 1:
            raise ValueError('invalid natural stage/reset probability')
        super().__init__(reset_source='reverse', target_remaining=84,
                         curriculum_targets=NC_SYNTHETIC_TARGETS,
                         curriculum_probs=(np.asarray(NC_SYNTHETIC_WEIGHTS)/sum(NC_SYNTHETIC_WEIGHTS)).tolist(),
                         backward_teacher_prob=backward_teacher_prob)
        self.nc_episode_probability = nc_episode_probability
        self._targets = bank.stages[:bank.stages.index(target)+1]
        self._states = {t: [cp for _, cp in bank.checkpoints('train', t)] for t in self._targets}
        if any(not states for states in self._states.values()):
            raise ValueError('natural training stage has no checkpoints')
        self._probs = ([1.] if len(self._targets) == 1 else
                       [rehearsal['older_weight']/(len(self._targets)-1)]*(len(self._targets)-1)+[rehearsal['current_weight']])

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if self.np_random.random() >= self.nc_episode_probability:
            return super().reset(options=options)
        target = int(self.np_random.choice(self._targets, p=self._probs))
        states = self._states[target]
        restore_checkpoint(self, states[int(self.np_random.integers(len(states)))])
        return self._get_obs(), self._get_info()


# Historical name retained for callers; implementation lives in common accounting.
WinUsageWrapper = SourceUsageWrapper


from src.common.checkpoints import read_checkpoint, save_checkpoint


def validate_runtime(c, bank, checkpoint, stop_at=None):
    from src.evaluation.natural import validate_natural_suite
    n = validate_natural_config(c, require_criteria=False)
    for key in ('num_envs', 'torch_threads', 'max_steps_per_stage', 'eval_interval'):
        if type(c.get(key)) is not int or c[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('n_steps', 'batch_size', 'n_epochs'):
        if type(c['ppo'].get(key)) is not int or c['ppo'][key] <= 0:
            raise ValueError(f'ppo.{key} must be a positive integer')
    if type(c.get('seed')) is not int or c['seed'] < 0:
        raise ValueError('seed must be nonnegative integer')
    rollout = c['num_envs']*c['ppo']['n_steps']
    budget = c['max_steps_per_stage']//rollout*rollout
    if rollout % c['ppo']['batch_size'] or budget < n['promotion']['min_stage_steps']:
        raise ValueError('batch must divide rollout; rounded budget must cover minimum stage steps')
    if tuple(n['stages']) != bank.stages:
        raise ValueError('bank extraction stages differ from config')
    if stop_at is not None and stop_at not in bank.stages:
        raise ValueError('--stop-at must be a configured stage')
    if len(bank.deck_ids) < n['minimum_winning_decks']:
        raise ValueError('not enough validated unique winning decks')
    counts = bank.counts()
    deficiencies = [f'{split}/NC{t}: {counts[split][t]} < {minimum}'
                    for split, minimum in n['minimum_stage_decks'].items()
                    for t in n['stages'] if counts[split][t] < minimum]
    e = n['evaluation']
    validate_natural_suite(games=e['natural_large_games'], seed=e['natural_seed'], forbidden_decks=bank.deck_ids)
    validate_natural_suite(games=e['natural_large_games'], seed=e.get('held_out_natural_seed', e['natural_seed']+10000000),
                           forbidden_decks=bank.deck_ids)
    metadata = read_checkpoint(checkpoint)
    if metadata['current_stage'].startswith('NC') and metadata['bank_sha256'] != bank.sha256:
        raise ValueError('selected checkpoint was trained with a different bank')
    return metadata, deficiencies


def run(c, checkpoint, bank, out, *, stop_at=None, checkpoint_every=None, stop_requested=None,
        scan_only=False, verbose=False, resume_run=False):
    """Advance a persisted state machine only at completed PPO/evaluation boundaries."""
    from src.common.checkpoints import run_lock
    out = Path(out)
    if not resume_run:
        out.mkdir(parents=True, exist_ok=False)
    with run_lock(out):
        return _run(c, checkpoint, bank, out, stop_at=stop_at, checkpoint_every=checkpoint_every,
                    stop_requested=stop_requested, scan_only=scan_only, resume_run=resume_run)


def _run(c, checkpoint, bank, out, *, stop_at, checkpoint_every, stop_requested, scan_only, resume_run):
    import contextlib
    import torch
    import yaml
    from stable_baselines3.common.env_util import make_vec_env
    from src.evaluation.natural import evaluate_nc, evaluate_natural, evaluate_reverse_panel
    from src.natural.difficulty_scan import difficulty_scan
    from src.common.checkpoints import (SCHEMA, HISTORY_FIELDS, atomic_json, canonical_hash, validate_run,
        scan_provenance, cached_scan, capture_runtime, restore_runtime, log_state, reconcile_logs)
    from src.common.checkpoints import load_or_create

    metadata, deficiencies = validate_runtime(c, bank, checkpoint, stop_at)
    validate_natural_config(c)
    if deficiencies:
        raise ValueError('insufficient stage split coverage: '+ '; '.join(deficiencies))
    n = c['natural_curriculum']
    stages, e = n['stages'], n['evaluation']
    codes = source_hashes()
    rollout = c['num_envs']*c['ppo']['n_steps']
    budget = c['max_steps_per_stage']//rollout*rollout
    interval = math.ceil(c['eval_interval']/rollout)*rollout
    checkpoint_every = c['eval_interval'] if checkpoint_every is None else checkpoint_every
    if type(checkpoint_every) is not int or checkpoint_every <= 0:
        raise ValueError('checkpoint_every must be a positive integer')
    checkpoint_interval = math.ceil(checkpoint_every/rollout)*rollout
    stop_requested = stop_requested or (lambda: False)
    origin = metadata.get('origin_global_steps', metadata['global_steps'])
    if resume_run:
        manifest = validate_run(out, c, bank, metadata, code_hashes=codes)
        if stop_at is not None and stop_at != metadata['stop_at']:
            raise ValueError('resume stop-at differs from saved run')
        stop_at = metadata['stop_at']
        checkpoint_interval = manifest['checkpoint_interval']
    else:
        stop_at = stages[-1] if stop_at is None else stop_at
        manifest = dict(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=file_sha256(checkpoint),
            checkpoint_metadata_sha256=file_sha256(Path(checkpoint).with_suffix('.json')),
            bank=str(bank.path.resolve()), bank_sha256=bank.sha256, bank_counts=bank.counts(),
            bank_teacher=bank.teacher_provenance, bank_distribution=DISTRIBUTION,
            config=c, config_sha256=canonical_hash(c), code_sha256=codes,
            evaluation_protocol=EVALUATION_PROTOCOL, checkpoint_interval=checkpoint_interval,
            origin_global_steps=origin, natural_stages=stages, split_deficiencies=deficiencies,
            promotion=n['promotion'], nc_episode_probability=n['nc_episode_probability'],
            rollout_transitions=rollout, max_steps_per_stage=budget, evaluation_interval=interval,
            resume_scope='model, optimizer, environment, RNG, stage, steps, usage, evaluation and promotion state')
        (out/'config.resolved.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    torch.set_num_threads(c['torch_threads'])
    env = make_vec_env(TheGameEnv, n_envs=c['num_envs'], seed=c['seed'])
    ui = Dashboard()
    try:
        with (out/'runtime.log').open('a') as logfile, contextlib.redirect_stdout(logfile):
            model = load_or_create(c, env, checkpoint)
        if model.num_timesteps != metadata['global_steps']:
            raise ValueError('selected model step differs from metadata')
        if resume_run:
            expected = {k: v for k, v in metadata.items() if k != 'model_sha256'}
            if getattr(model, '_natural_metadata_sha256', None) != canonical_hash(expected):
                raise ValueError('checkpoint continuation metadata hash mismatch')
            state = copy.deepcopy(metadata['continuation'])
            reconcile_logs(out, metadata, write=True)
        else:
            provenance = scan_provenance(ROOT, checkpoint, bank, e)
            rows, reused = cached_scan(out.parent/'.natural_scan_cache', provenance,
                                       lambda: difficulty_scan(model, bank, e))
            if metadata['current_stage'].startswith('NC'):
                selection = dict(stage=int(metadata['current_stage'][2:]), reason='explicit checkpoint stage')
            elif n['start_stage'] == 'auto':
                selection = select_start(n, rows)
            else:
                selection = dict(stage=n['start_stage'], reason=n['start_stage_reason'],
                                 selection_method='explicit_experiment_configuration', rationale_source='user_supplied')
            scan = dict(stages=rows, selection=selection, provenance=provenance, reused=reused)
            atomic_json(out/'difficulty_scan.json', scan)
            manifest.update(start_selection=selection, difficulty_scan_sha256=file_sha256(out/'difficulty_scan.json'))
            atomic_json(out/'manifest.json', manifest)
            if scan_only:
                atomic_json(out/'run_summary.json', dict(status='scan_only', training_started=False, selection=selection))
                return scan
            target = selection['stage']
            if stages.index(target) > stages.index(stop_at):
                raise ValueError('selected start/resume stage is after --stop-at')
            steps = metadata.get('stage_steps', 0) if metadata['current_stage'].startswith('NC') else 0
            if steps % rollout or steps > budget:
                raise ValueError('selected checkpoint stage steps incompatible with rollout/budget')
            state = dict(target=target, start=model.num_timesteps-steps, phase='evaluate', results={}, held={},
                next_eval=model.num_timesteps, next_large=e['natural_large_interval'], last_large_steps=0,
                usage_base=metadata.get('source_usage', {}), last_usage=metadata.get('source_usage', {}),
                references={str(t): row['validation']['win_rate'] for t, row in rows.items()
                            if row['validation']['win_rate'] is not None},
                promotion_history=[], retention_history=[], terminal_status=None)
            with (out/'history.csv').open('x', newline='') as stream:
                csv.DictWriter(stream, fieldnames=HISTORY_FIELDS).writeheader()

        def stage_env():
            return make_vec_env(NaturalReverseEnv, n_envs=c['num_envs'],
                seed=c['seed']+stages.index(state['target'])*10000,
                env_kwargs=dict(bank=bank, target=state['target'], nc_episode_probability=n['nc_episode_probability'],
                                rehearsal=n['rehearsal'], backward_teacher_prob=c['backward_teacher_prob']),
                wrapper_class=SourceUsageWrapper)
        env.close()
        env = stage_env()
        model.set_env(env)
        if resume_run:
            restore_runtime(model, env)

        def record_for(status):
            target = state['target']; idx = stages.index(target)
            raw = add_usage(state['usage_base'], source_usage(env))
            results = state['results']
            return dict(checkpoint_kind='natural_reverse', continuation_schema=SCHEMA,
                current_stage=f'NC{target}', next_stage=f'NC{stages[idx+1]}' if idx+1 < len(stages) else None,
                status=status, global_steps=model.num_timesteps, stage_steps=model.num_timesteps-state['start'],
                total_added_steps=model.num_timesteps-origin, origin_global_steps=origin, stage_budget=budget,
                bank_sha256=bank.sha256, config=c, manifest_sha256=file_sha256(out/'manifest.json'), stop_at=stop_at,
                results=copy.deepcopy(results), nc_current=results.get(f'NC{target}'),
                nc_next=results.get(f'NC{stages[idx+1]}') if idx+1 < len(stages) else None,
                natural_full=results.get('natural_full'), natural_full_large=results.get('natural_full_large'),
                source_usage=raw, last_evaluation_usage=state['last_usage'],
                usage_cumulative=usage_summary(raw), usage_window=usage_summary(subtract_usage(raw, state['last_usage'])),
                retention_references=state['references'], last_large_steps=state['last_large_steps'],
                promotion_history=state['promotion_history'], retention_history=state['retention_history'],
                next_evaluation=max(0, state['next_eval']-model.num_timesteps))

        def persist(status, pending=None):
            model._natural_runtime = capture_runtime(model, env)
            record = record_for(status)
            record.update(continuation=copy.deepcopy(state), log_state=log_state(out), pending_evaluation=pending)
            model._natural_metadata_sha256 = canonical_hash(record)
            save_checkpoint(model, out, record)
            saved = read_checkpoint(out/'latest.zip')
            if pending:
                reconcile_logs(out, saved, write=True)
            return saved

        with ui:
            while True:
                if stop_requested():
                    record = persist('paused')
                    atomic_json(out/'run_summary.json', record)
                    return record
                target = state['target']; idx = stages.index(target)
                steps = model.num_timesteps-state['start']
                additional = model.num_timesteps-origin
                if state['phase'] == 'train':
                    if model.num_timesteps >= state['next_eval']:
                        state.update(phase='evaluate', results={})
                        continue
                    model.learn(total_timesteps=rollout, reset_num_timesteps=False, callback=None)
                    ui.update(record_for('training'))
                    if (model.num_timesteps-state['start']) % checkpoint_interval == 0:
                        persist('checkpoint')
                    continue
                if state['phase'] == 'evaluate':
                    jobs = [(f'NC{t}', lambda t=t: evaluate_nc(model, bank, split='validation', bucket=t,
                             games=e['nc_games'], seed=e['nc_seed'])) for t in stages[:min(idx+2, len(stages))]]
                    jobs.append(('natural_full', lambda: evaluate_natural(model, games=e['natural_games'],
                                seed=e['natural_seed'], forbidden_decks=bank.deck_ids)))
                    if additional >= state['next_large'] or 'natural_full_large' in state['results']:
                        jobs.append(('natural_full_large', lambda: evaluate_natural(model, games=e['natural_large_games'],
                                    seed=e['natural_seed'], forbidden_decks=bank.deck_ids)))
                    for key, evaluate in jobs:
                        if key not in state['results']:
                            state['results'][key] = evaluate()
                            if key == 'natural_full_large':
                                state['last_large_steps'] = additional
                                state['next_large'] = (additional//e['natural_large_interval']+1)*e['natural_large_interval']
                            persist('evaluating')
                        if stop_requested():
                            break
                    if stop_requested():
                        continue
                    state['phase'] = 'assess'
                    continue
                if state['phase'] == 'assess':
                    results = state['results']
                    assessment = frontier_assessment(n, target,
                        {int(k[2:]): v for k,v in results.items() if k.startswith('NC')},
                        {int(k): v for k,v in state['references'].items()}, steps, budget)
                    complete = all(row['passed'] for row in assessment.values())
                    exhausted = steps >= budget
                    status = ('final_stage_complete' if idx == len(stages)-1 else 'promoted') if complete else ('stalled' if exhausted else 'training')
                    if complete:
                        state['references'][str(target)] = max(state['references'].get(str(target), 0), results[f'NC{target}']['win_rate'])
                        state['promotion_history'].append(dict(stage=target, global_steps=model.num_timesteps, status=status))
                    state['retention_history'].append(dict(stage=target, global_steps=model.num_timesteps, checks=assessment))
                    record = record_for(status)
                    record['promotion_checks'] = assessment
                    state['last_usage'] = copy.deepcopy(record['source_usage'])
                    state['terminal_status'] = status
                    if complete or exhausted:
                        state['phase'] = 'advance' if complete and target != stop_at else 'held'
                    else:
                        state['phase'] = 'train'
                        state['next_eval'] = min(state['start']+(steps//interval+1)*interval,
                            state['start']+budget, origin+math.ceil(state['next_large']/rollout)*rollout)
                    persist(status, pending=record)
                    if complete or exhausted:
                        name = f'stage_NC{target}_'+('complete' if complete else 'stalled')
                        for suffix in ('.zip', '.json'):
                            dest = out/(name+suffix)
                            if not dest.exists():
                                with dest.open('xb') as stream:
                                    stream.write((out/('latest'+suffix)).read_bytes())
                    ui.update(record, history=True)
                    continue
                if state['phase'] == 'advance':
                    state['usage_base'] = add_usage(state['usage_base'], source_usage(env))
                    state.update(target=stages[idx+1], start=model.num_timesteps, phase='evaluate', results={})
                    env.close(); env = stage_env(); model.set_env(env)
                    persist('stage_started')
                    continue
                if state['phase'] == 'held':
                    jobs = [(f'NC{t}_test', lambda t=t: evaluate_nc(model, bank, split='test', bucket=t,
                             games=e['nc_games'], seed=e['nc_seed'])) for t in sorted({target, stages[-1]})]
                    jobs.extend([
                        ('natural_full', lambda: evaluate_natural(model, games=e['natural_large_games'],
                            seed=e.get('held_out_natural_seed', e['natural_seed']+10000000), forbidden_decks=bank.deck_ids)),
                        ('synthetic_regression', lambda: evaluate_reverse_panel(model, games=e['synthetic_games'],
                            seed=e['synthetic_seed'], backward_teacher_prob=c['backward_teacher_prob']))])
                    for key, evaluate in jobs:
                        if key not in state['held']:
                            state['held'][key] = evaluate()
                            persist('held_out')
                        if stop_requested():
                            break
                    if stop_requested():
                        continue
                    state['phase'] = 'done'
                    metadata = persist(state['terminal_status'])
                    continue
                if state['phase'] == 'done':
                    if not (out/'final_held_out.json').exists():
                        atomic_json(out/'final_held_out.json', dict(results=state['held'], global_steps=model.num_timesteps,
                            bank_sha256=bank.sha256, evaluation_protocol=EVALUATION_PROTOCOL,
                            model_sha256=metadata['model_sha256']))
                    atomic_json(out/'run_summary.json', metadata)
                    return metadata
    finally:
        env.close()


def main():
    import shlex
    from datetime import datetime
    from src.common.checkpoints import atomic_json, validate_run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--resume', type=Path, help='Continue an existing run directory in place')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--set', action='append', default=[], dest='overrides')
    parser.add_argument('--checkpoint-every', type=int)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--stop-at', type=lambda s: int(s.removeprefix('NC')))
    parser.add_argument('--check', action='store_true', help='Validate bank, config and checkpoint; never train')
    parser.add_argument('--scan-only', action='store_true', help='Run or reuse the difficulty scan; never train')
    args = parser.parse_args()
    resuming = args.resume is not None
    try:
        if resuming:
            out = args.resume.resolve()
            if not out.is_dir():
                raise ValueError('--resume requires a run directory')
            if args.output_dir and args.output_dir.resolve() != out:
                raise ValueError('resume must append to the same output directory')
            if args.scan_only or args.checkpoint_every is not None:
                raise ValueError('resume retains the saved scan and checkpoint schedule')
            c = load_config(args.config or out/'config.resolved.yaml', args.overrides)
            manifest = json.loads((out/'manifest.json').read_text())
            bank_path = args.bank or Path(manifest['bank'])
            checkpoint = out/'latest.zip'
        else:
            out = (args.output_dir or ROOT/'runs'/('natural_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
            if out.exists():
                raise ValueError('output directory exists; use --resume to continue it')
            c = load_config(args.config or ROOT/'configs/natural_replay.yaml', args.overrides)
            checkpoint = Path(c['resume'])
            if not checkpoint.is_absolute():
                checkpoint = ROOT/checkpoint
            bank_path = args.bank or ROOT/'runs/natural_campaign/bank.jsonl'
            c.update(output_dir=str(out), resume=str(checkpoint.resolve()))
        print('Validating bank, provenance and checkpoint…', flush=True)
        bank = NaturalReverseBank(bank_path, c['natural_curriculum']['stages'])
        metadata, deficiencies = validate_runtime(c, bank, checkpoint, args.stop_at)
        validate_natural_config(c)
        if deficiencies:
            raise ValueError('insufficient stage split coverage: '+ '; '.join(deficiencies))
        if resuming:
            validate_run(out, c, bank, metadata, code_hashes=source_hashes())
            if args.stop_at is not None and args.stop_at != metadata['stop_at']:
                raise ValueError('resume stop-at differs from saved run')
        if args.check:
            report = dict(status='checked', bank_sha256=bank.sha256, bank_counts=bank.counts(),
                unique_winning_decks=len(bank.deck_ids), checkpoint=metadata,
                checkpoint_sha256=file_sha256(checkpoint), split_deficiencies=deficiencies,
                criteria_ready=True, model_executed=False, training_started=False)
            if not resuming:
                out.mkdir(parents=True)
                atomic_json(out/'preflight.json', report)
            print(f'Preflight passed · {len(bank.deck_ids):,} winning decks · {out}')
            return
        requested = False
        def request_stop(signum, frame):
            nonlocal requested
            requested = True  # Repeated interrupts must not tear a PPO update/checkpoint.
        previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            print(f'Run: {out}', flush=True)
            print('Restoring saved progress…' if resuming else 'Difficulty scan: checking matching provenance cache…', flush=True)
            result = run(c, checkpoint, bank, out, stop_at=args.stop_at, checkpoint_every=args.checkpoint_every,
                         stop_requested=lambda: requested, scan_only=args.scan_only, resume_run=resuming)
            if args.scan_only:
                print(f"Difficulty scan {'reused' if result['reused'] else 'complete'} · start NC{result['selection']['stage']} · no training")
            elif result['status'] == 'paused':
                print(f"Paused safely · {result['current_stage']} · stage steps {result['stage_steps']:,}")
                print('./run_natural_train.sh --resume '+shlex.quote(str(out)))
            else:
                print(f"{result['status']} · {result['current_stage']} · {out}")
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f'Natural training stopped: {exc}\n')


if __name__ == '__main__':
    main()

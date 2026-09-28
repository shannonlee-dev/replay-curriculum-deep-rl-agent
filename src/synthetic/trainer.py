from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3.common.env_util import make_vec_env

from src.synthetic.curriculum import promotion_checks, replay_distribution, validate_distribution
from src.evaluation.synthetic import evaluate_model
from src.tools.plot_history import plot_history
from src.env import TheGameEnv

from src.common.paths import ROOT


from src.common.checkpoints import resolve_checkpoint, load_or_create, save_model


def validate_config(c):
    stages = c['stages']
    validate_distribution(stages, [1/len(stages)] * len(stages)) if stages else None
    if not stages or sorted(stages) != stages:
        raise ValueError('stages must be nonempty and strictly increasing')
    for key in ['num_envs', 'torch_threads', 'max_steps_per_stage', 'eval_interval']:
        if not isinstance(c[key], int) or c[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if c['on_max_steps'] not in ('stop', 'advance'):
        raise ValueError('on_max_steps must be stop or advance')
    if not 0 <= c['min_steps_per_stage'] <= c['max_steps_per_stage']:
        raise ValueError('min_steps_per_stage must fit within stage budget')
    if not 0 < c['learning_rate_stage_decay'] <= 1:
        raise ValueError('learning_rate_stage_decay must be in (0,1]')
    if not 0 <= c['backward_teacher_prob'] <= 1:
        raise ValueError('backward_teacher_prob must be in [0,1]')
    for key, value in c['promotion'].items():
        if not 0 <= value <= 1:
            raise ValueError(f'{key} must be in [0,1]')
    ev = c['evaluation']
    if ev['games'] <= 0 or ev['balanced_metric'] not in ('mean', 'min'):
        raise ValueError('invalid evaluation games or balanced_metric')
    for key in ('anchors', 'balanced_targets'):
        targets = ev[key]
        validate_distribution(targets, [1/len(targets)] * len(targets))
    ppo = c['ppo']
    for key in ('n_steps', 'batch_size', 'n_epochs'):
        if not isinstance(ppo[key], int) or ppo[key] <= 0:
            raise ValueError(f'ppo.{key} must be positive integer')
    rollout = c['num_envs'] * ppo['n_steps']
    if ppo['batch_size'] > rollout or rollout % ppo['batch_size']:
        raise ValueError('batch_size must divide n_steps * num_envs')
    if c['max_steps_per_stage'] < rollout:
        raise ValueError('max_steps_per_stage must allow at least one rollout')
    for stage in stages:
        replay_distribution(stage, {**c['replay'], 'stages': stages})
    bridge = c['bridge']
    if bridge['enabled']:
        probs, thresholds = bridge['natural_probs'], bridge['natural_min_win_rates']
        if not probs or len(probs) != len(thresholds) or sorted(set(probs)) != probs or probs[-1] != 1.:
            raise ValueError('bridge needs increasing probabilities ending at 1 and matching thresholds')
        if any(not 0 < p <= 1 for p in probs) or any(not 0 <= t <= 1 for t in thresholds):
            raise ValueError('invalid bridge probability or win threshold')
        if not 0 <= bridge['reverse_floor_win_rate'] <= 1 or bridge['max_steps_per_phase'] < rollout:
            raise ValueError('invalid bridge floor or budget')
        if bridge['max_steps_per_phase'] < c['min_steps_per_stage']:
            raise ValueError('bridge budget must cover min_steps_per_stage')


def make_env(c, target, natural_prob, log_dir, seed):
    if natural_prob is None:
        targets, probs = replay_distribution(target, {**c['replay'], 'stages': c['stages']})
        source, natural_prob = 'reverse', 0.
    else:
        targets, probs = [98], [1.]
        source = 'mixed' if natural_prob < 1 else 'natural'
    print(f'Reset distribution: {dict(zip(targets, probs))}; natural={natural_prob:.0%}', flush=True)
    return make_vec_env(
        TheGameEnv, n_envs=c['num_envs'], seed=seed,
        env_kwargs=dict(reset_source=source, target_remaining=target,
                        curriculum_targets=targets, curriculum_probs=probs,
                        mixed_natural_prob=natural_prob,
                        backward_teacher_prob=c['backward_teacher_prob']),
        monitor_dir=str(log_dir),
        monitor_kwargs={'info_keywords': ('won', 'sampled_target_remaining', 'reset_source')},
    )


def run(c, checkpoint, *, stop_at=None, quick=False):
    validate_config(c)
    torch.set_num_threads(c['torch_threads'])
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    # Never silently mix different runs or overwrite previous model selections.
    if (out/'history.csv').exists() or (out/'latest.zip').exists():
        raise FileExistsError(f'{out} already contains a run; choose a new --output-dir')
    (out/'config.resolved.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    history = out/'history.csv'
    fields = ['global_steps', 'current_stage', 'stage_steps', 'status', 'balanced_score']
    fields += [f'test_R{t}' for t in range(2,99,2)] + ['natural_full']
    with history.open('w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    stages = [t for t in c['stages'] if t >= c.get('start_at', c['stages'][0]) and (stop_at is None or t <= stop_at)]
    if quick:
        stages = stages[:1]
    schedule = [(f'R{t}', t, None, None, c['max_steps_per_stage']) for t in stages]
    if stages and stages[-1] == 98 and c['bridge']['enabled'] and stop_at is None and not quick:
        schedule += [(f'bridge_{i+1}', 98, p, threshold, c['bridge']['max_steps_per_phase'])
                     for i, (p, threshold) in enumerate(zip(c['bridge']['natural_probs'], c['bridge']['natural_min_win_rates']))]
    model, env = None, None
    best_balanced = -1.
    seen = set()
    ev = c['evaluation']
    rollout = c['ppo']['n_steps'] * c['num_envs']
    interval = math.ceil(c['eval_interval']/rollout) * rollout
    final_status = 'no_stages'
    stage_outcomes = []
    try:
        for stage_index, (name, target, natural_prob, natural_threshold, max_steps) in enumerate(schedule):
            print(f'\nCURRENT STAGE: {name}', flush=True)
            if env is not None:
                env.close()
            env = make_env(c, target, natural_prob, out/'episodes'/name, c['seed']+stage_index*10000)
            if model is None:
                model = load_or_create(c, env, checkpoint)
            else:
                model.set_env(env)
            model.learning_rate = c['ppo']['learning_rate'] * c['learning_rate_stage_decay']**stage_index
            model._setup_lr_schedule()
            start_steps = model.num_timesteps
            budget = (max_steps//rollout)*rollout  # never exceed the configured budget
            if budget < c['min_steps_per_stage']:
                raise ValueError('rounded rollout budget is below min_steps_per_stage')
            seen.add(target)
            replay_targets, _ = replay_distribution(target, {**c['replay'], 'stages': c['stages']})
            core = sorted(set(replay_targets + [target]))
            panel = set(core) | set(ev['anchors']) | set(ev['balanced_targets'])
            if ev['all_seen_targets']:
                panel |= seen
            best_current = -1.
            while True:
                stage_steps = model.num_timesteps - start_steps
                results = {}
                # Fixed seeds and single-target environments, independent of training RNG.
                for t in sorted(panel):
                    result = evaluate_model(model, source='reverse', target_remaining=t,
                                            games=ev['games'], seed=ev['seed'],
                                            backward_teacher_prob=c['backward_teacher_prob'])
                    results[f'R{t}'] = result
                    print(f'R{t}: {result["win_rate"]:.1%} ({result["wins"]}/{ev["games"]})', flush=True)
                natural = evaluate_model(model, source='natural', games=ev['games'], seed=ev['seed'])
                results['natural_full'] = natural
                print(f'natural: {natural["win_rate"]:.1%}', flush=True)
                rates = {t: results[f'R{t}']['win_rate'] for t in panel}
                if natural_prob is None:
                    checks = promotion_checks(target, rates, core, c['promotion'])
                    current_score = rates[target]
                else:
                    checks = {'natural': natural['win_rate'] >= natural_threshold,
                              'reverse_floor': natural_prob == 1. or rates[98] >= c['bridge']['reverse_floor_win_rate']}
                    current_score = natural['win_rate']
                checks['minimum_steps'] = stage_steps >= c['min_steps_per_stage']
                promoted = all(checks.values())
                exhausted = stage_steps >= budget
                status = 'promoted' if promoted else ('stalled' if exhausted else ('baseline' if stage_steps == 0 else 'training'))
                if status == 'stalled' and c['on_max_steps'] == 'advance':
                    status = 'stalled_advanced'
                if quick:
                    status = 'smoke_complete' if exhausted else 'smoke_baseline'
                    promoted = False
                scores = [rates[t] for t in ev['balanced_targets']]
                if ev['balanced_include_natural']:
                    scores.append(natural['win_rate'])
                balanced = float(np.mean(scores) if ev['balanced_metric'] == 'mean' else min(scores))
                metadata = dict(global_steps=model.num_timesteps, current_stage=name,
                                stage_steps=stage_steps, status=status, checks=checks,
                                balanced_score=balanced, current_score=current_score,
                                quick=quick, results=results)
                save_model(model, out, 'latest', metadata)
                if current_score > best_current:
                    best_current = current_score
                    save_model(model, out, 'best_current_stage', metadata)
                    save_model(model, out, f'best_{name}', metadata)
                if balanced > best_balanced:
                    best_balanced = balanced
                    save_model(model, out, 'best_balanced', metadata)
                row = {k: metadata[k] for k in fields[:5]}
                row.update({f'test_R{t}': rates[t] for t in panel})
                row['natural_full'] = natural['win_rate']
                with history.open('a', newline='') as f:
                    csv.DictWriter(f, fieldnames=fields).writerow(row)
                with (out/'evaluations.jsonl').open('a') as f:
                    f.write(json.dumps(metadata)+'\n')
                plot_history(history)
                print('PROMOTION CONDITIONS: '+', '.join(f'{k}: {"PASS" if v else "FAIL"}' for k,v in checks.items()), flush=True)
                if promoted:
                    save_model(model, out, f'stage_{name}_complete', metadata)
                    next_name = schedule[stage_index+1][0] if stage_index+1 < len(schedule) else 'finished'
                    print(f'PROMOTED {name} -> {next_name}', flush=True)
                    final_status = 'complete'
                    break
                if exhausted:
                    save_model(model, out, f'stage_{name}_{status}', metadata)
                    print(f'{name}: {status}; steps={stage_steps}', flush=True)
                    final_status = status
                    break
                print('continue training...', flush=True)
                model.learn(total_timesteps=min(interval, budget-stage_steps), reset_num_timesteps=False)
            stage_outcomes.append({'stage': name, 'status': status, 'stage_steps': stage_steps})
            if exhausted and not promoted and (c['on_max_steps'] == 'stop' or quick):
                break
    finally:
        if env is not None:
            env.close()
    if final_status == 'complete' and any(s['status'] == 'stalled_advanced' for s in stage_outcomes):
        final_status = 'completed_with_stalls'
    (out/'run_summary.json').write_text(json.dumps(
        {'status': final_status, 'stages': stage_outcomes}, indent=2)+'\n')
    print(f'Run ended: {final_status}; output={out}', flush=True)
    return final_status


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT/'configs/synthetic.yaml'))
    group = p.add_mutually_exclusive_group()
    group.add_argument('--resume')
    group.add_argument('--from-scratch', action='store_true')
    p.add_argument('--quick', action='store_true', help='128 new steps, 4 games per target, first selected stage only')
    p.add_argument('--start-at', type=int)
    p.add_argument('--stop-at', type=int)
    p.add_argument('--output-dir')
    p.add_argument('--device')
    p.add_argument('--num-envs', type=int)
    args = p.parse_args()
    c = yaml.safe_load(Path(args.config).read_text())
    if args.start_at is not None:
        if args.start_at not in c['stages']:
            p.error('--start-at must be a configured stage')
        # Keep full stage list for predecessor calculation; mark active start below.
        c['start_at'] = args.start_at
    if args.stop_at is not None and args.stop_at not in c['stages']:
        p.error('--stop-at must be a configured stage')
    if args.stop_at is not None and args.stop_at < c.get('start_at', c['stages'][0]):
        p.error('--stop-at must not precede --start-at')
    if args.device:
        c['device'] = args.device
    if args.num_envs is not None:
        c['num_envs'] = args.num_envs
    if args.output_dir:
        c['output_dir'] = args.output_dir
    if args.quick:
        c.update(num_envs=2, max_steps_per_stage=128, min_steps_per_stage=128, eval_interval=128)
        c['ppo'].update(n_steps=32, batch_size=64, n_epochs=1)
        c['evaluation']['games'] = 4
        if not args.output_dir:
            from datetime import datetime
            c['output_dir'] = str(ROOT/'models'/('smoke_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    checkpoint = None if args.from_scratch else resolve_checkpoint(args.resume or c['resume'])
    c['loaded_checkpoint'] = str(checkpoint) if checkpoint else None
    if not args.output_dir and not args.quick:
        base = Path(c['output_dir'])
        if (base/'history.csv').exists() or (base/'latest.zip').exists():
            from datetime import datetime
            c['output_dir'] = str(base/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
            print(f'Existing results preserved in {base}; starting a new run.', flush=True)
    print(f'Output directory: {c["output_dir"]}', flush=True)
    run(c, checkpoint, stop_at=args.stop_at, quick=args.quick)


if __name__ == '__main__':
    main()

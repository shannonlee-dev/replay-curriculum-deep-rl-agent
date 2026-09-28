"""DEPRECATED direct-mix R80 -> R84 A/B experiment. Main trainer, config and outputs are untouched."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch
import yaml
from stable_baselines3.common.env_util import make_vec_env

from archive.privileged_teacher.build_bank import atomic_json
from src.synthetic.curriculum import promotion_checks
from src.natural.bank import NaturalBank, file_sha256
from archive.nc84_direct_mix.env import NaturalConditionedEnv, NC_SYNTHETIC_TARGETS, CONTROL_WEIGHTS, SourceUsageWrapper
from src.common.source_usage import source_usage, merge_counts as _merge_counts
from src.evaluation.natural import natural_eval_games, PANEL, evaluate_nc, evaluate_natural, evaluate_reverse_panel
from src.env import TheGameEnv
from src.synthetic.trainer import load_or_create, save_model, validate_config

from src.common.paths import ROOT


def load_config(path=None):
    return yaml.safe_load(Path(path or ROOT/'archive/nc84_direct_mix/config.yaml').read_text())


def validate_experiment(c):
    validate_config(c)
    ex = c['experiment']
    if ex['nc_probability'] != .30:
        raise ValueError('v1 experiment fixes nc_probability at 0.30')
    if c['start_at'] != 84 or c['bridge']['enabled'] or c['on_max_steps'] != 'stop':
        raise ValueError('v1 experiment is R84-only with bridge disabled and stop on exhaustion')
    for key in ('minimum_winning_decks', 'nc_games', 'natural_large_games', 'natural_large_interval'):
        if type(ex[key]) is not int or ex[key] <= 0:
            raise ValueError(f'experiment.{key} must be positive integer')
    for value in (c['seed'], c['evaluation']['seed'], ex['nc_seed'], ex['natural_seed']):
        if type(value) is not int or value < 0:
            raise ValueError('seeds must be nonnegative integers')
    expected = dict(current_min_win_rate=.6, previous_min_win_rate=.6, older_floor_win_rate=.5)
    if c['promotion'] != expected:
        raise ValueError('v1 preserves the 60/60/50 synthetic promotion gates')



from src.common.checkpoints import validate_start


def validate_bank(c, bank):
    if len(bank.deck_ids) < c['experiment']['minimum_winning_decks']:
        raise ValueError('bank has fewer winning decks than configured; use a pilot config/--quick for smoke')
    for split in ('train', 'validation', 'test'):
        if not bank.checkpoints(split, 84):
            raise ValueError(f'NC84 {split} bucket is empty')


def make_experiment_env(c, bank, arm, out):
    common = dict(n_envs=c['num_envs'], seed=c['seed'], monitor_dir=str(out/'episodes'),
                  wrapper_class=SourceUsageWrapper,
                  monitor_kwargs={'info_keywords': ('won', 'sampled_target_remaining', 'reset_source')})
    if arm == 'nc':
        return make_vec_env(NaturalConditionedEnv, env_kwargs=dict(bank=bank,
                            nc_probability=c['experiment']['nc_probability'],
                            backward_teacher_prob=c['backward_teacher_prob']), **common)
    return make_vec_env(TheGameEnv, env_kwargs=dict(reset_source='reverse', target_remaining=84,
                        curriculum_targets=NC_SYNTHETIC_TARGETS, curriculum_probs=CONTROL_WEIGHTS,
                        backward_teacher_prob=c['backward_teacher_prob']), **common)


def _provenance(c, checkpoint, bank, arm):
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    names = ['src/env.py', 'src/synthetic/curriculum.py', 'src/evaluation/synthetic.py', 'src/synthetic/trainer.py',
             'src/natural/bank.py', 'archive/nc84_direct_mix/env.py', 'src/evaluation/natural.py', 'archive/nc84_direct_mix/trainer.py']
    return dict(arm=arm, git_revision=revision, code_sha256={n: file_sha256(ROOT/n) for n in names},
                checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=file_sha256(checkpoint),
                checkpoint_metadata_sha256=file_sha256(Path(checkpoint).with_suffix('.json')),
                bank=str(bank.path), bank_sha256=bank.sha256, bank_counts=bank.counts(),
                bank_teacher=bank.teacher_provenance,
                config=c, synthetic_panel=list(PANEL),
                reset_probabilities=({'NC84': .30, 'R84': .25, 'R80': .15, 'R76': .09,
                                      'R72': .06, 'R20': .10, 'R12': .05} if arm == 'nc' else
                                     {f'R{t}': p for t, p in zip(NC_SYNTHETIC_TARGETS, CONTROL_WEIGHTS)}))



def run(c, checkpoint, bank, *, arm='nc'):
    if arm not in ('nc', 'control'):
        raise ValueError('arm must be nc or control')
    validate_experiment(c)
    validate_bank(c, bank)
    metadata = validate_start(checkpoint)
    out = Path(c['output_dir'])
    # Atomically claim a new directory; never mix histories or resume a run implicitly.
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(c['torch_threads'])
    (out/'config.resolved.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    atomic_json(out/'manifest.json', _provenance(c, checkpoint, bank, arm))
    rollout = c['num_envs']*c['ppo']['n_steps']
    budget = c['max_steps_per_stage']//rollout*rollout
    if budget < c['min_steps_per_stage']:
        raise ValueError('rounded budget cannot satisfy minimum_steps')
    interval = math.ceil(c['eval_interval']/rollout)*rollout
    env = make_experiment_env(c, bank, arm, out)
    ex, ev = c['experiment'], c['evaluation']
    fields = ['global_steps', 'stage_steps', 'status', 'balanced_score']
    fields += [f'R{t}' for t in PANEL] + ['NC84_validation', 'NC88_validation', 'natural_wins', 'natural_games']
    best_current = best_balanced = -1.
    last_large_steps = 0
    try:
        model = load_or_create(c, env, checkpoint)
        if model.num_timesteps != metadata['global_steps']:
            raise ValueError('checkpoint model timestep differs from R80 metadata')
        start = model.num_timesteps
        with (out/'history.csv').open('w', newline='') as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()
        while True:
            steps = model.num_timesteps-start
            results = evaluate_reverse_panel(model, games=ev['games'], seed=ev['seed'],
                                              backward_teacher_prob=c['backward_teacher_prob'])
            rates = {t: results[f'R{t}']['win_rate'] for t in PANEL}
            checks = promotion_checks(84, rates, NC_SYNTHETIC_TARGETS, c['promotion'])
            checks['minimum_steps'] = steps >= c['min_steps_per_stage']
            promoted, exhausted = all(checks.values()), steps >= budget
            status = 'promoted' if promoted else ('stalled' if exhausted else ('baseline' if not steps else 'training'))
            games = natural_eval_games(steps, last_large_steps, promoted, exhausted,
                                       regular_games=ev['games'], large_games=ex['natural_large_games'],
                                       interval=ex['natural_large_interval'])
            results['natural_full'] = evaluate_natural(model, games=games, seed=ex['natural_seed'],
                                                       forbidden_decks=bank.deck_ids)
            if games == ex['natural_large_games']:
                last_large_steps = steps
            for bucket in (84, 88):
                results[f'NC{bucket}_validation'] = evaluate_nc(model, bank, split='validation', bucket=bucket,
                                                               games=ex['nc_games'], seed=ex['nc_seed'])
            balanced_scores = [rates[t] for t in ev['balanced_targets']]
            if ev['balanced_include_natural']:
                balanced_scores.append(results['natural_full']['win_rate'])
            balanced = float(np.mean(balanced_scores) if ev['balanced_metric'] == 'mean' else min(balanced_scores))
            record = dict(global_steps=model.num_timesteps, start_global_steps=start, current_stage='R84',
                          stage_steps=steps, arm=arm, status=status, balanced_score=balanced,
                          checks=checks, results=results, source_usage=source_usage(env))
            save_model(model, out, 'latest', record)
            if rates[84] > best_current:
                best_current = rates[84]
                save_model(model, out, 'best_R84', record)
            if balanced > best_balanced:
                best_balanced = balanced
                save_model(model, out, 'best_balanced', record)
            with (out/'evaluations.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            row = dict(global_steps=model.num_timesteps, stage_steps=steps, status=status, balanced_score=balanced)
            row.update({f'R{t}': rates[t] for t in PANEL})
            row.update({f'NC{b}_validation': results[f'NC{b}_validation']['win_rate'] for b in (84, 88)})
            row.update(natural_wins=results['natural_full']['wins'], natural_games=games)
            with (out/'history.csv').open('a', newline='') as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow(row)
            print(json.dumps(row), flush=True)
            if promoted or exhausted:
                save_model(model, out, 'stage_R84_complete' if promoted else 'stage_R84_stalled', record)
                held_out = evaluate_nc(model, bank, split='test', bucket=84, games=ex['nc_games'], seed=ex['nc_seed'])
                atomic_json(out/'final_held_out.json', dict(global_steps=model.num_timesteps, stage_steps=steps,
                            bank_sha256=bank.sha256, model_sha256=file_sha256(out/'latest.zip'),
                            results={'R84': results['R84'], 'R98': results['R98'], 'NC84_test': held_out,
                                     'natural_full': results['natural_full']}))
                atomic_json(out/'run_summary.json', record)
                return record
            model.learn(total_timesteps=min(interval, budget-steps), reset_num_timesteps=False)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'archive/nc84_direct_mix/config.yaml'))
    parser.add_argument('--resume', required=True, help='R80 stage-complete .zip, not an existing experiment output')
    parser.add_argument('--bank', required=True)
    parser.add_argument('--arm', choices=['nc', 'control'], default='nc')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device')
    parser.add_argument('--quick', action='store_true', help='128-step real PPO smoke, for manual execution')
    parser.add_argument('--check', action='store_true', help='validate config/bank/metadata only; no model load or inference')
    args = parser.parse_args()
    c = load_config(args.config)
    c.update(output_dir=args.output_dir, resume=str(Path(args.resume).resolve()))
    if args.device:
        c['device'] = args.device
    if args.quick:
        c.update(num_envs=2, min_steps_per_stage=128, max_steps_per_stage=128, eval_interval=128)
        c['ppo'].update(n_steps=32, batch_size=64, n_epochs=1)
        c['evaluation']['games'] = 4
        c['experiment'].update(minimum_winning_decks=1, nc_games=4, natural_large_games=4)
    validate_experiment(c)
    bank = NaturalBank(args.bank)
    validate_bank(c, bank)
    metadata = validate_start(args.resume)
    if args.check:
        if Path(args.output_dir).exists():
            parser.error('output directory already exists')
        print(json.dumps(dict(checkpoint=metadata, bank_counts=bank.counts(), model_executed=False), indent=2))
        return
    run(c, Path(args.resume).resolve(), bank, arm=args.arm)


if __name__ == '__main__':
    main()

"""Independent NC validation/test and unconditional natural evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.synthetic import evaluate_model, wilson_interval
from src.natural.bank import NaturalBank, deck_id, file_sha256, restore_checkpoint
from src.env import TheGameEnv

PANEL = (12, 20, 32, 48, 64, 68, 72, 76, 80, 84, 98)


def natural_eval_games(stage_steps, last_large_steps, promoted, exhausted, *,
                       regular_games=1000, large_games=10000, interval=1000000):
    return (large_games if promoted or exhausted or stage_steps//interval > last_large_steps//interval
            else regular_games)


def next_large_step(additional_steps, interval):
    return (additional_steps//interval+1)*interval


def validate_natural_suite(*, games, seed, forbidden_decks):
    forbidden = set(forbidden_decks)
    for i in range(games):
        if deck_id(natural_permutation(seed+i)) in forbidden:
            raise ValueError('natural evaluation deck overlaps bank; choose a disjoint evaluation protocol')


def natural_permutation(seed):
    # Matches TheGameEnv._reset_natural, including dtype and shuffle API.
    cards = np.arange(2, 100, dtype=np.int16)
    np.random.default_rng(seed).shuffle(cards)
    return cards.tolist()


def _play(model, env, observation):
    for _ in range(196):
        action, _ = model.predict(observation, deterministic=True, action_masks=env.action_masks())
        observation, reward, done, truncated, info = env.step(action)
        if reward != float(done and info['won']) or truncated:
            raise RuntimeError('sparse reward/termination contract violated')
        if done:
            return int(info['won'])
    raise RuntimeError('episode exceeded rule-derived action bound')


def _result(wins, games, **kwargs):
    return dict(wins=wins, games=games, win_rate=wins/games if games else None,
                wilson_ci_95=wilson_interval(wins, games) if games else None, **kwargs)


def evaluate_nc(model, bank, *, split, bucket=84, games=1000, seed=200000):
    if split not in ('validation', 'test') or type(games) is not int or games <= 0:
        raise ValueError('NC evaluation requires validation/test split and positive games')
    states = sorted(bank.checkpoints(split, bucket), key=lambda item: item[0])
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(states))[:games]
    env, wins = TheGameEnv(), 0
    try:
        for index in indices:
            _, checkpoint = states[int(index)]
            restore_checkpoint(env, checkpoint)
            wins += _play(model, env, env._get_obs())
    finally:
        env.close()
    return _result(wins, len(indices), source='natural_conditioned', split=split, bucket=bucket,
                   unique_decks=len(indices), available_decks=len(states), seed=seed,
                   status='evaluated' if len(indices) else 'empty_bucket')


def evaluate_natural(model, *, games=1000, seed=900000, forbidden_decks=()):
    if type(games) is not int or games <= 0:
        raise ValueError('games must be positive')
    # Check the entire planned suite before policy inference; no rejection-resampling bias.
    validate_natural_suite(games=games, seed=seed, forbidden_decks=forbidden_decks)
    env, wins = TheGameEnv(reset_source='natural'), 0
    try:
        for i in range(games):
            obs, _ = env.reset(seed=seed+i)
            wins += _play(model, env, obs)
    finally:
        env.close()
    return _result(wins, games, source='natural', seed=seed)


def evaluate_reverse_panel(model, *, games=1000, seed=100000, backward_teacher_prob=.30):
    return {f'R{t}': evaluate_model(model, source='reverse', target_remaining=t, games=games,
                                   seed=seed, backward_teacher_prob=backward_teacher_prob)
            for t in PANEL}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--games', type=int, default=10000, help='unconditional natural games')
    parser.add_argument('--synthetic-games', type=int, default=1000)
    parser.add_argument('--nc-games', type=int, default=1000, help='cap; unique decks only')
    parser.add_argument('--natural-seed', type=int, default=900000)
    parser.add_argument('--synthetic-seed', type=int, default=100000)
    parser.add_argument('--nc-seed', type=int, default=200000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--torch-threads', type=int, default=1)
    args = parser.parse_args()
    if min(args.games, args.synthetic_games, args.nc_games, args.torch_threads) <= 0:
        parser.error('game counts and threads must be positive')
    output = Path(args.output)
    if output.exists():
        parser.error('output exists; choose a new filename')
    bank = NaturalBank(args.bank)
    if not bank.checkpoints(args.split, 84):
        parser.error(f'NC84 {args.split} bucket is empty')
    from sb3_contrib import MaskablePPO
    import torch
    torch.set_num_threads(args.torch_threads)
    model = MaskablePPO.load(args.model, device=args.device)
    results = evaluate_reverse_panel(model, games=args.synthetic_games, seed=args.synthetic_seed)
    results['NC84'] = evaluate_nc(model, bank, split=args.split, games=args.nc_games, seed=args.nc_seed)
    results['NC88'] = evaluate_nc(model, bank, split=args.split, bucket=88, games=args.nc_games, seed=args.nc_seed)
    results['natural_full'] = evaluate_natural(model, games=args.games, seed=args.natural_seed,
                                              forbidden_decks=bank.deck_ids)
    report = dict(model_sha256=file_sha256(args.model), bank_sha256=bank.sha256,
                  global_steps=model.num_timesteps, split=args.split,
                  protocol=vars(args), results=results)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

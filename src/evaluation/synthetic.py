from __future__ import annotations
import argparse
import json
import math
import numpy as np
from sb3_contrib import MaskablePPO
from src.env import TheGameEnv


def wilson_interval(wins: int, n: int, z: float = 1.96):
    if n <= 0:
        return 0., 0.
    p = wins / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0., center-half), min(1., center+half)


def evaluate_model(model, *, source='natural', target_remaining=98, games=1000,
                   seed=100000, deterministic=True, backward_teacher_prob=.30):
    """Independent fixed-target evaluation, never the training reset mixture."""
    if games <= 0:
        raise ValueError('games must be positive')
    env = TheGameEnv(reset_source=source, target_remaining=target_remaining,
                     backward_teacher_prob=backward_teacher_prob)
    wins, remaining, cards_played, turn_lengths = 0, [], [], []
    try:
        for i in range(games):
            obs, info = env.reset(seed=seed+i)
            for _ in range(200):  # 98 plays + at most 98 nonempty turns
                if model is None:
                    action = int(env.np_random.choice(np.flatnonzero(env.action_masks())))
                else:
                    action, _ = model.predict(obs, deterministic=deterministic,
                                              action_masks=env.action_masks())
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            else:
                raise RuntimeError('Episode exceeded rule-derived action bound')
            wins += int(info['won'])
            remaining.append(info['remaining_cards'])
            cards_played.append(info['agent_cards_played'])
            turn_lengths.extend(env.agent_turn_lengths)
            if not info['won'] and env.played_this_turn:
                turn_lengths.append(env.played_this_turn)
    finally:
        env.close()
    return dict(source=source, target_remaining=target_remaining, games=games,
                wins=wins, win_rate=wins/games, wilson_ci_95=wilson_interval(wins, games),
                avg_remaining_cards=float(np.mean(remaining)),
                avg_cards_played=float(np.mean(cards_played)),
                avg_turn_length=float(np.mean(turn_lengths)) if turn_lengths else 0.)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/latest.zip')
    p.add_argument('--source', choices=['natural', 'reverse', 'mixed'], default='natural')
    p.add_argument('--target-remaining', type=int, default=98)
    p.add_argument('--games', type=int, default=1000)
    p.add_argument('--seed', type=int, default=100000)
    p.add_argument('--device', default='cpu')
    p.add_argument('--torch-threads', type=int, default=1)
    p.add_argument('--backward-teacher-prob', type=float, default=.30)
    p.add_argument('--stochastic', action='store_true')
    args = p.parse_args()
    import torch
    torch.set_num_threads(args.torch_threads)
    model = MaskablePPO.load(args.model, device=args.device)
    result = evaluate_model(model, source=args.source, target_remaining=args.target_remaining,
                            games=args.games, seed=args.seed, deterministic=not args.stochastic,
                            backward_teacher_prob=args.backward_teacher_prob)
    print(json.dumps(result, indent=2))
    print('Remaining cards, cards played and turn length are diagnostics only.')


if __name__ == '__main__':
    main()

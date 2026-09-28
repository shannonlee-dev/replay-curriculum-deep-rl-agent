"""Offline observation-only comparison; never used by curriculum or promotion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from src.natural.bank import NaturalReverseBank, file_sha256, restore_checkpoint
from src.natural.curriculum import load_config, DISTRIBUTION
from src.env import TheGameEnv


def summarize(samples):
    array = np.asarray(samples, dtype=np.float32)
    return dict(sample_count=len(samples), feature_mean=array.mean(axis=0).tolist() if len(samples) else None,
                feature_std=array.std(axis=0).tolist() if len(samples) else None)


def collect_rollin(policy, *, stages, games, seed, heuristic):
    """One closest live turn boundary per deck/stage, independent of final outcome.

    Remaining count is used only to identify stage boundaries. Classifier features
    are copies of the exact 300-dimensional policy observation, with no metadata.
    """
    samples = {t: [] for t in stages}
    env = TheGameEnv(reset_source='natural')
    try:
        for index in range(games):
            obs, _ = env.reset(seed=seed+index)
            selected = {}
            for _ in range(196):
                action = policy.choose(env) if heuristic else policy.predict(obs, deterministic=True, action_masks=env.action_masks())[0]
                obs, reward, done, truncated, info = env.step(action)
                if truncated or reward != float(done and info['won']):
                    raise RuntimeError('audit sparse reward contract violated')
                if done:
                    break
                if env.played_this_turn == 0:
                    remaining = env._remaining_cards()
                    for target in stages:
                        distance = abs(target-remaining)
                        if distance <= 2 and (target not in selected or distance < selected[target][0]):
                            selected[target] = (distance, obs.copy())
            for target, (_, public_observation) in selected.items():
                samples[target].append(public_observation)
    finally:
        env.close()
    return samples


def bank_observations(bank, stages, limit, seed):
    samples = {}
    env = TheGameEnv()
    try:
        for target in stages:
            states = sorted([item for split in ('train', 'validation', 'test') for item in bank.checkpoints(split, target)], key=lambda item: item[0])
            samples[target] = []
            for index in np.random.default_rng(seed).permutation(len(states))[:limit]:
                restore_checkpoint(env, states[int(index)][1])
                samples[target].append(env._get_obs().copy())
    finally:
        env.close()
    return samples


def classifier_auc(first, second, seed):
    """Optional holdout logistic classifier; no stage/deck/teacher labels as inputs."""
    if min(len(first), len(second)) < 4:
        return dict(status='insufficient_samples', auc=None)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return dict(status='optional_sklearn_not_installed', auc=None)
    x = np.asarray(first+second)
    y = np.asarray([0]*len(first)+[1]*len(second))
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.3, stratify=y, random_state=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    model.fit(train_x, train_y)
    return dict(status='evaluated', auc=float(roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])),
                train_samples=len(train_y), test_samples=len(test_y))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', default='configs/natural_replay.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=3000000)
    parser.add_argument('--strategy', required=True, choices=['exponential', 'combined', 'combined_depth1', 'combined_depth2'])
    parser.add_argument('--classifier', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or args.games <= 0 or args.seed < 0:
        parser.error('use a new output, positive games and nonnegative seed')
    from sb3_contrib import MaskablePPO
    from src.natural.heuristics import StandalonePolicy
    import torch
    c = load_config(args.config)
    torch.set_num_threads(c['torch_threads'])
    stages = c['natural_curriculum']['stages']
    bank = NaturalReverseBank(args.bank, stages)
    from src.evaluation.natural import validate_natural_suite
    validate_natural_suite(games=args.games, seed=args.seed, forbidden_decks=bank.deck_ids)
    model = MaskablePPO.load(args.model, device=c['device'])
    populations = dict(A=bank_observations(bank, stages, args.games, args.seed),
        B=collect_rollin(StandalonePolicy(args.strategy), stages=stages, games=args.games, seed=args.seed, heuristic=True),
        C=collect_rollin(model, stages=stages, games=args.games, seed=args.seed, heuristic=False))
    rows = {t: {name: summarize(values[t]) for name, values in populations.items()} for t in stages}
    if args.classifier:
        for t in stages:
            rows[t]['A_vs_B'] = classifier_auc(populations['A'][t], populations['B'][t], args.seed)
            rows[t]['A_vs_C'] = classifier_auc(populations['A'][t], populations['C'][t], args.seed)
    report = dict(distributions=dict(A=DISTRIBUTION, B='heuristic all-reached natural states', C='PPO natural roll-in states'),
        features='exact public policy observation; no future deck, provenance or teacher info',
        strategy=args.strategy, bank_teacher=bank.teacher_provenance,
        caveat='A/B contrast also reflects generator differences unless the selected heuristic matches bank provenance',
        seed=args.seed, games=args.games, stages=rows, bank_sha256=bank.sha256, model_sha256=file_sha256(args.model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(f'Offline public-observation audit saved to {args.output}')


if __name__ == '__main__':
    main()

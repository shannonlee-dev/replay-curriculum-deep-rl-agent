"""Replay distributions and promotion decisions; only win rates select models."""
from collections import defaultdict
import numpy as np

DEFAULT_STAGES = list(range(20, 97, 4)) + [98]


def validate_distribution(targets, probs):
    if not targets or len(targets) != len(probs):
        raise ValueError('targets/probs must be nonempty and have matching lengths')
    if any(isinstance(t, bool) or not isinstance(t, (int, np.integer)) or t < 2 or t > 98 or t % 2 for t in targets):
        raise ValueError('targets must be even integers in [2, 98]')
    if len(set(targets)) != len(targets):
        raise ValueError('targets must be unique')
    p = np.asarray(probs, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any(p < 0) or not np.isclose(p.sum(), 1.):
        raise ValueError('probabilities must be finite, nonnegative, and sum to 1')
    return list(targets), (p / p.sum()).tolist()


def replay_distribution(current, config):
    override = config.get('distributions', {}).get(str(current))
    if override is None:
        override = config.get('distributions', {}).get(current)
    if override is not None:
        targets, probs = validate_distribution(override['targets'], override['probs'])
    else:
        # R98's immediate predecessor is R96, rather than R94.
        stages = config.get('stages', DEFAULT_STAGES)
        previous = sorted([t for t in stages if t < current], reverse=True)
        recent = previous[:3]
        for t in range(current - 4, 1, -4):
            if len(recent) == 3:
                break
            if t not in recent:
                recent.append(t)
        weights = config.get('recent_weights', [.35, .25, .15, .10])
        if len(weights) != 4:
            raise ValueError('recent_weights needs current and three previous weights')
        merged = defaultdict(float)
        for target, weight in zip([current] + recent, weights):
            merged[target] += weight
        for target, weight in config.get('anchors', {20: .10, 12: .05}).items():
            if int(target) <= current:
                merged[int(target)] += weight
        targets = sorted(merged)
        probs = [merged[t] for t in targets]
        targets, probs = validate_distribution(targets, probs)
    if current not in targets or probs[targets.index(current)] <= 0:
        raise ValueError('distribution must sample current target')
    if any(t > current for t in targets) or not any(t < current and p > 0 for t, p in zip(targets, probs)):
        raise ValueError('replay must include older targets and no future targets')
    return targets, probs


def promotion_checks(current, rates, core_targets, config):
    previous = max((t for t in core_targets if t < current), default=None)
    older = [t for t in rates if t < current and t != previous]
    return {
        'current': rates[current] >= config['current_min_win_rate'],
        'previous': previous is None or rates[previous] >= config['previous_min_win_rate'],
        'older_floor': all(rates[t] >= config['older_floor_win_rate'] for t in older),
    }

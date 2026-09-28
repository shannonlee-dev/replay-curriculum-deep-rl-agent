"""Dietrich 2019 §§3.4, 3.5.2 suffix evaluations, never environment rewards.

Recovery uses negative distance falloff: the printed Eq. 3.2 positive exponent
contradicts its decreasing-recovery explanation and [1,infinity) range.
"""
from dataclasses import replace
from functools import partial
import math

import numpy as np

from src.natural.bank import restore_checkpoint
from archive.privileged_teacher.teacher import _search_env, clone_env
from src.env import TheGameEnv


EXPONENTIAL = tuple(math.exp(-1.5*k) for k in range(5))


def pile_penalty(card, top, ascending, *, recovery=False):
    playable = (card > top or card == top-10) if ascending else (card < top or card == top+10)
    if playable:
        return 1.
    return 3.5 - math.exp(-.03*(abs(top-card)-1)) if recovery else 3.5


def penalty_score(env, *, recovery=False):
    tops = tuple(map(int, env.piles))
    total = 0.
    for card in range(2, 100):
        if card in env.played:
            continue
        if recovery:
            # Constant normalization keeps values manageable without changing rank.
            total += math.prod(pile_penalty(card, top, p < 2, recovery=True)
                               for p, top in enumerate(tops)) / 3.5**4
        else:
            count = sum((card > top or card == top-10) if p < 2
                        else (card < top or card == top+10) for p, top in enumerate(tops))
            total += EXPONENTIAL[count]
    return -total


class HeuristicPrior:
    def __init__(self, score):
        self.score = score

    def __call__(self, env):
        scores = np.full(393, -1e9)
        for action in np.flatnonzero(env.action_masks()):
            child = clone_env(env)
            child.step(int(action))
            scores[action] = self.score(child)
        return scores, 0.


def solve(checkpoint, prior, config, seed=0, *, mode='search', recovery=False):
    """Greedy turn rollout or heuristic beam search; no neural prior/value.

    Both use v3 local turn enumeration. Rollout commits the best complete turn
    without revisiting it. Search retains the configured inter-turn beam.
    """
    if mode not in ('rollout', 'search'):
        raise ValueError('unknown mode')
    env = TheGameEnv()
    restore_checkpoint(env, checkpoint)
    score = partial(penalty_score, recovery=recovery)
    if mode == 'rollout':
        config = replace(config, beam_width=1)
    return _search_env(env, HeuristicPrior(score), config, seed, score_fn=score)

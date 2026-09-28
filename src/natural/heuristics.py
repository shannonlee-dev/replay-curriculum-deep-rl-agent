"""Public-state, order-blind Dietrich-style policies for natural decks.

The policy reads the current hand, already-played cards, pile tops and legal
action mask.  It deliberately never reads ``env.deck``: remaining-card scores
use the known 2..99 card universe minus cards already played, and lookahead is
limited to cards currently in hand.  The environment alone performs draws.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import time

import numpy as np

from src.natural.bank import checkpoint_from_env, natural_start
from src.env import TheGameEnv


EXPONENTIAL = tuple(math.exp(-1.5 * count) for count in range(5))
STRATEGIES = ('exponential', 'combined', 'combined_depth1', 'combined_depth2')


@dataclass(frozen=True)
class _State:
    hand: frozenset[int]
    played: frozenset[int]
    piles: tuple[int, int, int, int]
    q: int


@dataclass
class NaturalResult:
    won: bool
    actions: list[int]
    remaining: int
    elapsed_seconds: float
    nc84_checkpoint: dict | None


def _playable(card: int, top: int, pile: int) -> bool:
    return card > top or card == top - 10 if pile < 2 else card < top or card == top + 10


def _state_from_env(env) -> _State:
    # Do not add env.deck here.  This is the policy's public state boundary.
    return _State(frozenset(env.hand), frozenset(env.played),
                  tuple(map(int, env.piles)), int(env.played_this_turn))


def _apply(state: _State, action: int) -> _State:
    card, pile = TheGameEnv.decode_action(action)
    if card not in state.hand or not _playable(card, state.piles[pile], pile):
        raise ValueError('attempted illegal public transition')
    piles = list(state.piles)
    piles[pile] = card
    return _State(state.hand - {card}, state.played | {card}, tuple(piles), state.q + 1)


def _card_actions(state: _State) -> list[int]:
    return [TheGameEnv.encode_action(card, pile) for card in sorted(state.hand)
            for pile in range(4) if _playable(card, state.piles[pile], pile)]


@lru_cache(maxsize=500000)
def exponential_score(state: _State) -> float:
    """Dietrich §3.4: -sum exp(-1.5 * playable-pile-count)."""
    cards = np.arange(2, 100)
    tops = np.asarray(state.piles)
    playable = np.empty((98, 4), dtype=bool)
    playable[:, :2] = (cards[:, None] > tops[:2]) | (cards[:, None] == tops[:2] - 10)
    playable[:, 2:] = (cards[:, None] < tops[2:]) | (cards[:, None] == tops[2:] + 10)
    unplayed = np.fromiter((card not in state.played for card in cards), dtype=bool, count=98)
    return -float(np.asarray(EXPONENTIAL)[playable.sum(axis=1)][unplayed].sum())


@lru_cache(maxsize=500000)
def _distance_score(state: _State) -> float:
    """Dietrich §3.5.2 distance-softened product penalty, order independent."""
    cards = np.arange(2, 100)
    tops = np.asarray(state.piles)
    playable = np.empty((98, 4), dtype=bool)
    playable[:, :2] = (cards[:, None] > tops[:2]) | (cards[:, None] == tops[:2] - 10)
    playable[:, 2:] = (cards[:, None] < tops[2:]) | (cards[:, None] == tops[2:] + 10)
    penalties = 3.5 - np.exp(-.03 * (np.abs(cards[:, None] - tops) - 1))
    penalties[playable] = 1.0
    unplayed = np.fromiter((card not in state.played for card in cards), dtype=bool, count=98)
    return -float(penalties.prod(axis=1)[unplayed].sum() / 3.5**4)


@lru_cache(maxsize=500000)
def combined_score(state: _State) -> float:
    """Equal-scale blend of §§3.4 and 3.5.2, plus exact backward recovery."""
    backward = sum(card == top - 10 if pile < 2 else card == top + 10
                   for card in state.hand for pile, top in enumerate(state.piles))
    return exponential_score(state) + _distance_score(state) + .10 * backward


class StandalonePolicy:
    def __init__(self, strategy: str):
        if strategy not in STRATEGIES:
            raise ValueError(f'unknown strategy: {strategy}')
        self.strategy = strategy
        self.depth = {'exponential': 0, 'combined': 0,
                      'combined_depth1': 1, 'combined_depth2': 2}[strategy]
        self.score = exponential_score if strategy == 'exponential' else combined_score

    def _value(self, state: _State, depth: int, cache: dict[tuple[_State, int], float]) -> float:
        key = (state, depth)
        if key in cache:
            return cache[key]
        value = self.score(state)
        if depth == 0 or not state.hand:
            cache[key] = value
            return value
        actions = _card_actions(state)
        if not actions:
            cache[key] = value
            return value
        # The next hand-only action is a true public lookahead: no END/draw.
        ranked = sorted((self.score(_apply(state, action)), action) for action in actions)
        # The public-state-only branch cap makes depth-2 practical without
        # becoming an accidental full-deck search.
        value = max(self._value(_apply(state, action), depth - 1, cache)
                    for _, action in ranked[-4:])
        cache[key] = value
        return value

    def choose(self, env) -> int:
        state = _state_from_env(env)
        mask = env.action_masks()
        legal = np.flatnonzero(mask)
        cards = [int(action) for action in legal if action != TheGameEnv.END_TURN]
        if not cards:
            return TheGameEnv.END_TURN
        cache: dict[tuple[_State, int], float] = {}
        if self.depth:
            ranked = sorted((self.score(_apply(state, action)), action) for action in cards)
            cards = [action for _, action in ranked[-4:]]
        candidates = [(self._value(_apply(state, action), self.depth, cache), action)
                      for action in cards]
        # Once ending is allowed, include it as a no-draw, no-peek choice.
        if mask[TheGameEnv.END_TURN]:
            candidates.append((self.score(state), TheGameEnv.END_TURN))
        return max(candidates, key=lambda item: (item[0], -item[1]))[1]


def play_natural_deck(permutation, strategy: str) -> NaturalResult:
    started = time.monotonic()
    env = natural_start(permutation)
    policy = StandalonePolicy(strategy)
    actions, checkpoint = [], None
    for _ in range(196):
        action = policy.choose(env)
        if not env.action_masks()[action]:
            raise RuntimeError('standalone policy chose an illegal action')
        _, reward, done, truncated, info = env.step(action)
        actions.append(action)
        if reward != float(done and info['won']) or truncated:
            raise RuntimeError('natural rollout violates sparse reward contract')
        if done:
            return NaturalResult(bool(info['won']), actions, env._remaining_cards(),
                                 time.monotonic() - started, checkpoint)
        if checkpoint is None and env.played_this_turn == 0 and 82 <= env._remaining_cards() <= 86:
            checkpoint = checkpoint_from_env(env, len(actions))
    raise RuntimeError('natural rollout exceeded action bound')

"""Bounded turn-level privileged search; scores are NEVER environment rewards."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time

import numpy as np

from src.natural.bank import natural_start, restore_checkpoint
from src.env import TheGameEnv


@dataclass(frozen=True)
class TeacherConfig:
    beam_width: int = 384
    candidate_turns_per_state: int = 10
    max_nodes_per_deck: int = 500000
    timeout_seconds_per_deck: float = 25.
    policy_top_k_actions: int = 6
    stochastic_candidates: int = 8
    policy_weight: float = .05
    value_weight: float = 1.
    future_weight: float = .10
    backward_weight: float = 2.
    legal_weight: float = .10
    damage_weight: float = .05
    progress_weight: float = 2.

    def __post_init__(self):
        for name in ('beam_width', 'candidate_turns_per_state', 'max_nodes_per_deck', 'policy_top_k_actions'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.stochastic_candidates) is not int or self.stochastic_candidates < 0:
            raise ValueError('stochastic_candidates must be a nonnegative integer')
        if not math.isfinite(self.timeout_seconds_per_deck) or self.timeout_seconds_per_deck <= 0:
            raise ValueError('timeout must be finite and positive')
        for name in ('policy', 'value', 'future', 'backward', 'legal', 'damage', 'progress'):
            if not math.isfinite(getattr(self, name+'_weight')) or getattr(self, name+'_weight') < 0:
                raise ValueError('heuristic weights must be finite and nonnegative')


@dataclass
class SearchResult:
    status: str
    actions: list[int]
    nodes: int
    elapsed_seconds: float
    deepest_remaining: int
    max_cards_played: int
    deepest_turn: int
    frontier_size: int


@dataclass
class _Node:
    env: TheGameEnv
    actions: tuple[int, ...]
    log_prior: float = 0.
    value: float = 0.
    score: float = 0.
    turns: int = 0


def clone_env(env):
    result = copy.copy(env)
    result.hand, result.played, result.deck = set(env.hand), set(env.played), list(env.deck)
    result.piles = env.piles.copy()
    result.agent_turn_lengths = list(env.agent_turn_lengths)
    return result


def state_key(env):
    # All calls at a search frontier are turn boundaries within the SAME deck.
    return (98-len(env.deck), tuple(sorted(env.hand)),
            tuple(sorted(map(int, env.piles[:2]))), tuple(sorted(map(int, env.piles[2:]))))


class PPOPrior:
    """The neural policy sees only public observations. Future order stays in search."""
    def __init__(self, model):
        self.model = model
        self.model.policy.set_training_mode(False)

    def __call__(self, env):
        import torch
        observation, _ = self.model.policy.obs_to_tensor(env._get_obs())
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(
                observation, action_masks=env.action_masks().reshape(1, -1))
            log_probs = distribution.distribution.logits[0].cpu().numpy()
            value = self.model.policy.predict_values(observation).item()
        return log_probs, value


def _score(node, config):
    env = node.env
    legal = env._legal_card_actions()
    backward = sum(env.decode_action(a)[0] == int(env.piles[env.decode_action(a)[1]])
                   + (-10 if env.decode_action(a)[1] < 2 else 10) for a in legal)
    # Privileged ORDER-sensitive compatibility: earlier future draws weigh more.
    future = sum((1. / (i+1)) * sum(env._legal_on_pile(card, p) for p in range(4))
                 for i, card in enumerate(env.deck[:16]))
    damage = int(env.piles[:2].sum()) - 2 + 200 - int(env.piles[2:].sum())
    return (config.progress_weight * len(env.played) + config.legal_weight * len(legal)
            + config.backward_weight * backward + config.future_weight * future
            - config.damage_weight * damage + config.value_weight * node.value
            + config.policy_weight * node.log_prior)


class _Limit(Exception):
    pass


class _Budget:
    def __init__(self, config, initial_remaining):
        self.config, self.nodes = config, 0
        self.initial_remaining = self.deepest_remaining = initial_remaining
        self.deepest_turn = 0
        self.frontier_size = 1
        self.start = time.monotonic()

    def observe(self, env, turns):
        self.deepest_remaining = min(self.deepest_remaining, env._remaining_cards())
        self.deepest_turn = max(self.deepest_turn, turns)

    def result(self, status, actions=()):
        return SearchResult(status, list(actions), self.nodes, time.monotonic()-self.start,
                            self.deepest_remaining, self.initial_remaining-self.deepest_remaining,
                            self.deepest_turn, self.frontier_size)

    def check(self):
        if time.monotonic() - self.start >= self.config.timeout_seconds_per_deck:
            raise _Limit('teacher_failed_timeout')

    def consume(self):
        self.check()
        if self.nodes >= self.config.max_nodes_per_deck:
            raise _Limit('teacher_failed_node_limit')
        self.nodes += 1


def _turn_candidates(root, prior, config, budget, rng, score_fn=None):
    partials, completed = [root], {}
    # No draws occur until END, so a turn can play at most the initial hand size.
    for _ in range(len(root.env.hand) + 1):
        following = []
        for node in partials:
            budget.check()
            mask = node.env.action_masks()
            legal = np.flatnonzero(mask)
            log_probs, value = prior(node.env)
            budget.check()
            log_probs = np.asarray(log_probs, dtype=float)
            if log_probs.shape != (393,) or not np.isfinite(log_probs[legal]).all() or not math.isfinite(value):
                raise ValueError('prior must supply finite legal log probabilities and value')
            card_actions = [int(a) for a in legal if a != TheGameEnv.END_TURN]
            ranked = sorted(card_actions, key=lambda a: (-log_probs[a], a))
            chosen = ranked[:config.policy_top_k_actions]
            extras = ranked[config.policy_top_k_actions:]
            if extras and config.stochastic_candidates:
                chosen += rng.choice(extras, size=min(len(extras), config.stochastic_candidates), replace=False).tolist()
            # Always consider END when legal, including forced-loss END.
            if mask[TheGameEnv.END_TURN]:
                chosen.append(TheGameEnv.END_TURN)
            for action in chosen:
                budget.consume()
                env = clone_env(node.env)
                env.step(action)
                child = _Node(env, node.actions+(action,), node.log_prior+float(log_probs[action]), float(value))
                child.turns = node.turns + int(action == TheGameEnv.END_TURN or env.won)
                budget.observe(env, child.turns)
                child.score = _score(child, config) if score_fn is None else score_fn(env)
                if env.won:
                    return [child]
                if env.terminated:
                    continue
                if action == TheGameEnv.END_TURN:
                    key = state_key(env)
                    if key not in completed or child.score > completed[key].score:
                        completed[key] = child
                else:
                    following.append(child)
        # Local beam bounds the combinatorial within-turn action enumeration.
        unique = {}
        for node in following:
            key = (state_key(node.env), node.env.played_this_turn)
            if key not in unique or node.score > unique[key].score:
                unique[key] = node
        partials = sorted(unique.values(), key=lambda n: n.score, reverse=True)[:config.candidate_turns_per_state]
        if not partials:
            break
    return sorted(completed.values(), key=lambda n: n.score, reverse=True)[:config.candidate_turns_per_state]


def _search_env(root, prior, config=None, seed=0, score_fn=None):
    config = config or TeacherConfig()
    budget = _Budget(config, root._remaining_cards())
    rng = np.random.default_rng(seed)
    frontier = [_Node(clone_env(root), ())]
    status = 'teacher_failed_exhausted'
    try:
        # Minimum one card per nonempty turn gives at most 98 turns.
        for _ in range(98):
            budget.frontier_size = len(frontier)
            budget.check()
            successors = {}
            for node in frontier:
                for child in _turn_candidates(node, prior, config, budget, rng, score_fn):
                    if child.env.won:
                        return budget.result('win_found', child.actions)
                    key = state_key(child.env)
                    if key not in successors or child.score > successors[key].score:
                        successors[key] = child
            frontier = sorted(successors.values(), key=lambda n: n.score, reverse=True)[:config.beam_width]
            budget.frontier_size = len(frontier)
            if not frontier:
                budget.frontier_size = 0
                break
    except _Limit as exc:
        status = str(exc)
    return budget.result(status)


def search_checkpoint(checkpoint, prior, config=None, seed=0):
    """Suffix-only search. Progress/turn counters are relative to this checkpoint."""
    env = TheGameEnv()
    restore_checkpoint(env, checkpoint)
    return _search_env(env, prior, config, seed)


def search(permutation, prior, config=None, seed=0):
    """Legacy full-deck API; NC84 generation uses search_checkpoint instead."""
    return _search_env(natural_start(permutation), prior, config, seed)

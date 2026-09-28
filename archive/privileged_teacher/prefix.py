"""Reach NC84 using only an R80 policy's public-observation natural rollout."""
from __future__ import annotations

from dataclasses import dataclass
import time

from src.natural.bank import checkpoint_from_env, natural_start


@dataclass
class PrefixResult:
    status: str
    reason: str | None
    actions: list[int]
    checkpoint: dict | None
    remaining: int
    cards_played: int
    turns: int
    elapsed_seconds: float


def rollout_prefix(permutation, policy, *, deterministic=True):
    """Stop at the FIRST live q=0 boundary with 82..86 cards remaining.

    No privileged environment/state object is passed to the policy. Overshooting
    the window is a reach failure; we never rewind or teacher-repair the prefix.
    """
    start = time.monotonic()
    env = natural_start(permutation)
    actions, turns = [], 0
    reason = 'action_limit'
    for _ in range(196):
        obs = env._get_obs()
        action, _ = policy.predict(obs, deterministic=deterministic, action_masks=env.action_masks())
        action = int(action)
        _, reward, done, truncated, info = env.step(action)
        actions.append(action)
        if reward != float(done and info['won']) or truncated:
            raise RuntimeError('prefix violates sparse reward/termination contract')
        if done:
            reason = 'policy_loss' if not info['won'] else 'passed_target_before_terminal_win'
            break
        if env.played_this_turn == 0:
            turns += 1
            if 82 <= env._remaining_cards() <= 86:
                return PrefixResult('prefix_reached', None, actions,
                                    checkpoint_from_env(env, len(actions)), env._remaining_cards(),
                                    len(env.played), turns, time.monotonic()-start)
        if env._remaining_cards() < 82:
            reason = 'overshot_target_window'
            break
    return PrefixResult('policy_failed_to_reach_nc84', reason, actions, None,
                        env._remaining_cards(), len(env.played), turns, time.monotonic()-start)

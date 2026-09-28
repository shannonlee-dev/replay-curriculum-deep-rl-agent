"""Versioned natural-deck witnesses. No policy or privileged metadata in observations."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.env import TheGameEnv

SCHEMA_VERSION = 1
BUCKETS = (84, 88, 92, 96, 98)
SPLITS = ('train', 'validation', 'test')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_permutation(permutation):
    if (not isinstance(permutation, (list, tuple)) or len(permutation) != 98
            or any(type(x) is not int for x in permutation)
            or sorted(permutation) != list(range(2, 100))):
        raise ValueError('full_permutation must contain each integer 2..99 exactly once')


def deck_id(permutation):
    validate_permutation(permutation)
    return hashlib.sha256(bytes(permutation)).hexdigest()


def split_for(identity):
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError('invalid deck_id')
    try:
        value = int(identity, 16)
    except ValueError as exc:
        raise ValueError('invalid deck_id') from exc
    slot = value % 10
    return 'train' if slot < 8 else ('validation' if slot == 8 else 'test')


def bucket_for(remaining):
    if remaining == 98:
        return 98
    candidates = [b for b in BUCKETS[:-1] if abs(b - remaining) <= 2]
    return min(candidates, key=lambda b: (abs(b - remaining), b)) if candidates else None


def natural_start(permutation):
    validate_permutation(permutation)
    env = TheGameEnv()
    env._clear_state()
    env.deck = list(permutation)
    env._refill_hand()
    env.actual_reset_source = 'natural'
    return env


def checkpoint_from_env(env, action_offset):
    if env.terminated or env.played_this_turn != 0:
        raise ValueError('checkpoint must be a live turn boundary')
    return dict(bucket=bucket_for(env._remaining_cards()),
                remaining_count=env._remaining_cards(), hand=sorted(env.hand),
                pile_tops=env.piles.tolist(), played=sorted(env.played),
                future_deck_order=list(env.deck), deck_position=98-len(env.deck),
                q=0, action_offset=action_offset)


def restore_checkpoint(env, checkpoint):
    """Restore a private state; deliberately omit provenance and teacher history."""
    try:
        hand, played, deck = (checkpoint[k] for k in ('hand', 'played', 'future_deck_order'))
        piles = checkpoint['pile_tops']
        cards = list(hand) + list(played) + list(deck)
        validate_permutation(cards)
        if (checkpoint['q'] != 0 or len(hand) != min(8, len(hand)+len(deck))
                or not hand or checkpoint['remaining_count'] != len(hand)+len(deck)
                or checkpoint['deck_position'] != 98-len(deck)
                or len(piles) != 4 or any(type(x) is not int for x in piles)
                or any(x not in played and x != (1 if i < 2 else 100)
                       for i, x in enumerate(piles))):
            raise ValueError('invalid turn-boundary checkpoint')
    except (KeyError, TypeError) as exc:
        raise ValueError('malformed checkpoint') from exc
    env._clear_state()
    env.hand, env.played, env.deck = set(hand), set(played), list(deck)
    env.piles = np.array(piles, dtype=np.int16)
    env.sampled_target_remaining = checkpoint['remaining_count']
    env.actual_reset_source = 'natural_conditioned'


def build_record(permutation, actions, *, prefix_action_offset=None):
    """Replay a WIN; v2 stores only the policy-reached NC84 checkpoint."""
    identity = deck_id(permutation)
    if (not isinstance(actions, (list, tuple)) or not actions or len(actions) > 196
            or any(type(a) is not int or not 0 <= a <= TheGameEnv.END_TURN for a in actions)):
        raise ValueError('invalid witness actions')
    if prefix_action_offset is not None and (type(prefix_action_offset) is not int
                                            or not 0 < prefix_action_offset < len(actions)):
        raise ValueError('invalid policy prefix action offset')
    env = natural_start(permutation)
    selected = {}

    def capture(offset):
        if prefix_action_offset is not None and offset != prefix_action_offset:
            return
        cp = checkpoint_from_env(env, offset)
        if prefix_action_offset is not None and cp['bucket'] != 84:
            raise ValueError('policy prefix must end at an NC84 boundary')
        bucket = cp['bucket']
        if bucket is None:
            return
        previous = selected.get(bucket)
        if previous is None or abs(cp['remaining_count']-bucket) < abs(previous['remaining_count']-bucket):
            selected[bucket] = cp

    capture(0)
    for offset, action in enumerate(actions, 1):
        if env.terminated or not env.action_masks()[action]:
            raise ValueError(f'invalid witness at action {offset-1}')
        _, reward, done, truncated, info = env.step(action)
        if reward != float(done and info['won']) or truncated:
            raise ValueError('reward/termination contract mismatch')
        if not done and env.played_this_turn == 0:
            capture(offset)
    if not env.terminated or not env.won:
        raise ValueError('witness does not finish with WIN')
    if prefix_action_offset is not None and not selected:
        raise ValueError('policy prefix does not end at a live turn boundary')
    record = dict(schema_version=SCHEMA_VERSION if prefix_action_offset is None else 2,
                deck_id=identity, split=split_for(identity),
                full_permutation=list(permutation), actions=list(actions),
                checkpoints=[selected[b] for b in sorted(selected)])
    if prefix_action_offset is not None:
        record.update(generation_mode='on_policy_natural_prefix', prefix_action_offset=prefix_action_offset)
    return record


def validate_record(record):
    try:
        if record['schema_version'] not in (1, 2):
            raise ValueError('unsupported bank schema')
        offset = record['prefix_action_offset'] if record['schema_version'] == 2 else None
        rebuilt = build_record(record['full_permutation'], record['actions'], prefix_action_offset=offset)
        for key, expected in rebuilt.items():
            if record[key] != expected:
                raise ValueError(f'bank record mismatch: {key}')
        # Validate each independently restored continuation, not only the full path.
        for checkpoint in record['checkpoints']:
            env = TheGameEnv()
            restore_checkpoint(env, checkpoint)
            for action in record['actions'][checkpoint['action_offset']:]:
                if env.terminated or not env.action_masks()[action]:
                    raise ValueError('invalid checkpoint continuation')
                env.step(action)
            if not env.won:
                raise ValueError('checkpoint continuation does not WIN')
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('malformed bank record') from exc
    return record


def validate_provenance(record):
    try:
        teacher, stats = record['teacher'], record['search']
        hashes = [teacher['model_sha256'], *teacher['code_sha256'].values()]
        if (not isinstance(teacher['config'], dict) or not teacher['code_sha256']
                or any(not isinstance(h, str) or len(h) != 64
                       or any(c not in '0123456789abcdef' for c in h) for h in hashes)
                or type(stats['nodes']) is not int or stats['nodes'] < 0
                or not math.isfinite(stats['elapsed_seconds']) or stats['elapsed_seconds'] < 0):
            raise ValueError('invalid teacher provenance/search statistics')
        if record['schema_version'] == 2:
            prefix = record['prefix_policy']
            remaining = record['checkpoints'][0]['remaining_count']
            if (prefix['model_sha256'] != teacher['model_sha256']
                    or type(prefix['deterministic']) is not bool
                    or type(prefix['seed']) is not int or prefix['seed'] < 0
                    or any(type(stats.get(k)) is not int or stats[k] < 0 for k in
                           ('deepest_remaining', 'max_cards_played', 'deepest_turn', 'frontier_size'))
                    or stats['deepest_remaining'] != 0
                    or stats['max_cards_played'] != remaining):
                raise ValueError('invalid on-policy prefix provenance/winning suffix diagnostics')
        json.dumps(teacher, allow_nan=False)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('missing/malformed teacher provenance or search statistics') from exc
    return teacher


class NaturalBank:
    """Fail-closed loader. Expensive witness validation happens once, before training."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.sha256 = file_sha256(self.path)
        self.deck_ids = set()
        self.teacher_provenance = None
        self.schema_version = None
        self._checkpoints = {(s, b): [] for s in SPLITS for b in BUCKETS}
        with self.path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = validate_record(json.loads(line))
                    provenance = validate_provenance(record)
                    if self.schema_version is None:
                        self.schema_version = record['schema_version']
                    elif self.schema_version != record['schema_version']:
                        raise ValueError('bank mixes generation schemas')
                    if self.teacher_provenance is None:
                        self.teacher_provenance = copy.deepcopy(provenance)
                    elif provenance != self.teacher_provenance:
                        raise ValueError('bank mixes teacher model/config/code provenance')
                    identity = record['deck_id']
                    if identity in self.deck_ids:
                        raise ValueError('duplicate deck_id')
                    self.deck_ids.add(identity)
                    for checkpoint in record['checkpoints']:
                        # No teacher witness/score enters a training environment.
                        self._checkpoints[record['split'], checkpoint['bucket']].append(
                            (identity, copy.deepcopy(checkpoint)))
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f'{self.path}:{line_number}: {exc}') from exc
        if not self.deck_ids:
            raise ValueError('empty bank')
        if file_sha256(self.path) != self.sha256:
            raise ValueError('bank changed while loading; freeze it before use')

    def checkpoints(self, split, bucket):
        if (split, bucket) not in self._checkpoints:
            raise ValueError('invalid split or bucket')
        return copy.deepcopy(self._checkpoints[split, bucket])

    def counts(self):
        return {s: {b: len(self._checkpoints[s, b]) for b in BUCKETS} for s in SPLITS}


class NaturalReverseBank:
    """Derive real turn boundaries from validated natural WINs; keep deck splits."""
    def __init__(self, path, stages):
        self.stages = tuple(stages)
        original = NaturalBank(path)  # Full natural-start and NC84 continuation replay.
        self.path, self.sha256, self.deck_ids = original.path, original.sha256, original.deck_ids
        self.teacher_provenance = original.teacher_provenance
        self._states = {(split, target): [] for split in ('train', 'validation', 'test') for target in self.stages}
        with self.path.open() as stream:
            for line in stream:
                record = json.loads(line)
                env = natural_start(record['full_permutation'])
                selected = {}
                for offset, action in enumerate(record['actions'], 1):
                    _, _, done, _, _ = env.step(action)
                    if done or env.played_this_turn:
                        continue
                    remaining = env._remaining_cards()
                    for target in self.stages:
                        if abs(remaining-target) > 2:
                            continue
                        previous = selected.get(target)
                        if previous is None or abs(remaining-target) < abs(previous['remaining_count']-target):
                            cp = checkpoint_from_env(env, offset)
                            cp['bucket'] = target
                            selected[target] = cp
                if not env.won:
                    raise ValueError('natural witness changed after validation')
                for target, cp in selected.items():
                    continuation = TheGameEnv()
                    try:
                        restore_checkpoint(continuation, cp)
                        for action in record['actions'][cp['action_offset']:]:
                            if continuation.terminated or not continuation.action_masks()[action]:
                                raise ValueError('invalid derived checkpoint continuation')
                            _, reward, done, truncated, info = continuation.step(action)
                            if truncated or reward != float(done and info['won']):
                                raise ValueError('derived checkpoint reward contract mismatch')
                        if not continuation.won:
                            raise ValueError('derived checkpoint continuation does not WIN')
                    finally:
                        continuation.close()
                    self._states[record['split'], target].append((record['deck_id'], cp))
                env.close()
        if file_sha256(self.path) != self.sha256:
            raise ValueError('bank changed during checkpoint extraction')

    def checkpoints(self, split, bucket):
        return copy.deepcopy(self._states[split, bucket])

    def counts(self):
        return {split: {target: len(self._states[split, target]) for target in self.stages}
                for split in ('train', 'validation', 'test')}

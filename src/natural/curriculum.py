"""Configuration and terminal-WIN frontier decisions; no environment mutation."""
from __future__ import annotations

import math
from pathlib import Path
import yaml

DISTRIBUTION = 'natural random deck | selected heuristic eventually WIN'


def validate_natural_config(c, *, require_criteria=True):
    def required(obj, key, prefix):
        if not isinstance(obj, dict) or key not in obj:
            raise ValueError(f'missing {prefix}.{key}; migrate to configs/natural_replay.yaml (no legacy fallback)')
        return obj[key]

    n = required(c, 'natural_curriculum', 'config')
    def value(obj, key, prefix, kind='rate', nullable=False):
        x = required(obj, key, prefix)
        if x is None and nullable and not require_criteria:
            return
        valid = (type(x) is int and x > 0 if kind == 'positive' else
                 type(x) is int and x >= 0 if kind == 'seed' else
                 type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1)
        if not valid:
            raise ValueError(f'{prefix}.{key} must be an explicit {kind} value; got {x!r}')
    stages = required(n, 'stages', 'natural_curriculum')
    if (not isinstance(stages, list) or not stages or
        any(type(t) is not int or not 2 <= t <= 84 for t in stages) or sorted(set(stages)) != stages or stages[-1] != 84):
        raise ValueError('natural_curriculum.stages must be unique increasing integers ending at NC84')
    start = required(n, 'start_stage', 'natural_curriculum')
    if start != 'auto' and (type(start) is not int or start not in stages):
        raise ValueError('start_stage must be auto or a configured stage')
    if start != 'auto' and (not isinstance(n.get('start_stage_reason'), str) or not n['start_stage_reason'].strip()):
        raise ValueError('explicit start_stage requires a nonempty start_stage_reason')
    value(n, 'nc_episode_probability', 'natural_curriculum')
    value(n, 'minimum_winning_decks', 'natural_curriculum', 'positive')
    counts = required(n, 'minimum_stage_decks', 'natural_curriculum')
    for split in ('train', 'validation', 'test'):
        value(counts, split, 'minimum_stage_decks', 'positive')
    rehearsal = required(n, 'rehearsal', 'natural_curriculum')
    for key in ('current_weight', 'older_weight'):
        value(rehearsal, key, 'rehearsal')
    if not math.isclose(sum(rehearsal[k] for k in ('current_weight', 'older_weight')), 1):
        raise ValueError('rehearsal weights must sum to one')
    auto = required(n, 'auto_start', 'natural_curriculum')
    for key, kind in (('min_wins', 'positive'), ('min_win_rate', 'rate')):
        candidate = required(auto, key, 'auto_start')
        if start != 'auto' and candidate is None:
            continue  # Explicit stage selection does not use auto criteria.
        value(auto, key, 'auto_start', kind, nullable=True)
    p = required(n, 'promotion', 'natural_curriculum')
    if required(p, 'mode', 'promotion') != 'frontier':
        raise ValueError('natural promotion.mode must be frontier')
    value(p, 'min_stage_steps', 'promotion', 'positive')
    value(p, 'next_stage_min_wins', 'promotion', 'positive', nullable=True)
    for k in ('next_stage_min_win_rate', 'retention_relative_floor', 'retention_absolute_floor'):
        value(p, k, 'promotion', nullable=True)
    if required(p, 'final_stage_rule', 'promotion') != 'budget_and_retention':
        raise ValueError('final_stage_rule must be budget_and_retention (completion is not final success)')
    e = required(n, 'evaluation', 'natural_curriculum')
    for k in ('nc_games', 'natural_games', 'natural_large_games', 'natural_large_interval', 'synthetic_games'):
        value(e, k, 'evaluation', 'positive')
    for k in ('nc_seed', 'natural_seed', 'synthetic_seed'):
        value(e, k, 'evaluation', 'seed')
    if e['natural_large_games'] < e['natural_games']:
        raise ValueError('natural_large_games must cover natural_games')
    return n


def load_config(path, overrides=()):
    c = yaml.safe_load(Path(path).read_text())
    for expression in overrides:
        key, sep, raw = expression.partition('=')
        if not sep:
            raise ValueError('--set requires existing.dotted.key=value')
        node = c
        parts = key.split('.')
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ValueError(f'unknown override {key}')
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise ValueError(f'unknown override {key}')
        node[parts[-1]] = yaml.safe_load(raw)
    validate_natural_config(c, require_criteria=False)
    return c


def select_start(n, rows):
    criteria = n['auto_start']
    if any(v is None for v in criteria.values()):
        raise ValueError('auto_start criteria are unset; scan saved, no automatic stage selection')
    eligible = [t for t in n['stages'] if t in rows and
                all(rows[t]['available'][s] >= minimum for s, minimum in n['minimum_stage_decks'].items()) and
                rows[t]['validation']['wins'] >= criteria['min_wins'] and
                rows[t]['validation']['win_rate'] is not None and
                rows[t]['validation']['win_rate'] >= criteria['min_win_rate']]
    if not eligible:
        raise ValueError('no eligible automatic start stage: insufficient terminal WIN signal or split coverage')
    return dict(stage=max(eligible), eligible=eligible, criteria=criteria,
                reason='hardest configured stage meeting explicit terminal WIN and split coverage criteria')


def frontier_assessment(n, target, results, references, steps, budget):
    """Auditable actual/required comparisons; all criteria come from configuration."""
    p = n['promotion']
    index = n['stages'].index(target)
    checks = {}
    def compare(name, actual, required, **context):
        passed = actual is not None and required is not None and actual >= required
        checks[name] = dict(actual=actual, required=required, operator='>=', passed=passed,
                            status='PASS' if passed else 'FAIL', **context)
    compare('minimum_stage_steps', steps, p['min_stage_steps'])
    if index+1 < len(n['stages']):
        next_stage = n['stages'][index+1]
        nxt = results[next_stage]
        compare('next_stage_wins', nxt['wins'], p['next_stage_min_wins'], stage=next_stage, games=nxt['games'])
        compare('next_stage_win_rate', nxt['win_rate'], p['next_stage_min_win_rate'], stage=next_stage, games=nxt['games'])
    else:
        compare('final_stage_budget', steps, budget, rule=p['final_stage_rule'])
    for t in n['stages'][:index]:
        rate = results[t]['win_rate']
        reference = references.get(t)
        compare(f'NC{t}_retention_absolute', rate, p['retention_absolute_floor'], stage=t)
        compare(f'NC{t}_retention_relative', rate,
                reference*p['retention_relative_floor'] if reference is not None else None,
                stage=t, reference=reference, relative_floor=p['retention_relative_floor'])
    return checks


def frontier_checks(n, target, results, references, steps, budget):
    return {name: row['passed'] for name, row in frontier_assessment(n, target, results, references, steps, budget).items()}


NC_SYNTHETIC_TARGETS = [12, 20, 72, 76, 80, 84]
NC_SYNTHETIC_WEIGHTS = [.05, .10, .06, .09, .15, .25]

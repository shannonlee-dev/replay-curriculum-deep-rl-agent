"""Local continuation transactions, scan provenance and private environment state."""
from __future__ import annotations

import copy
from src.common.paths import ROOT
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
from src.natural.bank import file_sha256

SCHEMA = 2
HISTORY_FIELDS = ['global_steps', 'stage_steps', 'current_stage', 'status', 'natural_wins', 'natural_games',
                  'natural_win_rate', 'nc_episode_share', 'nc_transition_share', 'model_sha256']


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


@contextmanager
def run_lock(out):
    with (Path(out)/'.run.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('run is already active in another process') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def log_state(out):
    result = {}
    for name in ('evaluations.jsonl', 'history.csv'):
        path = Path(out)/name
        data = path.read_bytes() if path.exists() else b''
        result[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return result


def event_bytes(record):
    natural = record['natural_full']
    row = {key: record[key] for key in HISTORY_FIELDS[:4]}
    row.update(natural_wins=natural['wins'], natural_games=natural['games'], natural_win_rate=natural.get('win_rate'),
        nc_episode_share=record['usage_cumulative']['episode_shares'].get('natural_conditioned', 0),
        nc_transition_share=record['usage_cumulative']['transition_shares'].get('natural_conditioned', 0),
        model_sha256=record['model_sha256'])
    stream = io.StringIO(newline='')
    csv.DictWriter(stream, fieldnames=HISTORY_FIELDS).writerow(row)
    return {'evaluations.jsonl': (json.dumps(record, sort_keys=True, allow_nan=False)+'\n').encode(),
            'history.csv': stream.getvalue().encode()}


def reconcile_logs(out, metadata, *, write=False):
    """Accept exactly the checkpoint's committed prefix plus its one pending event.

    A crash between JSONL/CSV appends can finish the missing append. Unknown or
    edited bytes are never truncated or replaced: continuation fails closed.
    """
    expected = metadata['log_state']
    event = copy.deepcopy(metadata.get('pending_evaluation'))
    suffixes = {}
    if event:
        event['model_sha256'] = metadata['model_sha256']
        suffixes = event_bytes(event)
    missing = []
    for name, item in expected.items():
        path = Path(out)/name
        data = path.read_bytes() if path.exists() else b''
        size = item['bytes']
        if len(data) < size or hashlib.sha256(data[:size]).hexdigest() != item['sha256']:
            raise ValueError(f'run log/history hash mismatch: {name}')
        suffix = suffixes.get(name, b'')
        if data[size:] not in (b'', suffix):
            raise ValueError(f'uncommitted or changed run log/history: {name}')
        if suffix and not data[size:]:
            missing.append((path, suffix))
    if write:
        for path, suffix in missing:
            with path.open('ab') as stream:
                stream.write(suffix); stream.flush(); os.fsync(stream.fileno())
    return event


def validate_run(out, c, bank, metadata, *, code_hashes):
    out = Path(out)
    manifest = json.loads((out/'manifest.json').read_text())
    if metadata.get('continuation_schema') != SCHEMA:
        raise ValueError('checkpoint lacks complete continuation state; cannot resume this run in place')
    if canonical_hash(c) != manifest.get('config_sha256') or canonical_hash(metadata['config']) != manifest['config_sha256']:
        raise ValueError('run config hash mismatch')
    import yaml
    if canonical_hash(yaml.safe_load((out/'config.resolved.yaml').read_text())) != manifest['config_sha256']:
        raise ValueError('resolved config hash mismatch')
    if bank.sha256 != manifest['bank_sha256'] or metadata['bank_sha256'] != manifest['bank_sha256']:
        raise ValueError('run bank hash mismatch')
    if file_sha256(out/'manifest.json') != metadata['manifest_sha256']:
        raise ValueError('run manifest/provenance hash mismatch')
    from src.common.provenance import validate_code_hashes
    validate_code_hashes(manifest['code_sha256'], code_hashes)
    if file_sha256(out/'difficulty_scan.json') != manifest['difficulty_scan_sha256']:
        raise ValueError('difficulty scan hash mismatch')
    origin = Path(manifest['checkpoint'])
    if file_sha256(origin) != manifest['checkpoint_sha256'] or file_sha256(origin.with_suffix('.json')) != manifest['checkpoint_metadata_sha256']:
        raise ValueError('initial checkpoint provenance hash mismatch')
    reconcile_logs(out, metadata)
    return manifest


def scan_provenance(root, checkpoint, bank, evaluation):
    names = ('src/natural/bank.py', 'src/evaluation/natural.py', 'src/env.py', 'src/evaluation/synthetic.py', 'src/natural/difficulty_scan.py',
             'src/synthetic/curriculum.py')
    return dict(model_sha256=file_sha256(checkpoint), checkpoint_metadata_sha256=file_sha256(Path(checkpoint).with_suffix('.json')),
                bank_sha256=bank.sha256, stages=list(bank.stages),
                nc_games=evaluation['nc_games'], nc_seed=evaluation['nc_seed'],
                code_sha256={name: file_sha256(Path(root)/name) for name in names},
                packages={name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'gymnasium', 'stable-baselines3', 'sb3-contrib')})


def cached_scan(directory, provenance, compute):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/(canonical_hash(provenance)+'.json')
    if path.exists():
        item = json.loads(path.read_text())
        if item['provenance'] != provenance or item['results_sha256'] != canonical_hash(item['results']):
            raise ValueError('difficulty scan cache provenance/hash mismatch')
        return {int(k): v for k, v in item['results'].items()}, True
    # JSON-normalize tuples and dictionary keys identically for fresh and cached results.
    rows = json.loads(json.dumps(compute()))
    atomic_json(path, dict(provenance=provenance, results=rows, results_sha256=canonical_hash(rows)))
    return {int(k): v for k, v in rows.items()}, False


GAME_FIELDS = ('deck', 'hand', 'played', 'piles', 'played_this_turn', 'terminated', 'won',
               'agent_cards_played', 'agent_turn_lengths', 'actual_reset_source', 'sampled_target_remaining',
               '_np_random', '_np_random_seed')
MONITOR_FIELDS = ('t_start', 'rewards', 'needs_reset', 'episode_returns', 'episode_lengths', 'episode_times',
                  'total_steps', 'current_reset_info')
USAGE_FIELDS = ('source_episodes', 'source_transitions', 'source_completed', 'source_completed_transitions',
               'source_wins', '_episode_source', '_episode_length')


def capture_runtime(model, env):
    """Private restore data is stored inside the hashed PPO zip, never policy input."""
    import torch
    rows = []
    for wrapper in env.envs:
        monitor, game = wrapper.env, wrapper.unwrapped
        rows.append(dict(game={k: copy.deepcopy(getattr(game, k)) for k in GAME_FIELDS if hasattr(game, k)},
            monitor={k: copy.deepcopy(getattr(monitor, k)) for k in MONITOR_FIELDS},
            usage={k: copy.deepcopy(getattr(wrapper, k)) for k in USAGE_FIELDS},
            action_rng=copy.deepcopy(game.action_space.np_random.bit_generator.state)))
    return dict(environments=rows, vector={k: copy.deepcopy(getattr(env, k)) for k in
                ('_seeds', '_options', 'reset_infos', 'buf_obs', 'buf_dones', 'buf_rews', 'buf_infos')},
        model={k: copy.deepcopy(getattr(model, k)) for k in
               ('_last_obs', '_last_episode_starts', '_last_original_obs') if hasattr(model, k)},
        python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_runtime(model, env):
    import torch
    state = getattr(model, '_natural_runtime', None)
    if not isinstance(state, dict) or len(state['environments']) != len(env.envs):
        raise ValueError('model lacks matching private continuation runtime')
    for wrapper, row in zip(env.envs, state['environments']):
        for obj, values in ((wrapper.unwrapped, row['game']), (wrapper.env, row['monitor']), (wrapper, row['usage'])):
            for name, value in values.items():
                setattr(obj, name, copy.deepcopy(value))
        wrapper.unwrapped.action_space.np_random.bit_generator.state = copy.deepcopy(row['action_rng'])
    for name, value in state['vector'].items():
        setattr(env, name, copy.deepcopy(value))
    for name, value in state['model'].items():
        setattr(model, name, copy.deepcopy(value))
    random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'].cpu())
    if state['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(state['cuda_rng'])


def validate_start(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    metadata = json.loads(checkpoint.with_suffix('.json').read_text())
    if (metadata.get('current_stage') != 'R80' or metadata.get('status') != 'promoted'
            or type(metadata.get('global_steps')) is not int):
        raise ValueError('resume must be the R80 stage-complete checkpoint, with promoted metadata')
    return metadata


def read_checkpoint(path):
    """Read the explicitly selected checkpoint; never substitute latest."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.with_suffix('.json').read_text())
    if data.get('current_stage') == 'R80':
        data = validate_start(path)
        if data.get('model_sha256') and file_sha256(path) != data['model_sha256']:
            raise ValueError('checkpoint model/metadata hash mismatch')
        return data
    if not (str(data.get('current_stage', '')).startswith('NC') and str(data['current_stage'])[2:].isdigit()):
        raise ValueError('expected R80 or a natural reverse NC checkpoint')
    for key in ('global_steps', 'stage_steps', 'total_added_steps'):
        if type(data.get(key)) is not int or data[key] < 0:
            raise ValueError(f'invalid checkpoint {key}')
    if not data['stage_steps'] <= data['total_added_steps'] <= data['global_steps']:
        raise ValueError('inconsistent checkpoint step counters')
    origin = data['global_steps']-data['total_added_steps']
    if data.get('origin_global_steps', origin) != origin:
        raise ValueError('inconsistent origin step')
    data['origin_global_steps'] = origin
    if data.get('checkpoint_kind') == 'natural_reverse' and not data.get('model_sha256'):
        raise ValueError('natural checkpoint missing model hash')
    if data.get('model_sha256') and file_sha256(path) != data['model_sha256']:
        raise ValueError('checkpoint model/metadata hash mismatch')
    if data.get('checkpoint_kind') != 'natural_reverse':
        # Support checkpoints written by the previous natural-reverse entry point.
        manifest = json.loads((path.parent/'manifest.json').read_text())
        if 'natural_stages' not in manifest:
            raise ValueError('checkpoint is not from natural reverse training')
        data.update(bank_sha256=manifest['bank_sha256'], config=manifest['config'])
    if not data.get('bank_sha256') or not isinstance(data.get('config'), dict):
        raise ValueError('checkpoint is missing bank/config provenance')
    return data


def save_checkpoint(model, out, metadata):
    """Keep an immutable step snapshot and atomically replace each latest file."""
    out = Path(out)
    snapshots = out/'checkpoints'
    snapshots.mkdir(parents=True, exist_ok=True)
    if model.num_timesteps != metadata['global_steps']:
        raise ValueError('model and checkpoint step differ')
    name = f"step_{metadata['total_added_steps']:09d}_{metadata['current_stage']}"
    snapshot = snapshots/(name+'.zip')
    temp = out/'.checkpoint_pending'
    model.save(str(temp))
    temp_zip = Path(str(temp)+'.zip')
    with temp_zip.open('rb') as stream:
        os.fsync(stream.fileno())
    data = dict(metadata, model_sha256=file_sha256(temp_zip))
    payload = json.dumps(data, indent=2)+'\n'
    if not snapshot.exists() and not snapshot.with_suffix('.json').exists():
        with snapshot.open('xb') as stream:
            stream.write(temp_zip.read_bytes())
            stream.flush(); os.fsync(stream.fileno())
        with snapshot.with_suffix('.json').open('x') as stream:
            stream.write(payload)
            stream.flush(); os.fsync(stream.fileno())
    else:
        # At an evaluation boundary weights may have been snapshotted already.
        prior = read_checkpoint(snapshot)
        if (prior['global_steps'], prior['current_stage']) != (data['global_steps'], data['current_stage']):
            raise ValueError('step checkpoint already exists with different progress')
    temp_json = out/'.checkpoint_pending.json'
    with temp_json.open('w') as stream:
        stream.write(payload)
        stream.flush(); os.fsync(stream.fileno())
    temp_zip.replace(out/'latest.zip')
    temp_json.replace(out/'latest.json')
    return snapshot





def resolve_checkpoint(value):
    path = Path(value)
    candidates = [path, ROOT/path, ROOT/'models'/path.name, ROOT/'models'/'1.0.0'/path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Checkpoint not found: {value}; tried {candidates}')


def load_or_create(c, env, checkpoint=None):
    from sb3_contrib import MaskablePPO
    ppo = copy.deepcopy(c['ppo'])
    net_arch = ppo.pop('net_arch')
    common = dict(env=env, device=c['device'], seed=c['seed'], verbose=0, **ppo)
    if checkpoint:
        # Override optimizer hyperparameters, not policy_kwargs or learned parameters.
        model = MaskablePPO.load(str(checkpoint), **common)
        print(f'Loaded {checkpoint}; inherited global steps={model.num_timesteps}', flush=True)
        return model
    return MaskablePPO('MlpPolicy', policy_kwargs={'net_arch': net_arch}, **common)


def save_model(model, out, name, metadata):
    model.save(str(out/name))
    (out/f'{name}.json').write_text(json.dumps(metadata, indent=2)+'\n')

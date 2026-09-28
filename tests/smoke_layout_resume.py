"""Real old-layout → package-layout continuation (tiny PPO, temporary files only)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]

# Run the exact old sources in a fresh interpreter: no flat-module aliases leak
# into the packaged process that subsequently resumes their checkpoint.
LEGACY = r'''
import sys, json
from pathlib import Path
root, repository = map(Path, sys.argv[1:])
sys.path[:0] = [str(Path.cwd()), str(repository/'tests'), str(repository)]
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from natural_bank import NaturalReverseBank, file_sha256
from the_game_env import TheGameEnv
from train_natural_reverse import run
from test_natural_bank import fixture, witness_record
from test_natural_pipeline import training_config
from unittest.mock import patch
from train_replay_curriculum import load_or_create

torch.set_num_threads(1)
bp = root/'bank.jsonl'
bp.write_text(''.join(json.dumps(witness_record(*fixture(seed=i)))+'\n' for i in range(50)))
c = training_config()
c.update(num_envs=2, max_steps_per_stage=128, eval_interval=128)
c['ppo'].update(n_steps=32, batch_size=64, n_epochs=1, net_arch=[16, 16])
n = c['natural_curriculum']
n.update(stages=[12, 84], start_stage=12)
n['promotion']['next_stage_min_wins'] = 100
n['evaluation'].update(nc_games=2, natural_games=2, natural_large_games=2, synthetic_games=2)
bank = NaturalReverseBank(bp, [12, 84])
env = make_vec_env(TheGameEnv, n_envs=2, seed=42)
model = MaskablePPO('MlpPolicy', env, n_steps=32, batch_size=64, n_epochs=1,
                    policy_kwargs={'net_arch': [16, 16]}, seed=42, device='cpu')
model.learn(64)
initial = root/'r80.zip'
model.save(initial)
initial.with_suffix('.json').write_text(json.dumps(dict(current_stage='R80', status='promoted',
    global_steps=64, model_sha256=file_sha256(initial))))
env.close()
c.update(resume=str(initial), output_dir=str(root/'split'))
run(c, initial, bank, root/'whole', stop_at=12)
loaded = []
def load(*args):
    learner = load_or_create(*args)
    loaded.append(learner)
    return learner
with patch('train_replay_curriculum.load_or_create', side_effect=load):
    paused = run(c, initial, bank, root/'split', stop_at=12,
                 stop_requested=lambda: bool(loaded) and loaded[0].num_timesteps >= 128)
assert paused['stage_steps'] == 64
'''


def main():
    from sb3_contrib import MaskablePPO
    from smoke_natural_continuation import equal
    with tempfile.TemporaryDirectory(prefix='layout-resume-') as tmp:
        root = Path(tmp)
        original = root/'original'
        with zipfile.ZipFile(ROOT/'archive/pre_package/source.zip') as source:
            source.extractall(original)
        subprocess.run([sys.executable, '-c', LEGACY, str(root), str(ROOT)], cwd=original, check=True)
        out = root/'split'
        preserved = {p: p.read_bytes() for p in (out/'manifest.json', out/'config.resolved.yaml', out/'difficulty_scan.json')}
        logs = {p: p.read_bytes() for p in (out/'history.csv', out/'evaluations.jsonl')}
        # Both preflight and actual continuation use the public one-command launcher.
        subprocess.run([str(ROOT/'run_natural_train.sh'), '--resume', str(out), '--check'], cwd=tmp, check=True)
        subprocess.run([str(ROOT/'run_natural_train.sh'), '--resume', str(out)], cwd=tmp, check=True)
        self_contained = json.loads((out/'latest.json').read_text())
        whole_metadata = json.loads((root/'whole/latest.json').read_text())
        whole = MaskablePPO.load(root/'whole/latest.zip', device='cpu')
        resumed = MaskablePPO.load(out/'latest.zip', device='cpu')
        equal(whole.policy.state_dict(), resumed.policy.state_dict())
        equal(whole.policy.optimizer.state_dict(), resumed.policy.optimizer.state_dict())
        equal(whole._natural_runtime['torch_rng'], resumed._natural_runtime['torch_rng'])
        equal(whole_metadata['source_usage'], self_contained['source_usage'])
        equal(whole_metadata['results'], self_contained['results'])
        assert self_contained['stage_steps'] == 128
        assert self_contained.keys() == whole_metadata.keys()
        assert preserved == {p: p.read_bytes() for p in preserved}
        assert all(p.read_bytes().startswith(prefix) for p, prefix in logs.items())
        print('PASS: exact legacy sources → shell preflight/resume; weights, optimizer, RNG, usage, evaluation, schemas and original provenance preserved')


if __name__ == '__main__':
    main()

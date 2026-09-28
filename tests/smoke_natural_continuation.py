"""Bounded real PPO continuation regression: two 64-transition updates per arm."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]


def equal(a, b):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b, (a, b)


def main():
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.env_util import make_vec_env
    from src.natural.bank import NaturalReverseBank, file_sha256
    from test_natural_bank import fixture, witness_record
    from test_natural_pipeline import training_config, result
    from src.env import TheGameEnv
    from src.natural.trainer import run
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bank_path = root/'bank.jsonl'
        bank_path.write_text(''.join(json.dumps(witness_record(*fixture(seed=i)))+'\n' for i in range(50)))
        c = training_config()
        c.update(num_envs=2, max_steps_per_stage=128, eval_interval=128)
        c['ppo'].update(n_steps=32, batch_size=64, n_epochs=1, net_arch=[16, 16])
        n = c['natural_curriculum']
        n.update(stages=[12, 84], start_stage=12)
        n['promotion']['next_stage_min_wins'] = 100
        n['evaluation'].update(nc_games=2, natural_games=2, natural_large_games=2, synthetic_games=2)
        bank = NaturalReverseBank(bank_path, [12, 84])
        env = make_vec_env(TheGameEnv, n_envs=2, seed=42)
        model = MaskablePPO('MlpPolicy', env, n_steps=32, batch_size=64, n_epochs=1,
                            policy_kwargs={'net_arch': [16, 16]}, seed=42, device='cpu')
        model.learn(64)
        initial = root/'r80.zip'
        model.save(initial)
        initial.with_suffix('.json').write_text(json.dumps(dict(current_stage='R80', status='promoted',
            global_steps=64, model_sha256=file_sha256(initial))))
        env.close()
        # Evaluation orchestration is tested separately; keep this test about PPO continuation.
        with patch('src.evaluation.natural.evaluate_nc', return_value=result(0, 2)), \
             patch('src.evaluation.natural.evaluate_natural', return_value=result(0, 2)), \
             patch('src.evaluation.natural.evaluate_reverse_panel', return_value={}):
            uninterrupted = run(c, initial, bank, root/'whole', stop_at=12)
            from src.common.checkpoints import load_or_create
            loaded = []
            def load(*args):
                learner = load_or_create(*args)
                loaded.append(learner)
                return learner
            with patch('src.common.checkpoints.load_or_create', side_effect=load):
                paused = run(c, initial, bank, root/'split', stop_at=12,
                             stop_requested=lambda: bool(loaded) and loaded[0].num_timesteps >= 128)
            assert paused['stage_steps'] == 64
            weights = copy.deepcopy(loaded[0].policy.state_dict())
            optimizer = copy.deepcopy(loaded[0].policy.optimizer.state_dict())
            restored = MaskablePPO.load(root/'split'/'latest.zip', device='cpu')
            equal(weights, restored.policy.state_dict())
            equal(optimizer, restored.policy.optimizer.state_dict())
            resumed = run(c, root/'split'/'latest.zip', bank, root/'split', resume_run=True)
        whole = MaskablePPO.load(root/'whole'/'latest.zip', device='cpu')
        split = MaskablePPO.load(root/'split'/'latest.zip', device='cpu')
        equal(whole.policy.state_dict(), split.policy.state_dict())
        equal(whole.policy.optimizer.state_dict(), split.policy.optimizer.state_dict())
        equal(uninterrupted['source_usage'], resumed['source_usage'])
        equal(whole._natural_runtime['torch_rng'], split._natural_runtime['torch_rng'])
        assert resumed['stage_steps'] == 128
        print('PASS: actual PPO pause/resume matches uninterrupted weights, optimizer, RNG and source counters')


if __name__ == '__main__':
    main()

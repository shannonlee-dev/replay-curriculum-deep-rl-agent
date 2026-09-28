"""Explicit real PPO smoke, deliberately outside model-free test discovery."""
from pathlib import Path
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))


def main():
    import numpy as np
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.env_util import make_vec_env
    from src.natural.bank import NaturalReverseBank, file_sha256
    from src.env import TheGameEnv
    from src.natural.trainer import run, read_checkpoint
    from test_natural_bank import fixture, witness_record
    from test_natural_pipeline import training_config
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bank_file = root/'bank.jsonl'
        bank_file.write_text(''.join(json.dumps(witness_record(*fixture(seed=i)))+'\n' for i in range(50)))
        c = training_config()
        c.update(num_envs=2, max_steps_per_stage=64, eval_interval=64)
        c['ppo'].update(n_steps=32, batch_size=64, n_epochs=1, net_arch=[16, 16])
        n = c['natural_curriculum']
        n.update(stages=[12, 84], start_stage=12, nc_episode_probability=.6)
        n['promotion'].update(next_stage_min_wins=100)  # Deliberately stall fixture after one update.
        n['evaluation'].update(nc_games=2, natural_games=2, natural_large_games=2, synthetic_games=2)
        bank = NaturalReverseBank(bank_file, n['stages'])
        env = make_vec_env(TheGameEnv, n_envs=2, seed=42)
        model = MaskablePPO('MlpPolicy', env, n_steps=32, batch_size=64, n_epochs=1,
                            policy_kwargs={'net_arch': [16, 16]}, seed=42, device='cpu')
        # Give the checkpoint real optimizer state, then verify both weights and optimizer on reload.
        model.learn(64)
        initial = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
        path = root/'r80.zip'
        model.save(path)
        path.with_suffix('.json').write_text(json.dumps(dict(current_stage='R80', status='promoted', global_steps=64,
                                                           model_sha256=file_sha256(path))))
        reloaded = MaskablePPO.load(path, device='cpu')
        for key, value in initial.items():
            assert torch.equal(value, reloaded.policy.state_dict()[key])
        for key, state in model.policy.optimizer.state_dict()['state'].items():
            for name, value in state.items():
                assert torch.equal(value, reloaded.policy.optimizer.state_dict()['state'][key][name])
        env.close()
        out = root/'smoke'
        report = run(c, path, bank, out, stop_at=12)
        assert report['total_added_steps'] == 64
        assert report['status'] == 'stalled'
        latest = read_checkpoint(out/'latest.zip')
        assert latest['global_steps'] == 128
        trained = MaskablePPO.load(out/'latest.zip', device='cpu')
        assert any(not torch.equal(value, trained.policy.state_dict()[key]) for key, value in initial.items())
        raw = report['source_usage']
        assert sum(raw['transitions'].values()) == 64
        assert sum(raw['completed_transitions'].values()) <= 64
        assert (out/'difficulty_scan.json').exists()
        assert (out/'final_held_out.json').exists()
        rows = [json.loads(line) for line in (out/'evaluations.jsonl').read_text().splitlines()]
        assert len(rows) == 2 and all(row['model_sha256'] for row in rows)
        for row in rows:
            assert row['natural_full']['games'] == 2
        print('PASS: real PPO 64-transition update, optimizer/weights resume, scan, metrics, checkpoints and held-out')


if __name__ == '__main__':
    main()

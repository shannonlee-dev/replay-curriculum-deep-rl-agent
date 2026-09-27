"""Export actual model games to a self-contained, offline replay table.

Run: .venv/bin/python html_model.py
Change only `html_model` below to watch another checkpoint.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from the_game_env import TheGameEnv

ROOT = Path(__file__).resolve().parent
# Hardcoded model input; relative paths are resolved against this file.
html_model = "models/best_model.zip"
OUTPUT = ROOT / "html_model.html"
SEEDS = (20260928, 20260929, 20260930)
REVERSE_TARGET = 48


def record_game(model, *, source: str, target: int, seed: int) -> dict:
    env = TheGameEnv(reset_source=source, target_remaining=target)
    turn = 1

    def snapshot():
        return dict(piles=env.piles.tolist(), hand=sorted(env.hand),
                    deck=len(env.deck), played=env.agent_cards_played,
                    turn=turn, turn_played=env.played_this_turn,
                    minimum=env._minimum_turn_play(),
                    legal=np.flatnonzero(env.action_masks()).tolist(),
                    done=bool(env.terminated), won=bool(env.won))

    try:
        obs, _ = env.reset(seed=seed)
        frames = [dict(state=snapshot(), move=None)]
        for _ in range(200):
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            action = int(action)
            if not mask[action]:
                raise RuntimeError(f"Model selected illegal action {action}")
            hand_before = set(env.hand)
            if action == env.END_TURN:
                move = dict(kind="end", turn=turn, action=action)
            else:
                card, pile = env.decode_action(action)
                before = int(env.piles[pile])
                backward = card == before + (-10 if pile < 2 else 10)
                move = dict(kind="play", action=action, turn=turn, card=card,
                            pile=pile, before=before, backward=backward)
            obs, _, terminated, truncated, info = env.step(action)
            move["drawn"] = sorted(env.hand - hand_before)
            if action == env.END_TURN and not terminated:
                turn += 1
            frames.append(dict(state=snapshot(), move=move))
            if terminated or truncated:
                break
        else:
            raise RuntimeError("Replay exceeded the game action bound")
        return dict(id=f"{source}-{seed}", source=source, target=target, seed=seed,
                    won=bool(info["won"]), frames=frames)
    finally:
        env.close()


def build_html(payload: dict) -> str:
    assets = ROOT / "web"
    # JSON is data, never executable markup (including user-selected filenames).
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    font = base64.b64encode((assets / "fonts/observatory-ko.woff2").read_bytes()).decode("ascii")
    license_text = (assets / "fonts/OFL.txt").read_text(encoding="utf-8")
    template = (assets / "replay.html").read_text(encoding="utf-8")
    template = template.replace("<!-- FONT_LICENSE -->", "<!--\n" + license_text + "\n-->")
    return (template.replace("/* REPLAY_STYLE */", (assets / "replay.css").read_text(encoding="utf-8"))
            .replace("/* REPLAY_SCRIPT */", (assets / "replay.js").read_text(encoding="utf-8"))
            .replace("__REPLAY_FONT__", font)
            .replace("__REPLAY_DATA__", data))


def main():
    import torch
    torch.set_num_threads(1)
    path = Path(html_model)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise SystemExit(f"Model not found: {path}\nEdit html_model in html_model.py.")
    # Load a single snapshot even if training later updates latest.zip.
    checkpoint = path.read_bytes()
    model = MaskablePPO.load(io.BytesIO(checkpoint), device="cpu")
    games = []
    for source, target in (("natural", 98), ("reverse", REVERSE_TARGET)):
        for seed in SEEDS:
            game = record_game(model, source=source, target=target, seed=seed)
            games.append(game)
            print(f"{source:7} seed={seed}: {len(game['frames'])-1} actions, "
                  f"{'WIN' if game['won'] else 'LOSS'}", flush=True)
    payload = dict(model=html_model, sha256=hashlib.sha256(checkpoint).hexdigest(),
                   timesteps=int(model.num_timesteps), games=games)
    OUTPUT.write_text(build_html(payload), encoding="utf-8")
    print(f"Open in your browser: {OUTPUT}")


if __name__ == "__main__":
    main()

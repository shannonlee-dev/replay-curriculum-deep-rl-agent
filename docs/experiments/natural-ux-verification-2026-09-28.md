# Natural training UX / continuation verification — 2026-09-28

> Historical experiment record. Active workflow: `./run_natural_train.sh`; see [natural replay curriculum](../natural-replay-curriculum.md).

Normal workflow:

```bash
./run_natural_train.sh
./run_natural_train.sh --resume runs/<run_dir>
```

The launcher validates the bank and R80 checkpoint, runs or reuses a provenance-matched difficulty scan, and starts NC12 with the configured frontier defaults (25 wins, 0.05 win rate, 0.70 relative / 0.05 absolute retention). No interactive threshold selection is required.

Continuation restores policy and optimizer, environment and RNG, current stage and all step counters, source usage, promotion/retention history, individual evaluation results, and the global large-natural evaluation schedule. It appends to the same run. Bank/config/code/manifest/scan/model/log mismatches fail closed; continuation JSON is also bound to a digest inside the PPO zip. A run lock prevents concurrent writers.

Checkpoint files are flushed before atomic replacement. Step snapshots are immutable. Evaluation JSONL/CSV appends use a checkpointed pending event so a crash between the two appends can finish the missing append without truncation. SIGINT/SIGTERM requests wait for a complete PPO update or evaluation item and print the exact resume command.

## Executed checks

- `tests/run_without_models.py`: 100 tests, 95 passed, 5 real-model tests intentionally skipped.
- Natural test suite after final checkpoint changes: 68 tests passed.
- `test_natural_continuation.py` after two additional regression cases: 10 tests passed. Covers mid-NC12, NC16 persistence, pre-promotion pause, large-evaluation scheduling, append recovery, config/bank/model/manifest/scan/log/continuation tampering, and completed-run idempotence.
- `tests/smoke_natural_continuation.py`: bounded real CPU PPO continuation matches uninterrupted weights, optimizer, Torch RNG and source counters. Each arm adds only 128 transitions to a temporary fixture model.
- `tests/smoke_natural_frontier.py`: real evaluation and one 64-transition fixture update, checkpoint and held-out checks passed.
- Python compilation, shell syntax and `git diff --check`: passed.
- Actual bank preflight: 5,000 unique winning decks, no split deficiencies. Artifact: `runs/natural_ux_preflight_20260928/preflight.json`.

Real-bank PPO training was not executed. Historical runs without complete continuation state are not silently migrated or restarted.

- Actual R80 difficulty scan: all 12 stages completed, NC12 selected. Artifact: `runs/natural_ux_scan_20260928/difficulty_scan.json`. Matching-provenance cache reuse was verified with recomputation forbidden. `training_started=false`; no training checkpoint was created.

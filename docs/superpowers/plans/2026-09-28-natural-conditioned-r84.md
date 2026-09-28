# Natural-conditioned R84 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan in the current session.

**Goal:** Verified natural checkpoint bank, privileged teacher, isolated NC84 A/B trainer and evaluator.
**Architecture:** Reuse TheGameEnv rules; add offline bank validation, reset subclass and separate CLI. Existing main code and artifacts remain untouched.
**Tech Stack:** Python, NumPy, Gymnasium, existing MaskablePPO, unittest.
**Spec:** ../specs/2026-09-28-natural-conditioned-r84-design.md

## Global Constraints

- User override: implement on develop; no worktree, model execution, bank generation or training/evaluation runs.
- Reward WIN=1, otherwise=0; observation=300; actions=393.
- New CLI/config/output only; existing main artifacts read-only.
- Pure environment tests and fake policy integration are permitted; real PPO smoke is deferred to the user.

## Review Focus

- Corrupted or partially written bank: fail closed; append recovery must never silently mix provenance.
- NC98 and odd remaining: exact full start and deterministic unique bucket allocation.
- Time/node limits within candidate expansion: bounded failure, never unwinnability.
- R84 restart panel and schedule: retain R64/R68 older floors and relative-step milestones.
- Output collision and data leakage: refuse existing runs and overlapping deck IDs.

## Tasks

### 1. Bank and reset
- [x] Write failing tests for full prefix/suffix replay, corruption, splits, bucket ties, leakage and masks.
- [x] Implement natural_bank.py: build_record(permutation, actions), validate_record(record), NaturalBank(path).
- [x] Implement natural_conditioned_env.py: NC/synthetic reset, reuse step/mask/obs.
- [x] Run environment-only unit tests.

### 2. Teacher and resumable generator
- [x] Write failing tests for turn transitions, winning path, symmetry and budget failure using hand-authored state/priors.
- [x] Implement natural_teacher.py: TeacherConfig, search(permutation, prior, config, seed).
- [x] Implement build_natural_bank.py: uniform shuffles, per-attempt journal, compatible resume, counts and hashes.
- [x] Run tests with no learned model loaded or executed.

### 3. Experiment and evaluation
- [x] Write failing tests for source mixture, fixed gates, evaluation milestones, output protection and fake-policy integration.
- [x] Implement natural_evaluation.py and train_natural_conditioned.py with archive/nc84_direct_mix/config.yaml.
- [x] Reuse original load_or_create/save_model and promotion_checks; no main trainer changes.
- [x] Provide standalone matched checkpoint evaluation, final held-out and leakage checks.
- [x] Run model-free integration tests; defer real PPO smoke explicitly.

### 4. Documentation and final verification
- [x] Add docs/natural-conditioned.md with pilot/build/train/control/evaluate commands and limits.
- [x] Run all new tests, model-free existing regression tests, CLI help, compile and diff checks.
- [x] Review final changes; report unexecuted model checks and completed checks separately.

## Ledger

- Ruling: User's latest instruction overrides isolated worktree and long experiment execution in the original spec.
- Ruling: Native implementation is authorized through completion; no repeated approval handoff.
- Ruling: Tests use hand-authored legal fixtures and fake policies; they do not demonstrate teacher throughput or learned-policy performance.

- Task 1 complete: bank witness validation, metadata isolation, split/bucket rules and real reset-mixture tests.
- Task 2 complete: turn search and bounded failures tested with a hand-authored prior; resumable journal tests.
- Task 3 complete: NC/control fake-learner integration, fixed synthetic panel, relative-step evaluation and provenance.
- Task 4 complete: CLI --help, py_compile, model-free repository suite and independent code review.
- Verification: `tests/run_without_models.py`: 40 discovered, 35 passed, 5 real-PPO tests explicitly skipped. Actual PPO construction/load/learn/predict patched to fail.
- Review fixes: control manifest zip TypeError, missing-journal resume, and missing/mixed teacher provenance each reproduced and fixed. Independent reviewer confirmed all three fixes.
- Deferred by user: actual teacher/model inference, production bank generation, real PPO smoke, long training and performance evaluation. No model performance or throughput claim is made.

# Natural frontier verification — 2026-09-28

> Historical experiment record. Active workflow: `./run_natural_train.sh`; see [natural replay curriculum](../natural-replay-curriculum.md).

최종 목적은 unconditional random natural full-game terminal WIN 확률이다. 실제 bank PPO 학습은 실행하지 않았다.

## 확정 설정

- 시작점 **NC12**: 사용자 지정. Auto 기능은 유지하되 이번 실험에는 사용하지 않는다.
- Next stage: held-out WIN ≥25 **그리고** win rate ≥5%.
- 이전 모든 natural anchor: 기준 성능의 ≥70% **그리고** 절대 win rate ≥5%.
- 사용자 지정 **first operational baseline**을 YAML에서 읽고 manifest에 기록한다.
- 각 promotion check는 actual, required, operator, passed, PASS/FAIL과 retention reference를 기록한다.

> NC12 selected because it is the hardest scanned stage with demonstrated PPO learnability from prior held-out validation improvement; harder stages had ≤2.3% baseline win rate and no demonstrated learnability.

선정 이유의 prior held-out 학습 근거는 사용자 제공이다. 이번 scan은 baseline만 측정한다. 인용문의 2.3%는 NC16을 소수점 한 자리로 반올림한 값이다.

## 실제 difficulty scan

모델: `models/restart_r64_from_line5/stage_R80_complete.zip`, global steps **19,398,656**, NC seed **200000**. 1,000판은 cap이며 unique validation decks만 평가했다. 두 scan의 결과가 모두 동일했다.

| Stage | Train / Validation / Test | WIN / Games | WIN rate | Wilson 95% CI |
| --- | ---: | ---: | ---: | ---: |
| NC4 | 2024 / 282 / 253 | 237 / 282 | 84.04% | 79.31%–87.86% |
| NC8 | 3569 / 474 / 440 | 234 / 474 | 49.37% | 44.89%–53.85% |
| NC12 | 3842 / 508 / 488 | 44 / 508 | 8.66% | 6.52%–11.43% |
| NC16 | 3903 / 513 / 492 | 12 / 513 | 2.34% | 1.34%–4.04% |
| NC20 | 3946 / 522 / 497 | 7 / 522 | 1.34% | 0.65%–2.74% |
| NC24 | 3959 / 525 / 499 | 2 / 525 | 0.38% | 0.10%–1.38% |
| NC32 | 3971 / 525 / 497 | 0 / 525 | 0.00% | 0.00%–0.73% |
| NC48 | 3970 / 526 / 498 | 0 / 526 | 0.00% | 0.00%–0.73% |
| NC64 | 3971 / 526 / 499 | 0 / 526 | 0.00% | 0.00%–0.73% |
| NC72 | 3974 / 526 / 499 | 0 / 526 | 0.00% | 0.00%–0.73% |
| NC80 | 3975 / 526 / 499 | 0 / 526 | 0.00% | 0.00%–0.73% |
| NC84 | 3975 / 526 / 499 | 0 / 526 | 0.00% | 0.00%–0.73% |

NC 성능은 heuristic-WIN-conditioned 분포의 성능이며 unconditional natural 성능이 아니다.

## 검증

- Model-free: **92개 실행, 87 통과**, 실제 모델 테스트 5개 분리.
- Natural 관련: **60개 통과**.
- Tiny PPO: optimizer/가중치 복원, **64-transition update**, scan/계측/checkpoint/held-out 통과.
- 실제 bank `--check`: **5,000 unique winning decks**, split 부족 0, **criteria ready=True**. 모든 witness 및 checkpoint continuation replay 유지.
- 실제 resume model `--scan-only`: 완료, **NC12 명시적 선택** 및 위 이유 저장. 실제 bank PPO `learn`은 실행하지 않았다.
- `git diff --check`: 통과.

## 변경 범위

- `src/natural/trainer.py`: 설정 기반 실행, preflight/scan, frontier, 정기 natural 평가, safe pause, 출력 schema.
- `src/natural/curriculum.py`, `configs/natural_replay.yaml`: 검증/override, auto 및 명시적 선택, 상세 판정.
- `src/natural/bank.py`, `src/evaluation/natural.py`: 동적 stage 추출/replay 검증, scan/스케줄/중복 fail-closed.
- `src/common/source_usage.py`, `archive/nc84_direct_mix/env.py`, `src/common/terminal_ui.py`: 누적/window, 완료/실행 transition 구분, Rich/non-TTY.
- `archive/nc84_direct_mix/trainer.py`: deprecated direct-mix 보존 및 공통 평가/계측 재사용.
- `src/evaluation/distribution_audit.py`: training과 독립된 observation-only audit.
- Natural reverse/resume/pipeline 테스트, tiny PPO smoke 및 natural 문서.
- 활성 natural trainer의 고정 STAGES, NC reset 30%, rehearsal 70/30, synthetic 60/60/50 gate 제거.

## 직접 학습 시작

```bash
.venv/bin/python -m src.natural.trainer \
  --config configs/natural_replay.yaml \
  --output-dir runs/natural_frontier_nc12_mix30
```

60% arm은 `--set natural_curriculum.nc_episode_probability=0.60`과 새 output 경로를 사용한다. 동일 model/optimizer, bank, seeds, scan, PPO budget 설정을 유지한다.

운영 문서: [natural-reverse.md](../natural-replay-curriculum.md). 원시 [difficulty scan](../data/natural_frontier_difficulty_scan_20260928.json). 확정 [실험 manifest](../data/natural_frontier_manifest_20260928.json).

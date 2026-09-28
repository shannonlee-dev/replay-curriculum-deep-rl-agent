# Natural reverse curriculum

최종 목적은 `P(WIN | unconditional random natural full-game start)`이다. 활성 entry point는 `./run_natural_train.sh`다. `archive/nc84_direct_mix/trainer.py`와 `archive/nc84_direct_mix/config.yaml`은 과거 direct-mix 실험 재현용이며 새 natural frontier 설정으로 자동 변환하지 않는다.

## 보존하는 계약

게임 규칙, 393개 action/mask, 300차원 observation은 그대로다. 보상은 terminal WIN=1, LOSS=0, intermediate=0이다. Remaining cards, cards played, turn length는 보상이나 승급 목적함수로 사용하지 않는다.

Bank의 분포는 **natural random deck | selected heuristic eventually WIN**이다. Unconditional natural 분포가 아니다. 전체 natural-start→WIN replay, 기존 checkpoint→WIN replay, provenance, 파일 hash와 deck_id 중복 검증을 수행하고, 새로 추출한 각 checkpoint의 continuation도 독립 restore→WIN replay로 검증한다. 각 stage는 target remaining ±2의 실제 살아 있는 turn boundary 중 가장 가까운 한 지점을 덱당 최대 하나 선택한다. 거리가 같으면 먼저 만난 경계를 유지한다. 기존 deck_id split을 유지한다. 미래 deck order는 환경 복원에만 쓰는 private state다.

## 설정과 사전 검사

새 schema는 `configs/natural_replay.yaml`이다. `natural_curriculum`에는 다음이 들어간다.

| 키 | 의미 |
| --- | --- |
| `stages`, `start_stage`, `start_stage_reason` | 증가하는 NC stage 목록, 마지막 NC84; 숫자와 선택 이유 또는 `auto` |
| `nc_episode_probability` | **매 episode reset에서** NC를 뽑는 확률 |
| `rehearsal.current_weight`, `older_weight` | NC 내부 현재/이전 stage 가중치; 이전 stage에는 균등 배분 |
| `minimum_winning_decks` | 실험에서 요구하는 전체 unique winning decks 수 |
| `minimum_stage_decks.{train,validation,test}` | 각 stage/split에 필요한 unique decks 수 |
| `auto_start.min_wins`, `min_win_rate` | 가장 어려운 시작 stage를 고르는 명시적 terminal WIN 기준 |
| `promotion.mode` | `frontier` |
| `promotion.min_stage_steps` | 승급 전 최소 PPO transitions |
| `promotion.next_stage_min_wins`, `next_stage_min_win_rate` | 다음 NC stage에서 필요한 terminal WIN signal |
| `promotion.retention_relative_floor`, `retention_absolute_floor` | 이전 모든 NC anchor가 각각 만족해야 하는 기준 |
| `promotion.final_stage_rule` | `budget_and_retention` |
| `evaluation` | NC/regular natural/large natural/synthetic 판수, seed, large 주기 |

5,000개는 과거 campaign의 production 규모이며 loader의 무결성 상수가 아니다. 예제의 deck 최소값 1은 비어 있지 않다는 기계적 조건일 뿐 통계적 충분성을 주장하지 않는다. 실험자는 필요한 표본 수를 명시해야 한다. NC 평가는 요청 판수와 available unique decks 중 작은 수만큼 수행하며, 재사용으로 판수를 부풀리지 않는다. Actual games와 available count를 함께 저장한다.

**이번 실험은 사용자가 NC12를 명시적 시작점으로 선택했다.** `start_stage_reason`에는 다음 근거를 그대로 기록하며, scan/manifest에도 전달한다. 이는 사용자 제공 실험 근거이며 이번 baseline scan만으로 학습 가능성을 새로 입증했다는 뜻은 아니다.

> NC12 selected because it is the hardest scanned stage with demonstrated PPO learnability from prior held-out validation improvement; harder stages had ≤2.3% baseline win rate and no demonstrated learnability.

**성능 기준은 임의로 정하지 않았다.** 사용하지 않는 `auto_start` 기준은 YAML에서 `null`이다. 명시적 시작점에서는 auto 기준을 요구하지 않는다. Frontier 기준은 사용자가 정한 첫 operational baseline으로 `next_stage_min_wins: 25`, `next_stage_min_win_rate: 0.05`, `retention_relative_floor: 0.70`, `retention_absolute_floor: 0.05`를 설정한다. 이 값들은 코드 상수가 아니라 설정이며 `criteria_status: first_operational_baseline`, `criteria_source: user_specified`와 함께 manifest에 남긴다. `--check`와 `--scan-only`는 이를 허용하고 상태를 보고하지만, 학습은 frontier promotion의 명시적 수치가 없으면 fail closed 한다. 예전 config, 잘못된 override, NaN, 범위 밖 확률은 명확한 오류를 낸다.

```bash
./run_natural_train.sh
./run_natural_train.sh --resume runs/<run_dir>

# 학습 없이 검증 / difficulty scan
./run_natural_train.sh --check
./run_natural_train.sh --scan-only
```

기본 bank는 `runs/natural_campaign/bank.jsonl`, 초기 checkpoint는 `models/restart_r64_from_line5/stage_R80_complete.zip`이다. 정상 실행은 추가 선택 없이 NC12부터 시작하며 `--resume`은 기존 run 디렉터리만 받는다. 초기 모델 변경은 config의 `resume` 필드를 사용한다. 새 실행은 timestamp가 붙은 디렉터리를 자동 생성하고, 재개는 같은 디렉터리에 append한다. `--check`는 모델을 실행하지 않고 `--scan-only`는 PPO `learn`을 호출하지 않는다. 동일 model/bank/evaluation/code/package provenance의 scan만 캐시에서 재사용한다.

Scan은 동일한 resume model로 모든 설정 stage의 validation terminal WIN을 평가하고 train/validation/test available counts, wins/games, win rate, Wilson 95% CI를 저장한다. Auto는 split 표본 조건과 두 WIN 기준을 모두 만족하는 가장 어려운 stage를 선택한다. 선택 이유와 eligible stages는 manifest에도 기록한다. 적합한 stage가 없거나 기준이 미지정이면 scan 결과를 보존하고 학습을 시작하지 않는다.

## 승급, 평가와 종료

Current NC mastery gate는 없다. 매 평가마다 current, next, 이전 모든 NC anchor를 검사한다. 최소 stage transitions, next-stage WIN count/rate, 모든 이전 anchor의 절대·상대 retention이 충족되면 진급한다. Retention reference는 preflight 성능으로 시작하며 stage 완료 시 그 stage의 성능이 더 높으면 올린다. 성능 하락 때문에 reference를 낮추지 않는다. Resume에는 reference도 보존한다.

마지막 NC84는 next가 없으므로 **설정된 stage 예산을 모두 사용하고 minimum steps와 이전 anchor retention을 만족할 때** `final_stage_complete`로 기록한다. 이것은 unconditional natural 성능의 성공 판정이 아니다. 다른 stage는 예산이 끝나도 frontier 조건이 부족하면 stalled로 종료한다. `--stop-at`은 설정된 stage에서 수동 종료하는 범위 제한이다.

학습 중 매 평가에는 고정 seed unconditional natural full game을 기본 1,000판 평가한다. Stage와 무관하게 누적 additional transitions가 1M 경계를 넘을 때 별도 10,000판 `natural_full_large` 평가를 기록한다. Full PPO rollout/update를 마친 첫 경계에서 실행하며, stage 전환이나 resume으로 주기를 초기화하지 않는다. Large 평가에서도 regular 결과를 별도 유지한다. 모든 planned natural deck을 bank와 대조해 중복이면 **재추출 없이 오류**를 낸다. Natural full은 최상위 진단 지표이며 promotion에는 쓰지 않는다.

정상 run 종료 시 current/final NC test, NC84 test, 학습 중 진단 seed와 분리한 unconditional natural 10,000판, 기존 synthetic regression panel을 평가한다. 자동 성공/실패 verdict는 만들지 않는다. SIGINT/SIGTERM 첫 요청은 완결된 PPO update 또는 평가 경계에서 모델·optimizer·진행상태를 저장하고 paused로 종료한다. 반복된 interrupt도 안전한 경계를 기다린다. 저장 후 정확한 `./run_natural_train.sh --resume ...` 명령을 출력한다. Paused run에는 비싼 final held-out을 수행하지 않는다.

## 계측과 출력

공통 `SourceUsageWrapper`는 시작 episode, 실제 모든 transition, 완료 episode, 완료 episode의 transition 합계, terminal WIN을 source별로 센다. 비율은 다음과 같다.

- Completed episode share = source 완료 episode / 전체 완료 episode.
- Completed transition share = source 완료 episode 길이 합계 / 전체 완료 episode 길이 합계.
- 진행 중 episode의 transition은 별도 원시 `transitions`와 `all_transition_shares`에 포함된다.

누적 값은 stage를 넘어 이어진다. Window 값은 직전 평가 이후의 차이다. 각 source의 완료 episodes, 실제 실행 transitions, 완료 episode 길이 합계인 completed_transitions, terminal wins/win rate와 두 share를 각각 누적/window로 저장한다. Window completed_transitions는 해당 window에서 완료된 episode의 전체 길이이므로 이전 window에서 실행한 일부 transition을 포함할 수 있다. 실제 window PPO 경험량은 transitions를 사용한다. Resume은 진행 중 episode, 환경 및 RNG를 복원한다. Stage 전환에서만 환경을 새로 만들며 전환 당시 미완료 episode는 완료 통계에 더하지 않는다. 모든 실제 PPO transition은 원시 누적 통계에 남는다. Transition share를 맞추는 sampler는 없다.

TTY에서는 Rich live dashboard를 갱신한다. Natural full을 맨 위에 보여주고 NC validation/next, 누적/window mix와 train WIN을 구분한다. Non-TTY에서는 평가당 간단한 한 줄만 남긴다. Rollout별 print 또는 raw JSON 출력은 없다.

출력은 `manifest.json`, `config.resolved.yaml`, `difficulty_scan.json`, `evaluations.jsonl`, `history.csv`, `latest.zip/json`, `checkpoints/*`, `stage_NC*_complete.zip/json` 또는 `stage_NC*_stalled.zip/json`, `final_held_out.json`, `run_summary.json`, `runtime.log`이다. Scan-only/preflight/paused에는 해당 단계에서 생성 가능한 산출물만 남는다. Evaluation JSONL에는 current/next/anchors, natural full과 large, raw source usage와 window, share, promotion checks, model hash를 저장한다. 각 promotion check는 실제 값(`actual`), 기준값(`required`), 비교 연산자, `passed`, `status: PASS/FAIL`을 포함한다. 상대 retention에는 reference와 relative floor도 기록한다. Step snapshot은 immutable이고 latest 파일은 atomic replace한다. Resume은 같은 run의 history/evaluations에 append만 한다. 평가 항목별 진행 위치와 next large evaluation, 승급/retention 이력, 환경/RNG를 checkpoint에 함께 저장한다. Bank/config/code/manifest/scan/model/log hash가 달라지면 fail closed한다. 동시 실행은 run lock으로 거부한다.

## 동일 조건 mix A/B

사용자가 확정한 NC12와 첫 operational baseline을 담은 **동일한 설정** `configs/natural_replay.yaml`, checkpoint, bank, seeds, PPO rollout와 stage budget을 두 arm에 사용한다. 아래 커맨드는 사용자가 직접 실행한다.

```bash
.venv/bin/python -m src.natural.trainer \
  --config configs/natural_replay.yaml \
  --set natural_curriculum.nc_episode_probability=0.30 \
  --output-dir runs/natural_mix30_new

.venv/bin/python -m src.natural.trainer \
  --config configs/natural_replay.yaml \
  --set natural_curriculum.nc_episode_probability=0.60 \
  --output-dir runs/natural_mix60_new
```

두 arm은 동일 checkpoint의 policy/optimizer state를 로드하고 동일 seed로 환경/RNG를 시작하며, 모델을 변경하기 전에 같은 difficulty scan을 수행한다. Model/bank/config/code hashes, seed와 budget은 manifest에 기록한다. Frontier 결과에 따라 실제 소모 예산과 stage가 달라질 수 있으므로 비교는 동일 additional-transition 위치에서 한다. Resume은 지정한 run의 latest에서 실제 continuation을 수행한다. 작은 CPU PPO 회귀 테스트에서 중단 없는 실행과 weights/optimizer/RNG/source counters 일치를 검증한다.

## Offline distribution audit

```bash
.venv/bin/python -m src.evaluation.distribution_audit \
  --bank runs/natural_campaign/bank.jsonl \
  --model models/restart_r64_from_line5/stage_R80_complete.zip \
  --strategy combined --games 1000 \
  --output verification/public_distribution_audit_new.json
```

A=heuristic WIN-conditioned bank, B=heuristic all-reached natural, C=PPO natural roll-in을 비교한다. B/C는 같은 고정 seed를 사용하며 최종 승리 여부로 선별하지 않는다. 각 stage당 덱당 가장 가까운 turn boundary 하나를 사용한다. Feature는 실제 300차원 public observation 그대로이며 미래 deck/provenance/teacher 정보가 없다. Stage별 sample count와 feature mean/std를 출력한다. `--classifier`는 scikit-learn이 있을 때 별도 holdout logistic classifier AUC를 추가하고, 없으면 설치 없이 상태만 기록한다. A bank의 heuristic과 B의 명시적 strategy가 다르면 그 차이도 포함된다는 점을 결과에 기록한다. Training에는 영향을 주지 않는다.

## 검증

```bash
.venv/bin/python tests/run_without_models.py
.venv/bin/python -m unittest discover -s tests -p 'test_natural_*.py'
.venv/bin/python tests/smoke_natural_frontier.py
.venv/bin/python tests/smoke_natural_continuation.py
```

Smoke는 임시 fixture bank, 작은 실제 PPO와 64-transition update만 사용한다. 실제 bank의 장시간 PPO 학습은 자동 실행하지 않는다.

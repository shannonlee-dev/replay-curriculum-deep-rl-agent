# NC84 natural trajectory handoff pilot v3

> Historical experiment record. Active workflow: `./run_natural_train.sh`; see [natural replay curriculum](../natural-replay-curriculum.md).

`archive/handoff_v3/build_bank.py`는 기존 `archive/privileged_teacher/build_bank.py`와 R84-start teacher를 변경하지 않는 별도 100-deck pilot이다. 기본 output은 `runs/nc84_bank_pilot_v3`이며 기존 디렉터리에 덮어쓰지 않는다.

```bash
.venv/bin/python -m archive.handoff_v3.build_bank \
  --model models/restart_r64_from_line5/stage_R80_complete.zip \
  --output-dir runs/nc84_bank_pilot_v3 --workers 4
```

## Rollout과 handoff

- Natural permutation은 기존 pilot과 같은 `SeedSequence([730000, index])`로 생성한다. Index 짝수는 deterministic, 홀수는 stochastic R80 policy를 사용한다. Policy sampling seed는 deck별로 독립이다.
- 처음 도달한 remaining 82~86의 살아 있는 turn boundary를 NC84로 저장한다. Prefix가 죽거나 이 범위를 건너뛰면 teacher가 prefix를 수리하지 않는다.
- NC84 이후에도 동일한 environment와 실제 deck에서 policy rollout을 계속한다. Policy에는 300차원 public observation과 393개 action mask만 전달한다.
- R80/R76/R72/R68/R64/R60/R56/R52/R48의 checkpoint를 저장한다. 실제 remaining이 target의 ±2인 turn boundary를 가장 가까운 target에 배정하며, 동률이면 작은 target이다. Target별 가장 가까운 상태 하나를 유지하고 동일 거리이면 먼저 도달한 상태를 유지한다. 도달률은 이 규칙에 따른 checkpoint 확보율이다. 건너뛴 target은 도달로 세지 않는다.
- Policy가 죽거나 처음으로 remaining ≤48인 살아 있는 turn boundary에 도달하면 rollout을 끝낸다. 종료된 상태나 turn 중간 상태는 teacher에 넘기지 않는다.
- 저장된 checkpoint 중 실제 remaining이 가장 작은 것부터 privileged suffix search를 실행한다. Exhausted이면 이전 checkpoint를 시도한다. Timeout/node limit이면 해당 deck을 종료한다. 설정의 25초·500000 nodes는 **모든 handoff가 공유하는 deck당 총 상한**이다. Inference 한 번 등의 원자적 작업 때문에 wall-clock은 소폭 초과할 수 있다.
- NC84에는 도달했지만 하위 checkpoint 없이 죽으면 `policy_failed_to_reach_handoff`이다. Prefix 실패 및 teacher 실패와 구분한다. R84-start fallback은 하지 않는다.

## Witness와 검증

Rk→WIN을 찾으면 원래 natural start부터 해당 handoff까지 저장한 policy actions와 teacher suffix를 연결한다. Policy가 handoff 이후 진행하다 죽은 actions는 witness에 넣지 않는다. `build_record` 및 `validate_record`로 처음부터 전체 WIN replay와 NC84에서의 독립 continuation replay를 검사한다. Journal에서도 policy trajectory와 checkpoint의 hand·pile·played·future deck·action offset을 실제 replay에 대조한다.

Bank는 기존 schema v2의 NC84 record를 저장한다. v3는 생성기/journal 버전이며 기존 bank schema를 바꾸지 않는다. Record의 `prefix_action_offset`은 NC84 위치이고, `handoff.policy_action_offset`은 teacher 시작 위치이다. `record.search`의 진행 수치는 기존 규약에 맞게 NC84 기준이며 각 `handoffs`의 검색 수치는 해당 Rk 기준이다. 다른 NC bucket은 이번 pilot에서는 저장하지 않는다. Private provenance와 teacher 정보를 observation에 추가하지 않는다.

## 출력과 통계

- `manifest.json`: 모델/metadata/code hash, seed, worker 수, handoff 규칙, deck당 탐색 상한.
- `attempts.jsonl`: fsync하는 authoritative journal. 실패한 deck도 full permutation, policy actions, 모든 checkpoint, handoff 검색 결과를 저장한다.
- `bank.jsonl`: replay 검증을 통과한 NC84 WIN records.
- `summary.json`: 기존 pilot 판정과 `handoff` 상세 통계.

`summary.handoff`에는 다음 값이 있다.

- `prefix_nc84_reach_rate`: NC84 prefix 도달 / 전체 unique natural decks.
- `reach.Rk`: handoff checkpoint 확보 수, 전체 deck 대비 도달률, NC84 도달 deck 대비 조건부 도달률.
- `selected_distribution.Rk`: 최초로 선택된 가장 깊은 handoff 수 및 handoff를 선택한 deck 중 비율.
- `by_handoff.Rk`: 모든 실제 시도 수 및 success(`win_found`)/timeout/exhausted/node_limit, 성공률, nodes, seconds. Fallback도 별도 시도로 센다.
- `nc84_witness_rate`: replay 검증 완료 NC84 witness / 전체 unique natural decks.
- `nc84_witness_rate_given_prefix`: NC84 도달 후 witness 확보율.

각 row의 `selected_handoff`와 `winning_handoff`는 fallback이 있으면 다를 수 있다. `unattempted_handoffs`는 성공 또는 예산 종료로 시도하지 않은 저장 지점이다. 도달하지 않은 지점이나 미시도 지점을 exhausted로 기록하지 않는다. 제한된 teacher의 exhausted는 해당 탐색 후보 소진이며 deck의 unwinnable 판정이 아니다.

Worker 4개는 독립 deck을 처리하고 index 순서로 journal에 기록한다. 탐색 상한은 wall-clock이므로 CPU 경쟁에 따라 검색량과 결과가 바뀔 수 있다. 이전 pilot과 시간당 성능을 비교할 때 worker 수와 nodes도 함께 확인해야 한다.

## 재개와 검증

동일한 명령에 `--resume`을 붙인다. Code/model/config/worker manifest가 일치해야 하며, 기존 v1/v2 output은 재개 대상으로 받지 않는다. 완료된 100-deck pilot도 재개 명령으로 검증하고 export할 수 있다. 100개 결과를 기록한 것과 생산 규모 확대에 충분한 witness를 확보한 것은 별개이며, 기존 `summary.pilot`의 최소 3개/확대 추정 10개 기준을 그대로 보고한다.

```bash
.venv/bin/python -m unittest discover -s tests -p test_natural_handoff.py -v
.venv/bin/python -m unittest discover -s tests -v
```

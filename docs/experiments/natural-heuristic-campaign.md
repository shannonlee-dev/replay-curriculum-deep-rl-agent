# Natural heuristic NC84 campaign

> Historical experiment record. Active workflow: `./run_natural_train.sh`; see [natural replay curriculum](../natural-replay-curriculum.md).

> **중단됨:** direct-mix A/B와 coordinator를 중단했다. 후속 실행은 [natural reverse curriculum](../natural-replay-curriculum.md)의 수동 커맨드를 사용한다. 아래 캠페인 명령은 과거 실행 기록이다.

2026-09-28 실행부터 privileged suffix teacher 대신 `exponential` / `combined`의 직접 natural rollout으로 WIN witness를 생성한다. 기존 teacher 결과와 모델은 보존하며 이 캠페인에서는 suffix search와 R80 prefix inference를 호출하지 않는다.

```bash
.venv/bin/python -u run_natural_campaign.py --output runs/natural_campaign --workers 4
```

출력 경로는 새 디렉터리여야 한다. 실행 중인 경로로 재실행하지 않는다. 현재 실행 로그는 `runs/natural_campaign.log`, 단계는 `runs/natural_campaign/status.json`이다.

1. 동일 seed 730000의 natural permutation 10,000개씩 `exponential`, `combined`를 순차 평가한다. 비교 지표는 win rate, wins/sec, runtime이다. Runtime에는 프로세스 시작·rollout·승리 replay 검증·출력 저장을 포함한다. 동일한 worker 4개를 사용한다.
2. wins/sec가 높은 전략 하나를 선택한다. 해당 전략의 10k 평가에서 얻은 NC84 witness를 재사용하고, seed 740000부터 batch별 1,000개의 새 덱을 생성한다. deck_id로 중복 제거하고 정확히 5,000개를 선택한다. 마지막 batch의 여분 witness는 batch 출력에 남고 최종 bank에는 포함되지 않는다.
3. 모든 WIN은 natural 초기 상태부터 action 전체를 replay 검증한다. NC84는 처음 도달한 remaining 82~86의 살아 있는 turn boundary이다. 이 boundary를 건너뛴 WIN은 승률에 포함하지만 NC84 bank에는 넣지 않는다. NC84부터의 continuation도 독립적으로 replay 검증한다.
4. deck_id SHA-256 modulo 10으로 train/validation/test를 약 80/10/10 분리한다. 한 덱당 NC84 checkpoint 하나만 저장하고 train만 PPO reset에 사용한다.
5. 최종 bank를 다시 `NaturalBank`로 전수 검증한 뒤 기존 R80 stage-complete에서 `nc`, `control`을 실행한다. 기존 `archive/nc84_direct_mix/config.yaml`의 seed 42, 최대 2,998,272 추가 transitions, 최소 250k 및 60/60/50 진급 기준을 적용한다. 각 arm은 독립 진급 시 조기 종료할 수 있으므로 최종 step이 다르면 같은 학습량의 비교라고 해석하지 않는다.

NC군의 episode reset 비율은 NC84 30%, synthetic R84 25%, R80 15%, R76 9%, R72 6%, R20 10%, R12 5%다. Control은 R84 35%, R80 25%, R76 15%, R72 10%, R20 10%, R12 5%다. 30/70은 episode reset 확률이며 transition 비율은 episode 길이에 따라 달라진다. `source_usage`에 두 비율의 원시 계수를 기록한다.

Policy는 hand·played·pile tops·legal mask만 읽는다. 미래 deck order는 환경 복원용 private checkpoint에만 저장한다. 기존 bank schema의 `teacher.model_sha256` 필드는 학습 모델 hash가 아닌 `standalone-natural:<strategy>` 문자열의 SHA-256 식별자이며, config의 `policy_kind=heuristic`, `privileged_suffix=false`가 출처를 명시한다.

출력:

- `manifest.json`: 코드/시작 모델/설정 hash와 실행 protocol
- `benchmark/summary.json`: 두 전략의 비교와 선택
- `production/batch_*/`: batch별 attempts, WIN bank, summary
- `bank.jsonl`, `bank_summary.json`: 정확히 5,000 unique decks, split 수와 replay 검증 결과
- `nc/`, `control/`: 모델 checkpoint, validation history, source usage, 최종 held-out test
- `comparison.json`: 두 arm의 최종 R84, R98, NC84 test, unconditional natural 결과
- `status.json`: benchmark / production / full_bank_validation / ppo_ab / complete / failed

평가 bank는 heuristic-selected winning-deck 분포이다. NC84 test 성능은 unconditional natural 승률과 별도 지표이며, 기존 A/B는 synthetic 비율도 달라 NC 데이터만의 인과 효과를 분리하지 않는다.

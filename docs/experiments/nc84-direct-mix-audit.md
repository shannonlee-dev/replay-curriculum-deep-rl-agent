# NC84 direct-mix 중단 및 수동 검사

> Historical experiment record. Active workflow: `./run_natural_train.sh`; see [natural replay curriculum](../natural-replay-curriculum.md).

2026-09-28 사용자의 요청으로 `runs/natural_campaign`의 coordinator, NC 혼합군, 대조군을 모두 중단했다. 기존 모델과 로그는 보존했다. 이후 실제 bank 전수 검사·모델 평가·학습은 사용자가 직접 커맨드를 실행한다. 자동 재시작이나 새 학습은 예약하지 않았다.

중단 후 기존 Monitor 로그를 읽은 결과:

| 학습 source | 완료 에피소드 | WIN (`won=True`) | 누적 보상 1 | WIN/보상 불일치 |
| --- | ---: | ---: | ---: | ---: |
| natural_conditioned | 5,699 | 0 | 0 | 0 |
| reverse | 13,286 | 5,399 | 5,399 | 0 |

양쪽 모델의 마지막 저장 평가 시점은 추가 1,048,576 transitions다. Monitor 집계는 중단 직전까지 완료된 rollout 에피소드이므로 마지막 저장 모델의 step과 시점이 다를 수 있다. 중단된 rollout 일부는 optimizer 업데이트에 반영되지 않았을 수 있다. validation 0승과 training 0승은 별도 집계다.

아래의 **실제 5,000개 bank 전수 검사 결과는 아직 없다.** 과거 생성 시 저장된 `bank_summary.json`의 검증 완료 표시는 이번 재검사의 결과로 대체하지 않는다.

## 직접 실행할 검사

프로젝트 루트에서 실행한다. 학습 프로세스가 중단된 상태여야 한다.

```bash
.venv/bin/python -u audit_nc_direct_mix.py \
  --run runs/natural_campaign/nc \
  --bank runs/natural_campaign/bank.jsonl \
  --output runs/natural_campaign/direct_mix_audit.json
```

모델을 생성·로드·추론·학습하지 않는다. 검사는 다음을 수행한다.

1. 학습 manifest와 bank hash 및 환경·bank·NC reset·학습 소스 코드 hash를 대조한다.
2. 모든 training Monitor 파일에서 source별 완료 에피소드와 실제 WIN을 집계한다. `won=True`와 보상 1이 일치하는지, NC source가 remaining 82~86인지 검사한다. 평가 게임은 포함하지 않는다. 손상된 행이나 잘린 행을 조용히 버리지 않는다.
3. 모든 bank record를 natural 처음부터 WIN까지 replay한다. 각 NC84 checkpoint를 독립 복원하여 continuation도 WIN까지 replay한다. deck_id 중복·분할·checkpoint·provenance를 검증한다.
4. 실제 `NaturalConditionedEnv.reset`을 통해 train 상태 64회를 샘플링하고 저장된 정답 continuation을 실행한다. train 분할의 checkpoint인지와 observation/action/reward 계약을 확인한다. 이때 발생한 WIN은 **PPO training WIN에 포함하지 않는다.**
5. 검사 도중 파일이 바뀌지 않았는지 hash를 재확인한다.

전수 replay는 수 분 걸릴 수 있다. 콘솔에는 1/3 로그 검사, 2/3 bank 전수 검사, 3/3 training reset 경로 검사를 표시하고 마지막에 NC training WIN 수와 판정을 출력한다. 출력 파일이 이미 존재하면 덮어쓰지 않으므로 재검사 시 새 `--output` 이름을 사용한다.

## 결과 판정

- `checks_passed: true`, `verdict: direct_mix_failed_for_observed_run`: 입력·보상·replay 검사를 통과했고 NC training WIN이 0이다. **관측한 예산에서 현재 direct-mix 설계가 실패한 것으로 판단하고 natural reverse curriculum 전환을 준비한다.** 모든 direct-mix 설정이 불가능하다는 일반적 증명은 아니다.
- `checks_passed: true`, `verdict: nc_training_wins_present`: NC 학습 중 WIN이 존재한다. 빈도와 validation 차이를 확인한 후 변경을 결정한다.
- `verdict: integrity_failure_investigate_first`: 입력 또는 replay 검사가 실패했다. `error`를 먼저 해결하고, 설계 실패로 판정하지 않는다.
- `verdict: no_completed_nc_training_episodes`: NC 학습이 실제로 일어났다는 증거가 부족하다. reset 분기와 로그부터 확인한다.

정상 검사 결과는 종료 코드 0, 손상 또는 증거 부족은 종료 코드 2다. 정상 검사 결과가 direct-mix 실패라는 판정이어도 검사 자체는 성공이므로 종료 코드 0이다. 어떠한 판정도 후속 학습을 자동 실행하지 않는다.

기존 Monitor에는 개별 `deck_id`가 기록되지 않았다. 따라서 과거 에피소드 각각의 split을 사후 재구성할 수는 없다. 대신 실행 코드 hash, bank split 및 실제 train reset 경로를 검사한다.

## 전환 조건이 충족된 뒤의 방향

같은 검증된 자연 승리 trajectory에서 쉬운 후반의 turn-boundary 상태부터 checkpoint를 추출한다. 예를 들어 natural R12→R20→…→NC84로 범위를 늘리며, 동일 deck_id는 모든 단계에서 같은 split을 유지한다. 새 덱을 미래 정보로 탐색하거나 합성 경로로 수리하지 않는다.

진급 여부는 natural validation 성능으로 판단하고 synthetic 성능은 회귀 지표로 별도 추적한다. 각 natural 단계에서 training WIN이 실제로 발생하는지 source별로 기록한다. 사용자 요청에 따라 [natural reverse 학습 진입점](../natural-replay-curriculum.md)을 준비했다. 직접 실행하면 학습 전에 bank 전수 검증을 수행한다. 새 학습은 실행하지 않았다.

# NC84 fixed-100 Dietrich suffix teacher

> Historical experiment record. Active workflow: `./run_natural_train.sh`; see [natural replay curriculum](../natural-replay-curriculum.md).

기존 v3 pilot은 0/100으로 종료했다. 원본 `runs/nc84_bank_pilot_v3`는 보존하며, 새 결과는 `runs/nc84_dietrich_fixed100`에 저장한다. Reward/environment/PPO 학습 코드는 수정하지 않는다.

## 고정 조건

- 원본 authoritative journal의 100개 permutation, policy actions, NC84 및 모든 handoff checkpoint를 replay 검증하고 그대로 사용한다. Policy를 다시 샘플링하지 않는다.
- SeedSequence([730000, index])로 permutation과 prefix/search seed를 재계산해 일치를 검증한다. Cohort와 원본 journal/manifest SHA256는 `frozen_cohort.json`에 저장한다.
- Deepest-first, exhausted에서만 shallower fallback, deck 전체 공유 25초/500000 nodes, worker 4를 유지한다. Handoff마다 25초를 새로 주지 않는다.
- Search beam 384, local candidate turns 10, heuristic top-k 6 + seeded random extras 최대 8, legal END 포함. Generic score와 PPO prior/value는 이 ablation에서 사용하지 않는다.
- Rollout은 동일한 local turn enumeration에서 가장 높은 점수의 완성 turn 하나를 확정해 진행한다(outer beam 1). Turn 사이 backtracking은 없다. 이는 단일-action greedy나 논문의 exhaustive turn enumeration과는 다르다.
- 지수형/거리형 각각 rollout/search를 평가한다. 100개에서 0승인 조건은 `stopped_zero_wins`로 종료하며 자동 확장·추가 튜닝하지 않는다.

## 수식

[Dietrich (2019), §§3.4 및 3.5.2](https://felixdietrich.com/_app/immutable/assets/Felix_Dietrich_Bachelor_Thesis_Document.fu-Qzr_5.pdf)를 참고했다.

지수형은 미사용 카드(hand와 future deck 전체)에 대해 `v(s) = -Σ exp(-1.5 * playable_pile_count)`이다. Backwards trick도 playable에 포함한다. 미래 deck 순서는 평가값에 영향을 주지 않지만 검색의 draw에는 실제 순서를 사용한다.

거리형은 playable이면 pile penalty 1, 아니면 `3.5 - exp(-0.03 * (abs(top-card)-1))`을 사용하고, 카드별 네 pile penalty의 곱을 합산해 음수화한다. `3.5**4`로 상수 정규화한다. 원문 식 3.2의 양수 지수는 거리 증가에 따른 recovery 감소 설명 및 penalty 범위와 모순되므로 음수 감쇠로 해석했다. 거리형은 지수형에 임의 가중치를 더하는 대신 논문 §3.5의 별도 product penalty variant다. 파라미터를 이 100개에 맞춰 탐색하지 않았다.

## 실행 및 해석

```bash
.venv/bin/python -m archive.handoff_v3.evaluate
# 중단된 동일 코드/설정의 실행을 재개하거나 완료된 journal 검증
.venv/bin/python -m archive.handoff_v3.evaluate --resume
.venv/bin/python tests/run_without_models.py
```

`comparison.json`은 모든 handoff의 attempts/wins, timeout/exhausted/node-limit, deepest_remaining 최소/평균, nodes 합계, seconds 합계와 witness rate를 담는다. 모든 성공은 원래 natural start 및 NC84 continuation에서 WIN replay 검증을 통과해야 한다.

Exhausted는 제한된 후보 집합 소진이지 해당 deck의 불가능 증명이 아니다. Rollout exhausted는 greedy 경로 종료를 의미한다. Exhausted 감소가 timeout 증가로 대체되었을 뿐이면 witness 개선으로 판단하지 않는다. Fallback 횟수 차이로 handoff별 attempted cohort가 달라질 수 있으므로 attempts를 함께 본다.

Nodes는 v3와 동일하게 검색 후보 transition 수이다. Heuristic action ranking용 임시 one-step 평가 횟수는 nodes에 포함하지 않으며, 그 비용은 seconds에 포함한다. 기존 v3의 neural inference도 nodes에는 포함하지 않는다. Wall-clock 제한 실험이므로 동일 seeds라도 CPU 부하에 따라 결과가 달라질 수 있다. 기존 v3 측정값을 재사용했으며 baseline을 재실행한 것은 아니다.

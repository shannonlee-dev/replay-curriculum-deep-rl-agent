# Natural-conditioned R84 A/B 실험 설계

상태: 구현 및 모델 비실행 검증 완료. 실제 teacher/PPO 실행과 성능 검증은 사용자가 직접 수행한다.

사용자 후속 지시: `develop`에서 작업하며 worktree를 만들지 않는다. 모델 실행·bank 생성·학습·평가는 실행하지 않는다. 아래 장기 실행 절차는 사용자 실행 대상으로 남긴다.

## 목적과 범위

동일한 R80 승급 모델에서 synthetic 능력을 유지하면서 NC84 및 unconditional natural 승률이 더 빨리 개선되는지 확인한다. 첫 실험은 NC84 학습 비중 30%, 승리 natural deck 5,000개 목표, R84 추가 학습 최대 3M steps로 제한한다. NC88~NC98 정식 커리큘럼과 unconditional natural 학습은 후속 범위다.

보상은 terminal WIN=1, 그 외=0. 기존 observation 300차원과 action 393개를 유지한다. Teacher heuristic은 탐색에만 사용한다.

## 확인한 로컬 기준

- Git: `develop`, HEAD `31f4f62`, 조사 시 작업 트리 clean.
- 시작 모델: `models/restart_r64_from_line5/stage_R80_complete.zip` 및 동명 JSON.
- R80 metadata: global_steps=19,398,656, status=promoted, R80=608/1000, natural=0/1000.
- 조사 시 latest metadata: R84, stage_steps=1,835,008, R84=549/1000, natural=0/1000.
- 프로세스 조회에서는 학습 프로세스를 찾지 못했다. 다른 세션/호스트 실행 여부는 미확인이다. 기존 실행을 종료하거나 재시작하지 않는다.
- `TheGameEnv`는 natural/reverse/mixed reset만 지원한다. 저장된 natural checkpoint 복원과 teacher search는 새 구현이 필요하다.
- 기존 trainer의 `seen`은 실행 시작 시 빈 집합이다. R84에서 재시작하면 main의 R64~R80 누적 평가 panel을 자동 복원하지 않는다.

## 구현 접근과 격리

최종 방식: 사용자 지시에 따라 develop 작업 트리에 NC 실험 코드를 추가하고, 기존 env의 규칙·observation·mask·reward를 상속하는 NC reset 환경과 독립 실험 진입점을 추가한다. 기존 trainer의 모델 로드와 평가 함수 등 재사용 가능한 함수를 활용한다. 기존 main 환경·trainer·config·output은 수정하지 않는다. 신규 실험은 별도 진입점과 `runs/` 출력 경로를 사용한다.

대안 1: 기존 trainer와 env에 opt-in 설정을 추가하면 중복은 적지만 main과 공유하는 코드의 회귀 위험이 커진다.
대안 2: 모든 환경 규칙을 복제하면 격리는 쉽지만 natural 규칙과 teacher simulator가 어긋날 위험이 크다.

출력은 새 실험 디렉터리로 제한하고 기존 출력이 있으면 실패한다. 실행 manifest에 코드 revision, 원본 모델 SHA-256, bank SHA-256, resolved config, seed, 시작 timestep을 저장한다. A/B는 실험 분기이며 기존 main을 재설정한다는 뜻이 아니다.

## Teacher와 bank

예정 파일: `archive/privileged_teacher/teacher.py`, `src/natural/bank.py`, `archive/privileged_teacher/build_bank.py`.

1. 전용 RNG로 2..99 permutation을 균등 shuffle한다. Synthetic plan으로 natural deck을 대체하지 않는다.
2. 고정 R80 PPO의 masked policy/value prior와 privileged heuristic을 이용해 turn 단위 beam search를 수행한다. PPO 입력에는 기존 observation만 전달한다.
3. 카드 후보를 확장하여 합법적 END와 refill까지 적용한 successor를 생성한다. 덱 소진 후 최소 1장, 마지막 카드에서 즉시 WIN 등 기존 규칙을 따른다.
4. 승리를 찾으면 즉시 중단하고 초기 natural 상태부터 실제 `TheGameEnv.step`으로 전체 경로를 replay한다. Replay 검증 실패는 bank에 저장하지 않고 오류로 기록한다.
5. 전체 경로와 checkpoint suffix를 검증한 후에만 bank record를 기록한다.

기본 상한: beam_width=2048, candidate_turns_per_state=32, max_nodes_per_deck=500000, timeout_seconds_per_deck=10, policy_top_k_actions=6, stochastic_candidates=8. 카드 후보 확장도 node budget에 포함하고 후보 생성 도중에도 timeout을 확인한다. 제한 시간은 환경에 따른 작은 체크 주기 지연을 기록한다.

Heuristic은 PPO prior/value, future-card compatibility, backward trick 기회, legal continuation count, pile damage로 구성한다. 계수와 구성은 config 및 provenance에 저장한다. 10초 내 승리 탐색률은 아직 측정되지 않았으므로 작은 pilot으로 throughput과 성공률을 먼저 측정한다.

중복 키는 같은 덱 안에서 `(deck_position, sorted(hand), sorted(asc_tops), sorted(desc_tops))`를 사용한다. 경계 상태의 q는 0이다. 실제 상태와 action 경로는 pile 순서를 유지하며 key만 canonicalize한다. PPO prior는 pile 순열에 불변이라고 가정하지 않는다.

종료 사유: `win_found`, `teacher_failed_timeout`, `teacher_failed_node_limit`, `teacher_failed_exhausted`. 제한 탐색의 실패를 unwinnable 또는 게임 LOSS로 보고하지 않는다.

## Record 및 bucket 계약

Deck record: schema version, permutation의 SHA-256인 deck_id, full_permutation, teacher actions, teacher config/model hash, search nodes/time, split, checkpoints.

Checkpoint: bucket, actual remaining_count, hand, pile_tops, played membership, future_deck_order, deck_position, q=0, action offset. Runtime 학습 reset에는 teacher score와 continuation을 전달하지 않는다. deck_id 등 provenance는 observation에 넣지 않는다.

- 지원 라벨: NC84, NC88, NC92, NC96, NC98.
- NC84~NC96은 목표 ±2 범위 안에서 가장 가까운 bucket 하나에만 배정한다. 동률이면 작은 라벨로 배정한다.
- NC98은 remaining=98인 초기 경계 상태만 허용한다. remaining=97 등을 NC98이라고 부르지 않는다.
- 같은 상태의 여러 bucket 등록을 금지한다. 한 덱에서 bucket별 가장 가까운 checkpoint 하나만 저장한다. 같은 거리면 경로에서 먼저 도달한 상태를 선택한다.
- 실제 remaining은 홀수일 수 있다. 기존 synthetic의 짝수 난이도 정의는 바꾸지 않는다.
- 경로가 어떤 bucket도 통과하지 않을 수 있으므로 빈 bucket을 synthetic으로 채우지 않는다.

Deck hash의 고정 구간으로 train/validation/test를 약 80/10/10 배정한다. 같은 permutation은 생성 seed나 append 시점에 관계없이 같은 split이다. 5,000은 전체 고유 winning deck 목표이며 NC84 train state 5,000개를 의미하지 않는다. Split/bucket별 실제 개수를 보고한다. Train bank에 validation/test deck이 섞이면 실행을 거부한다.

Bank 생성은 중간 결과를 보존하고 중복 deck을 검출한다. sampled decks, unique sampled decks, teacher wins, 실패 사유별 수, 성공률, 평균 nodes/time, split/bucket별 개수를 기록한다. 재개 시 schema/config/model hash가 다르면 새 bank가 필요하다.

## Reset와 학습 분포

예정 파일: `archive/nc84_direct_mix/env.py`, `archive/nc84_direct_mix/trainer.py`, `archive/nc84_direct_mix/config.yaml`.

NC env는 기존 `TheGameEnv`를 상속하여 reset만 확장한다. 복원 시 card partition, deck suffix, q, remaining을 검증하고 agent diagnostics는 0으로 초기화한다. `step`, `action_masks`, `_get_obs`는 기존 구현을 재사용한다. RNG seed별 reset이 재현 가능하고 vector env 사이에 mutable state를 공유하지 않아야 한다.

첫 R84 실험의 episode reset 확률:

| Source | 확률 |
| --- | ---: |
| NC84 train | 0.30 |
| Synthetic R84 | 0.25 |
| Synthetic R80 | 0.15 |
| Synthetic R76 | 0.09 |
| Synthetic R72 | 0.06 |
| Synthetic R20 | 0.10 |
| Synthetic R12 | 0.05 |

Recent replay 내부는 기존 0.25:0.15:0.10 비율을 유지한다. 이는 episode 비중이며 실제 transition 비중은 episode 길이에 따라 달라져 별도로 기록한다. NC bank가 비거나 손상되면 시작 전에 실패한다.

기존 main control은 그대로 보존한다. 필요시 같은 R80에서 새 control을 생성하되 기존 synthetic 비중 R84=.35, R80=.25, R76=.15, R72=.10, R20=.10, R12=.05를 사용한다. Teacher-selected data를 쓰는 효과와 synthetic 비중 재배분의 효과는 이 A/B만으로 완전히 분리되지 않는다는 해석 한계를 명시한다.

## 예산·진급·평가

- 시작 weight와 optimizer state는 동일 R80 archive에서 로드한다. 모든 PPO parameter 및 시작 timestep 동일성을 검사한다.
- 16 env × 1024 n_steps = 16384 transitions/rollout. 최대 3,000,000 예산은 2,998,272 transitions로 내림한다.
- 진급은 기존 current≥.60, previous≥.60, older floor≥.50, minimum steps≥250000을 유지한다. NC metric은 gate나 best-model 선택에 사용하지 않는다.
- A/B synthetic panel은 main R84와 동일한 R12/R20/R32/R48/R64/R68/R72/R76/R80/R84/R98로 명시적으로 고정한다. 기존 main의 history와 공통 추가 step에서 비교하고 early promotion이면 종료 지점 차이를 보고한다.
- 일반 평가: 기존 262144-step cadence, synthetic 및 natural 1000 games. 평가 RNG는 training RNG와 분리한다.
- Natural 10000 games: 추가 학습량이 1M, 2M 경계를 처음 넘긴 평가, stage complete, 최종 budget 종료 시. 같은 평가 시점에서는 중복 실행하지 않는다. inherited global_steps를 간격 기준으로 사용하지 않는다.
- NC validation은 고정된 별도 deck으로 주기 평가한다. 최종 held-out test는 학습 및 모델 선택에 사용하지 않는다. 하나의 고정 checkpoint를 여러 번 실행한 것을 독립 deck 수로 세지 않는다.
- 최종 네 지표: synthetic R84 승률, NC84 test 승률, reverse R98 승률, unconditional natural 10k wins/games 및 Wilson 구간. NC88은 bank 확보 시 보조 validation 지표로 추가한다.
- Natural 평가 seed 영역은 bank 생성과 분리한다. 평가 permutation이 bank에 포함되지 않았는지 hash로 검사한다. 충돌은 숨겨서 다시 뽑지 말고 평가 protocol 오류로 보고한다.
- 기존 main artifact는 다시 저장하지 않고 별도 evaluator가 읽어 동일한 held-out suite를 평가한다. 과거 동일 step 모델이 남아 있지 않으면 그 시점의 matched comparison은 불가능하다고 보고한다.

## 필수 검증

예정 파일: `tests/test_natural_bank.py`, `tests/test_natural_teacher.py`, `tests/test_natural_training.py`.

1. 모든 저장 checkpoint를 초기 permutation부터 prefix actions로 replay하여 state 전체가 일치하는지 확인.
2. 모든 checkpoint의 continuation을 replay하여 반드시 terminal WIN인지 확인.
3. Observation shape/값에 teacher metadata, source, future order가 포함되지 않는지 확인.
4. Future order만 바꾼 두 동일 상태의 observation과 현재 action mask가 일치하는지 확인.
5. 모든 checkpoint q=0 및 card partition/remaining/deck_position 정합성 확인.
6. 원본 natural env와 restored env의 action mask, step transition, 종료 조건 일치 확인.
7. 중간·LOSS reward=0, WIN=1 확인.
8. Train/validation/test deck ID 교집합이 비어 있고 중복 permutation도 같은 split인지 확인.
9. Bucket tie/홀수 remaining/NC98 정확성, 손상 bank 거부, timeout/node-limit 실패 분류 확인.
10. Source 확률, 독립 reset, PPO observation 계약, loaded weights 동일성, promotion panel 및 10k 평가 trigger 확인.
11. 작은 검증 bank로 실제 PPO smoke 실행, model save/load, 결과 로그 확인. Smoke의 승률은 성능 근거로 사용하지 않는다.
12. 기존 테스트 회귀 검사. Main output을 테스트 경로로 사용하지 않는다.

## 실행 순서와 완료 기준

1. develop에서 신규 파일과 독립 output 경로를 준비한다.
2. Teacher/bank/replay 검증을 먼저 구현하고 작은 pilot을 실행한다.
3. 검증된 bank만 받는 NC reset, 학습 분포 및 평가 연결을 구현한다.
4. 필수 테스트와 실제 PPO smoke를 통과시킨다.
5. Pilot 성공률과 자원 비용을 보고하고 5,000 winning deck 생성 및 2~3M 학습을 실행한다. 목표 미달이나 실패를 성공으로 보고하지 않는다.
6. 공통 R80 출발점과 추가 학습량을 기준으로 네 지표를 보고한다. Natural 첫 1승만으로 효과를 확정하지 않고 별도 seed 반복과 불확실성을 함께 본다.

구현 완료, bank 목표 달성, 장기 학습 완료, NC 효과 입증은 서로 다른 결과로 기록한다. 실제 모델을 실행하는 단계는 수행하지 않았으며, 수작업 fixture 및 가짜 prior/learner 기반 검증 결과만 생성했다.

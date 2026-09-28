> 이 문서는 deprecated direct-mix 실험 재현용입니다. 활성 natural frontier 경로는 [natural-reverse.md](../natural-replay-curriculum.md)를 따릅니다.

> Historical experiment record. Active workflow: `./run_natural_train.sh`; see [natural replay curriculum](../natural-replay-curriculum.md).

# Natural-conditioned R84 실험 실행

현재 R80 승급 모델에서 NC84 혼합 학습과 기존 synthetic 학습을 비교하는 별도 진입점입니다. 구현은 `develop`에 있으며 기존 `src/env.py`, `src/synthetic/trainer.py`, `configs/synthetic.yaml` 및 main 모델/출력을 변경하지 않습니다.

아래 명령은 사용자가 직접 실행하는 절차입니다. 구현 검증 중에는 실제 모델을 생성·로드·추론·학습하지 않았습니다. Teacher 처리량, natural 승리 탐색률, 실제 PPO 재개 및 성능은 아직 검증하지 않았습니다.

## 1. Teacher pilot

저장소 루트에서 실행합니다. 아래 pilot도 실제 R80 모델을 사용하는 작업입니다.

```bash
.venv/bin/python -m archive.privileged_teacher.build_bank \
  --model models/restart_r64_from_line5/stage_R80_complete.zip \
  --output-dir runs/nc84_bank \
  --prefix-mode mixed
```

Random natural deck에서 R80 policy가 public observation과 action mask만 보고 rollout합니다. 처음 도달한 remaining 82~86, q=0의 살아 있는 turn-boundary state를 NC84 checkpoint로 사용합니다. `--prefix-mode mixed`는 덱별 deterministic/stochastic을 번갈아 사용하며, 각각 단독 지정할 수도 있습니다. Window를 건너뛰거나 먼저 패배하면 teacher로 prefix를 수리하지 않습니다.

Teacher는 **해당 checkpoint부터만** future-deck-aware suffix beam search를 실행합니다. Suffix WIN이 발견되면 policy prefix와 teacher suffix를 처음부터 함께 replay하고, 실제 policy가 도달한 NC84 state 하나만 저장합니다. 기본값은 beam **384**, candidate turns **10**, timeout **25초**, 최대 500000 nodes입니다. 조정 범위는 beam 256~512, 후보 8~12, timeout 20~30초입니다. Timeout은 진행 중인 inference/정렬 한 번만큼 초과할 수 있습니다.

기본 pilot은 100 deck입니다. `summary.json`의 `pilot` 판정은 다음과 같습니다.

- **0/100**: `teacher_redesign_required`. Teacher를 재설계하고 새 디렉터리에서 다시 수행합니다. NC 아이디어 실패로 해석하지 않습니다.
- **1~2/100**: `insufficient_witnesses`. 최소 witness 기준에도 미달합니다.
- **3~9/100**: witness는 확보했지만 추가 튜닝이 필요하며 생산 확대는 차단합니다.
- **10개 이상/100**: 5000개 bank의 예상 덱 수와 시간, Wilson 구간 기반 덱 수 범위를 계산합니다. 이 기준을 통과해야 생산 확대가 가능합니다.

10/100이면 약 50000 deck, 20/100이면 약 25000 deck이 필요합니다. 평균 시도 시간이 25초라고 가정하면 직렬 실행 약 347시간/174시간입니다. 실제 시간 추정에는 관측한 prefix+suffix+record 생성 시간을 사용하며, 모델 로드와 export IO는 제외합니다. 현재 성공률이 유지된다는 조건의 추정이며 처리량 보장은 아닙니다.

`policy_failed_to_reach_nc84`와 `teacher_failed_timeout/node_limit/exhausted`는 별도로 기록합니다. Policy 실패 세부 사유는 `prefix.reason`에, teacher 진행 정도는 `deepest_remaining`, `max_cards_played`, `deepest_turn`, `frontier_size`에 남습니다. 카드 진행/turn 수는 suffix 시작점 기준이며, frontier는 마지막 외부 beam 크기입니다. Policy 도달 실패에는 teacher 지표가 null입니다. `summary.json`에서 도달률과 도달 후 teacher 성공률도 따로 확인할 수 있습니다.

100 deck에서 10개 미만이면 CLI는 실패 종료 코드를 반환하되 모든 완료 시도와 witness를 보존합니다. 설정을 바꿀 때는 **새 디렉터리**를 사용하세요. 제한 탐색 실패는 unwinnable 판정이 아닙니다.

## 2. 같은 bank 확대 및 재개

```bash
.venv/bin/python -m archive.privileged_teacher.build_bank \
  --model models/restart_r64_from_line5/stage_R80_complete.zip \
  --output-dir runs/nc84_bank \
  --winning-decks 5000 --max-sampled-decks 100000 --resume
```

`--max-sampled-decks`는 이번 실행의 추가 수가 아닌 누적 시도 상한입니다. Seed/model/config/code hash가 일치해야 재개됩니다. Timeout의 wall-clock 특성 때문에 중단된 미완료 시도를 재검색하면 결과가 달라질 수 있습니다.

- `attempts.jsonl`: 각 시도와 winning witness를 fsync하는 원본 journal.
- `bank.jsonl`: journal에서 내보낸 검증 완료 winning deck records.
- `manifest.json`: 모델·코드·teacher 설정·생성 seed 출처.
- `summary.json`: 검색률과 bucket/split 통계, bank hash.

Journal이 유실되거나 마지막 줄이 불완전하면 재개를 거부합니다. 원본을 보관하여 확인한 후 복구해야 합니다. 손상된 파생 `bank.jsonl`은 정상 journal을 사용한 resume/export로 복원할 수 있습니다. 생성기 실행 중인 bank를 학습에 사용하지 말고 생성이 끝난 파일을 고정하세요.

Bank record 자체에도 teacher 모델·설정·코드 hash와 덱별 nodes/time이 들어갑니다. 다른 teacher 출처의 record 혼합, 중복 덱, prefix 상태 불일치, 승리하지 않는 continuation은 로드 시 거부합니다.

5000은 **전체 고유 winning deck 수**입니다. Hash 기준 train/validation/test 약 80/10/10이므로 train NC84 5000개를 뜻하지 않습니다. 새 생성 경로는 winning deck마다 NC84 checkpoint를 정확히 하나 저장하며 split별 개수는 summary에 기록합니다. Synthetic record로 부족분을 채우지 않습니다.

## 3. 모델 없이 사전 검사

```bash
.venv/bin/python -m archive.nc84_direct_mix.trainer \
  --resume models/restart_r64_from_line5/stage_R80_complete.zip \
  --bank runs/nc84_bank/bank.jsonl \
  --output-dir runs/nc84_experiment --check
```

설정, R80 승급 metadata, bank의 전체 prefix/suffix, provenance 및 split/bucket 수를 확인합니다. 모델을 로드하지 않습니다. 기본 실험은 winning deck 최소 5000개와 NC84 train/validation/test 모두를 요구합니다. 작은 pilot은 `--quick --check`로 검사할 수 있으나 세 split의 NC84가 모두 있어야 합니다. 100-deck pilot에서 확보한 witness 수가 적으면 작은 split이 비어 있을 수 있습니다.

## 4. 실제 PPO smoke와 본 학습

Smoke는 모델을 실행하므로 사용자가 직접 수행합니다. 출력 디렉터리는 매번 새 경로여야 합니다.

```bash
.venv/bin/python -m archive.nc84_direct_mix.trainer \
  --resume models/restart_r64_from_line5/stage_R80_complete.zip \
  --bank runs/nc84_bank/bank.jsonl \
  --output-dir runs/nc84_smoke --quick
```

```bash
.venv/bin/python -m archive.nc84_direct_mix.trainer \
  --resume models/restart_r64_from_line5/stage_R80_complete.zip \
  --bank runs/nc84_bank/bank.jsonl \
  --output-dir runs/nc84_experiment --arm nc
```

`--resume`는 R80 모델을 학습 시작점으로 쓰는 옵션입니다. 중단된 NC run의 optimizer/환경/RNG까지 이어가는 재개 기능은 아닙니다. 기존 output은 거부합니다.

Episode reset 비율은 NC84 30%, synthetic R84 25%, R80 15%, R76 9%, R72 6%, R20 10%, R12 5%입니다. `source_usage`에는 source별 episode 시작 수와 transition 수를 별도 기록합니다. Episode 길이에 따라 실제 transition 비중은 다릅니다.

새 v2 bank는 NC84만 저장합니다. 기존 v1 bank의 NC84~NC96은 ±2 이내에서 가까운 라벨 하나로 배정하고 동률은 작은 라벨을 택합니다. 같은 덱에서는 라벨별 가장 가까운 상태 하나를 선택합니다. 실제 remaining은 홀수일 수도 있습니다. **NC98은 remaining=98인 초기 상태만** 받습니다. 기존 v1 bank의 NC88~NC98 로드는 지원하지만 학습은 R84에서 종료합니다. v1/v2 record를 같은 bank에 섞는 것은 거부합니다.

보상은 WIN=1, 나머지=0입니다. 300차원 observation/393개 action 및 기존 mask를 유지하며 미래 순서·deck_id·teacher score·continuation·source는 policy observation에 들어가지 않습니다.

최대 추가 학습량은 3M 설정을 rollout 크기 16384에 맞춰 내린 **2,998,272 transitions**입니다. 60%/60%/older 50% 및 최소 250k 조건을 만족하면 그 전에 종료합니다. 평가 panel은 R12/20/32/48/64/68/72/76/80/84/98로 고정하여 R84 재시작 시 older floor가 줄어들지 않도록 했습니다.

## 5. Control과 비교

기존 main은 그대로 보존됩니다. 동일 seed·R80 출발점의 새 control도 실행하려면 다음을 사용합니다.

```bash
.venv/bin/python -m archive.nc84_direct_mix.trainer \
  --resume models/restart_r64_from_line5/stage_R80_complete.zip \
  --bank runs/nc84_bank/bank.jsonl \
  --output-dir runs/nc84_control --arm control
```

Control은 synthetic R84 35%, R80 25%, R76 15%, R72 10%, R20 10%, R12 5%입니다. Bank는 validation/test와 natural 중복 검사에만 사용하며 학습에는 넣지 않습니다. 두 arm의 학습 예산·진급 panel·평가 seed는 같습니다. 기존 main은 이전 stage의 RNG와 학습 이력이 있으므로 새 control과 완전히 같은 실행 궤적은 아닙니다.

`history.csv`, `evaluations.jsonl`, `latest`, `best_R84`, `best_balanced`, `stage_R84_complete` 또는 `stage_R84_stalled`, `run_summary.json`을 기록합니다. `manifest.json`에 코드 content hash, 시작 모델 hash, bank hash, 설정과 확률을 저장합니다. `best_balanced`는 기존 synthetic+natural 기준을 유지하며 NC validation/test는 선택이나 진급에 쓰지 않습니다.

주기 NC 평가는 validation만 사용합니다. 종료 시 최종 `latest`에 대해 test를 한 번 평가하고 `final_held_out.json`에 네 지표를 저장합니다. NC 평가는 같은 checkpoint 반복을 독립 표본으로 세지 않고 최대 1000개의 고유 test deck만 평가합니다. Bank 규모에 따라 실제 평가 수는 더 적을 수 있습니다.

Natural은 보통 1000판, 추가 step 1M·2M 경계를 지난 첫 평가와 stage 완료/예산 종료에서 10000판입니다. Inherited global step이 아닌 이번 실험의 추가 step이 기준입니다. Natural 평가 permutation과 bank가 겹치면 재추출하지 않고 실패합니다.

## 6. 저장 모델의 공통 평가

기존 main 모델도 읽기만 하며 별도 결과 파일을 씁니다. NC 모델과 main 모델에 동일한 bank/seed/판수를 사용하세요.

```bash
.venv/bin/python -m src.evaluation.natural \
  --model models/restart_r64_from_line5/latest.zip \
  --bank runs/nc84_bank/bank.jsonl \
  --output runs/nc84_comparison/main.json --split test --games 10000
```

```bash
.venv/bin/python -m src.evaluation.natural \
  --model runs/nc84_experiment/latest.zip \
  --bank runs/nc84_bank/bank.jsonl \
  --output runs/nc84_comparison/nc.json --split test --games 10000
```

평가 대상 모델 파일은 실행 중 덮어써지지 않는 고정 checkpoint를 쓰세요. 공통 **추가 학습량**에서 synthetic R84, NC84 held-out, reverse R98, unconditional natural wins/games 및 Wilson 구간을 비교합니다. 동일 step의 과거 모델이 없다면 해당 시점의 matched comparison은 재구성할 수 없습니다. Test를 반복 확인하며 설정을 고르면 독립 held-out이 아니므로 최종 비교 protocol을 먼저 고정하세요.

NC는 teacher-selected 분포이며 unconditional natural과 다릅니다. 첫 1승만으로 효과를 단정하지 않습니다. 여러 seed 반복과 불확실성을 확인해야 합니다. 이 A/B는 synthetic replay 비율도 달라지므로 NC 자체 효과와 비율 재배분의 효과를 완전히 분리하지는 못합니다.

## 선택적 구현 검증

이번 prefix 전환 후에는 사용자 요청에 따라 테스트와 실제 pilot을 실행하지 않았습니다. 아래는 필요할 때 직접 실행하는 명령입니다.

모델 실행을 막고 신규 테스트와 기존 환경 회귀를 함께 실행합니다.

```bash
.venv/bin/python tests/run_without_models.py
```

이 runner는 실제 PPO 생성·로드·learn·predict를 차단하고 기존 실제 PPO 테스트 5개를 명시적으로 skip합니다. 수작업 승리 덱, 가짜 prior/learner 및 기존 환경을 사용하여 replay, sparse reward, hidden future, splits, 탐색 상한, 실제 reset 분포, A/B 제어 흐름, 평가 일정, journal 복구를 검사합니다. 이 fixture들은 성능 측정이나 실제 natural bank로 사용하지 않습니다.

실제 PPO 통합 검증을 포함한 기존 전체 테스트는 사용자가 필요할 때 실행할 수 있습니다.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

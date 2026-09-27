# 실행 가이드

## 설치와 학습

Linux / Python 3.12 기준입니다. 저장소 루트에서 실행합니다.

```bash
# 모델 없이 설치와 실행 확인
./run_train.sh --from-scratch --quick

# 기존 R20 모델로 R48까지 학습
./run_train.sh --resume /path/to/reverse_r20.zip --stop-at 48

# 새 모델로 정규 학습
./run_train.sh --from-scratch
```

스크립트는 `.venv`를 생성하고 매 실행 시 의존성을 설치합니다. PyTorch wheel 저장소는 `TORCH_INDEX_URL`로 지정할 수 있습니다. 기본 device는 CPU, thread 수는 1입니다. `--device cuda`도 지원하지만 GPU 실행은 검증하지 않았습니다.

`--from-scratch`를 생략하면 기본 모델 `models/1.0.0/reverse_r20.zip`을 불러옵니다. `--resume` 경로는 현재 경로 → 프로젝트 경로 → `models/` → `models/1.0.0/` 순으로 탐색하며 파일이 없으면 종료합니다. 모델 ZIP은 저장소에 포함되지 않습니다.

| 옵션 | 동작 |
| --- | --- |
| `--quick` | 첫 선택 단계에서 128 timestep, 환경 2개, 평가 난이도당 4판; bridge 없이 `smoke_complete`로 종료 |
| `--start-at 32` | R32에서 시작. 설정에 있는 단계만 지정 가능 |
| `--stop-at 48` | R48까지 진행. 진급 실패 시 앞 단계에서 중단될 수 있음 |
| `--output-dir models/my_run` | 새 실행의 출력 경로 지정 |
| `--config config.yaml` | 설정 파일 지정 |

## 평가

설치 이후에는 가상환경 Python으로 직접 실행할 수 있습니다. 아래 경로는 시작 로그의 `Output directory`에 나온 **실제 실행 폴더**로 바꿉니다.

```bash
.venv/bin/python evaluate.py --model models/my_run/latest.zip --source reverse --target-remaining 32 --games 1000
.venv/bin/python evaluate.py --model models/my_run/latest.zip --source natural --games 1000
.venv/bin/python random_baseline.py --source natural --games 1000
.venv/bin/python plot_history.py models/my_run/history.csv
```

자동 평가는 난이도당 1,000판, seed 100000~100999, deterministic policy입니다. 학습 환경과 독립된 평가 환경을 사용합니다. `evaluate.py`는 모든 판을 마친 뒤 출력합니다. 새로운 seed로 평가하려면 `--seed`를 지정합니다.

평가 대상은 replay 난이도, anchor, 이미 학습한 단계, 고정 balanced panel의 합집합이며 natural도 항상 평가합니다. `evaluation.all_seen_targets: false`로 과거 단계 추가 평가를 줄일 수 있습니다. 평가에 반복 사용한 seed는 개발용이며 최종 일반화 검증에는 별도 seed가 필요합니다.

## 난이도 분포와 진급

설정은 [config.yaml](../config.yaml)에 있습니다. 기본 분포는 현재 35%, 직전 세 단계 25/15/10%, R20 10%, R12 5%를 합산합니다. 중복 난이도의 비중은 더합니다. 예를 들어 R32는 R12/R20/R24/R28/R32에 5/20/15/25/35%를 배정합니다. R20의 이전 단계는 R16/R12/R8이며 R98의 직전 단계는 R96입니다.

단계별 분포를 직접 지정할 수도 있습니다.

```yaml
replay:
  recent_weights: [0.35, 0.25, 0.15, 0.10]
  anchors: {20: 0.10, 12: 0.05}
  distributions:
    32:
      targets: [12, 20, 24, 28, 32]
      probs: [0.10, 0.20, 0.15, 0.20, 0.35]
```

난이도는 2~98의 짝수, 확률 합은 1이어야 합니다. 현재·과거 난이도를 포함하고 미래 난이도는 제외합니다.

| 진급 조건 | 기본값 |
| --- | ---: |
| 현재 난이도 승률 | ≥60% |
| Replay 중 가장 가까운 이전 난이도 승률 | ≥60% |
| 평가된 나머지 과거 난이도의 승률 | 각각 ≥50% |
| 단계별 최소 학습량 | 250,000 timestep |
| 단계별 최대 예산 | 4,000,000 timestep |

학습 전 baseline을 평가하고 이후 약 250,000 timestep마다 평가합니다. Rollout 단위에 맞춰 평가 간격은 올림, 최대 예산은 내림합니다. 기본 16×1024 설정에서는 각각 262,144와 3,997,696 timestep입니다.

예산 소진 시 기본 `on_max_steps: stop`은 중단합니다. `advance`는 `stalled_advanced`로 기록하고 다음 단계로 이동하며 해당 단계의 완료 모델을 만들지 않습니다. 정체를 건너뛴 실행은 마지막에 성공해도 `completed_with_stalls`로 구분합니다. `learning_rate_stage_decay`를 1보다 작게 설정하면 단계마다 learning rate를 줄입니다.

## 출력 파일

기본 출력은 `models/`, quick 실행은 `models/smoke_<timestamp>/`입니다. 기본 출력에 `history.csv`나 `latest.zip`이 있으면 새 `run_<timestamp>/` 하위 폴더를 만듭니다. 직접 지정한 출력 폴더에 같은 파일이 있으면 오류로 종료합니다. 자동 폴더 생성은 이전 학습 재개를 뜻하지 않습니다.

| 파일 | 내용 |
| --- | --- |
| `latest.zip` | 마지막 평가 시점 모델 |
| `best_current_stage.zip`, `best_R*.zip`, `best_bridge_*.zip` | 현재·각 단계 최고 모델; bridge는 natural 승률 기준 |
| `best_balanced.zip` | 고정 panel 점수가 가장 높은 모델 |
| `stage_*_complete.zip`, `stage_*_stalled*.zip` | 진급 또는 예산 소진 시점 모델 |
| 각 ZIP과 동명의 JSON | 해당 모델의 평가 결과와 상태 |
| `history.csv`, `win_rate_by_target.png` | 단계·timestep·승률·상태와 그래프; 미평가 난이도는 빈칸 |
| `evaluations.jsonl` | 승수·판수·Wilson 95% 구간·진단값·진급 판정 |
| `episodes/<stage>/*.monitor.csv` | 완료 에피소드 보상·승패·시작 난이도·source |
| `config.resolved.yaml`, `run_summary.json` | 적용 설정과 단계별 종료 상태 |

Balanced 기본 panel은 R12/R20/R32/R48/R98와 natural입니다. `evaluation.balanced_targets`, `balanced_include_natural`, `balanced_metric`으로 변경합니다. Baseline도 best 후보이며 동점이면 먼저 저장한 모델을 유지합니다.

## 중단과 재개

학습을 중단한 뒤 보존할 ZIP과 동명 JSON을 복사하고 시작 단계를 명시합니다.

```bash
mkdir -p models/resume_points
cp models/my_run/latest.zip models/resume_points/paused.zip
cp models/my_run/latest.json models/resume_points/paused.json
./run_train.sh --resume models/resume_points/paused.zip --start-at 36 --output-dir models/new_run
```

`latest`는 마지막 평가 시점 모델이므로 중단 직전의 학습이 모두 저장되지는 않습니다. 평가 로그의 모든 행에 별도 모델이 남는 것도 아닙니다. JSON의 timestep을 확인하세요. 가중치·optimizer·global timestep은 복원하지만 단계 예산·best 기록·환경 RNG·진행 중 에피소드는 초기화합니다. 실행 중 config 수정은 이미 시작한 학습에 반영되지 않습니다.

## Natural bridge

R98 진급 후 natural 비중을 20→40→60→80→100%로 높이며 나머지는 reverse R98로 구성합니다. 각 단계 natural 진급 승률은 1/2/3/4/5%, 예산은 400만 timestep입니다. 이는 성능이 입증된 목표가 아닌 초기 설정입니다.

첫 네 단계는 reverse R98 승률 ≥50%도 요구합니다. 마지막은 natural 승률만 판정하고 reverse 승률은 계속 기록합니다. `--stop-at 98`은 reverse R98까지만 실행합니다.

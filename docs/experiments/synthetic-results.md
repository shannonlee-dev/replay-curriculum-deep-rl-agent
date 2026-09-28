# 실험 기록

> Historical experiment record. Active workflow: `./run_natural_train.sh`; see [natural replay curriculum](../natural-replay-curriculum.md).

## 평가 조건과 출처

2026-09-28 로컬 학습 기록에서 두 구간을 보관했습니다. 각 평가는 난이도당 1,000판, seed 100000~100999, deterministic policy를 사용합니다. 같은 seed를 진급과 모델 선택에도 사용했으므로 **개발 평가이며 독립 benchmark가 아닙니다.**

| 실행 | 보관한 평가 | 시작 모델 | 직전 난이도 진급 기준 |
| --- | --- | --- | ---: |
| A | 1~40번째, 누락 없음 | 기존 R20 | 70% |
| B | 1~15번째, 누락 없음 | A의 39번째 R36 | 60% |

B는 A의 마지막 평가에서 이어진 실행이 아닙니다. 그래프도 두 실행을 구분합니다. Global timestep은 시작 모델의 기존 학습량을 포함하며 저장소 기본 설정은 B와 같은 60% 기준입니다.

- [평가 CSV](../data/development_history.csv): 평가 시점 55개, 미평가 난이도는 빈칸.
- [A 설정](../data/a_config.yaml) / [B 설정](../data/b_config.yaml): 경로를 공개용 상대 경로로 정리한 실행 설정.
- [출처·SHA-256](../data/provenance.json): 원본 로그의 상대 경로, 포함 행 범위, 해당 구간 해시.
- [승률 그래프](../assets/development_win_rates.png): 실행별 주요 난이도 변화.

모델 가중치와 원본 전체 로그는 Git에서 제외합니다. CSV로 보고 수치를 확인할 수 있지만 과거 정책 실행이나 동일 학습 궤적 복원은 불가능합니다.

## 결과와 해석

| 시점 | R20 승률 | 현재 난이도 승률 |
| --- | ---: | ---: |
| A: R24 시작 | 62.2% | 43.9% |
| A: R24 진급 | 74.8% | 60.6% |
| A: R32 진급 | 89.0% | 61.7% |
| B: R56 진급 | 84.0% | 62.9% |

이 구간에서는 더 긴 reverse 난이도로 진행하면서 R20 능력을 유지하는 양상을 관측했습니다. Natural은 모든 평가에서 0/1,000승입니다. 실제 성공 확률이 정확히 0이라는 뜻은 아니며, 95% Wilson 구간 상한은 약 0.38%입니다.

Replay 없는 대조군, 여러 training seed, 별도 평가 seed가 없어 replay의 인과적 효과와 일반화는 확인하지 못했습니다. A와 B는 진급 기준도 달라 진급 속도를 직접 비교할 수 없습니다. **Natural 전이 성공 역시 아직 입증하지 않았습니다.**

## 구현 검증

문서 정리 시 재검증한 환경은 Python 3.12.3, PyTorch 2.14.0+cu130, Stable-Baselines3 / sb3-contrib 2.9.0, Gymnasium 1.3.0이며 CPU로 실행했습니다. 의존성은 최소 버전 범위로 지정되어 있어 완전히 고정된 재현 환경은 아닙니다. GPU 실행은 검증하지 않았습니다.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.synthetic.trainer --from-scratch --quick
```

테스트 17개는 sparse reward, 짝수 난이도별 완승 수순, 덱 순서 비노출, 분포·설정 검증, 독립 reset, Wilson 구간, 진급·정체·bridge, 출력 덮어쓰기 방지를 검사합니다. 이 중 원본 환경·R20 모델과의 비교 2개는 로컬 원본 파일이 없으면 건너뜁니다.

실제 짧은 PPO 학습과 모델 저장·로드, CSV·JSONL·PNG 출력을 포함합니다. Bridge 테스트는 제어 흐름 검증을 위해 승률 조건을 0으로 낮추므로 전이 성능의 근거가 아닙니다.

## 후속 검증

1. 동일 예산·시작 모델·seed에서 replay 없는 대조군과 비교.
2. 여러 training seed의 평균·분산과 별도 evaluation seed 승률 보고.
3. Reverse R98 → natural bridge의 실제 전이 성능 측정.
4. `gamma`, replay 비중, 진급 기준을 각각 바꿔 영향 확인.

현재 구현 범위는 학습·평가 시스템입니다. 모델 관전 HTML과 행동·전략 설명 기능은 포함하지 않습니다.

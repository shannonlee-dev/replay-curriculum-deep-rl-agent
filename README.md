# The Game — Replay Curriculum Deep RL Agent

Synthetic:

```bash
./run_train.sh
```

Natural replay:

```bash
./run_natural_train.sh
```

Resume:

```bash
./run_natural_train.sh --resume runs/<run_dir>
```

Linux / Python 3.12. 두 스크립트는 저장소 루트에서 실행하며 필요한 가상환경과 의존성을 준비합니다.
Natural 기본 입력은 `models/restart_r64_from_line5/stage_R80_complete.zip`와
`runs/natural_campaign/bank.jsonl`입니다. 모델과 bank는 로컬에 준비되어 있어야 합니다.

## 현재 연구 경로

```text
run_natural_train.sh
  → src/natural/trainer.py
    → bank.py             replay·split·provenance 검증
    → curriculum.py       stage·frontier·promotion
    → difficulty_scan.py  held-out NC baseline
    → src/evaluation/natural.py

R80 pretrained → NC12 → NC16 → NC20 → … → NC84
  → unconditional natural full-game evaluation
```

설정은 `configs/natural_replay.yaml`, synthetic 설정은 `configs/synthetic.yaml`입니다.
NC4/NC8은 scan·rehearsal anchor로 유지합니다. NC bank는 heuristic WIN으로 조건화된 분포이며,
최종 성능은 별도의 unconditional natural full-game terminal WIN으로 평가합니다.
Terminal WIN=1, LOSS=0, 중간 보상=0이며 미래 deck 순서는 observation에 노출하지 않습니다.

| 위치 | 역할 |
| --- | --- |
| `src/env.py` | 공통 게임·관측·행동 계약 |
| `src/natural/` | 현재 natural replay curriculum |
| `src/synthetic/` | synthetic replay curriculum |
| `src/evaluation/` | natural / synthetic / distribution audit |
| `src/common/` | checkpoint·resume·provenance·source usage·terminal UI |
| `archive/` | direct NC84 mix·privileged teacher·handoff v3, 원본 소스 스냅샷 |
| `docs/experiments/` | 과거 연구 기록 |

Resume은 저장된 설정으로 model·optimizer·환경/RNG·stage·steps·source counters·평가 위치를 복원합니다.
Ctrl+C는 완료된 작업 경계에서 안전하게 저장합니다. 기존 run 파일과 출력 schema는 유지하고,
model/config/bank/provenance 불일치는 거부합니다. 구조 이동 전 run의 코드 검증은
[정확한 소스 해시 대응](docs/project-layout.md)을 사용합니다.
TTY는 Rich dashboard, non-TTY는 평가 시점의 간결한 로그를 출력합니다.

## Advanced

학습 없는 검사와 scan:

```bash
./run_natural_train.sh --check
./run_natural_train.sh --scan-only
```

연구 설정 및 모듈 직접 실행:

```bash
./run_natural_train.sh --config configs/natural_replay.yaml --stop-at NC16
./run_natural_train.sh --set natural_curriculum.promotion.next_stage_min_wins=25
.venv/bin/python -m src.natural.trainer --help
.venv/bin/python -m src.evaluation.natural --help
.venv/bin/python -m src.evaluation.synthetic --help
.venv/bin/python -m src.evaluation.distribution_audit --help
.venv/bin/python -m src.evaluation.heuristic --help
```

검증:

```bash
.venv/bin/python tests/run_without_models.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/smoke_natural_frontier.py
.venv/bin/python tests/smoke_natural_continuation.py
.venv/bin/python tests/smoke_layout_resume.py
```

Smoke는 작은 PPO update만 실행합니다. 원본 모델·소스 ZIP 비교는 해당 로컬 파일이 없으면 skip합니다.

- [Natural replay curriculum](docs/natural-replay-curriculum.md)
- [Synthetic curriculum](docs/synthetic-curriculum.md)
- [파일 이동표와 resume 호환 정책](docs/project-layout.md)
- [게임 설계](docs/design.md)
- [과거 실험 재현](archive/README.md)

Research internals are configurable, normal workflow is one command.

# Package layout verification — 2026-09-28

검증 대상은 기존 작업 내용이 있던 `develop`의 현재 working tree입니다. 기존 변경을 되돌리지 않았으며,
실제 모델·bank·run 디렉터리는 수정하지 않았습니다. 검증 출력은 `/tmp`의 별도 디렉터리에 생성했습니다.

| 검증 | 결과 |
| --- | --- |
| 전체 unittest discovery | 107/107 PASS, skip 없음 |
| Model-free runner | 102 PASS, 실제 PPO 테스트 5개 의도적 skip |
| Synthetic 학습·checkpoint 테스트 | 9/9 PASS |
| `smoke_natural_frontier.py` | PASS: 실제 PPO 64-transition update, optimizer/weights, scan, held-out, source usage |
| `smoke_natural_continuation.py` | PASS: 중단/재개와 연속 실행의 weights·optimizer·RNG·usage 일치 |
| `smoke_layout_resume.py` | PASS: 원본 소스 snapshot에서 생성한 run을 새 shell로 preflight/resume; weights·optimizer·RNG·usage·evaluation·schema 일치 |
| 실제 기본 R80 / bank preflight | PASS: 5,000 unique winning decks, replay/provenance 검증, 모델 비실행 |
| 실제 기본 모델/bank difficulty scan | PASS: 12개 stage × 2개 validation decks, NC12 선택 유지, training_started=false |
| Package import / archive 경계 | PASS: 모든 모듈 import, active 소스와 R80 검사에서 archive import 없음 |
| Config / path / CLI | PASS: 모든 config 원본 바이트 보존, 다른 cwd의 두 shell launcher, 모듈 CLI |
| Game / bank contract | PASS: 원본과 import를 제외한 AST 동일; observation·mask·WIN 보상·bank replay 회귀 테스트 통과 |
| Hash mismatch 거부 | PASS: model/config/bank/manifest/scan/log/continuation/code 및 원본 snapshot 변조 |
| Non-TTY UI | PASS: escape sequence 없는 평가 로그, training update 중복 출력 없음 |
| 정적 검사 | PASS: compileall, bash -n, git diff --check, 로컬 문서 링크 |

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/run_without_models.py
.venv/bin/python tests/smoke_natural_frontier.py
.venv/bin/python tests/smoke_natural_continuation.py
.venv/bin/python tests/smoke_layout_resume.py
./run_natural_train.sh --check --output-dir /tmp/<verification>/preflight
./run_natural_train.sh --scan-only \
  --set natural_curriculum.evaluation.nc_games=2 \
  --output-dir /tmp/<verification>/scan
```

장시간 PPO 학습은 실행하지 않았습니다. 실제 PPO 실행은 테스트의 작은 update로 제한했습니다.

## Existing local legacy run

`runs/natural_reverse_nc12_to_nc84`는 작업 시작 시점부터 소스 5개의 hash가 달랐고,
`continuation_schema`와 `continuation` 상태가 없었습니다. 정확한 in-place resume에 필요한
환경/RNG·평가 상태가 없어 기존 코드에서도 재개할 수 없는 run입니다. 원본 파일을 수정하거나
검증을 우회하지 않았습니다. 현재 continuation schema를 저장한 이전 구조의 run은
`smoke_layout_resume.py`에서 실제 원본 코드로 생성하고 새 구조에서 정확한 재개를 검증했습니다.

[파일 이동표 및 호환 정책](project-layout.md)

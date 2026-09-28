# Historical experiments

현재 학습 경로는 `../run_natural_train.sh` → `src.natural.trainer`입니다.
아래 코드는 active trainer에서 import되지 않습니다.

| 영역 | 과거 실험 | 연구용 CLI (저장소 루트에서 실행) |
| --- | --- | --- |
| `nc84_direct_mix/` | NC84 direct mixture와 이전 campaign | `.venv/bin/python -m archive.nc84_direct_mix.trainer --help` |
| `privileged_teacher/` | privileged suffix teacher·prefix generator | `.venv/bin/python -m archive.privileged_teacher.build_bank --help` |
| `handoff_v3/` | handoff v3와 Dietrich teacher | `.venv/bin/python -m archive.handoff_v3.build_bank --help` |

이동된 모듈은 현재 공통 환경·bank·evaluation을 사용하며, 과거 config는 `nc84_direct_mix/config.yaml`에 있습니다.
실험 결과와 기존 bank/run 파일은 수정하지 않았습니다. 과거 hash에 고정된 generator resume나 audit에는
아래 원본 스냅샷을 사용합니다. 현재 패키지의 소스 해시를 과거 해시인 것처럼 기록하지 않습니다.

## Exact pre-package snapshot

`pre_package/source.zip`은 이동 직전 루트 Python 파일·shell·config의 원본입니다.
`pre_package/migration.json`에는 원본 파일 해시와 현재 패키지로의 이동표 및 승인된 대응 해시가 있습니다.
스냅샷은 active 코드로 import하지 않고 기존 run provenance 검증 데이터로만 읽습니다.

정확한 과거 실행이 필요하면 스냅샷을 별도 디렉터리에 풀고, 그 디렉터리에 원래 `models/`와 `runs/`
입력을 준비한 뒤 당시 CLI를 사용합니다. 기록에 절대 경로가 있으면 당시 경로도 보존해야 합니다.
과거 소스와 현재 소스를 섞어서 provenance 검사를 우회하지 마십시오.

과거 보고서는 [docs/experiments](../docs/experiments/)에 보존합니다.

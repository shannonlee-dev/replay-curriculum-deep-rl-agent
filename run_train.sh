#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
# Set TORCH_INDEX_URL to select a CPU/CUDA wheel source when creating the venv.
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  python -m pip install torch --index-url "$TORCH_INDEX_URL"
fi
python -m pip install -r requirements.txt
# The CLI resolves --resume and verifies the checkpoint before creating a run.
exec python train_replay_curriculum.py "$@"

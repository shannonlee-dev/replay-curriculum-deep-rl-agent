#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
# Set TORCH_INDEX_URL to select a CPU/CUDA wheel source when creating the venv.
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  .venv/bin/python -m pip install torch --index-url "$TORCH_INDEX_URL"
fi
if ! .venv/bin/python -c 'import numpy, gymnasium, torch, stable_baselines3, sb3_contrib, matplotlib, yaml' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements.txt
fi
# The CLI resolves --resume and verifies the checkpoint before creating a run.
exec .venv/bin/python -m src.synthetic.trainer "$@"

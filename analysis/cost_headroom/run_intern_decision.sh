#!/usr/bin/env bash
set -euo pipefail
INTERN_OUT=${1:-/mnt/llmd/results/exps/aristides/reason/intern_decision_20261001}
mkdir -p "${INTERN_OUT}/logs"
exec > >(tee -a "${INTERN_OUT}/logs/pilot.log") 2>&1
printf 'Starting pinned Intern-Decision-4B success pilot\n'
# Public weights: avoid expired ambient HF OAuth credentials.
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
INTERN_UV=/home/toolkit/.local/bin/uv
INTERN_ENV=${TMPDIR:-/tmp}/intern_decision_py312
if ! command -v python3.12 >/dev/null; then
  "${INTERN_UV}" python install 3.12
fi
"${INTERN_UV}" venv --python 3.12 "${INTERN_ENV}"
"${INTERN_UV}" pip install --python "${INTERN_ENV}/bin/python" torch==2.9.1 torchvision==0.24.1 transformers==5.14.1 'Pillow>=10.0.0' numpy aiohttp
"${INTERN_ENV}/bin/python" -u analysis/cost_headroom/intern_decision_pilot.py --out "${INTERN_OUT}"

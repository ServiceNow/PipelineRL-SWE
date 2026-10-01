#!/usr/bin/env bash
set -euo pipefail
INTERN_FT_POOL=$1
INTERN_FT_OUT=$2
INTERN_FT_LABEL=${INTERN_FT_POOL,,}
INTERN_FT_LABEL=${INTERN_FT_LABEL//-/_}
mkdir -p "${INTERN_FT_OUT}/logs"
exec > >(tee -a "${INTERN_FT_OUT}/logs/${INTERN_FT_LABEL}.log") 2>&1
printf 'Starting Intern LoRA success training: %s\n' "${INTERN_FT_POOL}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
INTERN_FT_UV=/home/toolkit/.local/bin/uv
INTERN_FT_ENV=${TMPDIR:-/tmp}/intern_ft_${INTERN_FT_LABEL}_py312
if ! command -v python3.12 >/dev/null; then "${INTERN_FT_UV}" python install 3.12; fi
"${INTERN_FT_UV}" venv --python 3.12 "${INTERN_FT_ENV}"
"${INTERN_FT_UV}" pip install --python "${INTERN_FT_ENV}/bin/python" torch==2.9.1 torchvision==0.24.1 transformers==5.14.1 peft==0.21.2 'Pillow>=10.0.0' numpy aiohttp scikit-learn
"${INTERN_FT_ENV}/bin/python" -u analysis/cost_headroom/intern_decision_finetune.py --pool "${INTERN_FT_POOL}" --out "${INTERN_FT_OUT}"

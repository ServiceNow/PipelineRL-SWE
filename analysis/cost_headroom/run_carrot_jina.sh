#!/usr/bin/env bash
set -euo pipefail
CARROT_JINA_POOL=$1
CARROT_JINA_OUT=$2
CARROT_JINA_LABEL=${CARROT_JINA_POOL,,}
CARROT_JINA_LABEL=${CARROT_JINA_LABEL//-/_}
mkdir -p "${CARROT_JINA_OUT}/logs"
exec > >(tee -a "${CARROT_JINA_OUT}/logs/${CARROT_JINA_LABEL}.log") 2>&1
printf 'Starting Jina CARROT-style comparison: %s\n' "${CARROT_JINA_POOL}"
python -u analysis/cost_headroom/carrot_jina.py --pool "${CARROT_JINA_POOL}" --out "${CARROT_JINA_OUT}"

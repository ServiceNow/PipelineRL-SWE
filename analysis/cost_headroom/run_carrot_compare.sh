#!/usr/bin/env bash
# Persist startup diagnostics as well as experiment output.
set -euo pipefail
CARROT_POOL=$1
CARROT_OUT=$2
CARROT_LOG_LABEL=${CARROT_POOL,,}
CARROT_LOG_LABEL=${CARROT_LOG_LABEL//-/_}
mkdir -p "${CARROT_OUT}/logs"
exec > >(tee -a "${CARROT_OUT}/logs/${CARROT_LOG_LABEL}.log") 2>&1
CARROT_PYTHON=${CARROT_PYTHON:-python}
CARROT_DEPS=${TMPDIR:-/tmp}/carrot_python_deps
printf 'Starting CARROT pool=%s\n' "${CARROT_POOL}"
"${CARROT_PYTHON}" --version
if [[ ! -f ${CARROT_DEPS}/sentence_transformers/__init__.py ]]; then
  "${CARROT_PYTHON}" -m pip install --no-deps --target "${CARROT_DEPS}" sentence-transformers==3.4.1
fi
export PYTHONPATH="${CARROT_DEPS}${PYTHONPATH:+:${PYTHONPATH}}"
"${CARROT_PYTHON}" -u analysis/cost_headroom/carrot_compare.py --pool "${CARROT_POOL}" --out "${CARROT_OUT}" --bootstrap 1000

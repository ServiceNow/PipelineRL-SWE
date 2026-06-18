#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SEED=${SEED:-43}
EVAL_N=${EVAL_N:-300}
ROLLOUTS=${ROLLOUTS:-3}
OUTPUT_BASE=${OUTPUT_BASE:-/mnt/llmd/results/exps/aristides/reason}
RUN_ROOT=${RUN_ROOT:-${OUTPUT_BASE}/offline_router_swe_smith_multi_rollout_trace_collect_eval300_no_overlap_${TIMESTAMP}}
ID_ROOT=${ID_ROOT:-${RUN_ROOT}/ids}

EXPANDED_ID_ROOT=${EXPANDED_ID_ROOT:-/mnt/llmd/results/exps/aristides/reason/swe_smith_real_label_trace_ids_expanded_1781073985}
EVAL_POOL_IDS=${EVAL_POOL_IDS:-${EXPANDED_ID_ROOT}/swe_smith_eval_1000_ids.txt}
EXCLUDE_IDS=${EXCLUDE_IDS:-${REPO_ROOT}/router_analysis/aws_eval_packages/swe_smith_multirollout_eval150_1781382734/ids/eval150.txt}
EVAL_IDS=${EVAL_IDS:-${ID_ROOT}/swe_smith_eval_${EVAL_N}_from_1000_excluding_eval150_seed${SEED}_ids.txt}
PYTHON_BIN=${PYTHON_BIN:-/home/toolkit/.conda/envs/pipeline-rl/bin/python}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable PYTHON_BIN=${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -s "${EVAL_POOL_IDS}" ]]; then
  echo "Missing EVAL_POOL_IDS=${EVAL_POOL_IDS}" >&2
  exit 1
fi
if [[ ! -s "${EXCLUDE_IDS}" ]]; then
  echo "Missing EXCLUDE_IDS=${EXCLUDE_IDS}" >&2
  exit 1
fi

mkdir -p "${ID_ROOT}"
"${PYTHON_BIN}" "${REPO_ROOT}/pipelinerl/swe/scripts/offline_router/sample_instance_ids_from_pool.py"   --pool-path "${EVAL_POOL_IDS}"   --exclude-path "${EXCLUDE_IDS}"   --output-path "${EVAL_IDS}"   --n "${EVAL_N}"   --seed "${SEED}"   --sort-output

common=$(comm -12 <(sort "${EVAL_IDS}") <(sort "${EXCLUDE_IDS}") | wc -l)
if [[ "${common}" != "0" ]]; then
  echo "Sample overlap check failed: ${common} IDs overlap with ${EXCLUDE_IDS}" >&2
  exit 1
fi

export TIMESTAMP
export SEED
export EVAL_N
export ROLLOUTS
export RUN_TRAIN_TRACES=0
export RUN_EVAL_TRACES=1
export RUN_ROOT
export ID_ROOT
export EVAL_POOL_IDS
export EVAL_IDS

bash "${SCRIPT_DIR}/launch_swe_smith_multi_rollout_trace_collect_parallel.sh"

echo "Submitted eval${EVAL_N} non-overlap multi-rollout collection."
echo "RUN_ROOT=${RUN_ROOT}"
echo "EVAL_IDS=${EVAL_IDS}"

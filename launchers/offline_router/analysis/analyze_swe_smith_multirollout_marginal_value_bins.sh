#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-/home/toolkit/.conda/envs/pipeline-rl/bin/python}
SCORE_JSONL=${SCORE_JSONL:-/mnt/llmd/results/exps/aristides/reason/score_swe_smith_multirollout_eval150_proxy_init_random128_verifier_1781857392/scores/eval_verifier_scores.jsonl}
REPORT_ROOT=${REPORT_ROOT:-router_analysis/uploaded_eval_full_20260617/swe_smith_multirollout_eval150_1781382734/logs/run_evaluation}
OUTPUT_DIR=${OUTPUT_DIR:-router_analysis/swe_smith_multirollout_marginal_value_bins_proxyinit_eval150}
BINS=${BINS:-6}
LAMBDAS=${LAMBDAS:-0,1,2,5,10,20}

"${PYTHON_BIN}" pipelinerl/swe/scripts/offline_router/analyze_multirollout_marginal_value_bins.py \
  --score-jsonl "${SCORE_JSONL}" \
  --report-root "${REPORT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --bins "${BINS}" \
  --lambdas "${LAMBDAS}"

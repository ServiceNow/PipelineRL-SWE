#!/usr/bin/env bash
set -euo pipefail

SCORE_JSONL=${SCORE_JSONL:?Set SCORE_JSONL to eval_verifier_scores.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR for calibration diagnostic output}
REPORT_ROOT=${REPORT_ROOT:-router_analysis/uploaded_eval_full_20260617/swe_smith_multirollout_eval150_1781382734/logs/run_evaluation}
N_SPLITS=${N_SPLITS:-50}
MAX_THRESHOLDS=${MAX_THRESHOLDS:-5}
STEPS=${STEPS:-1000}
PYTHON=${PYTHON:-/home/toolkit/.conda/envs/pipeline-rl/bin/python}

"${PYTHON}" \
  pipelinerl/swe/scripts/offline_router/analyze_verifier_calibration_diagnostics.py \
  --scores-jsonl "${SCORE_JSONL}" \
  --report-root "${REPORT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --n-splits "${N_SPLITS}" \
  --max-thresholds "${MAX_THRESHOLDS}" \
  --steps "${STEPS}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}

JOB_NAME=${JOB_NAME:-analyze_swe_smith_multirollout_controller_ceiling}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
SCORE_JSONL=${SCORE_JSONL:-/mnt/llmd/results/exps/aristides/reason/score_swe_smith_multirollout_eval150_proxy_init_random128_verifier_1781857392/scores/eval_verifier_scores.jsonl}
REPORT_ROOT=${REPORT_ROOT:-router_analysis/uploaded_eval_full_20260617/swe_smith_multirollout_eval150_1781382734/logs/run_evaluation}
ROUTE_COST_WEIGHTS=${ROUTE_COST_WEIGHTS:-2.78e-7,1.299e-6,4.64e-6,1.113e-5}
LAMBDAS=${LAMBDAS:-0,1,2,5,10,20}
MAX_THRESHOLDS=${MAX_THRESHOLDS:-5}
N_SPLITS=${N_SPLITS:-50}
SEED=${SEED:-31}

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  GPU=0 \
  NPROC=1 \
  CPU=${CPU:-4} \
  CPU_MEM=${CPU_MEM:-32} \
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${OUTPUT_DIR}; set -o pipefail; \
    python pipelinerl/swe/scripts/offline_router/analyze_multirollout_controller_pathways.py \
      --analysis ceiling \
      --scores-jsonl ${SCORE_JSONL} \
      --report-root ${REPORT_ROOT} \
      --output-dir ${OUTPUT_DIR} \
      --route-cost-weights ${ROUTE_COST_WEIGHTS} \
      --lambdas ${LAMBDAS} \
      --max-thresholds ${MAX_THRESHOLDS} \
      --n-splits ${N_SPLITS} \
      --seed ${SEED} \
      2>&1 | tee -a ${OUTPUT_DIR}/launch.out"

echo "Submitted: ${OUTPUT_DIR}"

#!/usr/bin/env bash
# Launch Daytona eval jobs for all model JSONL files produced by launch_collect.sh.
# Submits one EAI job per model; they run in parallel.
#
# Daytona writes to logs/run_evaluation/<run_id>/report.json (relative to REPO_ROOT).
# This script uses deterministic run_ids (or_sweep_<slug>) so the analysis script
# can find them without a manifest file.
#
# Usage:
#   PREDICTIONS_DIR=/mnt/.../openrouter_sweep_collect_XYZ bash launch_daytona.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=$(date +%s)

# Directory containing per-model *.jsonl prediction files from launch_collect.sh
PREDICTIONS_DIR=${PREDICTIONS_DIR:?Need PREDICTIONS_DIR set to the collect output dir}

# SWE-smith HuggingFace dataset name (used by Daytona for test sandboxes)
HF_DATASET=${HF_DATASET:-SWE-bench/SWE-smith-py}
CONCURRENCY=${CONCURRENCY:-50}

# Daytona reports land at: logs/run_evaluation/<run_id>/ (relative to REPO_ROOT)
# We use deterministic run_ids so the analysis script can find them.
RUN_ID_PREFIX="or_sweep"

for jsonl_file in "${PREDICTIONS_DIR}"/*.jsonl; do
  [[ -f "${jsonl_file}" ]] || continue
  slug=$(basename "${jsonl_file}" .jsonl)
  run_id="${RUN_ID_PREFIX}_${slug}"
  job_name="or_daytona_${slug:0:35}_${TIMESTAMP}"

  echo "[submit] slug=${slug}  run_id=${run_id}"

  make -C "${REPO_ROOT}" job \
    JOB_NAME="${job_name}" \
    ENV=pipeline-rl \
    CONDA_EXE=/opt/conda/bin/conda \
    SNAPSHOT=1 \
    NPROC=1 \
    GPU=0 \
    GPU_MEM=0 \
    CPU=8 \
    CPU_MEM=32 \
    COMMAND="cd ${REPO_ROOT}; \
      python pipelinerl/swe/scripts/offline_router/run_swesmith_eval_daytona.py \
        --predictions_path ${jsonl_file} \
        --run_id ${run_id} \
        --hf_dataset ${HF_DATASET} \
        --concurrency ${CONCURRENCY} \
      2>&1 | tee ${PREDICTIONS_DIR}/${slug}_daytona.log"

  sleep 2  # Stagger submissions slightly
done

echo ""
echo "Daytona jobs submitted. Reports land at:"
echo "  ${REPO_ROOT}/logs/run_evaluation/or_sweep_<slug>/report.json"
echo ""
echo "Once all jobs finish, run the analysis:"
echo "  python pipelinerl/swe/scripts/openrouter_sweep/analyze_openrouter_sweep.py \\"
echo "    --daytona-log-dir ${REPO_ROOT}/logs/run_evaluation \\"
echo "    --run-id-prefix or_sweep \\"
echo "    --existing-parquet-dir <eval parquet dir> \\"
echo "    --output-dir ${PREDICTIONS_DIR}/analysis"

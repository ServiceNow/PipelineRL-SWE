#!/usr/bin/env bash
# Collect REAL SWE-bench Verified labels for gpt-oss-120b (and optionally other
# routes) by running existing model patches through Daytona's official test harness.
#
# The 5-route proxy collection already has patches for all models; this script
# re-evaluates them with proper Docker execution instead of the proxy reward.
#
# Steps:
#   1. Extract patches from parquet → per-model predictions JSONL  (local, fast)
#   2. Submit Daytona eval job for each model
#
# Required env vars:
#   DAYTONA_API_KEY  (or in .env)
#
# Optional:
#   SOURCE_PARQUET_DIR  -- 5-route verified parquet eval dir (default below)
#   VERIFIED_DATASET_PATH -- local SWE-bench Verified HF dataset (default below)
#   ROUTES              -- comma-separated route indices to eval (default: 1,2,3)
#   CONCURRENCY         -- Daytona sandbox concurrency (default: 32)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=/home/toolkit/.conda/envs/pipeline-rl/bin/python3

TIMESTAMP=$(date +%s)
JOB_NAME=verified_real_label_eval_${TIMESTAMP}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

SOURCE_PARQUET_DIR=${SOURCE_PARQUET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_bench_train_all_16k_verified_eval_collect_5route_4b_scout_oss20_qwen30_oss120_gemini/collect/eval}
VERIFIED_DATASET_PATH=${VERIFIED_DATASET_PATH:-/mnt/llmd/data/swebench_verified/all_16k/ds}
ROUTES=${ROUTES:-1,2,3}  # oss-20b, qwen30, oss-120b (skip 4b-scout and gemini)
CONCURRENCY=${CONCURRENCY:-32}

# Load DAYTONA_API_KEY from .env if not already set
if [[ -z "${DAYTONA_API_KEY:-}" ]]; then
  ENV_FILE="${REPO_ROOT}/.env"
  if [[ -f "${ENV_FILE}" ]]; then
    DAYTONA_API_KEY=$(grep -E '^DAYTONA_API_KEY=' "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"'"'")
  fi
fi
: "${DAYTONA_API_KEY:?Need DAYTONA_API_KEY — set it in .env or the environment}"

mkdir -p "${OUTPUT_DIR}"
PREDS_DIR="${OUTPUT_DIR}/predictions"

# --- Step 1: extract patches from parquet (local, no GPU needed) ---
echo "=== Extracting model patches from parquet (routes: ${ROUTES}) ==="
"${PYTHON}" "${REPO_ROOT}/pipelinerl/swe/scripts/offline_router/extract_verified_route_predictions.py" \
  --parquet-dir           "${SOURCE_PARQUET_DIR}" \
  --verified-dataset-path "${VERIFIED_DATASET_PATH}" \
  --output-dir            "${PREDS_DIR}" \
  --routes                "${ROUTES}"
echo ""

# --- Step 2: build runner script ---
RUNNER="${OUTPUT_DIR}/run_daytona_eval.sh"
cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
# Load DAYTONA_API_KEY from .env (avoids exposing it in the job command line)
if [[ -z "\${DAYTONA_API_KEY:-}" ]]; then
  for _env_file in /home/toolkit/PipelineRL-SWE/.env /home/toolkit/.env; do
    if [[ -f "\${_env_file}" ]]; then
      DAYTONA_API_KEY=\$(grep -E '^DAYTONA_API_KEY=' "\${_env_file}" | head -1 | cut -d'=' -f2- | tr -d '"' | tr -d "'")
      break
    fi
  done
fi
export DAYTONA_API_KEY
: "\${DAYTONA_API_KEY:?Need DAYTONA_API_KEY in .env}"
cd "${REPO_ROOT}"
SCRIPT_EOF

for pred_file in "${PREDS_DIR}"/predictions_*.jsonl; do
  run_id=$(basename "${pred_file}" .jsonl)_${TIMESTAMP}
  cat >> "${RUNNER}" << SCRIPT_EOF

echo "=== Evaluating ${pred_file} ==="
python ${REPO_ROOT}/pipelinerl/swe/scripts/offline_router/run_swebench_eval_daytona.py \
  --predictions-path ${pred_file} \
  --run-id           ${run_id} \
  --concurrency      ${CONCURRENCY} \
  2>&1 | tee ${OUTPUT_DIR}/${run_id}.log
SCRIPT_EOF
done

chmod +x "${RUNNER}"

# --- Step 3: submit EAI job ---
echo "=== Submitting Daytona eval job: ${JOB_NAME} ==="
make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=1 \
  GPU=0 \
  GPU_MEM=0 \
  CPU=8 \
  CPU_MEM=32 \
  COMMAND="bash ${RUNNER}"

echo ""
echo "Summary JSONLs: ${OUTPUT_DIR}/*.results.jsonl"
echo "Logs: ${OUTPUT_DIR}/"

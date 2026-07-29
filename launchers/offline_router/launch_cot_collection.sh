#!/usr/bin/env bash
# Collect CoT (thinking) trajectories from Qwen3-4B-Thinking-2507 via local vLLM.
#
# Both train (1146) and eval (286) parquet instances come from ds_train —
# ds_test has zero overlap with our labeled parquet.
#
# Outputs per split in OUTPUT_DIR:
#   predictions_{train,eval}.jsonl  -- Daytona-compatible (optional scout eval)
#   trajectories_{train,eval}.jsonl -- input for train_autoregressive_verifier.py
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=/home/toolkit/.conda/envs/pipeline-rl/bin/python3

TIMESTAMP=$(date +%s)
JOB_NAME=cot_trajectories_${TIMESTAMP}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}
MODEL_PATH=${MODEL_PATH:-${MODEL_NAME}}
DATASET_PATH=${DATASET_PATH:-/mnt/llmd/data/swe_smith_bugged_context/ds_train}
REAL_LABEL_DATASET_DIR=${REAL_LABEL_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}

CONCURRENCY=${CONCURRENCY:-16}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
NPROC=1
TENSOR_PARALLEL=1

# --- Step 1: export instance ID lists from parquet (local, fast) ---
mkdir -p "${OUTPUT_DIR}"
TRAIN_IDS_FILE="${OUTPUT_DIR}/train_instance_ids.json"
EVAL_IDS_FILE="${OUTPUT_DIR}/eval_instance_ids.json"

echo "=== Exporting instance IDs from parquet ==="
"${PYTHON}" "${REPO_ROOT}/pipelinerl/swe/scripts/offline_router/export_parquet_ids.py" \
  --parquet-dir "${REAL_LABEL_DATASET_DIR}/train" \
  --output-file "${TRAIN_IDS_FILE}"
"${PYTHON}" "${REPO_ROOT}/pipelinerl/swe/scripts/offline_router/export_parquet_ids.py" \
  --parquet-dir "${REAL_LABEL_DATASET_DIR}/eval" \
  --output-file "${EVAL_IDS_FILE}"
echo ""

# --- Step 2: write runner script (avoids make $ expansion issues) ---
RUNNER="${OUTPUT_DIR}/run_cot_collection.sh"
cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"

echo '[vllm] Starting vLLM server...'
python -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_PATH} \
  --port ${VLLM_PORT} \
  --tensor-parallel-size ${TENSOR_PARALLEL} \
  --gpu-memory-utilization ${VLLM_GPU_UTIL} \
  --max-model-len ${MAX_MODEL_LEN} \
  --trust-remote-code \
  --served-model-name ${MODEL_NAME} \
  > ${OUTPUT_DIR}/vllm_server.log 2>&1 &
VLLM_PID=\$!

echo "[vllm] Waiting for server to be ready (PID=\${VLLM_PID})..."
for i in \$(seq 1 120); do
  if curl -sf http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
    echo '[vllm] Server ready.'; break
  fi
  sleep 5
done

echo '[collect] Starting trajectory collection...'
python pipelinerl/swe/scripts/offline_router/collect_cot_trajectories.py \
  --vllm-base-url http://localhost:${VLLM_PORT} \
  --model-name ${MODEL_NAME} \
  --train-dataset-path ${DATASET_PATH} \
  --eval-dataset-path ${DATASET_PATH} \
  --train-instance-ids-file ${TRAIN_IDS_FILE} \
  --eval-instance-ids-file ${EVAL_IDS_FILE} \
  --output-dir ${OUTPUT_DIR} \
  --concurrency ${CONCURRENCY} \
  2>&1 | tee ${OUTPUT_DIR}/collect.log

echo '[vllm] Shutting down vLLM server...'
kill \${VLLM_PID} 2>/dev/null || true
echo "[done] Trajectories written to ${OUTPUT_DIR}"
SCRIPT_EOF
chmod +x "${RUNNER}"

# --- Step 3: submit EAI job ---
echo "=== Submitting CoT collection job: ${JOB_NAME} ==="
make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${NPROC} \
  GPU=1 \
  GPU_MEM=80 \
  CPU=8 \
  CPU_MEM=64 \
  COMMAND="bash ${RUNNER}"

echo ""
echo "Output dir:         ${OUTPUT_DIR}"
echo "Train trajectories: ${OUTPUT_DIR}/trajectories_train.jsonl  (1146 instances)"
echo "Eval trajectories:  ${OUTPUT_DIR}/trajectories_eval.jsonl   (286 instances)"

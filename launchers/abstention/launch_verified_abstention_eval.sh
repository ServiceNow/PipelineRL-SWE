#!/usr/bin/env bash
# Zero-shot generalization: run trained abstention predictor on SWE-bench Verified.
#
# Does NOT retrain — loads an existing checkpoint and scores Verified instances.
# Labels come from an existing Verified parquet collection.
# Scout CoT traces are collected fresh via local vLLM (no Daytona needed).
#
# Required env vars:
#   PREDICTOR_DIR   -- output dir of train_cot_abstention_predictor.py
#   CHECKPOINT_EPOCH -- epoch number to load, e.g. 6 (default: best by eval_auc)
#
# Optional:
#   VERIFIED_PARQUET_DIR -- eval parquet with existing 120b labels (default below)
#   LABEL_ROUTE_IDX      -- route index for strong-model labels (default: 3)
#   MODEL_NAME / MODEL_PATH -- scout model (default: Qwen/Qwen3-4B-Thinking-2507)
#   VLLM_PORT / VLLM_GPU_UTIL / MAX_MODEL_LEN
#   CONCURRENCY
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=$(date +%s)
JOB_NAME=verified_abstention_eval_${TIMESTAMP}

PREDICTOR_DIR=${PREDICTOR_DIR:?Need PREDICTOR_DIR set to train_cot_abstention_predictor.py output dir}
VERIFIED_PARQUET_DIR=${VERIFIED_PARQUET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_bench_train_all_16k_verified_eval_collect_5route_4b_scout_oss20_qwen30_oss120_gemini/collect/eval}
LABEL_ROUTE_IDX=${LABEL_ROUTE_IDX:-3}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}
MODEL_PATH=${MODEL_PATH:-${MODEL_NAME}}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}
CONCURRENCY=${CONCURRENCY:-16}

OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

# Pick checkpoint epoch: explicit override or best by grep-ing train.log
if [[ -n "${CHECKPOINT_EPOCH:-}" ]]; then
  EPOCH_STR=$(printf "%04d" "${CHECKPOINT_EPOCH}")
else
  # Find epoch with highest eval_auc — parse epoch + auc from the same line (mawk-compatible)
  EPOCH_STR=$(grep "eval_auc" "${PREDICTOR_DIR}/train.log" \
    | sed 's/Epoch \([0-9]*\):.*eval_auc=\([0-9.]*\).*/\1 \2/' \
    | awk 'BEGIN{best=-1;best_e=0} {if($2+0>best+0){best=$2+0;best_e=$1+0}} END{printf "%04d\n",best_e}')
fi
CHECKPOINT_DIR="${PREDICTOR_DIR}/checkpoints/epoch_${EPOCH_STR}"
echo "Using checkpoint: ${CHECKPOINT_DIR}"

mkdir -p "${OUTPUT_DIR}"
RUNNER="${OUTPUT_DIR}/run_verified_eval.sh"

cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"

echo '[vllm] Starting vLLM server...'
python -m vllm.entrypoints.openai.api_server \\
  --model ${MODEL_PATH} \\
  --port ${VLLM_PORT} \\
  --tensor-parallel-size ${TENSOR_PARALLEL} \\
  --gpu-memory-utilization ${VLLM_GPU_UTIL} \\
  --max-model-len ${MAX_MODEL_LEN} \\
  --trust-remote-code \\
  --served-model-name ${MODEL_NAME} \\
  > ${OUTPUT_DIR}/vllm_server.log 2>&1 &
VLLM_PID=\$!

echo "[vllm] Waiting for server (PID=\${VLLM_PID})..."
for i in \$(seq 1 120); do
  if curl -sf http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
    echo '[vllm] Server ready.'; break
  fi
  if ! kill -0 "\${VLLM_PID}" 2>/dev/null; then
    echo '[vllm] Server died. Check ${OUTPUT_DIR}/vllm_server.log'; exit 1
  fi
  sleep 5
done

echo '[collect] Collecting CoT traces on SWE-bench Verified...'
python pipelinerl/swe/scripts/offline_router/collect_verified_cot_trajectories.py \\
  --parquet-dir    ${VERIFIED_PARQUET_DIR} \\
  --label-route-idx ${LABEL_ROUTE_IDX} \\
  --vllm-base-url  http://localhost:${VLLM_PORT} \\
  --scout-model    ${MODEL_NAME} \\
  --output-dir     ${OUTPUT_DIR} \\
  --concurrency    ${CONCURRENCY} \\
  2>&1 | tee ${OUTPUT_DIR}/collect.log

echo '[vllm] Shutting down...'
kill \${VLLM_PID} 2>/dev/null || true
wait \${VLLM_PID} 2>/dev/null || true

echo '[score] Running abstention predictor (zero-shot)...'
python pipelinerl/swe/scripts/offline_router/score_cot_abstention_predictor.py \\
  --checkpoint-dir ${CHECKPOINT_DIR} \\
  --train-config   ${PREDICTOR_DIR}/train_config.json \\
  --trajectories   ${OUTPUT_DIR}/trajectories_verified.jsonl \\
  --parquet-dir    ${OUTPUT_DIR} \\
  --label-route-idx 0 \\
  --output-path    ${OUTPUT_DIR}/verified_predictions.jsonl \\
  2>&1 | tee ${OUTPUT_DIR}/score.log

echo "[done] Output: ${OUTPUT_DIR}"
echo "  Predictions: ${OUTPUT_DIR}/verified_predictions.jsonl"
SCRIPT_EOF
chmod +x "${RUNNER}"

echo "=== Submitting verified abstention eval: ${JOB_NAME} ==="
make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=1 \
  GPU=1 \
  GPU_MEM=80 \
  CPU=8 \
  CPU_MEM=64 \
  COMMAND="bash ${RUNNER}"

echo ""
echo "Predictions will be at: ${OUTPUT_DIR}/verified_predictions.jsonl"
echo "AUC will appear at the end of:  ${OUTPUT_DIR}/score.log"

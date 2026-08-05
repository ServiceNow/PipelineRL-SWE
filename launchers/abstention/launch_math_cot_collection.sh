#!/usr/bin/env bash
# Collect MATH domain CoT trajectories for abstention predictor training.
#
# Self-contained: starts a local vLLM server for the scout phase, then runs
# the OpenRouter labels phase once scout collection completes.
#
# Outputs in OUTPUT_DIR:
#   trajectories_train.jsonl   -- scout thinking traces (train split)
#   trajectories_eval.jsonl    -- scout thinking traces (eval split)
#   labels_train.jsonl         -- strong model label records (train)
#   labels_eval.jsonl          -- strong model label records (eval)
#   labels_train.parquet       -- merged parquet (use with --label-route-idx 0)
#   labels_eval.parquet        -- merged parquet
#
# Optional env vars:
#   OUTPUT_DIR          -- override default timestamped output directory
#   MODEL_NAME          -- scout model (default: Qwen/Qwen3-4B-Thinking-2507)
#   MODEL_PATH          -- local path/HF id for vLLM (default: MODEL_NAME)
#   STRONG_MODEL        -- OpenRouter model for labels (default: openai/gpt-oss-120b)
#   TRAIN_MAX_SAMPLES   -- cap on training problems (default: 1500)
#   EVAL_MAX_SAMPLES    -- cap on eval problems (default: 500)
#   CONCURRENCY         -- async concurrency (default: 32)
#   VLLM_PORT           -- vLLM port (default: 8000)
#   VLLM_GPU_UTIL       -- vLLM GPU memory fraction (default: 0.90)
#   MAX_MODEL_LEN       -- vLLM max sequence length (default: 16384)
#   TENSOR_PARALLEL     -- tensor parallel size (default: 1)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=$(date +%s)
JOB_NAME=math_cot_trajectories_${TIMESTAMP}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}
MODEL_PATH=${MODEL_PATH:-${MODEL_NAME}}
STRONG_MODEL=${STRONG_MODEL:-openai/gpt-oss-120b}

TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-1500}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-500}
CONCURRENCY=${CONCURRENCY:-32}

VLLM_PORT=${VLLM_PORT:-8000}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}

NPROC=1

# --- Write self-contained runner script ---
mkdir -p "${OUTPUT_DIR}"
RUNNER="${OUTPUT_DIR}/run_math_cot_collection.sh"

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

echo "[vllm] Waiting for server to be ready (PID=\${VLLM_PID})..."
for i in \$(seq 1 120); do
  if curl -sf http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
    echo '[vllm] Server ready.'; break
  fi
  if ! kill -0 "\${VLLM_PID}" 2>/dev/null; then
    echo '[vllm] Server process died. Check ${OUTPUT_DIR}/vllm_server.log'; exit 1
  fi
  sleep 5
done

echo '[scout] Starting MATH scout collection...'
python pipelinerl/swe/scripts/offline_router/collect_math_cot_trajectories.py \\
  --phase scout \\
  --vllm-base-url http://localhost:${VLLM_PORT} \\
  --scout-model ${MODEL_NAME} \\
  --output-dir ${OUTPUT_DIR} \\
  --train-max-samples ${TRAIN_MAX_SAMPLES} \\
  --eval-max-samples ${EVAL_MAX_SAMPLES} \\
  --concurrency ${CONCURRENCY} \\
  2>&1 | tee ${OUTPUT_DIR}/scout.log

echo '[vllm] Shutting down vLLM server...'
kill \${VLLM_PID} 2>/dev/null || true
wait \${VLLM_PID} 2>/dev/null || true

echo '[labels] Starting strong model label collection...'
python pipelinerl/swe/scripts/offline_router/collect_math_cot_trajectories.py \\
  --phase labels \\
  --strong-model ${STRONG_MODEL} \\
  --output-dir ${OUTPUT_DIR} \\
  --concurrency ${CONCURRENCY} \\
  2>&1 | tee ${OUTPUT_DIR}/labels.log

echo "[done] Output: ${OUTPUT_DIR}"
echo "  Train trajectories: ${OUTPUT_DIR}/trajectories_train.jsonl"
echo "  Eval  trajectories: ${OUTPUT_DIR}/trajectories_eval.jsonl"
echo "  Train parquet:      ${OUTPUT_DIR}/labels_train.parquet"
echo "  Eval  parquet:      ${OUTPUT_DIR}/labels_eval.parquet"
SCRIPT_EOF
chmod +x "${RUNNER}"

# --- Submit EAI job ---
echo "=== Submitting MATH CoT collection job: ${JOB_NAME} ==="
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
echo "Scout log:          ${OUTPUT_DIR}/scout.log"
echo "Labels log:         ${OUTPUT_DIR}/labels.log"
echo "vLLM log:           ${OUTPUT_DIR}/vllm_server.log"
echo ""
echo "To train the abstention predictor on MATH results:"
echo "  TRAJECTORIES_DIR=${OUTPUT_DIR} \\"
echo "  TRAIN_PARQUET_DIR=${OUTPUT_DIR} \\"
echo "  EVAL_PARQUET_DIR=${OUTPUT_DIR} \\"
echo "  LABEL_ROUTE_IDX=0 \\"
echo "  bash launchers/offline_router/launch_cot_abstention_predictor.sh"

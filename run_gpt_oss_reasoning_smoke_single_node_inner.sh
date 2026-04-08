#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:?RUN_NAME must be set}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT must be set}

VLLM_ENV=${VLLM_ENV:-vllm-env}
EVAL_ENV=${EVAL_ENV:-pipeline-rl}
MODEL_NAME=${MODEL_NAME:-openai/gpt-oss-120b}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8380}
CUDA_DEVICES=${CUDA_DEVICES:-}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-26720}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

REASONING_LEVELS=${REASONING_LEVELS:-"low medium high"}
SUBSAMPLE=${SUBSAMPLE:-32}
TEST_MAX_SAMPLES=${TEST_MAX_SAMPLES:-500}
MAX_TOKENS=${MAX_TOKENS:-15000}
TEMPERATURE=${TEMPERATURE:-0.7}
SERVER_START_TIMEOUT_SECS=${SERVER_START_TIMEOUT_SECS:-900}

SERVER_LOG=${OUTPUT_ROOT}/server.log
mkdir -p "${OUTPUT_ROOT}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ -n "${CUDA_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
fi

echo "Starting gpt-oss server on ${HOST}:${PORT}"
(
  exec conda run --no-capture-output -n "${VLLM_ENV}" \
    python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --model "${MODEL_NAME}" \
      --served-model-name "${MODEL_NAME}" \
      --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
      --dtype bfloat16 \
      --enable-prefix-caching \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "Waiting for server readiness..."
SECONDS_WAITED=0
until curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server exited before becoming ready. Tail of ${SERVER_LOG}:"
    tail -n 80 "${SERVER_LOG}" || true
    exit 1
  fi
  if [[ "${SECONDS_WAITED}" -ge "${SERVER_START_TIMEOUT_SECS}" ]]; then
    echo "Timed out waiting for server after ${SERVER_START_TIMEOUT_SECS}s. Tail of ${SERVER_LOG}:"
    tail -n 80 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 5
  SECONDS_WAITED=$((SECONDS_WAITED + 5))
done

echo "Server is ready. Running reasoning-effort evals sequentially."
for EFFORT in ${REASONING_LEVELS}; do
  EFFORT_OUTPUT_DIR=${OUTPUT_ROOT}/${EFFORT}
  mkdir -p "${EFFORT_OUTPUT_DIR}"
  echo "=== reasoning_effort=${EFFORT} ==="
  conda run --no-capture-output -n "${EVAL_ENV}" \
    python -m pipelinerl.swe.scripts.run_expert_repair_eval --config-dir conf --config-name swe \
      output_dir="${EFFORT_OUTPUT_DIR}" \
      wandb.use_wandb=false \
      expert_eval.base_url="http://${HOST}:${PORT}" \
      expert_eval.model_name="${MODEL_NAME}" \
      expert_eval.parameters.max_tokens="${MAX_TOKENS}" \
      expert_eval.parameters.temperature="${TEMPERATURE}" \
      expert_eval.parameters.reasoning_effort="${EFFORT}" \
      expert_eval.subsample="${SUBSAMPLE}" \
      dataset_loader_params.test_max_samples="${TEST_MAX_SAMPLES}" \
    2>&1 | tee -a "${EFFORT_OUTPUT_DIR}/launch.out"
done

echo "Completed all reasoning-effort smoke runs under ${OUTPUT_ROOT}"

#!/usr/bin/env bash
# Collect CoT (thinking) trajectories from Qwen3-4B-Thinking-2507 via local vLLM.
#
# This job:
#   1. Starts a vLLM server for the model
#   2. Runs collect_cot_trajectories.py to generate predictions + trajectories
#   3. Shuts down the vLLM server
#
# Outputs two JSONL files per split:
#   predictions_train.jsonl  -- feed to run_swesmith_eval_daytona.py for labels
#   trajectories_train.jsonl -- feed to train_autoregressive_verifier.py
#
# After this job, run Daytona eval on predictions_*.jsonl, then launch
# launch_autoregressive_verifier_train.sh with the resulting report paths.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=$(date +%s)
JOB_NAME=cot_trajectories_${TIMESTAMP}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B-Thinking-2507}
# Local models should be pre-downloaded to HF cache or a local path
MODEL_PATH=${MODEL_PATH:-${MODEL_NAME}}

TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-/mnt/llmd/data/swe_smith_bugged_context/ds_train}
EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-/mnt/llmd/data/swe_smith_bugged_context/ds_test}

TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-2000}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-500}
CONCURRENCY=${CONCURRENCY:-16}

VLLM_PORT=${VLLM_PORT:-8000}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}

# Single GPU is enough for 4B model
NPROC=1
TENSOR_PARALLEL=1

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
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${OUTPUT_DIR}; set -euo pipefail; { \
    echo '[vllm] Starting vLLM server...'; \
    python -m vllm.entrypoints.openai.api_server \
      --model ${MODEL_PATH} \
      --port ${VLLM_PORT} \
      --tensor-parallel-size ${TENSOR_PARALLEL} \
      --gpu-memory-utilization ${VLLM_GPU_UTIL} \
      --max-model-len ${MAX_MODEL_LEN} \
      --trust-remote-code \
      --served-model-name ${MODEL_NAME} \
    > ${OUTPUT_DIR}/vllm_server.log 2>&1 & \
    VLLM_PID=\$!; \
    echo '[vllm] Waiting for server to be ready (PID='\${VLLM_PID}')...'; \
    for i in \$(seq 1 120); do \
      if curl -sf http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then \
        echo '[vllm] Server ready.'; break; \
      fi; \
      sleep 5; \
    done; \
    echo '[collect] Starting trajectory collection...'; \
    python pipelinerl/swe/scripts/offline_router/collect_cot_trajectories.py \
      --vllm-base-url http://localhost:${VLLM_PORT} \
      --model-name ${MODEL_NAME} \
      --train-dataset-path ${TRAIN_DATASET_PATH} \
      --eval-dataset-path ${EVAL_DATASET_PATH} \
      --output-dir ${OUTPUT_DIR} \
      --train-max-samples ${TRAIN_MAX_SAMPLES} \
      --eval-max-samples ${EVAL_MAX_SAMPLES} \
      --concurrency ${CONCURRENCY} \
    2>&1 | tee ${OUTPUT_DIR}/collect.log; \
    echo '[vllm] Shutting down vLLM server...'; \
    kill \${VLLM_PID} 2>/dev/null || true; \
    echo '[done] Trajectories written to ${OUTPUT_DIR}'; \
    echo 'Next: run Daytona eval on predictions_train.jsonl, then launch_autoregressive_verifier_train.sh'; \
  }"

echo "Output dir: ${OUTPUT_DIR}"

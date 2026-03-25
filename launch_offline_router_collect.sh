#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP=$(date +%s)
JOB_NAME=${JOB_NAME:-offline_router_collect}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
LOG_DIR=${OUTPUT_DIR}/server_logs

# Reserve a full 8-GPU node by default. The collection process itself is light,
# and the primary-model + GPT-OSS throughput test still fits comfortably on one 8x80GB node.
NPROC=${NPROC:-5}
CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}
COLLECTOR_ENV=${COLLECTOR_ENV:-pipeline-rl}

# Edit these directly for the collection run you want.
PRIMARY_MODEL_PATH=${PRIMARY_MODEL_PATH:-/mnt/llmd/results/exps/aristides/reason/swe_smith_policy_conditioned_no_devstral_1773812579/finetune/current}
PRIMARY_TOKENIZER_NAME=${PRIMARY_TOKENIZER_NAME:-Qwen/Qwen2.5-Coder-7B-Instruct}
PRIMARY_SERVED_MODEL_NAME=${PRIMARY_SERVED_MODEL_NAME:-primary_model}

COLLECT_TRAIN=${COLLECT_TRAIN:-true}
COLLECT_EVAL=${COLLECT_EVAL:-true}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-10000}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-500}
# Keep concurrency modest so GPT-OSS does not get buried in queueing.
MAX_CONCURRENT_PROBLEMS=${MAX_CONCURRENT_PROBLEMS:-8}
SHARD_SIZE=${SHARD_SIZE:-64}
EXTRA_ARGS=${EXTRA_ARGS:-}

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=${COLLECTOR_ENV} \
  CONDA_EXE=${CONDA_EXE} \
  SNAPSHOT=1 \
  NPROC=${NPROC} \
  COMMAND="cd /home/toolkit/PipelineRL-SWE; mkdir -p ${LOG_DIR}; \
    python -m pipelinerl.swe.scripts.offline_router.run_collection_job \
    output_dir=${OUTPUT_DIR} \
    offline_router.primary_model.model_path=${PRIMARY_MODEL_PATH} \
    offline_router.primary_model.tokenizer_name=${PRIMARY_TOKENIZER_NAME} \
    offline_router.primary_model.served_model_name=${PRIMARY_SERVED_MODEL_NAME} \
    offline_router.primary_model.model_name=${PRIMARY_SERVED_MODEL_NAME} \
    offline_router.collection.collect_train=${COLLECT_TRAIN} \
    offline_router.collection.collect_eval=${COLLECT_EVAL} \
    offline_router.dataset.train_max_samples=${TRAIN_MAX_SAMPLES} \
    offline_router.dataset.eval_max_samples=${EVAL_MAX_SAMPLES} \
    offline_router.collection.max_concurrent_problems=${MAX_CONCURRENT_PROBLEMS} \
    offline_router.collection.shard_size=${SHARD_SIZE} \
    ${EXTRA_ARGS}"

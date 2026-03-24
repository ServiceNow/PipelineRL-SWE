#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP=$(date +%s)
JOB_NAME=${JOB_NAME:-offline_router_collect}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
LOG_DIR=${OUTPUT_DIR}/server_logs

# Reserve a full 8-GPU node by default. The collection process itself is light,
# but the primary-model + expert vLLM servers fit comfortably on one 8x80GB node.
NPROC=${NPROC:-8}
CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}
COLLECTOR_ENV=${COLLECTOR_ENV:-pipeline-rl}

COLLECT_TRAIN=${COLLECT_TRAIN:-true}
COLLECT_EVAL=${COLLECT_EVAL:-true}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-null}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-500}
MAX_CONCURRENT_PROBLEMS=${MAX_CONCURRENT_PROBLEMS:-32}
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
    offline_router.collection.collect_train=${COLLECT_TRAIN} \
    offline_router.collection.collect_eval=${COLLECT_EVAL} \
    offline_router.collection.max_samples.train=${TRAIN_MAX_SAMPLES} \
    offline_router.collection.max_samples.eval=${EVAL_MAX_SAMPLES} \
    offline_router.collection.max_concurrent_problems=${MAX_CONCURRENT_PROBLEMS} \
    offline_router.collection.shard_size=${SHARD_SIZE} \
    ${EXTRA_ARGS}"

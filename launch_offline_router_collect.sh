#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP=$(date +%s)
JOB_NAME=${JOB_NAME:-offline_router_collect}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}

# The collector expects OpenAI-compatible HTTP endpoints to already be available.
# Override these if your servers are elsewhere.
POLICY_BASE_URL=${POLICY_BASE_URL:-http://127.0.0.1:8000}
EXPERT_BASE_URLS=${EXPERT_BASE_URLS:-'["http://127.0.0.1:8280","http://127.0.0.1:8380"]'}

COLLECT_TRAIN=${COLLECT_TRAIN:-true}
COLLECT_EVAL=${COLLECT_EVAL:-true}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-null}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-500}
MAX_CONCURRENT_PROBLEMS=${MAX_CONCURRENT_PROBLEMS:-32}
SHARD_SIZE=${SHARD_SIZE:-64}

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=1 \
  COMMAND="cd /home/toolkit/PipelineRL-SWE; python -m pipelinerl.swe.scripts.offline_router.collect_router_dataset \
    output_dir=${OUTPUT_DIR} \
    offline_router.policy.base_url=${POLICY_BASE_URL} \
    offline_router.expert_base_urls=${EXPERT_BASE_URLS} \
    offline_router.collection.collect_train=${COLLECT_TRAIN} \
    offline_router.collection.collect_eval=${COLLECT_EVAL} \
    offline_router.collection.max_samples.train=${TRAIN_MAX_SAMPLES} \
    offline_router.collection.max_samples.eval=${EVAL_MAX_SAMPLES} \
    offline_router.collection.max_concurrent_problems=${MAX_CONCURRENT_PROBLEMS} \
    offline_router.collection.shard_size=${SHARD_SIZE}"

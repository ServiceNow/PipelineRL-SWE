#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP=${TIMESTAMP:-$(date +%s)}

SOURCE_RUN_DIR=${SOURCE_RUN_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_bench_router_split_collect_1776749732}
COLLECT_OUTPUT_DIR=${COLLECT_OUTPUT_DIR:-${SOURCE_RUN_DIR}/collect}

JOB_NAME=${JOB_NAME:-offline_router_swe_bench_router_split_vector_bin_20bucket_train_only}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_text_lora_vector_bin_20bucket_random}

JOB_NPROC=${JOB_NPROC:-4}
TRAIN_NPROC=${TRAIN_NPROC:-4}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-24000}
RUN_COLLECT=0
RUN_TRAIN=1
ROUTER_SUPERVISION_MODE=text_reward_vector_bin
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:-offline_router.train.text_reward.bin_count=21 offline_router.train.text_reward.bin_value_order=ascending offline_router.train.text_reward.drop_overlength_rows=true}

if [[ ! -f "${COLLECT_OUTPUT_DIR}/metadata.json" ]]; then
  echo "Missing offline-router collection metadata: ${COLLECT_OUTPUT_DIR}/metadata.json" >&2
  echo "Set SOURCE_RUN_DIR or COLLECT_OUTPUT_DIR to an existing collection run." >&2
  exit 1
fi

export JOB_NAME
export TIMESTAMP
export OUTPUT_ROOT
export TRAIN_OUTPUT_DIR
export JOB_NPROC
export TRAIN_NPROC
export MAX_SEQ_LENGTH
export RUN_COLLECT
export RUN_TRAIN
export COLLECT_OUTPUT_DIR
export ROUTER_SUPERVISION_MODE
export TRAIN_EXTRA_ARGS

bash launch_offline_router_swe_smith_id.sh

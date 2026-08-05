#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}

SOURCE_RUN_DIR=${SOURCE_RUN_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_id_1776113427}
COLLECT_OUTPUT_DIR=${COLLECT_OUTPUT_DIR:-${SOURCE_RUN_DIR}/collect}

JOB_NAME=${JOB_NAME:-offline_router_swe_smith_id_bin_train_only}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_text_lora_bin_expectation_random}

JOB_NPROC=${JOB_NPROC:-4}
TRAIN_NPROC=${TRAIN_NPROC:-4}
RUN_COLLECT=0
RUN_TRAIN=1
ROUTER_SUPERVISION_MODE=text_reward_bin
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:-offline_router.train.text_reward.bin_count=21}

export JOB_NAME
export TIMESTAMP
export OUTPUT_ROOT
export TRAIN_OUTPUT_DIR
export JOB_NPROC
export TRAIN_NPROC
export RUN_COLLECT
export RUN_TRAIN
export COLLECT_OUTPUT_DIR
export ROUTER_SUPERVISION_MODE
export TRAIN_EXTRA_ARGS

bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_id.sh"

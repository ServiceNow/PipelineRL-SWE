#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

SOURCE_RUN_DIR=${SOURCE_RUN_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_id_1776113427}
COLLECT_OUTPUT_DIR=${COLLECT_OUTPUT_DIR:-${SOURCE_RUN_DIR}/collect}

JOB_NAME=${JOB_NAME:-offline_router_swe_smith_id_train_only}
JOB_NPROC=${JOB_NPROC:-4}
TRAIN_NPROC=${TRAIN_NPROC:-4}
RUN_COLLECT=0
RUN_TRAIN=1

export JOB_NAME
export JOB_NPROC
export TRAIN_NPROC
export RUN_COLLECT
export RUN_TRAIN
export COLLECT_OUTPUT_DIR

bash "${SCRIPT_DIR}/launch_offline_router_swe_smith_id.sh"

#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP=$(date +%s)
JOB_NAME=offline_router_train
OUTPUT_DIR=/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}

# Edit these directly when you want a different launch configuration.
NPROC=1
DATASET_DIR=/mnt/llmd/results/exps/aristides/reason/offline_router_collect_1774417576
MODEL_PATH=/mnt/llmd/results/exps/aristides/reason/swe_smith_policy_conditioned_no_devstral_1773812579/finetune/current
EXTRA_ARGS=

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${NPROC} \
  COMMAND="cd /home/toolkit/PipelineRL-SWE; python -m pipelinerl.swe.scripts.offline_router.train_router_offline \
    output_dir=${OUTPUT_DIR} \
    offline_router.train.dataset_dir=${DATASET_DIR} \
    offline_router.train.model_path=${MODEL_PATH} \
    ${EXTRA_ARGS}"

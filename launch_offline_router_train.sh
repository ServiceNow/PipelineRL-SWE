#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP=$(date +%s)
JOB_NAME=offline_router_train_text_mode_test
OUTPUT_DIR=/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}

# Edit these directly when you want a different launch configuration.
NPROC=4
DATASET_DIR=/mnt/llmd/results/exps/aristides/reason/offline_router_collect_1774417576
MODEL_PATH=/mnt/llmd/results/exps/aristides/reason/swe_smith_policy_conditioned_no_devstral_1773812579/finetune/current
MIXED_PRECISION=bf16
ACCELERATE_CONFIG=deepspeed
DEEPSPEED_CONFIG=deepspeed_stage3_bf16
EXTRA_ARGS="offline_router.train.supervision_mode=text_reward_per_route offline_router.train.mode=full_backbone offline_router.train.max_train_rows=64 offline_router.train.max_eval_rows=16 offline_router.train.num_epochs=1 offline_router.train.save_checkpoints=true" \

TRAIN_CMD="python -m pipelinerl.swe.scripts.offline_router.train_router_offline"
if [[ "${NPROC}" -gt 1 ]]; then
  TRAIN_CMD="python -m accelerate.commands.launch \
    --use_deepspeed \
    --mixed_precision ${MIXED_PRECISION} \
    --num_processes ${NPROC} \
    --config_file conf/accelerate/${ACCELERATE_CONFIG}.yaml \
    --deepspeed_config_file conf/deepspeed/${DEEPSPEED_CONFIG}.json \
    pipelinerl/swe/scripts/offline_router/train_router_offline.py"
fi

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${NPROC} \
  COMMAND="cd /home/toolkit/PipelineRL-SWE; mkdir -p ${OUTPUT_DIR}; set -o pipefail; { ${TRAIN_CMD} \
    output_dir=${OUTPUT_DIR} \
    offline_router.train.dataset_dir=${DATASET_DIR} \
    offline_router.train.model_path=${MODEL_PATH} \
    ${EXTRA_ARGS}; } 2>&1 | tee -a ${OUTPUT_DIR}/launch.out"

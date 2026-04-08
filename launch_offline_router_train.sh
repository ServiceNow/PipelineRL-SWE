#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP=$(date +%s)
JOB_NAME=offline_router_text_lora_full
OUTPUT_DIR=/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}

# Edit these directly when you want a different launch configuration.
NPROC=4
DATASET_DIR=/mnt/llmd/results/exps/aristides/reason/offline_router_collect_1774417576
MODEL_PATH=/mnt/llmd/results/exps/aristides/reason/swe_smith_policy_conditioned_no_devstral_1773812579/finetune/current
MIXED_PRECISION=bf16
ACCELERATE_CONFIG=base_mp
EXTRA_ARGS="offline_router.train.supervision_mode=text_reward_vector offline_router.train.mode=full_backbone offline_router.train.max_seq_length=32000 offline_router.train.num_epochs=3 offline_router.train.save_checkpoints=true offline_router.train.text_reward.debug_step_logging=false"

TRAIN_CMD="python -m pipelinerl.swe.scripts.offline_router.train_router_offline"
if [[ "${NPROC}" -gt 1 ]]; then
  TRAIN_CMD="python -m accelerate.commands.launch \
    --multi_gpu \
    --mixed_precision ${MIXED_PRECISION} \
    --num_processes ${NPROC} \
    --config_file conf/accelerate/${ACCELERATE_CONFIG}.yaml \
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

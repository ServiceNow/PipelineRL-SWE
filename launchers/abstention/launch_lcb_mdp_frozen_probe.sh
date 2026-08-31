#!/usr/bin/env bash
# Is the LoRA fine-tuning doing anything but memorizing?
#
# The current recipe's calibration loss is BEST at epoch 0 (0.448) and rises from
# there (0.454, 0.547) while train loss halves (0.535 -> 0.264 -> 0.203). That is
# memorization of 551 problems, not learning from them. If a probe on frozen
# features matches the LoRA model, the fine-tuning contributes only overfitting and
# the method gets substantially simpler and more defensible.
#
# Frozen encoder, no LoRA, same head, same data, same split. More epochs than the
# LoRA recipe (which is stopped at epoch 0 by early selection) because a small head
# on fixed features should tolerate a real training run -- and if it still peaks at
# epoch 0, that localizes the overfitting to the head rather than the encoder.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

SUBMIT=${SUBMIT:-0}
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SEED=${SEED:-17}
SOURCE_ARTIFACT=${SOURCE_ARTIFACT:-/mnt/llmd/results/exps/aristides/reason/lcb_mdp_temporal_551_341_counts_last_unweighted_1787939458b}
TENSORS_DIR=${TENSORS_DIR:-${SOURCE_ARTIFACT}/tensors_v3}
DATASET_DIR=${DATASET_DIR:-${SOURCE_ARTIFACT}/reachable_dataset}
JOB_NAME=${JOB_NAME:-lcb_mdp_frozen_probe_seed${SEED}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}
MODEL_DIR=${MODEL_DIR:-${OUTPUT_DIR}/model}
REPLAY_DIR=${REPLAY_DIR:-${OUTPUT_DIR}/replay}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LR=${LR:-1e-3}
NPROC=${NPROC:-4}
NUM_ORDERINGS=${NUM_ORDERINGS:-5}
SNAPSHOT=${SNAPSHOT:-1}

if [[ "${SUBMIT}" != "1" ]]; then
  cat <<EOF
Prepared but not submitted.
  job: ${JOB_NAME}
  dataset (reused): ${DATASET_DIR}
  stage 1: train head only, encoder frozen, no LoRA (${NUM_EPOCHS} epochs, lr ${LR})
  stage 2: replay, Bellman H=2, same RoR baseline
  output: ${OUTPUT_DIR}
  compare against: ${SOURCE_ARTIFACT} (LoRA, best_calibration_epoch=0)

Submit explicitly with:
  SUBMIT=1 bash ${BASH_SOURCE[0]}
EOF
  exit 0
fi

if [[ "${NPROC}" -gt 1 ]]; then
  # Explicit --config_file: a bare `accelerate launch` picks up the user default,
  # which points at a deepspeed config that does not exist in the job image.
  TRAIN_LAUNCH="python -m accelerate.commands.launch --multi_gpu --mixed_precision bf16 --num_processes ${NPROC} --config_file conf/accelerate/base_mp.yaml"
else
  TRAIN_LAUNCH="python"
fi

# No cd into the live tree; SNAPSHOT=1 already points --workdir and PYTHONPATH at the snapshot.
COMMAND="${TRAIN_LAUNCH} pipelinerl/swe/scripts/livecodebench/train_mdp_reachable_policy.py \
  --dataset-dir ${DATASET_DIR} --output-dir ${MODEL_DIR} --seed ${SEED} \
  --num-epochs ${NUM_EPOCHS} --lr ${LR} --max-seq-length 8192 --frozen-encoder && \
python pipelinerl/swe/scripts/livecodebench/replay_mdp_full_execution.py \
  --tensors-dir ${TENSORS_DIR} --output-dir ${REPLAY_DIR} \
  --sequential-model-dir ${MODEL_DIR} \
  --num-orderings ${NUM_ORDERINGS} --start-protocol scout_first \
  --state-layout counts_last --cost-mode usd \
  --bellman-horizons 2 --retention-grid 0.90,0.95,0.98,1.0"

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" NPROC="${NPROC}" GPU=1 GPU_MEM=80 CPU=16 CPU_MEM=128 \
  COMMAND="${COMMAND}"

echo "Job: ${JOB_NAME}"
echo "Output: ${OUTPUT_DIR}"

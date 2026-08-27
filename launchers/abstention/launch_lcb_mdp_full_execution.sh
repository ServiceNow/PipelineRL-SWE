#!/usr/bin/env bash
# Corrected scout-first LCB MDP: full execution, reachable failure histories,
# train/calibration/test separation, and end-to-end learned-policy replay.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${LCB_COLLECTION_DIR:?Set the corrected temporal LCB collection directory}"
: "${EXPERT_MULTIDRAW_DIR:?Set the oss20/oss120 multidraw collection directory}"
: "${SCOUT_MULTIDRAW_DIR:?Set the scout multidraw collection directory}"

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SEED=${SEED:-17}
NPROC=${NPROC:-4}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LR=${LR:-2e-5}
NUM_ORDERINGS=${NUM_ORDERINGS:-5}
HISTORIES_PER_PROBLEM=${HISTORIES_PER_PROBLEM:-10}
SNAPSHOT=${SNAPSHOT:-1}
JOB_NAME=${JOB_NAME:-lcb_mdp_latest_attempt_seed${SEED}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

TENSORS_DIR=${OUTPUT_DIR}/tensors_v2
DATASET_DIR=${OUTPUT_DIR}/reachable_dataset
MODEL_DIR=${OUTPUT_DIR}/model
REPLAY_DIR=${OUTPUT_DIR}/replay

if [[ "${NPROC}" -gt 1 ]]; then
  TRAIN_LAUNCH="python -m accelerate.commands.launch --multi_gpu --mixed_precision bf16 --num_processes ${NPROC} --config_file conf/accelerate/base_mp.yaml"
else
  TRAIN_LAUNCH="python"
fi

COMMAND="cd ${REPO_ROOT} && \
python pipelinerl/swe/scripts/livecodebench/build_mdp_tensors_v2.py \
  --collection-dir ${EXPERT_MULTIDRAW_DIR} \
  --collection-dir ${SCOUT_MULTIDRAW_DIR} \
  --source-collection-dir ${LCB_COLLECTION_DIR} \
  --output-dir ${TENSORS_DIR} && \
python pipelinerl/swe/scripts/livecodebench/build_mdp_reachable_dataset.py \
  --tensors-dir ${TENSORS_DIR} --output-dir ${DATASET_DIR} \
  --histories-per-problem ${HISTORIES_PER_PROBLEM} --seed 0 && \
${TRAIN_LAUNCH} pipelinerl/swe/scripts/livecodebench/train_mdp_reachable_policy.py \
  --dataset-dir ${DATASET_DIR} --output-dir ${MODEL_DIR} \
  --seed ${SEED} --num-epochs ${NUM_EPOCHS} --lr ${LR} --max-seq-length 8192 && \
python pipelinerl/swe/scripts/livecodebench/replay_mdp_full_execution.py \
  --tensors-dir ${TENSORS_DIR} --output-dir ${REPLAY_DIR} \
  --sequential-model-dir ${MODEL_DIR} --num-orderings ${NUM_ORDERINGS} --cost-mode usd"

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" NPROC="${NPROC}" GPU=1 GPU_MEM=80 CPU=16 CPU_MEM=128 \
  COMMAND="${COMMAND}"

echo "Job: ${JOB_NAME}"
echo "Output: ${OUTPUT_DIR}"

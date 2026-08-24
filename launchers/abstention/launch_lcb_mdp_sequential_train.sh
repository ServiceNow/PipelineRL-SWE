#!/usr/bin/env bash
# Train the thread-(a) sequential MDP policy (depth-1, 4 BCE heads).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${TENSORS_DIR:?Set TENSORS_DIR to the mdp tensors dir}"
DATASET_DIR=${DATASET_DIR:-${TENSORS_DIR%/mdp_tensors*}/mdp_seq_dataset_v1}
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
NPROC=${NPROC:-4}
SNAPSHOT=${SNAPSHOT:-1}

JOB_NAME=${JOB_NAME:-mdp_sequential_policy_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

if [[ "${NPROC}" -gt 1 ]]; then
  TRAIN_LAUNCH="python -m accelerate.commands.launch --multi_gpu --mixed_precision bf16 --num_processes ${NPROC} --config_file conf/accelerate/base_mp.yaml"
else
  TRAIN_LAUNCH="python"
fi

COMMAND="cd ${REPO_ROOT} && \
python pipelinerl/swe/scripts/livecodebench/build_mdp_sequential_dataset.py --tensors-dir ${TENSORS_DIR} --output-dir ${DATASET_DIR} && \
${TRAIN_LAUNCH} pipelinerl/swe/scripts/livecodebench/train_mdp_sequential_policy.py \
  --dataset-dir ${DATASET_DIR} --output-dir ${OUTPUT_DIR} \
  --lr 1e-4 --num-epochs 8 --max-seq-length 8192 --seed 17"

make -C "${REPO_ROOT}" job JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" NPROC=${NPROC} GPU=1 GPU_MEM=80 CPU=16 CPU_MEM=128 COMMAND="${COMMAND}"

echo "Job: ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"

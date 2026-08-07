#!/usr/bin/env bash
# Train abstention predictor on LCB scout trajectories + oracle labels.
# Labels = did oss-120b solve the problem? (route index 3 in labels parquet)
#
# Optional env vars:
#   LCB_COLLECTION_DIR  -- output dir from launch_lcb_collection.sh (required)
#   INPUT_ONLY          -- "true" to train IO variant (no scout output)
#   NUM_EPOCHS / NPROC
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${LCB_COLLECTION_DIR:?Set LCB_COLLECTION_DIR to the output of launch_lcb_collection.sh}"

TIMESTAMP=$(date +%s)
INPUT_ONLY=${INPUT_ONLY:-false}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
NPROC=${NPROC:-4}
LABEL_ROUTE_IDX=3  # oracle = oss-120b

if [[ "${INPUT_ONLY}" == "true" ]]; then
  IO_ARG="--input-only"
  IO_SUFFIX="_input_only"
else
  IO_ARG=""
  IO_SUFFIX=""
fi

JOB_NAME=lcb_abstention_route${LABEL_ROUTE_IDX}_nocot${IO_SUFFIX}_${NUM_EPOCHS}epoch_${TIMESTAMP}
OUTPUT_DIR=/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}

mkdir -p "${OUTPUT_DIR}"

RUNNER="${OUTPUT_DIR}/run_train.sh"

TRAIN_CMD="python pipelinerl/swe/scripts/offline_router/train_cot_abstention_predictor.py"
if [[ "${NPROC}" -gt 1 ]]; then
  TRAIN_CMD="python -m accelerate.commands.launch \
    --multi_gpu \
    --mixed_precision bf16 \
    --num_processes ${NPROC} \
    --config_file conf/accelerate/base_mp.yaml \
    pipelinerl/swe/scripts/offline_router/train_cot_abstention_predictor.py"
fi

cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "=== Training LCB abstention predictor ==="
${TRAIN_CMD} \\
  --train-trajectories ${LCB_COLLECTION_DIR}/trajectories_train.jsonl \\
  --eval-trajectories  ${LCB_COLLECTION_DIR}/trajectories_eval.jsonl \\
  --train-parquet-dir  ${LCB_COLLECTION_DIR}/train \\
  --eval-parquet-dir   ${LCB_COLLECTION_DIR}/eval \\
  --output-dir         ${OUTPUT_DIR} \\
  --label-route-idx    ${LABEL_ROUTE_IDX} \\
  --no-include-thinking \\
  ${IO_ARG} \\
  --max-seq-length     8192 \\
  --num-epochs         ${NUM_EPOCHS} \\
  --batch-size         1 \\
  --eval-batch-size    1 \\
  --gradient-accumulation-steps 8 \\
  --lr                 2e-5 \\
  --lora-r             ${LORA_R} \\
  --lora-alpha         ${LORA_ALPHA} \\
  --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \\
  --gradient-checkpointing \\
  --checkpoint-every-epoch \\
  2>&1 | tee ${OUTPUT_DIR}/train.log

echo "[done] Output: ${OUTPUT_DIR}"
SCRIPT_EOF
chmod +x "${RUNNER}"

echo "=== Submitting LCB abstention train: ${JOB_NAME} ==="
make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${NPROC} \
  GPU=1 \
  GPU_MEM=80 \
  CPU=16 \
  CPU_MEM=128 \
  COMMAND="bash ${RUNNER}"

echo ""
echo "Job:        ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Train log:  ${OUTPUT_DIR}/train.log"

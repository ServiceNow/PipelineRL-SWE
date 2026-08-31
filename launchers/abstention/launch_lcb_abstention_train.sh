#!/usr/bin/env bash
# Train abstention predictor on LCB scout trajectories + oracle labels.
# Labels = did oss-120b solve the problem? (route index 3 in labels parquet)
#
# Optional env vars:
#   LCB_COLLECTION_DIR  -- output dir from launch_lcb_collection.sh (required)
#   INPUT_ONLY          -- "true" to train IO variant (no scout output)
#   INCLUDE_THINKING    -- include scout reasoning (default: false)
#   INCLUDE_TEST_FEEDBACK -- include official scout test results (default: false)
#   TEST_FEEDBACK_FORMAT  -- full | names_only | count_only (default: full)
#   NUM_EPOCHS / NPROC
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${LCB_COLLECTION_DIR:?Set LCB_COLLECTION_DIR to the output of launch_lcb_collection.sh}"

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
INPUT_ONLY=${INPUT_ONLY:-false}
INCLUDE_THINKING=${INCLUDE_THINKING:-false}
INCLUDE_TEST_FEEDBACK=${INCLUDE_TEST_FEEDBACK:-false}
TEST_FEEDBACK_FORMAT=${TEST_FEEDBACK_FORMAT:-full}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
NPROC=${NPROC:-4}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-24000}
SNAPSHOT=${SNAPSHOT:-1}
# The 0.5521 input-only vs 0.7693 post-scout gap is one seed. It is the spine of the
# "scout before you route" claim, so it needs replication before anything is built on it.
SEED=${SEED:-17}
LABEL_ROUTE_IDX=3  # oracle = oss-120b

if [[ "${INPUT_ONLY}" == "true" ]]; then
  IO_ARG="--input-only"
  IO_SUFFIX="_input_only"
else
  IO_ARG=""
  IO_SUFFIX="_post_scout"
fi

if [[ "${INPUT_ONLY}" == "true" && "${INCLUDE_TEST_FEEDBACK}" == "true" ]]; then
  echo "INPUT_ONLY=true is incompatible with INCLUDE_TEST_FEEDBACK=true" >&2
  exit 1
fi

if [[ "${INCLUDE_THINKING}" == "true" ]]; then
  THINKING_ARG="--include-thinking"
  THINKING_SUFFIX="_cot"
else
  THINKING_ARG="--no-include-thinking"
  THINKING_SUFFIX="_nocot"
fi

if [[ "${INCLUDE_TEST_FEEDBACK}" == "true" ]]; then
  TEST_FEEDBACK_ARG="--include-test-feedback --test-feedback-format ${TEST_FEEDBACK_FORMAT}"
  TEST_FEEDBACK_SUFFIX="_testfb_${TEST_FEEDBACK_FORMAT}"
else
  TEST_FEEDBACK_ARG="--no-include-test-feedback"
  TEST_FEEDBACK_SUFFIX=""
fi

JOB_NAME=${JOB_NAME:-lcb_corrected_abstention_route${LABEL_ROUTE_IDX}${THINKING_SUFFIX}${IO_SUFFIX}${TEST_FEEDBACK_SUFFIX}_${NUM_EPOCHS}epoch_seed${SEED}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

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
  --eval-oracle-results ${LCB_COLLECTION_DIR}/oracle_eval.jsonl \\
  --output-dir         ${OUTPUT_DIR} \\
  --label-route-idx    ${LABEL_ROUTE_IDX} \\
  ${THINKING_ARG} \\
  ${IO_ARG} \\
  ${TEST_FEEDBACK_ARG} \\
  --max-seq-length     ${MAX_SEQ_LENGTH} \\
  --num-epochs         ${NUM_EPOCHS} \\
  --batch-size         1 \\
  --eval-batch-size    1 \\
  --gradient-accumulation-steps 8 \\
  --lr                 2e-5 \\
  --seed               ${SEED} \\
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
  SNAPSHOT=${SNAPSHOT} \
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

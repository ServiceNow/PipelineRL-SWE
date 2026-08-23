#!/usr/bin/env bash
# Transfer experiment: train abstention predictor on SWE-Smith 4B-Instruct traces
# (real route-3 oss-120b labels from the 4-route parquet), then score zero-shot on
# SWE-bench Verified 4B-Instruct traces (real Daytona route-3 oss-120b labels).
#
# Label = did oss-120b succeed? — the predictor learns to route based on whether
# the strong model (not the 4B scout itself) can handle the instance.
#
# No vLLM needed — traces are already collected on both sides.
#
# Optional env vars:
#   TRAIN_TRAJECTORIES_DIR      -- SWE-Smith instruct trajectories (default below)
#   EVAL_TRAJECTORIES_JSONL     -- Verified instruct trajectories JSONL (default below)
#   VERIFIED_REAL_LABELS_JSONL  -- Daytona results for Verified route 3 oss-120b (default below)
#   REAL_LABEL_DATASET_DIR      -- SWE-Smith 4-route real-label parquet dir
#   LABEL_ROUTE_IDX             -- route index for train labels (default: 3 = oss-120b)
#   INPUT_ONLY                  -- "true" to ablate patch text (default: false)
#   NUM_EPOCHS / LORA_R / NPROC
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=$(date +%s)

TRAIN_TRAJECTORIES_DIR=${TRAIN_TRAJECTORIES_DIR:-/mnt/llmd/results/exps/aristides/reason/instruct_patches_trajectories_1785884242}
EVAL_TRAJECTORIES_JSONL=${EVAL_TRAJECTORIES_JSONL:-/mnt/llmd/results/exps/aristides/reason/verified_zeroshot_nocot/trajectories_verified.jsonl}
# Route 3 = oss-120b real Daytona labels on Verified (194/369 resolved)
VERIFIED_REAL_LABELS_JSONL=${VERIFIED_REAL_LABELS_JSONL:-/mnt/llmd/results/exps/aristides/reason/verified_real_label_eval_1785965509/predictions/predictions_route_3.results.jsonl}

REAL_LABEL_DATASET_DIR=${REAL_LABEL_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}
TRAIN_PARQUET_DIR=${TRAIN_PARQUET_DIR:-${REAL_LABEL_DATASET_DIR}/train}
EVAL_PARQUET_DIR=${EVAL_PARQUET_DIR:-${REAL_LABEL_DATASET_DIR}/eval}
LABEL_ROUTE_IDX=${LABEL_ROUTE_IDX:-3}  # route 3 = oss-120b

INPUT_ONLY=${INPUT_ONLY:-false}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-24000}
NPROC=${NPROC:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-base_mp}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-2e-5}
SEED=${SEED:-17}

if [[ "${INPUT_ONLY}" == "true" ]]; then
  INPUT_ONLY_ARG="--input-only"
  INPUT_ONLY_SUFFIX="_input_only"
else
  INPUT_ONLY_ARG=""
  INPUT_ONLY_SUFFIX=""
fi

JOB_NAME=swe_smith_instruct_to_verified_transfer_route${LABEL_ROUTE_IDX}_nocot${INPUT_ONLY_SUFFIX}_${NUM_EPOCHS}epoch_seed${SEED}_${TIMESTAMP}
OUTPUT_DIR=/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}

RUNNER="${OUTPUT_DIR}/run_transfer.sh"
mkdir -p "${OUTPUT_DIR}"

TRAIN_CMD="python pipelinerl/swe/scripts/offline_router/train_cot_abstention_predictor.py"
if [[ "${NPROC}" -gt 1 ]]; then
  TRAIN_CMD="python -m accelerate.commands.launch \
    --multi_gpu \
    --mixed_precision ${MIXED_PRECISION} \
    --num_processes ${NPROC} \
    --config_file conf/accelerate/${ACCELERATE_CONFIG}.yaml \
    pipelinerl/swe/scripts/offline_router/train_cot_abstention_predictor.py"
fi

cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"

mkdir -p "${OUTPUT_DIR}"

echo "=== [1/2] Training abstention predictor on SWE-Smith 4B-Instruct traces ==="
${TRAIN_CMD} \\
  --train-trajectories ${TRAIN_TRAJECTORIES_DIR}/trajectories_train.jsonl \\
  --eval-trajectories  ${TRAIN_TRAJECTORIES_DIR}/trajectories_eval.jsonl \\
  --train-parquet-dir  ${TRAIN_PARQUET_DIR} \\
  --eval-parquet-dir   ${EVAL_PARQUET_DIR} \\
  --output-dir         ${OUTPUT_DIR} \\
  --label-route-idx    ${LABEL_ROUTE_IDX} \\
  --no-include-thinking \\
  ${INPUT_ONLY_ARG} \\
  --max-seq-length     ${MAX_SEQ_LENGTH} \\
  --num-epochs         ${NUM_EPOCHS} \\
  --batch-size         ${BATCH_SIZE} \\
  --eval-batch-size    ${BATCH_SIZE} \\
  --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \\
  --lr                 ${LR} \\
  --lora-r             ${LORA_R} \\
  --lora-alpha         ${LORA_ALPHA} \\
  --lora-target-modules ${LORA_TARGET_MODULES} \\
  --gradient-checkpointing \\
  --seed ${SEED} \\
  --checkpoint-every-epoch \\
  2>&1 | tee ${OUTPUT_DIR}/train.log

# Pick best checkpoint by eval_auc from train.log
EPOCH_STR=\$(python3 -c "
import re, sys
lines = open('${OUTPUT_DIR}/train.log').readlines()
best_auc, best_epoch = -1, 1
for line in lines:
    m = re.search(r'Epoch (\d+):.*eval_auc=([\d.]+)', line)
    if m:
        epoch, auc = int(m.group(1)), float(m.group(2))
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
print(f'{best_epoch:04d}')
sys.stderr.write(f'Best epoch: {best_epoch} (eval_auc={best_auc:.4f})\n')
")
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/epoch_\${EPOCH_STR}"
echo "Best checkpoint: \${CHECKPOINT_DIR}"

echo ""
echo "=== [2/2] Zero-shot transfer: scoring on SWE-bench Verified ==="
python pipelinerl/swe/scripts/offline_router/score_cot_abstention_predictor.py \\
  --checkpoint-dir \${CHECKPOINT_DIR} \\
  --train-config   ${OUTPUT_DIR}/train_config.json \\
  --trajectories   ${EVAL_TRAJECTORIES_JSONL} \\
  --real-labels-jsonl ${VERIFIED_REAL_LABELS_JSONL} \\
  --output-path    ${OUTPUT_DIR}/verified_transfer_predictions.jsonl \\
  2>&1 | tee ${OUTPUT_DIR}/score.log

echo "[done] Output: ${OUTPUT_DIR}"
SCRIPT_EOF
chmod +x "${RUNNER}"

echo "=== Submitting SWE-Smith→Verified transfer job: ${JOB_NAME} ==="
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
echo "Output dir:   ${OUTPUT_DIR}"
echo "Train log:    ${OUTPUT_DIR}/train.log"
echo "Transfer AUC: tail of ${OUTPUT_DIR}/score.log"
echo "Predictions:  ${OUTPUT_DIR}/verified_transfer_predictions.jsonl"

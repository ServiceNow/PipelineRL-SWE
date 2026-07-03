#!/usr/bin/env bash
# Inner loop for sample efficiency sweep.
# All config comes from environment variables set by the launcher.
set -euo pipefail

IFS=',' read -r -a BUDGETS <<< "${SWEEP_BUDGETS}"
for BUDGET in "${BUDGETS[@]}"; do
  BUDGET=$(echo "${BUDGET}" | xargs)
  LABEL="random_${BUDGET}_seed${SEED}"
  DATASET_DIR="${OUTPUT_ROOT}/datasets/${LABEL}"
  TRAIN_DIR="${OUTPUT_ROOT}/${LABEL}/train_finetune_${NUM_EPOCHS}epoch"
  SCORE_DIR="${OUTPUT_ROOT}/${LABEL}/scores_eval286"
  mkdir -p "${DATASET_DIR}" "${TRAIN_DIR}" "${SCORE_DIR}"

  echo "=== Materializing ${BUDGET} real-label tasks ==="
  python pipelinerl/swe/scripts/offline_router/materialize_active_real_label_verifier_subset.py \
    --source-dataset-dir "${REAL_SOURCE_DATASET_DIR}" \
    --output-dir "${DATASET_DIR}" \
    --strategy random \
    --budget-instances "${BUDGET}" \
    --seed "${SEED}" \
    2>&1 | tee -a "${TRAIN_DIR}/materialize.out"

  echo "=== Fine-tuning from proxy checkpoint (N=${BUDGET}) ==="
  python -m accelerate.commands.launch \
    --multi_gpu \
    --mixed_precision "${MIXED_PRECISION}" \
    --num_processes "${TRAIN_NPROC}" \
    --config_file "conf/accelerate/${ACCELERATE_CONFIG}.yaml" \
    pipelinerl/swe/scripts/offline_router/train_qwen_embedding_cascade_baseline.py \
    --dataset-dir "${DATASET_DIR}" \
    --output-dir "${TRAIN_DIR}" \
    --model-name "${MODEL_NAME}" \
    --cascade-order "${CASCADE_ORDER}" \
    --max-seq-length "${MAX_SEQ_LENGTH}" \
    --num-epochs "${NUM_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --seed "${SEED}" \
    --dropout "${DROPOUT}" \
    --mlp-hidden-size "${MLP_HIDDEN_SIZE}" \
    --torch-dtype "${TORCH_DTYPE}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --no-encoder-frozen \
    --use-lora \
    --lora-r "${LORA_R}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --lora-target-modules "${LORA_TARGET_MODULES}" \
    --gradient-checkpointing \
    --max-threshold-candidates "${MAX_THRESHOLD_CANDIDATES}" \
    --loss-type "${LOSS_TYPE}" \
    --utility-lambdas "${UTILITY_LAMBDAS}" \
    --epoch-report-every 1 \
    --checkpoint-every-epoch \
    --init-from-model-checkpoint "${INIT_FROM_MODEL_CHECKPOINT}" \
    --save-model \
    2>&1 | tee -a "${TRAIN_DIR}/launch.out"

  echo "=== Scoring 286-task eval set (N=${BUDGET}) ==="
  python pipelinerl/swe/scripts/offline_router/score_qwen_embedding_cascade_verifier.py \
    --dataset-dir "${REAL_SOURCE_DATASET_DIR}" \
    --checkpoint-dir "${TRAIN_DIR}" \
    --output-dir "${SCORE_DIR}" \
    --split eval \
    --route-order "${CASCADE_ORDER}" \
    --max-seq-length "${MAX_SEQ_LENGTH}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --loss-type "${LOSS_TYPE}" \
    --device cuda \
    2>&1 | tee -a "${SCORE_DIR}/launch.out"

  echo "=== Done N=${BUDGET} ==="
done

echo "Sample efficiency sweep complete: ${OUTPUT_ROOT}" \
  2>&1 | tee -a "${OUTPUT_ROOT}/launch.out"

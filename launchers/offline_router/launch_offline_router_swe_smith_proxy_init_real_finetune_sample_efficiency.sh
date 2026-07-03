#!/usr/bin/env bash
# Sample efficiency sweep: proxy-init cascade verifier fine-tuned with N real labels.
# For each N in SWEEP_BUDGETS, materializes a random real-label subset, fine-tunes
# from the proxy checkpoint, then scores the 286-task eval set so we can compute
# abstention AUC vs N.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}

JOB_NAME=${JOB_NAME:-offline_router_swe_smith_proxy_init_real_finetune_sample_efficiency}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-Embedding-8B}
REAL_SOURCE_DATASET_DIR=${REAL_SOURCE_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}
INIT_FROM_MODEL_CHECKPOINT=${INIT_FROM_MODEL_CHECKPOINT:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_trace_expanded_4route_qwen3_embedding_8b_lora_proxy_verifier_soft_bce_r32_qkvo_mlp_5epoch_1781684950/train_qwen3_embedding_8b_lora_proxy_verifier_soft_bce_r32_qkvo_mlp_5epoch/checkpoints/epoch_0004}

# Comma-separated list of real-label budgets to sweep over
SWEEP_BUDGETS=${SWEEP_BUDGETS:-50,100,200,500}

JOB_NPROC=${JOB_NPROC:-4}
TRAIN_NPROC=${TRAIN_NPROC:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-base_mp}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-24000}
NUM_EPOCHS=${NUM_EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
LR=${LR:-1.0e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.06}
SEED=${SEED:-17}
CASCADE_ORDER=${CASCADE_ORDER:-0,1,2,3}
LOSS_TYPE=${LOSS_TYPE:-soft_bce}
UTILITY_LAMBDAS=${UTILITY_LAMBDAS:-0.0,1.0e-5,2.0e-5,5.0e-5,1.0e-4,2.0e-4}
MAX_THRESHOLD_CANDIDATES=${MAX_THRESHOLD_CANDIDATES:-21}
MLP_HIDDEN_SIZE=${MLP_HIDDEN_SIZE:-1024}
DROPOUT=${DROPOUT:-0.1}
TORCH_DTYPE=${TORCH_DTYPE:-bf16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}

if [[ ! -f "${REAL_SOURCE_DATASET_DIR}/metadata.json" ]]; then
  echo "Missing real-label dataset: ${REAL_SOURCE_DATASET_DIR}/metadata.json" >&2
  exit 1
fi
if [[ ! -f "${INIT_FROM_MODEL_CHECKPOINT}/model.safetensors" ]]; then
  echo "Missing proxy init checkpoint: ${INIT_FROM_MODEL_CHECKPOINT}/model.safetensors" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${JOB_NPROC} \
  COMMAND="cd ${REPO_ROOT}; set -euo pipefail; \
    SWEEP_BUDGETS='${SWEEP_BUDGETS}'; \
    IFS=',' read -r -a BUDGETS <<< \"\${SWEEP_BUDGETS}\"; \
    for BUDGET in \"\${BUDGETS[@]}\"; do \
      BUDGET=\$(echo \"\${BUDGET}\" | xargs); \
      LABEL=\"random_\${BUDGET}_seed${SEED}\"; \
      DATASET_DIR=\"${OUTPUT_ROOT}/datasets/\${LABEL}\"; \
      TRAIN_DIR=\"${OUTPUT_ROOT}/\${LABEL}/train_finetune_${NUM_EPOCHS}epoch\"; \
      SCORE_DIR=\"${OUTPUT_ROOT}/\${LABEL}/scores_eval286\"; \
      mkdir -p \"\${DATASET_DIR}\" \"\${TRAIN_DIR}\" \"\${SCORE_DIR}\"; \
      echo \"=== Materializing \${BUDGET} real-label tasks ===\"; \
      python pipelinerl/swe/scripts/offline_router/materialize_active_real_label_verifier_subset.py \
        --source-dataset-dir ${REAL_SOURCE_DATASET_DIR} \
        --output-dir \"\${DATASET_DIR}\" \
        --strategy random \
        --budget-instances \"\${BUDGET}\" \
        --seed ${SEED} \
        2>&1 | tee -a \"\${TRAIN_DIR}/materialize.out\"; \
      echo \"=== Fine-tuning from proxy checkpoint (N=\${BUDGET}) ===\"; \
      python -m accelerate.commands.launch \
        --multi_gpu \
        --mixed_precision ${MIXED_PRECISION} \
        --num_processes ${TRAIN_NPROC} \
        --config_file conf/accelerate/${ACCELERATE_CONFIG}.yaml \
        pipelinerl/swe/scripts/offline_router/train_qwen_embedding_cascade_baseline.py \
        --dataset-dir \"\${DATASET_DIR}\" \
        --output-dir \"\${TRAIN_DIR}\" \
        --model-name ${MODEL_NAME} \
        --cascade-order ${CASCADE_ORDER} \
        --max-seq-length ${MAX_SEQ_LENGTH} \
        --num-epochs ${NUM_EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --eval-batch-size ${EVAL_BATCH_SIZE} \
        --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
        --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --warmup-ratio ${WARMUP_RATIO} \
        --seed ${SEED} \
        --dropout ${DROPOUT} \
        --mlp-hidden-size ${MLP_HIDDEN_SIZE} \
        --torch-dtype ${TORCH_DTYPE} \
        --attn-implementation ${ATTN_IMPLEMENTATION} \
        --no-encoder-frozen \
        --use-lora \
        --lora-r ${LORA_R} \
        --lora-alpha ${LORA_ALPHA} \
        --lora-dropout ${LORA_DROPOUT} \
        --lora-target-modules ${LORA_TARGET_MODULES} \
        --gradient-checkpointing \
        --max-threshold-candidates ${MAX_THRESHOLD_CANDIDATES} \
        --loss-type ${LOSS_TYPE} \
        --utility-lambdas ${UTILITY_LAMBDAS} \
        --epoch-report-every 1 \
        --checkpoint-every-epoch \
        --init-from-model-checkpoint ${INIT_FROM_MODEL_CHECKPOINT} \
        --save-model \
        2>&1 | tee -a \"\${TRAIN_DIR}/launch.out\"; \
      echo \"=== Scoring 286-task eval set (N=\${BUDGET}) ===\"; \
      python pipelinerl/swe/scripts/offline_router/score_qwen_embedding_cascade_verifier.py \
        --dataset-dir ${REAL_SOURCE_DATASET_DIR} \
        --checkpoint-dir \"\${TRAIN_DIR}\" \
        --output-dir \"\${SCORE_DIR}\" \
        --split eval \
        --route-order ${CASCADE_ORDER} \
        --max-seq-length ${MAX_SEQ_LENGTH} \
        --batch-size ${EVAL_BATCH_SIZE} \
        --loss-type ${LOSS_TYPE} \
        --device cuda \
        2>&1 | tee -a \"\${SCORE_DIR}/launch.out\"; \
      echo \"=== Done N=\${BUDGET} ===\"; \
    done; \
    echo \"Sample efficiency sweep complete: ${OUTPUT_ROOT}\" \
    2>&1 | tee -a ${OUTPUT_ROOT}/launch.out"

echo "Submitted: ${OUTPUT_ROOT}"
echo "Sweeping N = ${SWEEP_BUDGETS}"

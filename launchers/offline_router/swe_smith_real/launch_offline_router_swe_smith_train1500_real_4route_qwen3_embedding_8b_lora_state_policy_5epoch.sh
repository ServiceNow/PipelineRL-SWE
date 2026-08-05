#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-Embedding-8B}
JOB_NAME=${JOB_NAME:-offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_state_policy_5epoch}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
REAL_LABEL_DATASET_DIR=${REAL_LABEL_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_qwen3_embedding_8b_lora_state_policy_5epoch}

JOB_NPROC=${JOB_NPROC:-4}
TRAIN_NPROC=${TRAIN_NPROC:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-base_mp}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-24000}
TARGET_ROUTE_IDXS=${TARGET_ROUTE_IDXS:-}
ATTEMPTED_STATE_MODE=${ATTEMPTED_STATE_MODE:-single}
INCLUDE_BARE_STATE=${INCLUDE_BARE_STATE:-true}
INCLUDE_COSTS_IN_PROMPT=${INCLUDE_COSTS_IN_PROMPT:-true}
ROUTE_OUTPUT_COST_WEIGHTS=${ROUTE_OUTPUT_COST_WEIGHTS:-2.78e-7,1.299e-6,4.64e-6,1.113e-5}
UTILITY_LAMBDAS=${UTILITY_LAMBDAS:-0,5,10,15,20,25,30,35,40,50,75,100,150,200}
SCOUT_ROUTE_IDX=${SCOUT_ROUTE_IDX:-0}
MAX_POLICY_STEPS=${MAX_POLICY_STEPS:-2}
NUM_EPOCHS=${NUM_EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
LR=${LR:-2.0e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.06}
MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS:-0}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-0}
SEED=${SEED:-17}
DELTA_AUX_WEIGHT=${DELTA_AUX_WEIGHT:-0.0}
DELTA_AUX_HUBER_DELTA=${DELTA_AUX_HUBER_DELTA:-0.0}
DECISION_AUX_WEIGHT=${DECISION_AUX_WEIGHT:-0.0}
DECISION_AUX_LAMBDAS=${DECISION_AUX_LAMBDAS:-${UTILITY_LAMBDAS}}
DECISION_AUX_LAMBDA_SAMPLING=${DECISION_AUX_LAMBDA_SAMPLING:-none}
DECISION_AUX_LAMBDA_SAMPLE_COUNT=${DECISION_AUX_LAMBDA_SAMPLE_COUNT:-1}
DECISION_AUX_LAMBDA_MIN=${DECISION_AUX_LAMBDA_MIN:-0}
DECISION_AUX_LAMBDA_MAX=${DECISION_AUX_LAMBDA_MAX:-0}
DECISION_AUX_TEMPERATURE=${DECISION_AUX_TEMPERATURE:-0.1}
DECISION_AUX_COST_MODE=${DECISION_AUX_COST_MODE:-fixed_train_mean}
DECISION_AUX_STOP_TIE_BONUS=${DECISION_AUX_STOP_TIE_BONUS:-1.0e-4}
DECISION_AUX_BARE_OUT_ACTION=${DECISION_AUX_BARE_OUT_ACTION:-false}
DECISION_AUX_REGRET_WEIGHT_MODE=${DECISION_AUX_REGRET_WEIGHT_MODE:-none}
DECISION_AUX_REGRET_WEIGHT_SCALE=${DECISION_AUX_REGRET_WEIGHT_SCALE:-0.0}
DECISION_AUX_REGRET_WEIGHT_POWER=${DECISION_AUX_REGRET_WEIGHT_POWER:-1.0}
DECISION_AUX_REGRET_WEIGHT_MIN=${DECISION_AUX_REGRET_WEIGHT_MIN:-1.0}
DECISION_AUX_REGRET_WEIGHT_MAX=${DECISION_AUX_REGRET_WEIGHT_MAX:-8.0}
POLICY_BARE_OUT_ACTION=${POLICY_BARE_OUT_ACTION:-false}
SAMPLE_WEIGHTING=${SAMPLE_WEIGHTING:-uniform}
REGRET_LAMBDAS=${REGRET_LAMBDAS:-${UTILITY_LAMBDAS}}
REGRET_DEFAULT_ROUTE_IDX=${REGRET_DEFAULT_ROUTE_IDX:-0}
REGRET_WEIGHT_SCALE=${REGRET_WEIGHT_SCALE:-4.0}
REGRET_WEIGHT_POWER=${REGRET_WEIGHT_POWER:-1.0}
REGRET_WEIGHT_MIN=${REGRET_WEIGHT_MIN:-1.0}
REGRET_WEIGHT_MAX=${REGRET_WEIGHT_MAX:-8.0}
NORMALIZE_SAMPLE_WEIGHTS=${NORMALIZE_SAMPLE_WEIGHTS:-true}
MLP_HIDDEN_SIZE=${MLP_HIDDEN_SIZE:-1024}
DROPOUT=${DROPOUT:-0.1}
TORCH_DTYPE=${TORCH_DTYPE:-bf16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}
CHECKPOINT_EVERY_EPOCH=${CHECKPOINT_EVERY_EPOCH:-true}
EPOCH_REPORT_EVERY=${EPOCH_REPORT_EVERY:-1}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
SAVE_MODEL=${SAVE_MODEL:-false}

TRAIN_CMD="python pipelinerl/swe/scripts/offline_router/train_qwen_embedding_state_policy.py"
if [[ "${TRAIN_NPROC}" -gt 1 ]]; then
  TRAIN_CMD="python -m accelerate.commands.launch \
    --multi_gpu \
    --mixed_precision ${MIXED_PRECISION} \
    --num_processes ${TRAIN_NPROC} \
    --config_file conf/accelerate/${ACCELERATE_CONFIG}.yaml \
    pipelinerl/swe/scripts/offline_router/train_qwen_embedding_state_policy.py"
fi

SAVE_MODEL_ARG=""
if [[ "${SAVE_MODEL}" == "true" ]]; then
  SAVE_MODEL_ARG="--save-model"
fi

CHECKPOINT_ARG=""
if [[ "${CHECKPOINT_EVERY_EPOCH}" == "true" ]]; then
  CHECKPOINT_ARG="--checkpoint-every-epoch"
fi

RESUME_ARG=""
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARG="--resume-from-checkpoint ${RESUME_FROM_CHECKPOINT}"
fi

ATTN_ARG=""
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  ATTN_ARG="--attn-implementation ${ATTN_IMPLEMENTATION}"
fi

TARGET_ROUTE_ARG=""
if [[ -n "${TARGET_ROUTE_IDXS}" ]]; then
  TARGET_ROUTE_ARG="--target-route-idxs ${TARGET_ROUTE_IDXS}"
fi

BARE_STATE_ARG="--include-bare-state"
if [[ "${INCLUDE_BARE_STATE}" != "true" ]]; then
  BARE_STATE_ARG="--no-include-bare-state"
fi

COSTS_IN_PROMPT_ARG="--include-costs-in-prompt"
if [[ "${INCLUDE_COSTS_IN_PROMPT}" != "true" ]]; then
  COSTS_IN_PROMPT_ARG="--no-include-costs-in-prompt"
fi

DECISION_AUX_BARE_OUT_ARG=""
if [[ "${DECISION_AUX_BARE_OUT_ACTION}" == "true" ]]; then
  DECISION_AUX_BARE_OUT_ARG="--decision-aux-bare-out-action"
fi

POLICY_BARE_OUT_ARG=""
if [[ "${POLICY_BARE_OUT_ACTION}" == "true" ]]; then
  POLICY_BARE_OUT_ARG="--policy-bare-out-action"
fi

NORMALIZE_SAMPLE_WEIGHTS_ARG="--normalize-sample-weights"
if [[ "${NORMALIZE_SAMPLE_WEIGHTS}" != "true" ]]; then
  NORMALIZE_SAMPLE_WEIGHTS_ARG="--no-normalize-sample-weights"
fi

if [[ ! -f "${REAL_LABEL_DATASET_DIR}/metadata.json" ]]; then
  echo "Missing real-label router dataset: ${REAL_LABEL_DATASET_DIR}/metadata.json" >&2
  echo "Generate it once with pipelinerl/swe/scripts/offline_router/materialize_real_label_router_dataset.py, or set REAL_LABEL_DATASET_DIR." >&2
  exit 1
fi

make job \
  JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=${JOB_NPROC} \
  COMMAND="cd ${REPO_ROOT}; mkdir -p ${OUTPUT_ROOT} ${TRAIN_OUTPUT_DIR}; set -o pipefail; \
    ${TRAIN_CMD} \
      --dataset-dir ${REAL_LABEL_DATASET_DIR} \
      --output-dir ${TRAIN_OUTPUT_DIR} \
      --model-name ${MODEL_NAME} \
      ${TARGET_ROUTE_ARG} \
      --max-seq-length ${MAX_SEQ_LENGTH} \
      --attempted-state-mode ${ATTEMPTED_STATE_MODE} \
      ${BARE_STATE_ARG} \
      ${COSTS_IN_PROMPT_ARG} \
      --route-output-cost-weights ${ROUTE_OUTPUT_COST_WEIGHTS} \
      --utility-lambdas ${UTILITY_LAMBDAS} \
      --scout-route-idx ${SCOUT_ROUTE_IDX} \
      --max-policy-steps ${MAX_POLICY_STEPS} \
      --num-epochs ${NUM_EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --eval-batch-size ${EVAL_BATCH_SIZE} \
      --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
      --lr ${LR} \
      --weight-decay ${WEIGHT_DECAY} \
      --warmup-ratio ${WARMUP_RATIO} \
      --max-train-rows ${MAX_TRAIN_ROWS} \
      --max-eval-rows ${MAX_EVAL_ROWS} \
      --seed ${SEED} \
      --dropout ${DROPOUT} \
      --mlp-hidden-size ${MLP_HIDDEN_SIZE} \
      --torch-dtype ${TORCH_DTYPE} \
      ${ATTN_ARG} \
      --no-encoder-frozen \
      --use-lora \
      --lora-r ${LORA_R} \
      --lora-alpha ${LORA_ALPHA} \
      --lora-dropout ${LORA_DROPOUT} \
      --lora-target-modules ${LORA_TARGET_MODULES} \
      --gradient-checkpointing \
      --delta-aux-weight ${DELTA_AUX_WEIGHT} \
      --delta-aux-huber-delta ${DELTA_AUX_HUBER_DELTA} \
      --decision-aux-weight ${DECISION_AUX_WEIGHT} \
      --decision-aux-lambdas ${DECISION_AUX_LAMBDAS} \
      --decision-aux-lambda-sampling ${DECISION_AUX_LAMBDA_SAMPLING} \
      --decision-aux-lambda-sample-count ${DECISION_AUX_LAMBDA_SAMPLE_COUNT} \
      --decision-aux-lambda-min ${DECISION_AUX_LAMBDA_MIN} \
      --decision-aux-lambda-max ${DECISION_AUX_LAMBDA_MAX} \
      --decision-aux-temperature ${DECISION_AUX_TEMPERATURE} \
      --decision-aux-cost-mode ${DECISION_AUX_COST_MODE} \
      --decision-aux-stop-tie-bonus ${DECISION_AUX_STOP_TIE_BONUS} \
      ${DECISION_AUX_BARE_OUT_ARG} \
      --decision-aux-regret-weight-mode ${DECISION_AUX_REGRET_WEIGHT_MODE} \
      --decision-aux-regret-weight-scale ${DECISION_AUX_REGRET_WEIGHT_SCALE} \
      --decision-aux-regret-weight-power ${DECISION_AUX_REGRET_WEIGHT_POWER} \
      --decision-aux-regret-weight-min ${DECISION_AUX_REGRET_WEIGHT_MIN} \
      --decision-aux-regret-weight-max ${DECISION_AUX_REGRET_WEIGHT_MAX} \
      ${POLICY_BARE_OUT_ARG} \
      --sample-weighting ${SAMPLE_WEIGHTING} \
      --regret-lambdas ${REGRET_LAMBDAS} \
      --regret-default-route-idx ${REGRET_DEFAULT_ROUTE_IDX} \
      --regret-weight-scale ${REGRET_WEIGHT_SCALE} \
      --regret-weight-power ${REGRET_WEIGHT_POWER} \
      --regret-weight-min ${REGRET_WEIGHT_MIN} \
      --regret-weight-max ${REGRET_WEIGHT_MAX} \
      ${NORMALIZE_SAMPLE_WEIGHTS_ARG} \
      --epoch-report-every ${EPOCH_REPORT_EVERY} \
      ${CHECKPOINT_ARG} \
      ${RESUME_ARG} \
      ${SAVE_MODEL_ARG} \
      2>&1 | tee -a ${TRAIN_OUTPUT_DIR}/launch.out"

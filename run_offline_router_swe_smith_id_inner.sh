#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:?RUN_NAME must be set}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT must be set}

RUN_COLLECT=${RUN_COLLECT:-1}
RUN_TRAIN=${RUN_TRAIN:-1}
CLEANUP_VLLM_BEFORE_TRAIN=${CLEANUP_VLLM_BEFORE_TRAIN:-1}
POST_COLLECT_CLEANUP_SLEEP_SECS=${POST_COLLECT_CLEANUP_SLEEP_SECS:-30}

COLLECT_OUTPUT_DIR=${COLLECT_OUTPUT_DIR:-${OUTPUT_ROOT}/collect}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/train_text_lora_random}

PRIMARY_MODEL_PATH=${PRIMARY_MODEL_PATH:-/mnt/llmd/results/exps/aristides/reason/swe_smith_policy_conditioned_no_devstral_1773812579/finetune/current}
PRIMARY_TOKENIZER_NAME=${PRIMARY_TOKENIZER_NAME:-Qwen/Qwen2.5-Coder-7B-Instruct}
PRIMARY_SERVED_MODEL_NAME=${PRIMARY_SERVED_MODEL_NAME:-primary_model}

TRAIN_DATASET_NAMES=${TRAIN_DATASET_NAMES:-swe_smith_train}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-/mnt/llmd/data/swe_smith/ds_train}
EVAL_DATASET_NAMES=${EVAL_DATASET_NAMES:-swe_smith_test}
EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-/mnt/llmd/data/swe_smith/ds_test}

COLLECT_TRAIN=${COLLECT_TRAIN:-true}
COLLECT_EVAL=${COLLECT_EVAL:-true}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-4096}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-500}
MAX_CONCURRENT_PROBLEMS=${MAX_CONCURRENT_PROBLEMS:-8}
SHARD_SIZE=${SHARD_SIZE:-64}
COLLECT_EXTRA_ARGS=${COLLECT_EXTRA_ARGS:-}

TRAIN_NPROC=${TRAIN_NPROC:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-base_mp}
ROUTER_MODEL_PATH=${ROUTER_MODEL_PATH:-${PRIMARY_MODEL_PATH}}
ROUTER_SUPERVISION_MODE=${ROUTER_SUPERVISION_MODE:-text_reward_vector}
ROUTER_TRAIN_MODE=${ROUTER_TRAIN_MODE:-full_backbone}
ROUTER_TRAIN_SAMPLING=${ROUTER_TRAIN_SAMPLING:-random}
MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS:-4096}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-500}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-32000}
NUM_EPOCHS=${NUM_EPOCHS:-1}
SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-false}
TEXT_REWARD_DEBUG_STEP_LOGGING=${TEXT_REWARD_DEBUG_STEP_LOGGING:-false}
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:-}

mkdir -p "${OUTPUT_ROOT}" "${COLLECT_OUTPUT_DIR}" "${TRAIN_OUTPUT_DIR}"

if [[ "${RUN_COLLECT}" == "1" ]]; then
  echo "=== Collecting SWE-smith offline-router traces into ${COLLECT_OUTPUT_DIR} ==="
  COLLECT_ARGS=(
    output_dir="${COLLECT_OUTPUT_DIR}"
    offline_router.primary_model.model_path="${PRIMARY_MODEL_PATH}"
    offline_router.primary_model.tokenizer_name="${PRIMARY_TOKENIZER_NAME}"
    offline_router.primary_model.served_model_name="${PRIMARY_SERVED_MODEL_NAME}"
    offline_router.primary_model.model_name="${PRIMARY_SERVED_MODEL_NAME}"
    "offline_router.dataset.train_dataset_names=[${TRAIN_DATASET_NAMES}]"
    offline_router.dataset.train_dataset_path="${TRAIN_DATASET_PATH}"
    "offline_router.dataset.eval_dataset_names=[${EVAL_DATASET_NAMES}]"
    offline_router.dataset.eval_dataset_path="${EVAL_DATASET_PATH}"
    offline_router.collection.collect_train="${COLLECT_TRAIN}"
    offline_router.collection.collect_eval="${COLLECT_EVAL}"
    offline_router.dataset.train_max_samples="${TRAIN_MAX_SAMPLES}"
    offline_router.dataset.eval_max_samples="${EVAL_MAX_SAMPLES}"
    offline_router.collection.max_concurrent_problems="${MAX_CONCURRENT_PROBLEMS}"
    offline_router.collection.shard_size="${SHARD_SIZE}"
  )
  if [[ -n "${COLLECT_EXTRA_ARGS}" ]]; then
    COLLECT_ARGS+=(${COLLECT_EXTRA_ARGS})
  fi
  python -m pipelinerl.swe.scripts.offline_router.run_collection_job "${COLLECT_ARGS[@]}" \
    2>&1 | tee -a "${COLLECT_OUTPUT_DIR}/launch.out"
else
  echo "=== Skipping collection; using existing dataset at ${COLLECT_OUTPUT_DIR} ==="
fi

echo "=== Summarizing collected route distribution ==="
python pipelinerl/swe/scripts/offline_router/summarize_collected_dataset.py \
  --dataset-dir "${COLLECT_OUTPUT_DIR}" \
  --output-json "${COLLECT_OUTPUT_DIR}/route_distribution_summary.json" \
  2>&1 | tee -a "${COLLECT_OUTPUT_DIR}/route_distribution_summary.out"

if [[ "${RUN_TRAIN}" != "1" ]]; then
  echo "RUN_TRAIN=${RUN_TRAIN}; stopping after collection and summary."
  exit 0
fi

if [[ "${CLEANUP_VLLM_BEFORE_TRAIN}" == "1" ]]; then
  echo "=== Cleaning up any vLLM server processes before router training ==="
  VLLM_CLEANUP_PATTERNS=(
    "vllm.entrypoints.openai.api_server"
    "vllm.v1.engine"
    "vllm_worker"
  )
  for PATTERN in "${VLLM_CLEANUP_PATTERNS[@]}"; do
    mapfile -t VLLM_PIDS < <(pgrep -u "$(id -u)" -f "${PATTERN}" || true)
    for PID in "${VLLM_PIDS[@]}"; do
      if [[ "${PID}" == "$$" || "${PID}" == "${BASHPID}" || "${PID}" == "${PPID}" ]]; then
        continue
      fi
      echo "Stopping leftover vLLM process ${PID} matching ${PATTERN}"
      kill "${PID}" 2>/dev/null || true
    done
  done
  if [[ "${POST_COLLECT_CLEANUP_SLEEP_SECS}" -gt 0 ]]; then
    sleep "${POST_COLLECT_CLEANUP_SLEEP_SECS}"
  fi
fi

TRAIN_CMD=(python -m pipelinerl.swe.scripts.offline_router.train_router_offline)
if [[ "${TRAIN_NPROC}" -gt 1 ]]; then
  TRAIN_CMD=(
    python -m accelerate.commands.launch
    --multi_gpu
    --mixed_precision "${MIXED_PRECISION}"
    --num_processes "${TRAIN_NPROC}"
    --config_file "conf/accelerate/${ACCELERATE_CONFIG}.yaml"
    pipelinerl/swe/scripts/offline_router/train_router_offline.py
  )
fi

TRAIN_ARGS=(
  output_dir="${TRAIN_OUTPUT_DIR}"
  offline_router.train.dataset_dir="${COLLECT_OUTPUT_DIR}"
  offline_router.train.model_path="${ROUTER_MODEL_PATH}"
  offline_router.train.supervision_mode="${ROUTER_SUPERVISION_MODE}"
  offline_router.train.mode="${ROUTER_TRAIN_MODE}"
  offline_router.train.train_sampling_strategy="${ROUTER_TRAIN_SAMPLING}"
  offline_router.train.max_train_rows="${MAX_TRAIN_ROWS}"
  offline_router.train.max_eval_rows="${MAX_EVAL_ROWS}"
  offline_router.train.max_seq_length="${MAX_SEQ_LENGTH}"
  offline_router.train.num_epochs="${NUM_EPOCHS}"
  offline_router.train.save_checkpoints="${SAVE_CHECKPOINTS}"
  offline_router.train.text_reward.debug_step_logging="${TEXT_REWARD_DEBUG_STEP_LOGGING}"
)
if [[ -n "${TRAIN_EXTRA_ARGS}" ]]; then
  TRAIN_ARGS+=(${TRAIN_EXTRA_ARGS})
fi

echo "=== Training SWE-smith in-distribution text-LoRA router into ${TRAIN_OUTPUT_DIR} ==="
"${TRAIN_CMD[@]}" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${TRAIN_OUTPUT_DIR}/launch.out"

echo "Completed ${RUN_NAME}"
echo "Collection: ${COLLECT_OUTPUT_DIR}"
echo "Route distribution summary: ${COLLECT_OUTPUT_DIR}/route_distribution_summary.json"
echo "Training: ${TRAIN_OUTPUT_DIR}"

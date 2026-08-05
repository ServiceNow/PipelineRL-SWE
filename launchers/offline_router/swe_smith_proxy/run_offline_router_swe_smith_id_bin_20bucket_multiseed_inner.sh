#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

RUN_NAME=${RUN_NAME:?RUN_NAME must be set}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT must be set}

SOURCE_RUN_DIR=${SOURCE_RUN_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_id_1776113427}
COLLECT_OUTPUT_DIR=${COLLECT_OUTPUT_DIR:-${SOURCE_RUN_DIR}/collect}

ROUTER_SEEDS=${ROUTER_SEEDS:-42 43 44}
TRAIN_NPROC=${TRAIN_NPROC:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-base_mp}
ROUTER_MODEL_PATH=${ROUTER_MODEL_PATH:-/mnt/llmd/results/exps/aristides/reason/swe_smith_policy_conditioned_no_devstral_1773812579/finetune/current}
ROUTER_TRAIN_MODE=${ROUTER_TRAIN_MODE:-full_backbone}
ROUTER_TRAIN_SAMPLING=${ROUTER_TRAIN_SAMPLING:-random}
MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS:-4096}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-500}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-32000}
NUM_EPOCHS=${NUM_EPOCHS:-1}
SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-false}
TEXT_REWARD_DEBUG_STEP_LOGGING=${TEXT_REWARD_DEBUG_STEP_LOGGING:-false}
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:-}
CLEANUP_VLLM_BEFORE_TRAIN=${CLEANUP_VLLM_BEFORE_TRAIN:-1}
POST_COLLECT_CLEANUP_SLEEP_SECS=${POST_COLLECT_CLEANUP_SLEEP_SECS:-30}

mkdir -p "${OUTPUT_ROOT}"

if [[ ! -f "${COLLECT_OUTPUT_DIR}/metadata.json" ]]; then
  echo "Missing offline-router collection metadata: ${COLLECT_OUTPUT_DIR}/metadata.json" >&2
  exit 1
fi

echo "=== Multi-seed bin20 router training ==="
echo "Run name: ${RUN_NAME}"
echo "Collection: ${COLLECT_OUTPUT_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Seeds: ${ROUTER_SEEDS}"

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

RUN_DIRS=()
for SEED in ${ROUTER_SEEDS}; do
  TRAIN_OUTPUT_DIR="${OUTPUT_ROOT}/seed_${SEED}/train_text_lora_bin_expectation_20bucket_random"
  mkdir -p "${TRAIN_OUTPUT_DIR}"
  RUN_DIRS+=("${TRAIN_OUTPUT_DIR}")

  TRAIN_ARGS=(
    seed="${SEED}"
    output_dir="${TRAIN_OUTPUT_DIR}"
    offline_router.train.dataset_dir="${COLLECT_OUTPUT_DIR}"
    offline_router.train.model_path="${ROUTER_MODEL_PATH}"
    offline_router.train.supervision_mode="text_reward_bin"
    offline_router.train.mode="${ROUTER_TRAIN_MODE}"
    offline_router.train.train_sampling_strategy="${ROUTER_TRAIN_SAMPLING}"
    offline_router.train.max_train_rows="${MAX_TRAIN_ROWS}"
    offline_router.train.max_eval_rows="${MAX_EVAL_ROWS}"
    offline_router.train.max_seq_length="${MAX_SEQ_LENGTH}"
    offline_router.train.num_epochs="${NUM_EPOCHS}"
    offline_router.train.save_checkpoints="${SAVE_CHECKPOINTS}"
    offline_router.train.text_reward.debug_step_logging="${TEXT_REWARD_DEBUG_STEP_LOGGING}"
    offline_router.train.text_reward.bin_count=21
    offline_router.train.text_reward.bin_value_order=ascending
  )
  if [[ -n "${TRAIN_EXTRA_ARGS}" ]]; then
    TRAIN_ARGS+=(${TRAIN_EXTRA_ARGS})
  fi

  echo "=== Training bin20 seed=${SEED} into ${TRAIN_OUTPUT_DIR} ==="
  "${TRAIN_CMD[@]}" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${TRAIN_OUTPUT_DIR}/launch.out"
done

REPORT_ARGS=(
  python -m pipelinerl.swe.scripts.offline_router.report_router_calibration
  --output-dir "${OUTPUT_ROOT}/calibration_report"
  --bucket-count 20
  --near-zero-threshold 0.05
  --lambdas 0.0
)
for RUN_DIR in "${RUN_DIRS[@]}"; do
  SEED_NAME="$(basename "$(dirname "${RUN_DIR}")")"
  REPORT_ARGS+=(--run "${SEED_NAME}=${RUN_DIR}")
done

echo "=== Writing multi-seed calibration report ==="
"${REPORT_ARGS[@]}" 2>&1 | tee -a "${OUTPUT_ROOT}/calibration_report.out"

echo "Completed ${RUN_NAME}"
echo "Collection: ${COLLECT_OUTPUT_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Report: ${OUTPUT_ROOT}/calibration_report"

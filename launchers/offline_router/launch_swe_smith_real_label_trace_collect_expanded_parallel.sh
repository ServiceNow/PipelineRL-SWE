#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SEED=${SEED:-123}
# These are additional IDs after exclusions, not total IDs including old batches.
TRAIN_N=${TRAIN_N:-3000}
EVAL_N=${EVAL_N:-1000}
RUN_TRAIN_TRACES=${RUN_TRAIN_TRACES:-1}
RUN_EVAL_TRACES=${RUN_EVAL_TRACES:-1}
ID_ROOT=${ID_ROOT:-/mnt/llmd/results/exps/aristides/reason/swe_smith_real_label_trace_ids_expanded_${TIMESTAMP}}
OUTPUT_BASE=${OUTPUT_BASE:-/mnt/llmd/results/exps/aristides/reason}
SWE_SMITH_DATA_ROOT=${SWE_SMITH_DATA_ROOT:-/mnt/llmd/data/swe_smith_bugged_context}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-${SWE_SMITH_DATA_ROOT}/ds_train}
EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-${SWE_SMITH_DATA_ROOT}/ds_test}
TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-swe_smith_train_bugged_context}
EVAL_DATASET_NAME=${EVAL_DATASET_NAME:-swe_smith_test_bugged_context}
SUBMIT_SLEEP_SECS=${SUBMIT_SLEEP_SECS:-5}

# Avoid recollecting IDs from earlier 1.5k/500 SWE-Smith batches. Override these env vars
# if you intentionally want overlap or have a newer exclusion file.
DEFAULT_EXCLUDE_TRAIN="/mnt/llmd/results/exps/aristides/reason/swe_smith_real_label_trace_ids_1780301438/swe_smith_train_1500_ids.txt /mnt/llmd/results/exps/aristides/reason/swe_smith_real_label_trace_ids_1780461385/swe_smith_train_1500_ids.txt"
DEFAULT_EXCLUDE_EVAL="/mnt/llmd/results/exps/aristides/reason/swe_smith_real_label_trace_ids_1780301438/swe_smith_eval_500_ids.txt /mnt/llmd/results/exps/aristides/reason/swe_smith_real_label_trace_ids_1780461385/swe_smith_eval_500_ids.txt"
EXCLUDE_TRAIN_IDS_PATHS=${EXCLUDE_TRAIN_IDS_PATHS:-${DEFAULT_EXCLUDE_TRAIN}}
EXCLUDE_EVAL_IDS_PATHS=${EXCLUDE_EVAL_IDS_PATHS:-${DEFAULT_EXCLUDE_EVAL}}
EXCLUDE_IDS_PATHS=${EXCLUDE_IDS_PATHS:-}

# Filter absent default exclusion files so a fresh environment still works.
filter_existing_paths() {
  local out=()
  for path in "$@"; do
    if [[ -s "${path}" ]]; then
      out+=("${path}")
    fi
  done
  printf '%s ' "${out[@]}"
}
EXCLUDE_TRAIN_IDS_PATHS=$(filter_existing_paths ${EXCLUDE_TRAIN_IDS_PATHS})
EXCLUDE_EVAL_IDS_PATHS=$(filter_existing_paths ${EXCLUDE_EVAL_IDS_PATHS})

TRAIN_IDS=${TRAIN_IDS:-${ID_ROOT}/swe_smith_train_${TRAIN_N}_ids.txt}
EVAL_IDS=${EVAL_IDS:-${ID_ROOT}/swe_smith_eval_${EVAL_N}_ids.txt}

if [[ ! -s "${TRAIN_IDS}" || ! -s "${EVAL_IDS}" ]]; then
  TIMESTAMP="${TIMESTAMP}" SEED="${SEED}" TRAIN_N="${TRAIN_N}" EVAL_N="${EVAL_N}" ID_ROOT="${ID_ROOT}" \
  SWE_SMITH_DATA_ROOT="${SWE_SMITH_DATA_ROOT}" TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH}" EVAL_DATASET_PATH="${EVAL_DATASET_PATH}" \
  TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME}" EVAL_DATASET_NAME="${EVAL_DATASET_NAME}" \
  EXCLUDE_TRAIN_IDS_PATHS="${EXCLUDE_TRAIN_IDS_PATHS}" EXCLUDE_EVAL_IDS_PATHS="${EXCLUDE_EVAL_IDS_PATHS}" EXCLUDE_IDS_PATHS="${EXCLUDE_IDS_PATHS}" \
    bash "${SCRIPT_DIR}/prepare_swe_smith_real_label_trace_ids.sh"
fi

submit_one() {
  local split=$1
  local dataset_path=$2
  local dataset_name=$3
  local ids_path=$4
  local model=$5
  local slug=$6
  local gpu_count=$7
  local tensor_parallel_size=$8
  local cpu_mem=$9
  local max_model_len=${10}
  local max_tokens=${11}
  local temperature=${12}
  local top_p=${13}
  local top_k=${14}
  local repetition_penalty=${15}

  local job_name="offline_router_swe_smith_real_${split}_collect_${slug}_expanded"
  local output_root="${OUTPUT_BASE}/${job_name}_${TIMESTAMP}"
  echo "[submit] split=${split} model=${model} ids=${ids_path} output=${output_root} temp=${temperature} top_p=${top_p} top_k=${top_k} repetition_penalty=${repetition_penalty}"

  TIMESTAMP="${TIMESTAMP}" \
  MODEL="${model}" \
  MODEL_SLUG="${slug}" \
  JOB_NAME="${job_name}" \
  OUTPUT_ROOT="${output_root}" \
  DATASET_PATH="${dataset_path}" \
  DATASET_NAME="${dataset_name}" \
  INSTANCE_IDS_PATH="${ids_path}" \
  GPU_COUNT="${gpu_count}" \
  TENSOR_PARALLEL_SIZE="${tensor_parallel_size}" \
  CPU_MEM="${cpu_mem}" \
  MAX_MODEL_LEN="${max_model_len}" \
  MAX_TOKENS="${max_tokens}" \
  TEMPERATURE="${temperature}" \
  TOP_P="${top_p}" \
  TOP_K="${top_k}" \
  REPETITION_PENALTY="${repetition_penalty}" \
  bash "${SCRIPT_DIR}/launch_swebench_train_local_model_collect.sh"

  sleep "${SUBMIT_SLEEP_SECS}"
}

submit_split() {
  local split=$1
  local dataset_path=$2
  local dataset_name=$3
  local ids_path=$4

  # Qwen model-card defaults: temp=0.7 top_p=0.8 top_k=20; Coder also recommends repetition_penalty=1.05.
  submit_one "${split}" "${dataset_path}" "${dataset_name}" "${ids_path}" \
    "Qwen/Qwen3-4B-Instruct-2507" "qwen3_4b_instruct_2507" 1 1 256 32768 15000 0.7 0.8 20 0
  submit_one "${split}" "${dataset_path}" "${dataset_name}" "${ids_path}" \
    "openai/gpt-oss-20b" "gpt_oss_20b" 4 4 512 32768 15000 0.7 1.0 0 0
  submit_one "${split}" "${dataset_path}" "${dataset_name}" "${ids_path}" \
    "Qwen/Qwen3-Coder-30B-A3B-Instruct" "qwen3_coder_30b_a3b" 4 4 512 32768 15000 0.7 0.8 20 1.05
  submit_one "${split}" "${dataset_path}" "${dataset_name}" "${ids_path}" \
    "openai/gpt-oss-120b" "gpt_oss_120b" 8 8 1024 32768 15000 0.7 1.0 0 0
}

if [[ "${RUN_TRAIN_TRACES}" == "1" ]]; then
  submit_split "train${TRAIN_N}" "${TRAIN_DATASET_PATH}" "${TRAIN_DATASET_NAME}" "${TRAIN_IDS}"
fi

if [[ "${RUN_EVAL_TRACES}" == "1" ]]; then
  submit_split "eval${EVAL_N}" "${EVAL_DATASET_PATH}" "${EVAL_DATASET_NAME}" "${EVAL_IDS}"
fi

echo "Submitted expanded SWE-Smith real-label trace jobs."
echo "ID_ROOT=${ID_ROOT}"
echo "TRAIN_IDS=${TRAIN_IDS}"
echo "EVAL_IDS=${EVAL_IDS}"
echo "EXCLUDE_TRAIN_IDS_PATHS=${EXCLUDE_TRAIN_IDS_PATHS}"
echo "EXCLUDE_EVAL_IDS_PATHS=${EXCLUDE_EVAL_IDS_PATHS}"

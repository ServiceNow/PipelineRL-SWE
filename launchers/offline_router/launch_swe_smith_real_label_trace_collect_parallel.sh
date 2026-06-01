#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SEED=${SEED:-42}
TRAIN_N=${TRAIN_N:-1500}
EVAL_N=${EVAL_N:-500}
RUN_TRAIN_TRACES=${RUN_TRAIN_TRACES:-1}
RUN_EVAL_TRACES=${RUN_EVAL_TRACES:-1}
ID_ROOT=${ID_ROOT:-/mnt/llmd/results/exps/aristides/reason/swe_smith_real_label_trace_ids_${TIMESTAMP}}
OUTPUT_BASE=${OUTPUT_BASE:-/mnt/llmd/results/exps/aristides/reason}
SUBMIT_SLEEP_SECS=${SUBMIT_SLEEP_SECS:-5}

TRAIN_IDS=${TRAIN_IDS:-${ID_ROOT}/swe_smith_train_${TRAIN_N}_ids.txt}
EVAL_IDS=${EVAL_IDS:-${ID_ROOT}/swe_smith_eval_${EVAL_N}_ids.txt}

if [[ ! -s "${TRAIN_IDS}" || ! -s "${EVAL_IDS}" ]]; then
  TIMESTAMP="${TIMESTAMP}" SEED="${SEED}" TRAIN_N="${TRAIN_N}" EVAL_N="${EVAL_N}" ID_ROOT="${ID_ROOT}" \
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

  local job_name="offline_router_swe_smith_real_${split}_collect_${slug}"
  local output_root="${OUTPUT_BASE}/${job_name}_${TIMESTAMP}"
  echo "[submit] split=${split} model=${model} ids=${ids_path} output=${output_root}"

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
  bash "${SCRIPT_DIR}/launch_swebench_train_local_model_collect.sh"

  sleep "${SUBMIT_SLEEP_SECS}"
}

submit_split() {
  local split=$1
  local dataset_path=$2
  local dataset_name=$3
  local ids_path=$4

  submit_one "${split}" "${dataset_path}" "${dataset_name}" "${ids_path}" \
    "Qwen/Qwen3-4B-Instruct-2507" "qwen3_4b_instruct_2507" 1 1 256 32768 15000
  submit_one "${split}" "${dataset_path}" "${dataset_name}" "${ids_path}" \
    "openai/gpt-oss-20b" "gpt_oss_20b" 4 4 512 32768 15000
  submit_one "${split}" "${dataset_path}" "${dataset_name}" "${ids_path}" \
    "Qwen/Qwen3-Coder-30B-A3B-Instruct" "qwen3_coder_30b_a3b" 4 4 512 32768 15000
  submit_one "${split}" "${dataset_path}" "${dataset_name}" "${ids_path}" \
    "openai/gpt-oss-120b" "gpt_oss_120b" 8 8 1024 32768 15000
}

if [[ "${RUN_TRAIN_TRACES}" == "1" ]]; then
  submit_split "train1500" "/mnt/llmd/data/swe_smith/ds_train" "swe_smith_train" "${TRAIN_IDS}"
fi

if [[ "${RUN_EVAL_TRACES}" == "1" ]]; then
  submit_split "eval500" "/mnt/llmd/data/swe_smith/ds_test" "swe_smith_test" "${EVAL_IDS}"
fi

echo "Submitted SWE-Smith real-label trace jobs."
echo "ID_ROOT=${ID_ROOT}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SEED=${SEED:-42}
TRAIN_N=${TRAIN_N:-500}
EVAL_N=${EVAL_N:-500}
ROLLOUTS=${ROLLOUTS:-3}
RUN_TRAIN_TRACES=${RUN_TRAIN_TRACES:-1}
RUN_EVAL_TRACES=${RUN_EVAL_TRACES:-1}
SUBMIT_SLEEP_SECS=${SUBMIT_SLEEP_SECS:-20}

OUTPUT_BASE=${OUTPUT_BASE:-/mnt/llmd/results/exps/aristides/reason}
RUN_ROOT=${RUN_ROOT:-${OUTPUT_BASE}/offline_router_swe_smith_multi_rollout_trace_collect_${TIMESTAMP}}
ID_ROOT=${ID_ROOT:-${RUN_ROOT}/ids}

EXPANDED_ID_ROOT=${EXPANDED_ID_ROOT:-/mnt/llmd/results/exps/aristides/reason/swe_smith_real_label_trace_ids_expanded_1781073985}
TRAIN_POOL_IDS=${TRAIN_POOL_IDS:-${EXPANDED_ID_ROOT}/swe_smith_train_3000_ids.txt}
EVAL_POOL_IDS=${EVAL_POOL_IDS:-${EXPANDED_ID_ROOT}/swe_smith_eval_1000_ids.txt}

SWE_SMITH_DATA_ROOT=${SWE_SMITH_DATA_ROOT:-/mnt/llmd/data/swe_smith_bugged_context}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-${SWE_SMITH_DATA_ROOT}/ds_train}
EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-${SWE_SMITH_DATA_ROOT}/ds_test}
TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-swe_smith_train_bugged_context}
EVAL_DATASET_NAME=${EVAL_DATASET_NAME:-swe_smith_test_bugged_context}

PYTHON_BIN=${PYTHON_BIN:-/home/toolkit/.conda/envs/pipeline-rl/bin/python}
VLLM_PYTHON=${VLLM_PYTHON:-/home/toolkit/.conda/envs/vllm-env/bin/python}
CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}
COLLECTOR_ENV=${COLLECTOR_ENV:-pipeline-rl}
DRY_RUN=${DRY_RUN:-0}
SNAPSHOT=${SNAPSHOT:-1}
LOCAL=${LOCAL:-0}

PORT=${PORT:-8390}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
DTYPE=${DTYPE:-bfloat16}
MAX_CONCURRENT_PROBLEMS=${MAX_CONCURRENT_PROBLEMS:-8}
CONNECTOR_LIMIT=${CONNECTOR_LIMIT:-32}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-1800}
HEALTHCHECK_TIMEOUT=${HEALTHCHECK_TIMEOUT:-5400}
HEALTHCHECK_POLL=${HEALTHCHECK_POLL:-10}
MAX_TOKENS=${MAX_TOKENS:-15000}
MAX_TOKEN_FALLBACKS=${MAX_TOKEN_FALLBACKS:-8192,4096,2048}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.8}
SLEEP_AFTER_MODEL=${SLEEP_AFTER_MODEL:-5}
GPU_MEM=${GPU_MEM:-80}
CPU=${CPU:-48}
EXTRA_ARGS=${EXTRA_ARGS:-}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}

TRAIN_IDS=${TRAIN_IDS:-${ID_ROOT}/swe_smith_train_${TRAIN_N}_from_3000_seed${SEED}_ids.txt}
EVAL_IDS=${EVAL_IDS:-${ID_ROOT}/swe_smith_eval_${EVAL_N}_from_1000_seed${SEED}_ids.txt}

if [[ "${RUN_TRAIN_TRACES}" != "1" && "${RUN_EVAL_TRACES}" != "1" ]]; then
  echo "Nothing to launch: RUN_TRAIN_TRACES=0 and RUN_EVAL_TRACES=0" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing executable PYTHON_BIN=${PYTHON_BIN}" >&2
    exit 1
  fi
  if [[ ! -x "${VLLM_PYTHON}" ]]; then
    echo "Missing executable VLLM_PYTHON=${VLLM_PYTHON}" >&2
    exit 1
  fi
  if [[ "${RUN_TRAIN_TRACES}" == "1" && ! -s "${TRAIN_POOL_IDS}" ]]; then
    echo "Missing TRAIN_POOL_IDS=${TRAIN_POOL_IDS}" >&2
    exit 1
  fi
  if [[ "${RUN_EVAL_TRACES}" == "1" && ! -s "${EVAL_POOL_IDS}" ]]; then
    echo "Missing EVAL_POOL_IDS=${EVAL_POOL_IDS}" >&2
    exit 1
  fi
fi

if [[ "${RUN_TRAIN_TRACES}" == "1" && ! -s "${TRAIN_IDS}" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/pipelinerl/swe/scripts/offline_router/sample_instance_ids_from_pool.py" \
    --pool-path "${TRAIN_POOL_IDS}" \
    --output-path "${TRAIN_IDS}" \
    --n "${TRAIN_N}" \
    --seed "${SEED}"
fi

if [[ "${RUN_EVAL_TRACES}" == "1" && ! -s "${EVAL_IDS}" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/pipelinerl/swe/scripts/offline_router/sample_instance_ids_from_pool.py" \
    --pool-path "${EVAL_POOL_IDS}" \
    --output-path "${EVAL_IDS}" \
    --n "${EVAL_N}" \
    --seed "${SEED}"
fi

mkdir -p "${RUN_ROOT}"

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

  local job_name="offline_router_swe_smith_${split}_multirollout_collect_${slug}"
  local output_root="${RUN_ROOT}/${split}/${slug}"
  echo "[submit] split=${split} rollouts=${ROLLOUTS} model=${model} ids=${ids_path} output=${output_root}"

  local vllm_extra_arg=""
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    vllm_extra_arg="--vllm-extra-arg '${VLLM_EXTRA_ARGS}'"
  fi

  local collect_commands=""
  local rollout_idx
  for rollout_idx in $(seq 0 $((ROLLOUTS - 1))); do
    local rollout_seed=$((SEED + 1009 * rollout_idx))
    local rollout_root="${output_root}/rollout_${rollout_idx}"
    collect_commands+="mkdir -p '${rollout_root}'; "
    collect_commands+="echo '[rollout] split=${split} model=${model} rollout=${rollout_idx}/${ROLLOUTS} seed=${rollout_seed} output=${rollout_root}'; "
    collect_commands+="python -u -m pipelinerl.swe.scripts.offline_router.collect_model_discovery_candidates --output-dir '${rollout_root}/collect' --dataset-path '${dataset_path}' --dataset-name '${dataset_name}' --instance-ids-path '${ids_path}' --models '${model}' --port ${PORT} --vllm-python '${VLLM_PYTHON}' --tensor-parallel-size ${tensor_parallel_size} --max-model-len ${max_model_len} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} --dtype '${DTYPE}' --healthcheck-timeout ${HEALTHCHECK_TIMEOUT} --healthcheck-poll ${HEALTHCHECK_POLL} --max-concurrent-problems ${MAX_CONCURRENT_PROBLEMS} --connector-limit ${CONNECTOR_LIMIT} --request-timeout ${REQUEST_TIMEOUT} --max-tokens ${max_tokens} --max-token-fallbacks '${MAX_TOKEN_FALLBACKS}' --temperature ${temperature} --top-p ${top_p} --top-k ${top_k} --repetition-penalty ${repetition_penalty} --success-threshold ${SUCCESS_THRESHOLD} --sleep-after-model ${SLEEP_AFTER_MODEL} --seed ${rollout_seed} ${vllm_extra_arg} ${EXTRA_ARGS} 2>&1 | tee -a '${rollout_root}/launch.out'; "
  done

  make job \
    ENV="${COLLECTOR_ENV}" \
    CONDA_EXE="${CONDA_EXE}" \
    CONDA=1 \
    ACCELERATE=0 \
    DEEPSPEED=0 \
    NPROC=1 \
    CPU="${CPU}" \
    CPU_MEM="${cpu_mem}" \
    GPU="${gpu_count}" \
    GPU_MEM="${GPU_MEM}" \
    SNAPSHOT="${SNAPSHOT}" \
    LOCAL="${LOCAL}" \
    DRY_RUN="${DRY_RUN}" \
    JOB_NAME="${job_name}_${TIMESTAMP}" \
    COMMAND="cd ${REPO_ROOT}; mkdir -p '${output_root}'; set -euo pipefail; ${collect_commands} echo '[done] split=${split} model=${model} rollouts=${ROLLOUTS}' 2>&1 | tee -a '${output_root}/launch.out'"

  sleep "${SUBMIT_SLEEP_SECS}"
}

submit_split() {
  local split=$1
  local dataset_path=$2
  local dataset_name=$3
  local ids_path=$4

  # Qwen model-card defaults: temp=0.7 top_p=0.8 top_k=20; Qwen Coder also recommends repetition_penalty=1.05.
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

cat > "${RUN_ROOT}/manifest.txt" <<EOF
RUN_ROOT=${RUN_ROOT}
ID_ROOT=${ID_ROOT}
TRAIN_IDS=${TRAIN_IDS}
EVAL_IDS=${EVAL_IDS}
TRAIN_POOL_IDS=${TRAIN_POOL_IDS}
EVAL_POOL_IDS=${EVAL_POOL_IDS}
TRAIN_N=${TRAIN_N}
EVAL_N=${EVAL_N}
ROLLOUTS=${ROLLOUTS}
SEED=${SEED}
JOBS=8 when train and eval are both enabled: 2 splits * 4 models; each job runs ROLLOUTS sequential collections.
EOF

echo "Submitted SWE-Smith multi-rollout trace jobs."
echo "RUN_ROOT=${RUN_ROOT}"
echo "TRAIN_IDS=${TRAIN_IDS}"
echo "EVAL_IDS=${EVAL_IDS}"

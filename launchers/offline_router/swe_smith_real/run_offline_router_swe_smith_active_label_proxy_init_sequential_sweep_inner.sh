#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-Embedding-8B}
RUN_ROOT=${RUN_ROOT:?Set RUN_ROOT}
DATASET_ROOT=${DATASET_ROOT:-${RUN_ROOT}/datasets}
SOURCE_DATASET_DIR=${SOURCE_DATASET_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}
SWEEP_SPECS=${SWEEP_SPECS:-random:128,top2_margin:128,mean_uncertainty:128,high_variance:128,high_score:128}
PROXY_PREDICTIONS=${PROXY_PREDICTIONS:-}
INIT_FROM_MODEL_CHECKPOINT=${INIT_FROM_MODEL_CHECKPOINT:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_trace_expanded_4route_qwen3_embedding_8b_lora_proxy_verifier_soft_bce_r32_qkvo_mlp_5epoch_1781684950/train_qwen3_embedding_8b_lora_proxy_verifier_soft_bce_r32_qkvo_mlp_5epoch/checkpoints/epoch_0004}

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
MAX_TRAIN_ROWS=${MAX_TRAIN_ROWS:-0}
MAX_EVAL_ROWS=${MAX_EVAL_ROWS:-0}
SEED=${SEED:-17}
TARGET_ROUTE_IDXS=${TARGET_ROUTE_IDXS:-}
CASCADE_ORDER=${CASCADE_ORDER:-0,1,2,3}
MAX_THRESHOLD_CANDIDATES=${MAX_THRESHOLD_CANDIDATES:-21}
LOSS_TYPE=${LOSS_TYPE:-soft_bce}
UTILITY_LAMBDAS=${UTILITY_LAMBDAS:-0.0,1.0e-5,2.0e-5,5.0e-5,1.0e-4,2.0e-4}
MLP_HIDDEN_SIZE=${MLP_HIDDEN_SIZE:-1024}
DROPOUT=${DROPOUT:-0.1}
TORCH_DTYPE=${TORCH_DTYPE:-bf16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
CHECKPOINT_EVERY_EPOCH=${CHECKPOINT_EVERY_EPOCH:-true}
EPOCH_REPORT_EVERY=${EPOCH_REPORT_EVERY:-1}
SAVE_MODEL=${SAVE_MODEL:-true}
SUBRUN_SLEEP_SECS=${SUBRUN_SLEEP_SECS:-5}

mkdir -p "${RUN_ROOT}" "${DATASET_ROOT}"

if [[ ! -f "${SOURCE_DATASET_DIR}/metadata.json" ]]; then
  echo "Missing source real-label dataset: ${SOURCE_DATASET_DIR}/metadata.json" >&2
  exit 1
fi
if [[ ! -f "${INIT_FROM_MODEL_CHECKPOINT}/model.safetensors" && ! -f "${INIT_FROM_MODEL_CHECKPOINT}" ]]; then
  echo "Missing init checkpoint model.safetensors: ${INIT_FROM_MODEL_CHECKPOINT}" >&2
  exit 1
fi

TRAIN_CMD=(python pipelinerl/swe/scripts/offline_router/train_qwen_embedding_cascade_baseline.py)
if [[ "${TRAIN_NPROC}" -gt 1 ]]; then
  TRAIN_CMD=(
    python -m accelerate.commands.launch
    --multi_gpu
    --mixed_precision "${MIXED_PRECISION}"
    --num_processes "${TRAIN_NPROC}"
    --config_file "conf/accelerate/${ACCELERATE_CONFIG}.yaml"
    pipelinerl/swe/scripts/offline_router/train_qwen_embedding_cascade_baseline.py
  )
fi

PROXY_ARGS=()
if [[ -n "${PROXY_PREDICTIONS}" ]]; then
  IFS=',' read -r -a PROXY_PATHS <<< "${PROXY_PREDICTIONS}"
  for proxy_path in "${PROXY_PATHS[@]}"; do
    PROXY_ARGS+=(--proxy-predictions "${proxy_path}")
  done
fi

SAVE_MODEL_ARG=()
if [[ "${SAVE_MODEL}" == "true" ]]; then
  SAVE_MODEL_ARG=(--save-model)
fi

CHECKPOINT_ARG=()
if [[ "${CHECKPOINT_EVERY_EPOCH}" == "true" ]]; then
  CHECKPOINT_ARG=(--checkpoint-every-epoch)
fi

ATTN_ARG=()
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  ATTN_ARG=(--attn-implementation "${ATTN_IMPLEMENTATION}")
fi

TARGET_ROUTE_ARG=()
if [[ -n "${TARGET_ROUTE_IDXS}" ]]; then
  TARGET_ROUTE_ARG=(--target-route-idxs "${TARGET_ROUTE_IDXS}")
fi

MANIFEST="${RUN_ROOT}/sweep_manifest.jsonl"
echo "Sequential active-label verifier sweep"
echo "Run root: ${RUN_ROOT}"
echo "Source real-label dataset: ${SOURCE_DATASET_DIR}"
echo "Proxy init checkpoint: ${INIT_FROM_MODEL_CHECKPOINT}"
echo "Sweep specs: ${SWEEP_SPECS}"
echo "Manifest: ${MANIFEST}"

IFS=',' read -r -a SPECS <<< "${SWEEP_SPECS}"
for raw_spec in "${SPECS[@]}"; do
  spec=$(echo "${raw_spec}" | xargs)
  if [[ -z "${spec}" ]]; then
    continue
  fi
  if [[ "${spec}" != *:* ]]; then
    echo "Invalid SWEEP_SPECS item '${spec}'. Expected strategy:budget, e.g. top2_margin:128" >&2
    exit 1
  fi

  strategy="${spec%%:*}"
  budget="${spec##*:}"
  label="${strategy}_${budget}_seed${SEED}"
  dataset_dir="${DATASET_ROOT}/${label}"
  train_output_dir="${RUN_ROOT}/${label}/train_qwen3_embedding_8b_lora_verifier_proxy_init_real_${strategy}_${budget}_${NUM_EPOCHS}epoch"

  echo
  echo "=== [${label}] materialize dataset ==="
  python "${REPO_ROOT}/pipelinerl/swe/scripts/offline_router/materialize_active_real_label_verifier_subset.py" \
    --source-dataset-dir "${SOURCE_DATASET_DIR}" \
    --output-dir "${dataset_dir}" \
    --strategy "${strategy}" \
    --budget-instances "${budget}" \
    --seed "${SEED}" \
    "${PROXY_ARGS[@]}"

  mkdir -p "${train_output_dir}"
  python - "${strategy}" "${budget}" "${dataset_dir}" "${train_output_dir}" "${INIT_FROM_MODEL_CHECKPOINT}" >> "${MANIFEST}" <<'PY'
import json
import sys

strategy, budget, dataset_dir, train_output_dir, init_checkpoint = sys.argv[1:]
print(json.dumps({
    "strategy": strategy,
    "budget_instances": int(budget),
    "dataset_dir": dataset_dir,
    "train_output_dir": train_output_dir,
    "init_from_model_checkpoint": init_checkpoint,
}, sort_keys=True))
PY

  echo "=== [${label}] train verifier fine-tune ==="
  set -o pipefail
  "${TRAIN_CMD[@]}" \
    --dataset-dir "${dataset_dir}" \
    --output-dir "${train_output_dir}" \
    --model-name "${MODEL_NAME}" \
    --cascade-order "${CASCADE_ORDER}" \
    "${TARGET_ROUTE_ARG[@]}" \
    --max-seq-length "${MAX_SEQ_LENGTH}" \
    --num-epochs "${NUM_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --max-train-rows "${MAX_TRAIN_ROWS}" \
    --max-eval-rows "${MAX_EVAL_ROWS}" \
    --seed "${SEED}" \
    --dropout "${DROPOUT}" \
    --mlp-hidden-size "${MLP_HIDDEN_SIZE}" \
    --torch-dtype "${TORCH_DTYPE}" \
    "${ATTN_ARG[@]}" \
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
    --epoch-report-every "${EPOCH_REPORT_EVERY}" \
    --init-from-model-checkpoint "${INIT_FROM_MODEL_CHECKPOINT}" \
    "${CHECKPOINT_ARG[@]}" \
    "${SAVE_MODEL_ARG[@]}" \
    2>&1 | tee -a "${train_output_dir}/launch.out"

  echo "=== [${label}] done ==="
  sleep "${SUBRUN_SLEEP_SECS}"
done

echo
echo "Sequential active-label verifier sweep complete: ${RUN_ROOT}"

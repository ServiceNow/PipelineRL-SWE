#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:?Set RUN_DIR to a gpt_oss_reasoning_* run directory}
DATASET_PATH=${DATASET_PATH:-/mnt/llmd/data/swe_smith/ds_test}
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_DIR}/sb_cli_exports}
LEVELS=${LEVELS:-"low medium high"}
SB_SUBSET=${SB_SUBSET:-swe-bench_lite}
SB_SPLIT=${SB_SPLIT:-test}
RUN_SB_CLI=${RUN_SB_CLI:-0}
SKIP_EMPTY_PATCHES=${SKIP_EMPTY_PATCHES:-0}
CONDA_ENV=${CONDA_ENV:-pipeline-rl}

mkdir -p "${OUTPUT_DIR}"

for LEVEL in ${LEVELS}; do
  INPUT_JSONL="${RUN_DIR}/${LEVEL}/expert_eval.jsonl"
  if [[ ! -f "${INPUT_JSONL}" ]]; then
    echo "Skipping ${LEVEL}: missing ${INPUT_JSONL}"
    continue
  fi

  ARGS=(
    python -m pipelinerl.swe.scripts.new.export_expert_eval_to_sb_cli
    --input-jsonl "${INPUT_JSONL}"
    --dataset-path "${DATASET_PATH}"
    --output-json "${OUTPUT_DIR}/${LEVEL}_predictions.json"
    --output-jsonl "${OUTPUT_DIR}/${LEVEL}_predictions.jsonl"
    --output-summary "${OUTPUT_DIR}/${LEVEL}_summary.json"
    --model-name-or-path "openai/gpt-oss-120b-${LEVEL}"
    --sb-subset "${SB_SUBSET}"
    --sb-split "${SB_SPLIT}"
  )

  if [[ "${RUN_SB_CLI}" == "1" ]]; then
    ARGS+=(--run-sb-cli)
  fi
  if [[ "${SKIP_EMPTY_PATCHES}" == "1" ]]; then
    ARGS+=(--skip-empty-patches)
  fi

  echo "=== Exporting ${LEVEL} ==="
  conda run --no-capture-output -n "${CONDA_ENV}" "${ARGS[@]}"
done

echo "Exports written under ${OUTPUT_DIR}"

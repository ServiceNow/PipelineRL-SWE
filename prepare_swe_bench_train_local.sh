#!/usr/bin/env bash
set -euo pipefail

CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}
CONDA_ENV=${CONDA_ENV:-pipeline-rl}

HF_DATASET=${HF_DATASET:-princeton-nlp/SWE-bench}
HF_SPLIT=${HF_SPLIT:-train}
OUTPUT_PATH=${OUTPUT_PATH:-/mnt/llmd/data/swebench/ds_train}
REPOS_BASE_DIR=${REPOS_BASE_DIR:-/mnt/llmd/data/swebench/repos}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-16000}
MAX_NORMALIZED_ROWS=${MAX_NORMALIZED_ROWS:-6000}
ROW_SAMPLE_SEED=${ROW_SAMPLE_SEED:-42}
GIT_TIMEOUT_SECONDS=${GIT_TIMEOUT_SECONDS:-900}
FETCH_EXISTING_REPOS=${FETCH_EXISTING_REPOS:-0}
PARTIAL_CLONE=${PARTIAL_CLONE:-1}
GOLD_FILE_SOURCE=${GOLD_FILE_SOURCE:-raw-url}

EXTRA_ARGS=(
  --max-normalized-rows "${MAX_NORMALIZED_ROWS}"
  --row-sample-seed "${ROW_SAMPLE_SEED}"
  --git-timeout-seconds "${GIT_TIMEOUT_SECONDS}"
  --gold-file-source "${GOLD_FILE_SOURCE}"
)
if [[ "${FETCH_EXISTING_REPOS}" == "1" ]]; then
  EXTRA_ARGS+=(--fetch-existing-repos)
fi
if [[ "${PARTIAL_CLONE}" != "1" ]]; then
  EXTRA_ARGS+=(--no-partial-clone)
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")" "${REPOS_BASE_DIR}"

"${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" python -m pipelinerl.swe.scripts.new.prepare_swe_smith_dataset \
  --hf-dataset "${HF_DATASET}" \
  --hf-split "${HF_SPLIT}" \
  --single-output-path "${OUTPUT_PATH}" \
  --reconstruct-missing-gold-files \
  --repos-base-dir "${REPOS_BASE_DIR}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS}" \
  "${EXTRA_ARGS[@]}"

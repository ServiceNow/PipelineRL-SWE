#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP=${TIMESTAMP:-$(date +%s)}

JOB_NAME=${JOB_NAME:-offline_router_swe_bench_router_split_collect}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}}
COLLECT_OUTPUT_DIR=${COLLECT_OUTPUT_DIR:-${OUTPUT_ROOT}/collect}

CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}
COLLECTOR_ENV=${COLLECTOR_ENV:-pipeline-rl}
JOB_NPROC=${JOB_NPROC:-5}
SNAPSHOT=${SNAPSHOT:-1}
DRY_RUN=${DRY_RUN:-0}
LOCAL=${LOCAL:-0}

PRIMARY_MODEL_PATH=${PRIMARY_MODEL_PATH:-/mnt/llmd/results/exps/aristides/reason/swe_smith_policy_conditioned_no_devstral_1773812579/finetune/current}
PRIMARY_TOKENIZER_NAME=${PRIMARY_TOKENIZER_NAME:-Qwen/Qwen2.5-Coder-7B-Instruct}
PRIMARY_SERVED_MODEL_NAME=${PRIMARY_SERVED_MODEL_NAME:-primary_model}

# Router train split: official SWE-bench train, preprocessed into local SWE format.
TRAIN_DATASET_NAMES=${TRAIN_DATASET_NAMES:-swe_bench_train}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-/mnt/llmd/data/swebench/ds_train}

# Router test split: existing local SWE-bench Lite test set by default.
# Override these if a full preprocessed SWE-bench test set is available.
EVAL_DATASET_NAMES=${EVAL_DATASET_NAMES:-swebench_lite}
EVAL_DATASET_PATH=${EVAL_DATASET_PATH:-/mnt/llmd/data/swebench_lite/ds}

COLLECT_TRAIN=${COLLECT_TRAIN:-true}
COLLECT_EVAL=${COLLECT_EVAL:-true}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-4096}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-500}
MAX_CONCURRENT_PROBLEMS=${MAX_CONCURRENT_PROBLEMS:-8}
SHARD_SIZE=${SHARD_SIZE:-64}
COLLECT_EXTRA_ARGS=${COLLECT_EXTRA_ARGS:-}

RUN_COLLECT=1
RUN_TRAIN=0
TRAIN_NPROC=4

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ "${COLLECT_TRAIN}" == "true" && ! -d "${TRAIN_DATASET_PATH}" ]]; then
    echo "Missing TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH}" >&2
    echo "Run prepare_swe_bench_train_local.sh first, or override TRAIN_DATASET_PATH to an existing preprocessed dataset." >&2
    exit 1
  fi
  if [[ "${COLLECT_EVAL}" == "true" && ! -d "${EVAL_DATASET_PATH}" ]]; then
    echo "Missing EVAL_DATASET_PATH=${EVAL_DATASET_PATH}" >&2
    exit 1
  fi
fi

export JOB_NAME
export TIMESTAMP
export OUTPUT_ROOT
export COLLECT_OUTPUT_DIR
export CONDA_EXE
export COLLECTOR_ENV
export JOB_NPROC
export SNAPSHOT
export DRY_RUN
export LOCAL
export PRIMARY_MODEL_PATH
export PRIMARY_TOKENIZER_NAME
export PRIMARY_SERVED_MODEL_NAME
export TRAIN_DATASET_NAMES
export TRAIN_DATASET_PATH
export EVAL_DATASET_NAMES
export EVAL_DATASET_PATH
export COLLECT_TRAIN
export COLLECT_EVAL
export TRAIN_MAX_SAMPLES
export EVAL_MAX_SAMPLES
export MAX_CONCURRENT_PROBLEMS
export SHARD_SIZE
export COLLECT_EXTRA_ARGS
export RUN_COLLECT
export RUN_TRAIN
export TRAIN_NPROC

bash launch_offline_router_swe_smith_id.sh

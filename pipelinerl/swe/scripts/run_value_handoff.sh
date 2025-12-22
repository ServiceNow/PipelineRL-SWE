#!/usr/bin/env bash

# Usage:
#   bash pipelinerl/swe/scripts/run_value_handoff.sh
#
# This script runs actor repair evaluation with value-head scoring,
# then runs handoff analysis (mean and last value thresholds).

set -euo pipefail

# Paths (edit as needed)
ACTOR_EVAL_JSONL="/mnt/llmd/results/exps/aristides/reason/pure_ppo_swe/actor_eval_value_2.jsonl"
EXPERT_JSONL="/mnt/llmd/results/exps/aristides/reason/pareto/expert_claude.jsonl"
MODEL_PATH="/mnt/llmd/results/exps/aristides/reason/pure_ppo_swe/finetune/current"
OUTPUT_DIR="/home/toolkit/PipelineRL-SWE/value_handoff_2"
ANALYSIS_OUTPUT="${OUTPUT_DIR}/handoff_analysis.json"

# 1) Run actor repair eval with value-head scoring
if [[ -z "${SKIP_ACTOR_GEN:-}" ]]; then
  python -m pipelinerl.swe.scripts.run_actor_repair_eval \
    --config-name swe \
    output_dir="${OUTPUT_DIR}" \
    small_eval.model_name="${MODEL_PATH}" \
    small_eval.output_path="${ACTOR_EVAL_JSONL}" \
    small_eval.value_model_path="${MODEL_PATH}"
else
  echo "Skipping actor repair generation (SKIP_ACTOR_GEN set)"
fi

# 2) Run handoff analysis using value scores (mean/last)
python pipelinerl/swe/scripts/analyze_handoff.py \
  --actor_glob="${ACTOR_EVAL_JSONL}" \
  --expert_jsonl="${EXPERT_JSONL}" \
  --output_path="${ANALYSIS_OUTPUT}" \
  --small_token_cost_per_1k=0.15 \
  --expert_token_cost_per_1k=1.20

if [[ ! -f "${ANALYSIS_OUTPUT}" ]]; then
  echo "ERROR: Expected ${ANALYSIS_OUTPUT} but it was not created" >&2
  exit 1
fi

ls -1 "${OUTPUT_DIR}"/handoff_analysis*

echo "Done. Outputs under ${MODEL_PATH} (handoff_analysis*.json/png/csv)."

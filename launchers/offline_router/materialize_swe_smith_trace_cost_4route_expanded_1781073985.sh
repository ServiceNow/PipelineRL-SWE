#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_trace_cost_4route_expanded_1781073985/collect}
OVERWRITE=${OVERWRITE:-false}
PYTHON=${PYTHON:-/home/toolkit/.conda/envs/pipeline-rl/bin/python}

OVERWRITE_ARG=""
if [[ "${OVERWRITE}" == "true" ]]; then
  OVERWRITE_ARG="--overwrite"
fi

"${PYTHON}" pipelinerl/swe/scripts/offline_router/materialize_trace_cost_router_dataset.py \
  --output-dir "${OUTPUT_DIR}" \
  --train-dataset-path /mnt/llmd/data/swe_smith_bugged_context/ds_train \
  --eval-dataset-path /mnt/llmd/data/swe_smith_bugged_context/ds_test \
  --train-dataset-name swe_smith_train_bugged_context \
  --eval-dataset-name swe_smith_test_bugged_context \
  --train-route "qwen3_4b_instruct_2507=scout:Qwen/Qwen3-4B-Instruct-2507=/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_real_train3000_collect_qwen3_4b_instruct_2507_expanded_1781073985/collect/models/Qwen_Qwen3-4B-Instruct-2507/outputs.jsonl" \
  --train-route "gpt_oss_20b=solver:openai/gpt-oss-20b=/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_real_train3000_collect_gpt_oss_20b_expanded_1781073985/collect/models/openai_gpt-oss-20b/outputs.jsonl" \
  --train-route "qwen3_coder_30b_a3b=solver:Qwen/Qwen3-Coder-30B-A3B-Instruct=/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_real_train3000_collect_qwen3_coder_30b_a3b_expanded_1781073985/collect/models/Qwen_Qwen3-Coder-30B-A3B-Instruct/outputs.jsonl" \
  --train-route "gpt_oss_120b=solver:openai/gpt-oss-120b=/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_real_train3000_collect_gpt_oss_120b_expanded_1781073985/collect/models/openai_gpt-oss-120b/outputs.jsonl" \
  --eval-route "qwen3_4b_instruct_2507=scout:Qwen/Qwen3-4B-Instruct-2507=/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_real_eval1000_collect_qwen3_4b_instruct_2507_expanded_1781073985/collect/models/Qwen_Qwen3-4B-Instruct-2507/outputs.jsonl" \
  --eval-route "gpt_oss_20b=solver:openai/gpt-oss-20b=/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_real_eval1000_collect_gpt_oss_20b_expanded_1781073985/collect/models/openai_gpt-oss-20b/outputs.jsonl" \
  --eval-route "qwen3_coder_30b_a3b=solver:Qwen/Qwen3-Coder-30B-A3B-Instruct=/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_real_eval1000_collect_qwen3_coder_30b_a3b_expanded_1781073985/collect/models/Qwen_Qwen3-Coder-30B-A3B-Instruct/outputs.jsonl" \
  --eval-route "gpt_oss_120b=solver:openai/gpt-oss-120b=/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_real_eval1000_collect_gpt_oss_120b_expanded_1781073985/collect/models/openai_gpt-oss-120b/outputs.jsonl" \
  --seed 17 \
  ${OVERWRITE_ARG}

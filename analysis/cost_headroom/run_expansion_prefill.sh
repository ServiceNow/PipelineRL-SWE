#!/usr/bin/env bash
set -euo pipefail
PREFILL_DATASET=$1
PREFILL_OUT=/mnt/llmd/results/exps/aristides/reason/expansion_prefills_20261001
PREFILL_MODEL=Qwen/Qwen3-4B-Instruct-2507
if [[ ${PREFILL_DATASET} == omni500 ]]; then PREFILL_MODEL=Qwen/Qwen3-4B-Thinking-2507; fi
mkdir -p "${PREFILL_OUT}/logs"
exec > >(tee -a "${PREFILL_OUT}/logs/${PREFILL_DATASET}.log") 2>&1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python analysis/cost_headroom/prepare_expansion_prefills.py --dataset "${PREFILL_DATASET}"
export PYTHONPATH="/mnt/llmd/results/exps/aristides/envs/accel:${PYTHONPATH:-}"
/home/toolkit/.conda/envs/vllm-env/bin/python -u pipelinerl/swe/scripts/livecodebench/pool_activation_probe.py \
  --phase extract --model "${PREFILL_MODEL}" --route-label "${PREFILL_DATASET}_expansion" \
  --prompts-file "${PREFILL_OUT}/${PREFILL_DATASET}_prompts.jsonl" \
  --activations "${PREFILL_OUT}/${PREFILL_DATASET}_prefill.npz" --max-len 8192 \
  --system-prompt "You are a helpful assistant."

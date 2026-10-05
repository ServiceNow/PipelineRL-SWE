#!/usr/bin/env bash
# Non-Qwen prefills for the size sweep (NEW_PATH 4.A.54): one GPU job per model; same prompts, system prompt, 8 relative layers.
# Models pre-downloaded into /home/toolkit/.cache/huggingface. Commit + push first.
set -euo pipefail
O=/mnt/llmd/results/exps/aristides/reason/prefill_size_20261005
for spec in microsoft/Phi-4-mini-instruct:phi4mini ibm-granite/granite-3.3-2b-instruct:granite2b HuggingFaceTB/SmolLM2-1.7B-Instruct:smol17b; do
  M=${spec%%:*}; TAG=${spec##*:}
  if [ "${SUBMIT:-0}" != "1" ]; then echo "dry run: $M -> $O/$TAG"; continue; fi
  make job JOB_NAME="prefill_${TAG}_$(date -u +%H%M%S)" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=1 GPU_MEM=48 CPU=8 CPU_MEM=64 SNAPSHOT=1 \
    COMMAND="bash $O/run_extract.sh $M $TAG"
  sleep 35
done

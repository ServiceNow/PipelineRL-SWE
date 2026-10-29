#!/usr/bin/env bash
# Prefill-size scale table (NEW_PATH 4.A.122): extract the paper's 8 relative layers x {last, mean} for one model, as run_extract.sh
# (prefill_size_20261005), on the datasets given: omni500 / mmlupro (math system prompt) and lcb (the paper's LCB prompts and coding
# system prompt, as pv2_scout_prefill). Usage: run_extract_scale.sh MODEL TAG "lcb omni500 mmlupro"
set -euo pipefail
M=$1; TAG=$2; D=/mnt/llmd/results/exps/aristides/reason; O=$D/prefill_size_20261005
export HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTHONPATH=/mnt/llmd/results/exps/aristides/envs/accel:${PYTHONPATH:-}
cd /home/toolkit/PipelineRL-SWE; mkdir -p $O/$TAG
for ds in $3; do
  if [[ $ds == lcb ]]; then
    python pipelinerl/swe/scripts/livecodebench/pool_activation_probe.py --phase extract --model "$M" --route-label $TAG \
      --prompts-file $D/pool_probe_prompts.jsonl --activations $O/$TAG/lcb.npz --max-len 8192 > $O/$TAG/lcb.log 2>&1
  else
    python pipelinerl/swe/scripts/livecodebench/pool_activation_probe.py --phase extract --model "$M" --route-label $TAG \
      --prompts-file $O/${ds}_prompts.jsonl --activations $O/$TAG/${ds}.npz --max-len 8192 \
      --system-prompt-file $D/math_system_prompt.txt > $O/$TAG/${ds}.log 2>&1
  fi
  echo "EXIT 0" >> $O/$TAG/$ds.log
done

#!/usr/bin/env bash
# 2026-11-23 (free, GPU): regenerate 100 problems' scout activations with the release repo's extractor (prompts rebuilt by its
# loader) and compare with the original files: LCB (Instruct), MMLU-Pro (Instruct), expanded Omni-MATH (Thinking).
set -euo pipefail
O=/mnt/llmd/results/exps/aristides/release_check_20261123; mkdir -p $O; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3
CD="cd /home/toolkit/scout-before-you-route && export HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 &&"
sub() { make -C /home/toolkit/PipelineRL-SWE job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=0 GPU=1 GPU_MEM=40 CPU=8 CPU_MEM=64 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
for spec in "lcb::rel_anchor_lcb" "mmlupro::rel_anchor_mmlupro" "omni:expanded:rel_anchor_omnix"; do
  IFS=: read ds sb name <<< "$spec"
  echo -n "$name "; sub $name "$CD $PY -u -m scoutroute.extract $ds $sb --limit 100 --out $O/$name.npz > $O/$name.txt 2>&1; echo EXIT \$? >> $O/$name.txt"
done

#!/usr/bin/env bash
# 2026-11-23 (free): run one stage of the release repo's reproduce.sh on eai, one job per command (the baseline for the refactor).
# Usage: SUBMIT=1 [TAG=suffix] [GREP=pattern] [SKIP=pattern] bash launch_stage.sh <stage>
# GPU for features / embeddings / finetune and for run lines marked "# GPU"; 80 GB for the 14B scout (loaded from base_models).
set -euo pipefail
ST=$1; REPO=/home/toolkit/scout-before-you-route; O=/mnt/llmd/results/exps/aristides/release_check_20261123/jobs/$ST; mkdir -p $O
PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3
ENV="cd $REPO && export PY=$PY HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TRANSFORMERS_OFFLINE=1 &&"
sub() { make -C /home/toolkit/PipelineRL-SWE job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=0 GPU=$3 GPU_MEM=$4 CPU=16 CPU_MEM=128 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
i=0
while IFS= read -r c; do
  i=$((i + 1)); [[ -z $c ]] && continue
  [[ -n ${GREP:-} && ! $c =~ $GREP ]] && continue; [[ -n ${SKIP:-} && $c =~ $SKIP ]] && continue
  c=${c//python /$PY }; c=${c//--model q14b/--model q14b --model-path \/mnt\/llmd\/base_models\/Qwen3-14B}
  g=0; m=0; [[ $ST =~ ^(features|embeddings|finetune)$ || $c == *"# GPU"* ]] && { g=1; m=40; }; [[ $c == *q14b* && $ST == features ]] && m=80
  n="rel_${ST:0:4}${TAG:-}_$i"; echo "$i $c" > $O/$i.cmd
  [[ ${SUBMIT:-0} == 1 ]] || { echo "[dry] $n g=$g m=$m: ${c:0:150}"; continue; }
  echo -n "$n "; sub $n "$ENV ( ${c%%# GPU*} ) > $O/$i.log 2>&1; echo EXIT \$? >> $O/$i.log" $g $m
done < <(bash $REPO/reproduce.sh $ST --list)

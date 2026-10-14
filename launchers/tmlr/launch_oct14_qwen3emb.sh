#!/usr/bin/env bash
# 2026-10-14 (user-approved; free, GPU): Qwen3-Embedding-4B on all nine pools, 8B on the seven not yet embedded (embed_qwen3.py; NEW_PATH 4.A.101)
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/qwen3emb; mkdir -p $D
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
for SZ in 4B 8B; do
  DIRS=""; [[ $SZ == 8B ]] && DIRS="pool_v2_tensors_5rung,apps_tensors,bcb_tensors_5r,cc_tensors,aime_tensors,supergpqa_tensors,bbeh_tensors"
  n=qwen3emb_$(echo $SZ | tr A-Z a-z); echo -n "$n "
  make job JOB_NAME=$n ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=1 GPU_MEM=40 CPU=8 CPU_MEM=64 SNAPSHOT=1 \
    COMMAND="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && /home/toolkit/.conda/envs/pipeline-rl/bin/python3 -u embed_qwen3.py $SZ $DIRS > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true
done

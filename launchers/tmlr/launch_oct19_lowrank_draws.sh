#!/usr/bin/env bash
# 2026-10-19 (user-approved; free, stored features only): shared rank-k bottleneck for success and cost on the 10 training sets
# (lowrank_draws.py; NEW_PATH 4.A.117), one eai job per (pool, 5 training sets).
# Usage: SUBMIT=1 bash launch_oct19_lowrank_draws.sh
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/lowrankdraws; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned NTHREADS=16 RESULT_TAG=_pinned NBOOT=300"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=128 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
for P in ${POOLS:-LCB Omni MMLU-Pro AIME APPS BCB CC SuperGPQA BBEH}; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  for s in ${STARTS:-0 5}; do n=lr_${p}_$s; echo -n "$n "; sub $n "$CD OUT_DIR=$D && $PY -u lowrank_draws.py $P $s $((s + 5)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt"; echo; done
done

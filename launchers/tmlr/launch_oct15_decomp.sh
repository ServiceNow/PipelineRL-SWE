#!/usr/bin/env bash
# 2026-10-15 (user-approved; free): predictor-free data analyses -- variance decomposition of log length (route / problem / interaction /
# noise) for every route subset, and headroom split into shared-level and route-specific oracles on 10 training sets (data_decomp.py;
# NEW_PATH 4.A.109). One eai job per pool. Usage: SUBMIT=1 bash launch_oct15_decomp.sh
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/decomp; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned OUT_DIR=$D NTHREADS=16 RESULT_TAG=_pinned &&"
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
for P in ${POOLS:-LCB Omni MMLU-Pro AIME APPS BCB CC SuperGPQA BBEH}; do p=$(echo $P | tr -d '-' | tr A-Z a-z); n=dec_$p; echo -n "$n "
  make job JOB_NAME=$n ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=128 COMMAND="$CD $PY -u data_decomp.py $P > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true
done

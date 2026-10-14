#!/usr/bin/env bash
# 2026-10-14 (user-approved; free): one router for mixed traffic over the nine pools, with the per-(route, benchmark) median control,
# 10 training sets, one eai job each (mixed_pool.py; NEW_PATH 4.A.103). Usage: SUBMIT=1 bash launch_oct14_mixed.sh
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/mixed; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
for s in ${SEEDS:-0 1 2 3 4 5 6 7 8 9}; do n=mix_$s; echo -n "$n "
  make job JOB_NAME=$n ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=192 \
    COMMAND="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && REASON_ROOT=$R/../reason_pinned OUT_DIR=$D NTHREADS=16 NBOOT=300 DRAW=$s RESULT_TAG=_pinned_d$s $PY -u mixed_pool.py > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true
done

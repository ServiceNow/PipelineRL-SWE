#!/usr/bin/env bash
# 2026-10-14 (user-approved; free): the eight-route pools (newpool, all-but-deepseek-v4-flash, all8) on the 10 training sets + the
# manifest split: estimator arms, router vs every single route, router vs a random mix of single routes (newfamily_draws.py; NEW_PATH 4.A.96).
# One eai job per (benchmark, training set). Usage: SUBMIT=1 bash launch_oct14_newfamily_draws.sh
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/newfam_draws; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned OUT_DIR=$D NTHREADS=16 RESULT_TAG=_pinned NBOOT=300 &&"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=128 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
for P in ${POOLS:-LCB Omni MMLU-Pro}; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  for s in ${SEEDS:--1 0 1 2 3 4 5 6 7 8 9}; do n=nfd_${p}_${s/-/m}
    echo -n "$n "; sub $n "$CD $PY -u newfamily_draws.py $P $s $((s + 1)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt"
  done
done

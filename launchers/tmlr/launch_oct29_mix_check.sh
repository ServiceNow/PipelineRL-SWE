#!/usr/bin/env bash
# 2026-10-29 (user-approved; free, GPU): every estimator vs a random mix of single routes (mix_check.py; NEW_PATH 4.A.123), one GPU job per (pool, 5 sets).
# Usage: SUBMIT=1 bash launch_oct29_mix_check.sh      SMOKE=1: one APPS training set, NBOOT=3, into $D/smoke
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/mixcheck; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D/smoke
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned NTHREADS=8 RESULT_TAG=_pinned HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=1 GPU_MEM=16 CPU=8 CPU_MEM=96 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
if [[ ${SMOKE:-0} == 1 ]]; then
  n=mx_smoke_$(date +%s); echo -n "$n "; sub $n "$CD NBOOT=3 OUT_DIR=$D/smoke && $PY -u mix_check.py APPS 0 1 > $D/smoke/mx_smoke.txt 2>&1; echo EXIT >> $D/smoke/mx_smoke.txt"; echo; exit 0
fi
for P in ${POOLS:-APPS CC BCB LCB AIME SuperGPQA BBEH Omni MMLU-Pro}; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  W=5; ST="0 5"
  for s in ${STARTS:-$ST}; do n=mx_${p}_$s; echo -n "$n "; sub $n "$CD NBOOT=200 OUT_DIR=$D && $PY -u mix_check.py $P $s $((s + W)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt"; echo; done
done

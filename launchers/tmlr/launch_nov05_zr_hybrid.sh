#!/usr/bin/env bash
# 2026-11-05 (user-approved; free, GPU): ZeroRouter as published and with the 4B scout, with and without our cost probe, on the ten datasets:
# direct paired comparison, NBOOT=300, per-accuracy savings and per-query predictions (zr_faithful_draws.py). One GPU job per (pool, 5 training
# sets), into zrfaithful/hybrid; stored results are untouched.  Usage: SUBMIT=1 bash launch_nov05_zr_hybrid.sh
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/zrfaithful/hybrid; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned NTHREADS=8 RESULT_TAG=_pinned HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=1 GPU_MEM=32 CPU=8 CPU_MEM=96 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
for P in ${POOLS:-LCB Omni MMLU-Pro AIME APPS BCB CC SuperGPQA BBEH SWESmith}; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  for s in 0 5; do n=zrh_${p}_$s; echo -n "$n "; sub $n "$CD NBOOT=300 SAVE_PREDS=1 OUT_DIR=$D && $PY -u zr_faithful_draws.py $P $s $((s + 5)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt"; echo; done
done

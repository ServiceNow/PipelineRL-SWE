#!/usr/bin/env bash
# 2026-10-29 (user-approved; free, GPU): learning curves, cost-only and whole-system panels, n in {50, 100, 200, all} training problems
# on the 10 training sets (learning_curve.py; NEW_PATH 4.A.121). One GPU job per (pool, block of training sets): 2 sets per job on
# Omni-MATH and MMLU-Pro (1,000 / 6,500 test problems), 5 elsewhere.
# Usage: SUBMIT=1 bash launch_oct29_learning_curve.sh      SMOKE=1: one LCB training set, n in {50, all}, NBOOT=3, into $D/smoke
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/lcurve; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D/smoke
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned NTHREADS=8 RESULT_TAG=_pinned HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=1 GPU_MEM=16 CPU=8 CPU_MEM=96 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
if [[ ${SMOKE:-0} == 1 ]]; then
  n=lc_smoke_$(date +%s); echo -n "$n "; sub $n "$CD NBOOT=3 NS=50,all OUT_DIR=$D/smoke && $PY -u learning_curve.py LCB 0 1 > $D/smoke/lc_smoke.txt 2>&1; echo EXIT >> $D/smoke/lc_smoke.txt"; echo; exit 0
fi
for P in ${POOLS:-LCB Omni MMLU-Pro AIME APPS BCB CC SuperGPQA BBEH}; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  if [[ $P == Omni || $P == MMLU-Pro ]]; then W=2; ST="0 2 4 6 8"; else W=5; ST="0 5"; fi
  for s in ${STARTS:-$ST}; do n=lc_${p}_$s; echo -n "$n "; sub $n "$CD NBOOT=200 OUT_DIR=$D && $PY -u learning_curve.py $P $s $((s + W)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt"; echo; done
done

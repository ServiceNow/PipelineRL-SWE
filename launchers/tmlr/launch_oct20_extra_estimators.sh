#!/usr/bin/env bash
# 2026-10-20 (user-approved; free, GPU): Route-To-Reason (faithful: frozen mpnet + route embeddings + shared MLP) and SCX-style shared length
# 10 training sets (extra_estimators_draws.py; NEW_PATH 4.A.118), one GPU job per (pool, 5 training sets).
# Usage: SUBMIT=1 bash launch_oct19_zr_faithful.sh      SMOKE=1: one LCB training set, NBOOT=3, into $D/smoke
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/extraest; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D/smoke
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned NTHREADS=8 RESULT_TAG=_pinned HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=1 GPU_MEM=16 CPU=8 CPU_MEM=96 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
if [[ ${SMOKE:-0} == 1 ]]; then
  n=xe_smoke; echo -n "$n "; sub $n "$CD NBOOT=3 OUT_DIR=$D/smoke && $PY -u extra_estimators_draws.py LCB 0 1 > $D/smoke/$n.txt 2>&1; echo EXIT >> $D/smoke/$n.txt"; echo; exit 0
fi
for P in ${POOLS:-LCB Omni MMLU-Pro AIME APPS BCB CC SuperGPQA BBEH}; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  for s in ${STARTS:-0 5}; do n=xe_${p}_$s; echo -n "$n "; sub $n "$CD NBOOT=300 OUT_DIR=$D && $PY -u extra_estimators_draws.py $P $s $((s + 5)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt"; echo; done
done

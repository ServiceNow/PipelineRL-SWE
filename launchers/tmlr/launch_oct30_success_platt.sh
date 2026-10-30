#!/usr/bin/env bash
# 2026-10-30 (user-approved cheap check; free, GPU): our success probes with per-route Platt, without, and with one pooled Platt map (success_platt.py; NEW_PATH 4.A.128), one GPU job per (pool, 5 sets).
# Usage: SUBMIT=1 bash launch_oct30_success_platt.sh      SMOKE=1: one APPS training set, NBOOT=3, into $D/smoke
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/succplatt; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D/smoke
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned NTHREADS=8 RESULT_TAG=_pinned HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1"
sub() { cd /home/toolkit/PipelineRL-SWE && make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=1 GPU_MEM=16 CPU=8 CPU_MEM=96 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
if [[ ${SMOKE:-0} == 1 ]]; then
  n=sq_smoke_$(date +%s); echo -n "$n "; sub $n "$CD NBOOT=3 OUT_DIR=$D/smoke && $PY -u success_platt.py APPS 0 1 > $D/smoke/sq_smoke.txt 2>&1; echo EXIT >> $D/smoke/sq_smoke.txt"; echo; exit 0
fi
for P in ${POOLS:-APPS CC BCB LCB AIME SuperGPQA BBEH Omni MMLU-Pro SWESmith}; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  W=5; ST="0 5"
  for s in ${STARTS:-$ST}; do n=sq_${p}_$s; echo -n "$n "; sub $n "$CD NBOOT=200 OUT_DIR=$D && $PY -u success_platt.py $P $s $((s + W)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt"; echo; done
done

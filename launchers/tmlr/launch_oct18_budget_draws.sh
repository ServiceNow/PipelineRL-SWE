#!/usr/bin/env bash
# 2026-10-18 (user-approved; free): per-query hard budgets on the 10 training sets against deepseek-v4-flash alone, with a direct
# prefill readout of "fits within the cap" (budget_draws.py; NEW_PATH 4.A.114), one eai job per (pool, 5 training sets).
# Usage: SUBMIT=1 bash launch_oct18_budget_draws.sh
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/budgetdraws; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned NTHREADS=16 RESULT_TAG=_pinned NBOOT=300"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=128 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
for P in ${POOLS:-LCB Omni MMLU-Pro}; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  for s in ${STARTS:-0 5}; do n=bud_${p}_$s; echo -n "$n "; sub $n "$CD OUT_DIR=$D && $PY -u budget_draws.py $P $s $((s + 5)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt"; echo; done
done

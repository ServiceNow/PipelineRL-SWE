#!/usr/bin/env bash
# 2026-10-16 (user-approved; free): (1) other prefills on the 10 training sets (prefill_draws.py; NEW_PATH 4.A.112; Qwen3-8B excluded
# by user decision), one eai job per (pool, prefill, 5 training sets); (2) every cost estimator on the non-reasoning routes
# (nonreason_estimators.py; NEW_PATH 4.A.113), one job per (pool, 5 training sets). Usage: SUBMIT=1 bash launch_oct16_prefill_nonreason.sh
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D1=$R/reason_pinned_logs/prefilldraws; D2=$R/reason_pinned_logs/nrest; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D1 $D2
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned NTHREADS=16 RESULT_TAG=_pinned NBOOT=300"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=128 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
if [[ ${PART:-all} != nr ]]; then
for P in Omni MMLU-Pro; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  for T in ${TAGS:-q06b q17b q4bhyb phi4mini granite2b smol17b}; do
    for s in 0 5; do n=pfd_${p}_${T}_$s; echo -n "$n "; sub $n "$CD OUT_DIR=$D1 && $PY -u prefill_draws.py $P $T $s $((s + 5)) > $D1/$n.txt 2>&1; echo EXIT >> $D1/$n.txt"; echo; done
  done
done
fi
if [[ ${PART:-all} != pf ]]; then
for P in LCB Omni MMLU-Pro; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  for s in 0 5; do n=nre_${p}_$s; echo -n "$n "; sub $n "$CD OUT_DIR=$D2 && $PY -u nonreason_estimators.py $P $s $((s + 5)) > $D2/$n.txt 2>&1; echo EXIT >> $D2/$n.txt"; echo; done
done
fi

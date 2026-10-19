#!/usr/bin/env bash
# 2026-10-19 (user-approved; free, GPU): ZeroRouter as published (fine-tuned DistilBERT stage 2) next to the paper's 4B stand-in, on the
# 10 training sets (zr_faithful_draws.py; NEW_PATH 4.A.118), one GPU job per (pool, 5 training sets).
# Usage: SUBMIT=1 bash launch_oct19_zr_faithful.sh      SMOKE=1: one LCB training set, NBOOT=3, into $D/smoke
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/zrfaithful; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D/smoke
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned NTHREADS=8 RESULT_TAG=_pinned HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=1 GPU_MEM=32 CPU=8 CPU_MEM=96 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
if [[ ${SMOKE:-0} == 1 ]]; then
  n=zrf_smoke; echo -n "$n "; sub $n "$CD NBOOT=3 OUT_DIR=$D/smoke && $PY -u zr_faithful_draws.py LCB 0 1 > $D/smoke/$n.txt 2>&1; echo EXIT >> $D/smoke/$n.txt"; echo; exit 0
fi
for P in ${POOLS:-LCB Omni MMLU-Pro AIME APPS BCB CC SuperGPQA BBEH}; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  for s in ${STARTS:-0 5}; do n=zrf_${p}_$s; echo -n "$n "; sub $n "$CD NBOOT=300 OUT_DIR=$D && $PY -u zr_faithful_draws.py $P $s $((s + 5)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt"; echo; done
done

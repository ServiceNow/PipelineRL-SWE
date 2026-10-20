#!/usr/bin/env bash
# 2026-10-20 (user-approved; free, GPU): MixLLM as published on the mixed-traffic pool, benchmarks as its domains (fine-tuned
# bert-base-uncased + per-route length regressor), 10 training sets, one GPU job each (mixllm_mixed.py; NEW_PATH 4.A.119).
# Usage: SUBMIT=1 bash launch_oct20_mixllm_mixed.sh      SMOKE=1: training set 0, NBOOT=3, EPOCHS=1, into $D/smoke
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/mixllmmixed; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D/smoke
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned NTHREADS=16 HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=1 GPU_MEM=32 CPU=16 CPU_MEM=192 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
if [[ ${SMOKE:-0} == 1 ]]; then
  n=mxm_smoke; echo -n "$n "; sub $n "$CD NBOOT=3 EPOCHS=1 DRAW=0 RESULT_TAG=_smoke OUT_DIR=$D/smoke && $PY -u mixllm_mixed.py > $D/smoke/$n.txt 2>&1; echo EXIT >> $D/smoke/$n.txt"; echo; exit 0
fi
for s in ${SEEDS:-0 1 2 3 4 5 6 7 8 9}; do n=mxm_$s; echo -n "$n "
  sub $n "$CD NBOOT=300 DRAW=$s RESULT_TAG=_pinned_d$s OUT_DIR=$D && $PY -u mixllm_mixed.py > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt"; echo
done

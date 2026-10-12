#!/usr/bin/env bash
# 2026-10-11/12: re-run the generalisation battery on the refreshed BBEH / SuperGPQA pools (gpt-oss error draws refilled).
# Every reasoning-domain target is affected (BBEH / SuperGPQA are sources for the others). Coding targets are unchanged.
# The ad-hoc launch of Oct 11 lost its arguments (IndexError on sys.argv); this replaces it. One job per (mode, pool).
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; L=$R/reason_pinned_logs; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned RESULT_TAG=_pinned OUT_DIR=$L NTHREADS=16 &&"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=0 GPU_MEM=0 CPU=$3 CPU_MEM=$4 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
for M in transfer:cgxfer fulltransfer:cgfullxfer pooled:cgpooled whytransfer:cgwhy robust:cgrobust; do
  mode=${M%%:*}; tag=${M##*:}
  for P in Omni500 MMLU-Pro AIME SuperGPQA BBEH; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
    echo -n "${tag}_$p "; sub "${tag}_$p" "$CD $PY -u cost_generalization.py $mode $P > $L/costgen_${tag}_$p.txt 2>&1" 16 128
  done
done
for T in slopes gaps subjects; do echo -n "whytests_$T "; sub "whytests_$T" "$CD $PY -u why_transfer_tests.py $T > $L/why_transfer_tests_$T.txt 2>&1" 16 128; done

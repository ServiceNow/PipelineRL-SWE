#!/usr/bin/env bash
# 2026-10-19 (user-approved; free): pricing robustness of tab:fresh on the 10 training sets (NEW_PATH 4.A.116).
#  rft_<pool>_<a>  RATE_FIT=train: every model priced at its effective rate fitted on TRAINING calls only (LCB, Omni, MMLU-Pro)
#  nob_<pool>_<a>  NO_BILLED=1: Omni-MATH / MMLU-Pro test calls costed as tokens x rate instead of their billed cost
# Usage: SUBMIT=1 bash launch_oct19_price_checks.sh
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/pricechecks; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned OUT_DIR=$D NTHREADS=16 NBOOT=300"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=128 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
for P in LCB Omni MMLU-Pro; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  for s in 0 5; do n=rft_${p}_$s; echo -n "$n "; sub $n "$CD RATE_FIT=train RESULT_TAG=_ratetrain && $PY -u tmlr_strengthen.py draws $P $s $((s + 5)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt"; echo; done
done
for P in Omni MMLU-Pro; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  for s in 0 5; do n=nob_${p}_$s; echo -n "$n "; sub $n "$CD NO_BILLED=1 RESULT_TAG=_nobilled && $PY -u tmlr_strengthen.py draws $P $s $((s + 5)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt"; echo; done
done

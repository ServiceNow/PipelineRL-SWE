#!/usr/bin/env bash
# 2026-10-14 (user-approved; free): rerun of launch_oct13_singles.sh with the random-mix baseline (router vs the convex hull of single routes; NEW_PATH 4.A.97)
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/draws; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned OUT_DIR=$D NTHREADS=16 RESULT_TAG=_pinned NBOOT=300 &&"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=128 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
for P in LCB Omni MMLU-Pro AIME APPS BCB CC SuperGPQA BBEH; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  echo -n "dsm_$p "; sub dsm_$p "$CD $PY -u tmlr_strengthen.py singles $P 0 10 > $D/singles_$p.txt 2>&1; echo EXIT >> $D/singles_$p.txt"
done

#!/usr/bin/env bash
# 2026-10-14 (user-approved; free): size-matched text baselines (Qwen3-Embedding-4B; 8B as a one-line check) on the 10 training sets,
# 9 pools (tmlr_strengthen.py textemb; NEW_PATH 4.A.101). One eai job per (pool, 2 training sets). Usage: SUBMIT=1 bash launch_oct14_textemb.sh
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/textemb; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned OUT_DIR=$D NTHREADS=16 RESULT_TAG=_pinned NBOOT=300 &&"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=128 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
for P in ${POOLS:-LCB Omni MMLU-Pro AIME APPS BCB CC SuperGPQA BBEH}; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  for s in 0 2 4 6 8; do n=temb_${p}_$s; echo -n "$n "; sub $n "$CD $PY -u tmlr_strengthen.py textemb $P $s $((s + 2)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt"; done
done

#!/usr/bin/env bash
# 2026-11-06 (user-approved; free): few-label table relative to the median fitted on the SAME n problems, not the full training set.
# Same subsets as launch_oct13_draws.sh (dl_*), one job per pool and training set; the old baseline stays in the output as a check.
# STAGE=smoke: BBEH, set 0, one subset per n, three resamples. STAGE=full: 5 pools x 10 sets.
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/draws/labels_samen; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned OUT_DIR=$D NTHREADS=16 SAME_N_MEDIAN=1 SAVE_BS=1 HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 &&"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=128 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
if [[ ${STAGE:-smoke} == smoke ]]; then
  echo -n "lsn_smoke "; sub lsn_smoke "$CD export DRAW=0 RESULT_TAG=_smoke SEEDS=1 NBOOT=3 && $PY -u cost_generalization.py labels BBEH > $D/smoke.txt 2>&1; echo EXIT \$? >> $D/smoke.txt"
else
  for P in LCB Omni MMLU-Pro SuperGPQA BBEH; do p=$(echo $P | tr -d '-' | tr A-Z a-z); for s in 0 1 2 3 4 5 6 7 8 9; do
    echo -n "lsn_${p}_$s "; sub lsn_${p}_$s "$CD export DRAW=$s RESULT_TAG=_pinned_d$s SEEDS=2 NBOOT=200 && $PY -u cost_generalization.py labels $P > $D/labels_${p}_$s.txt 2>&1; echo EXIT \$? >> $D/labels_${p}_$s.txt"
  done; done
fi

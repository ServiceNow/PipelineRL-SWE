#!/usr/bin/env bash
# 2026-10-13 (user-approved; free, GPU): fine-tuned text encoders as cost-estimator arms of tab:fresh, NEW_PATH 4.A.90.
#  STAGE=1  ft_<pool>_<enc>  finetune_cost_estimator.py, one job per (pool, encoder) -> $L/ft_cost_<pool>_<enc>.jsonl + ft_<pool>_<enc>.txt
#  STAGE=2  ftb_fresh, ftb_lcb  fresh_baselines.py with FT_DIR=$L (every tab:fresh arm + ft-<enc>), RESULT_TAG=_pinned_ft
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; L=$R/reason_pinned_logs; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned OUT_DIR=$L HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 &&"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=$5 GPU_MEM=$6 CPU=$3 CPU_MEM=$4 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
if [[ ${STAGE:-1} == 1 ]]; then
  for P in LCB Omni MMLU-Pro; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
    for E in distilbert jina qwen06; do G=32; [[ $E == qwen06 ]] && G=80
      echo -n "ft_${p}_$E "; sub ft_${p}_$E "$CD $PY -u finetune_cost_estimator.py --pool $P --enc $E > $L/ft_${p}_$E.txt 2>&1; echo EXIT \$? >> $L/ft_${p}_$E.txt" 8 64 1 $G
    done
  done
else
  FT="export FT_DIR=$L RESULT_TAG=_pinned_ft NTHREADS=16 &&"
  echo -n "ftb_fresh "; sub ftb_fresh "$CD $FT $PY -u fresh_baselines.py > $L/ftb_fresh.txt 2>&1; echo EXIT \$? >> $L/ftb_fresh.txt" 16 128 0 0
  echo -n "ftb_lcb "; sub ftb_lcb "$CD $FT $PY -u fresh_baselines.py --pool LCB > $L/ftb_lcb.txt 2>&1; echo EXIT \$? >> $L/ftb_lcb.txt" 16 128 0 0
fi

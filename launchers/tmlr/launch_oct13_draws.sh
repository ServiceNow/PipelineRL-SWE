#!/usr/bin/env bash
# 2026-10-13 (user-approved; free): every quantitative claim averaged over 10 training draws (NEW_PATH 4.A.92). A draw = DRAW=s of
# decompose.split_manifest: the pool's train + calibration problems reshuffled into train / calibration of the same sizes, test kept.
#  STAGE=smoke  BBEH: draws (seeds -1, 0; NBOOT 3), labels and onboarding on draw 0 (tiny); STAGE=smoke2 the draws part only
#  STAGE=1      (PART=draws | rest | all, default all)
#               dr_<pool>_<a>   tmlr_strengthen.py draws, seeds -1..4 and 5..9 (tab:fresh, tab:pools, tab:allest; -1 = paper split)
#               dl_<pool>_<s>   cost_generalization.py labels on draw s, 2 subsets per n (tab:labels, fig:labels)
#               do_<pool>_<s>   onboard_compare.py HOLD=groups on draw s, 4 anchor draws per k (tab:onboard)
#               dft_lcb_<s>     finetune_cost_estimator.py LCB qwen06 on draw s (GPU)
#  STAGE=ft20   dft20_lcb_<s>   20-epoch rerun for the draws in FTS whose best epoch was the last of 10 -> $D/ft_e20 (keep lower calib. MSE)
#  STAGE=2      drf_lcb_<a>     LCB draws again with the fine-tuned predictions of STAGE=1 (FT_DIR)
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; L=$R/reason_pinned_logs; D=$L/draws; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned OUT_DIR=$D NTHREADS=16 HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 &&"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=${5:-0} GPU_MEM=${6:-0} CPU=$3 CPU_MEM=$4 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
mkdir -p $D
ALL="LCB Omni MMLU-Pro AIME APPS BCB CC SuperGPQA BBEH"; FIVE="LCB Omni MMLU-Pro SuperGPQA BBEH"
lc() { echo $1 | tr -d '-' | tr A-Z a-z; }
case ${STAGE:-smoke} in
smoke)
  echo -n "dsmoke_draws "; sub dsmoke_draws "$CD export RESULT_TAG=_smoke NBOOT=3 && $PY -u tmlr_strengthen.py draws BBEH -1 1 > $D/smoke_draws.txt 2>&1; echo EXIT \$? >> $D/smoke_draws.txt" 16 128
  echo -n "dsmoke_labels "; sub dsmoke_labels "$CD export DRAW=0 RESULT_TAG=_smoke SEEDS=1 NBOOT=3 SAVE_BS=1 && $PY -u cost_generalization.py labels BBEH > $D/smoke_labels.txt 2>&1; echo EXIT \$? >> $D/smoke_labels.txt" 16 128
  echo -n "dsmoke_onboard "; sub dsmoke_onboard "$CD export DRAW=0 RESULT_TAG=_smoke HOLD=groups SMOKE=1 SAVE_BS=1 && $PY -u onboard_compare.py BBEH > $D/smoke_onboard.txt 2>&1; echo EXIT \$? >> $D/smoke_onboard.txt" 16 128
  ;;
smoke2)
  echo -n "dsmoke2_draws "; sub dsmoke2_draws "$CD export RESULT_TAG=_smoke2 NBOOT=3 && $PY -u tmlr_strengthen.py draws BBEH -1 1 > $D/smoke2_draws.txt 2>&1; echo EXIT \$? >> $D/smoke2_draws.txt" 16 128
  ;;
1)
  [[ ${PART:-all} != rest ]] && for P in $ALL; do p=$(lc $P); [[ $P == LCB ]] && continue
    for a in -1 5; do b=$([[ $a == -1 ]] && echo 5 || echo 10); n=${p}_$((a < 0 ? 0 : a))
      echo -n "dr_$n "; sub dr_$n "$CD export RESULT_TAG=_pinned NBOOT=300 && $PY -u tmlr_strengthen.py draws $P $a $b > $D/draws_$n.txt 2>&1; echo EXIT \$? >> $D/draws_$n.txt" 16 128
    done
  done
  [[ ${PART:-all} == draws ]] && exit 0
  for P in $FIVE; do p=$(lc $P)
    for s in 0 1 2 3 4 5 6 7 8 9; do
      echo -n "dl_${p}_$s "; sub dl_${p}_$s "$CD export DRAW=$s RESULT_TAG=_pinned_d$s SEEDS=2 NBOOT=200 SAVE_BS=1 && $PY -u cost_generalization.py labels $P > $D/labels_${p}_$s.txt 2>&1; echo EXIT \$? >> $D/labels_${p}_$s.txt" 16 128
      echo -n "do_${p}_$s "; sub do_${p}_$s "$CD export DRAW=$s RESULT_TAG=_pinned_d$s HOLD=groups NDRAW=4 NBOOT=200 SAVE_BS=1 && $PY -u onboard_compare.py $P > $D/onboard_${p}_$s.txt 2>&1; echo EXIT \$? >> $D/onboard_${p}_$s.txt" 16 128
    done
  done
  for s in 0 1 2 3 4 5 6 7 8 9; do
    echo -n "dft_lcb_$s "; sub dft_lcb_$s "$CD export DRAW=$s DRAW_TAG=_d$s && $PY -u finetune_cost_estimator.py --pool LCB --enc qwen06 > $D/ft_lcb_qwen06_$s.txt 2>&1; echo EXIT \$? >> $D/ft_lcb_qwen06_$s.txt" 8 64 1 80
  done
  ;;
ft20)
  mkdir -p $D/ft_e20
  for s in $FTS; do
    echo -n "dft20_lcb_$s "; sub dft20_lcb_$s "$CD export OUT_DIR=$D/ft_e20 DRAW=$s DRAW_TAG=_d$s && $PY -u finetune_cost_estimator.py --pool LCB --enc qwen06 --epochs 20 > $D/ft_e20/ft_lcb_qwen06_$s.txt 2>&1; echo EXIT \$? >> $D/ft_e20/ft_lcb_qwen06_$s.txt" 8 64 1 80
  done
  ;;
2)
  for a in -1 5; do b=$([[ $a == -1 ]] && echo 5 || echo 10); n=lcb_$((a < 0 ? 0 : a))
    echo -n "drf_$n "; sub drf_$n "$CD export RESULT_TAG=_pinned NBOOT=300 FT_DIR=$D && $PY -u tmlr_strengthen.py draws LCB $a $b > $D/draws_$n.txt 2>&1; echo EXIT \$? >> $D/draws_$n.txt" 16 128
  done
  ;;
esac

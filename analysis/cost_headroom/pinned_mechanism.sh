#!/usr/bin/env bash
# NEW_PATH 4.A.66: mechanism + deployment analyses rerun on the PINNED shadow root (outputs tagged _pinned). One part per eai job.
# Usage: bash pinned_mechanism.sh why|lvr|drivers|onboard|budget
cd /home/toolkit/PipelineRL-SWE
PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; CH=analysis/cost_headroom; L=/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs
export REASON_ROOT=/mnt/llmd/results/exps/aristides/reason_pinned RESULT_TAG=_pinned HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1
POOLS="pool_v2_tensors_5rung:cost_preds_probe.jsonl omni500_tensors:cost_preds_probe_thinking.jsonl mmlupro_tensors:cost_preds_probe_instruct.jsonl"
case $1 in
  why) $PY -u $CH/why_decompose.py > $L/mech_why.txt 2>&1 ;;
  lvr) for pc in $POOLS; do p=${pc%%:*}; c=${pc##*:}; $PY -u $CH/level_vs_relative.py $p $c >> $L/mech_lvr.txt 2>&1
         t=${c#cost_preds_}; t=${t%.jsonl}
         $PY -u $CH/decompose.py --out lvr_$p $p:cost_preds_lvr_oracle_full.jsonl:market $p:cost_preds_lvr_oracle_level.jsonl:market \
           $p:cost_preds_lvr_oracle_diff.jsonl:market $p:cost_preds_lvr_${t}_level.jsonl:market $p:cost_preds_lvr_${t}_diff.jsonl:market $p:$c:market >> $L/mech_lvr.txt 2>&1; done ;;
  drivers) $PY -u $CH/mmlupro_length_driver.py > $L/mech_drivers.txt 2>&1
         D=mmlupro_tensors; fs=$(ls $REASON_ROOT/$D | grep '^cost_preds_driver_' | sed "s#^#$D:#; s#\$#:market#" | tr '\n' ' ')
         $PY -u $CH/decompose.py --out drivers_mmlupro $fs $D:cost_preds_probe_instruct.jsonl:market >> $L/mech_drivers.txt 2>&1 ;;
  onboard) for pc in $POOLS; do $PY -u $CH/onboard_full.py ${pc%%:*} ${pc##*:} >> $L/mech_onboard.txt 2>&1; done ;;
  budget) $PY -u $CH/budget_cap.py > $L/mech_budget.txt 2>&1 ;;
esac
echo "EXIT $? ($1)" >> $L/mech_$1.txt

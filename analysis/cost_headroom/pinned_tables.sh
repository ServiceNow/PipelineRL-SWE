#!/usr/bin/env bash
# NEW_PATH 4.A.59: the remaining fresh-set tables/figures with deepseek-v4-flash pinned (REASON_ROOT shadow, outputs tagged _pinned).
cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom
PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; L=/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs
export REASON_ROOT=/mnt/llmd/results/exps/aristides/reason_pinned RESULT_TAG=_pinned HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1
pids=""
for s in billed_reprice rep_head_grid second_family_eval size_sweep_eval; do $PY -u $s.py > $L/$s.txt 2>&1 & pids="$pids $!"; done
$PY -u ../../paper_nowai/make_billed_curves.py > $L/make_billed_curves.txt 2>&1 & pids="$pids $!"
wait $pids; echo "PINNED TABLES DONE" >> $L/pinned_tables.txt

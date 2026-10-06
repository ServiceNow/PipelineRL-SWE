#!/usr/bin/env bash
# NEW_PATH 4.A.59: unpinned vs pinned dsv4f through the SAME script (reprice_original.py: list + billed, paired bootstrap) for every pool.
# Unpinned = real root (archived readouts); pinned = reason_pinned (refit readouts). Usage: bash reprice_matched.sh [pool ...]
cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom
PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; REAL=/mnt/llmd/results/exps/aristides/reason; SH=$REAL/../reason_pinned; L=$REAL/reason_pinned_logs
declare -A SPEC=( [lcb]="LCB:pool_v2_tensors_5rung:cost_preds_probe.jsonl" [omni]="Omni:omni500_tensors:cost_preds_probe_thinking.jsonl"
  [mmlupro]="MMLU-Pro:mmlupro_tensors:cost_preds_probe_instruct.jsonl" [aime]="AIME:aime_tensors:cost_preds_probe_instruct.jsonl"
  [apps]="APPS:apps_tensors:cost_preds_probe.jsonl" [bcb]="BCB:bcb_tensors_5r:cost_preds_probe.jsonl" [cc]="CC:cc_tensors:cost_preds_probe.jsonl" )
pids=""
for p in ${@:-lcb omni mmlupro aime apps bcb}; do
  ( RESULT_TAG=_unpinned $PY reprice_original.py ${SPEC[$p]} > $L/${p}_reprice_unpinned.txt 2>&1 ) & pids="$pids $!"
  ( REASON_ROOT=$SH RESULT_TAG=_pinned $PY reprice_original.py ${SPEC[$p]} > $L/${p}_reprice.txt 2>&1 ) & pids="$pids $!"
done; wait $pids
for p in ${@:-lcb omni mmlupro aime apps bcb}; do echo "== $p"; grep -h "learned saves" $L/${p}_reprice_unpinned.txt | sed 's/^/  unpinned /'; grep -h "learned saves" $L/${p}_reprice.txt | sed 's/^/  pinned   /'; done
echo REPRICE MATCHED DONE

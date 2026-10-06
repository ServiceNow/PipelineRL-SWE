#!/usr/bin/env bash
# NEW_PATH 4.A.60: pinned estimator suite on LCB's own temporal split (fresh_baselines.py --pool LCB, incl. MixLLM-style embeddings)
# and the pinned refit bootstrap on the fresh sets (refit_bootstrap.py 300). Outputs tagged _pinned.
cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom
PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; L=/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs
export REASON_ROOT=/mnt/llmd/results/exps/aristides/reason_pinned RESULT_TAG=_pinned HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1
( $PY baseline_cost_heads.py pool_v2_tensors_5rung /mnt/llmd/results/exps/aristides/reason/pv2_scout_prefill_1756715297/scout.npz --only mixllm > $L/lcb_mixllm.txt 2>&1
  $PY -u fresh_baselines.py --pool LCB > $L/lcb_baselines.txt 2>&1 ) & a=$!
$PY -u refit_bootstrap.py 300 > $L/refit_bootstrap.txt 2>&1 & b=$!
wait $a $b; echo "LCB SUITE DONE" >> $L/lcb_suite.txt

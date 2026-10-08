#!/usr/bin/env bash
# NEW_PATH 4.A.61: full pinned estimator suite on one original pool (fresh_baselines.py --pool P): median, mean, prompt GBM,
# MixLLM-style embeddings, full ZeroRouter, our two difficulty ablations, and the headroom oracle; test split, billed rates.
# Usage: bash pool_suite.sh LCB|BCB|APPS|AIME|CC
cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom
PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; L=/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs
export REASON_ROOT=/mnt/llmd/results/exps/aristides/reason_pinned RESULT_TAG=_pinned HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1
P=$1; read T FEAT < <($PY -c "src=open('fresh_baselines.py').read(); ns={}; exec(src[src.index('ORIG ='):src.index('POOL =')], ns); o=ns['ORIG']['$P']; print(o[0], o[2])")
[ -s $REASON_ROOT/$T/cost_preds_mixllm.jsonl ] || $PY baseline_cost_heads.py $T $FEAT --only mixllm > $L/${P}_mixllm.txt 2>&1
$PY -u fresh_baselines.py --pool $P > $L/${P}_suite.txt 2>&1
echo "EXIT $?" >> $L/${P}_suite.txt

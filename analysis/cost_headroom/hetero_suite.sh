#!/usr/bin/env bash
# NEW_PATH 4.A.70: heterogeneous pools (SuperGPQA, BIG-Bench Extra Hard), collected with deepseek-v4-flash already pinned to StreamLake.
# Builds the tensors (build_math_tensors.py, seed-0 55/15/30 split), links them into the pinned shadow root, fits the paper readouts
# (success: activation_content_preds --rich --select-C; cost: probe_model_compare on the Qwen3-4B Instruct prefill), then runs
# the full estimator suite (pool_suite.sh: MixLLM embeddings need the GPU) and the mechanism analyses in parallel.
# Usage: bash hetero_suite.sh supergpqa|bbeh
set -uo pipefail
cd /home/toolkit/PipelineRL-SWE
PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; REAL=/mnt/llmd/results/exps/aristides/reason; SH=$REAL/../reason_pinned; CH=analysis/cost_headroom
export HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
DS=$1; LABEL=$([[ $DS == supergpqa ]] && echo SuperGPQA || echo BBEH); T=${DS}_tensors; FEAT=$REAL/${DS}_probe/instruct.npz
L=$REAL/reason_pinned_logs; mkdir -p $L; exec > >(tee -a $L/${DS}_hetero.log) 2>&1
$PY $CH/build_math_tensors.py $DS || exit 1
ln -sfn $REAL/$T $SH/$T
D=$SH/$T
$PY pipelinerl/swe/scripts/livecodebench/activation_content_preds.py --activations $FEAT --tensors-dir $D --rich --select-C \
  --out $D/content_preds.jsonl > $D/content.log 2>&1 || { tail -5 $D/content.log; exit 1; }
REASON_ROOT=$SH $PY $CH/probe_model_compare.py $T instruct=$FEAT > $D/cost.log 2>&1 || { tail -5 $D/cost.log; exit 1; }
tail -n 1 $D/cost.log
export REASON_ROOT=$SH RESULT_TAG=_pinned OUT_DIR=$L
bash $CH/pool_suite.sh $LABEL & p1=$!
$PY -u $CH/tmlr_free_analyses.py $LABEL > $L/${DS}_free.txt 2>&1 & p2=$!
$PY -u $CH/label_efficiency.py $LABEL > $L/${DS}_label_efficiency.txt 2>&1 & p3=$!
$PY -u $CH/prefill_router_repro.py $LABEL > $L/${DS}_prefill_router_repro.txt 2>&1 & p4=$!
$PY -u $CH/cascade_oracle.py $LABEL > $L/${DS}_cascade.txt 2>&1 & p5=$!
wait $p1 $p2 $p3 $p4 $p5          # explicit PIDs: a bare wait also waits for the tee process substitution
tail -n 25 $L/${LABEL}_suite.txt; grep -h "=====\|saves\|n=" $L/${DS}_label_efficiency.txt $L/${DS}_prefill_router_repro.txt $L/${DS}_cascade.txt
echo "HETERO SUITE $DS DONE"

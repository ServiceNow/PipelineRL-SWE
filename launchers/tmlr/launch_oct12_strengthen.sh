#!/usr/bin/env bash
# 2026-10-12 (user-approved items 1-4; all free): deployable policies, random re-splits, price-gap sweep, prefill overhead.
#  deploy_<pool>      calibration-chosen operating points applied once to test, every cost arm (tmlr_strengthen.py deploy)
#  resplit_<pool>_<a> whole pipeline on random re-splits, seeds a..a+4 (tmlr_strengthen.py resplit)
#  pricegap_<pool>    price spread sweep, flat .. ~270x (tmlr_strengthen.py pricegap)
#  prefill_overhead   Qwen3-4B prompt lengths and forward-pass latency on one GPU (prefill_overhead.py)
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; L=$R/reason_pinned_logs; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3
ENVS="export HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 &&"
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned RESULT_TAG=_pinned OUT_DIR=$L NTHREADS=16 &&"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=${5:-0} GPU_MEM=${6:-0} CPU=$3 CPU_MEM=$4 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
for P in LCB Omni MMLU-Pro AIME APPS BCB CC SuperGPQA BBEH; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  echo -n "deploy_$p "; sub deploy_$p "$CD $PY -u tmlr_strengthen.py deploy $P > $L/strengthen_deploy_$p.txt 2>&1" 16 128
  echo -n "pricegap_$p "; sub pricegap_$p "$CD $PY -u tmlr_strengthen.py pricegap $P > $L/strengthen_pricegap_$p.txt 2>&1" 16 128
  for a in 0 5; do echo -n "resplit_${p}_$a "; sub resplit_${p}_$a "$CD export NBOOT=100 && $PY -u tmlr_strengthen.py resplit $P $a $((a+5)) > $L/strengthen_resplit_${p}_$a.txt 2>&1" 16 128; done
done
echo -n "prefill_overhead "; sub prefill_overhead "$ENVS cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && OUT_DIR=$L $PY -u prefill_overhead.py > $L/prefill_overhead.txt 2>&1" 8 64 1 40

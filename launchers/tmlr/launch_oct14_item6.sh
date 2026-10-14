#!/usr/bin/env bash
# 2026-10-14 (user-approved; free): the remaining manifest-split analyses on the 10 training sets (DRAW=s; NEW_PATH 4.A.102):
# deployable policies (tmlr_strengthen.py deploy, 9 pools), full transfer (cost_generalization.py fulltransfer, 7 targets),
# reasoning vs non-reasoning (nonreason_compare.py, 3 pools), new families incl. onboarding (newfamily_eval.py). One eai job per
# (analysis, pool, 2 training sets). Outputs: $D/<analysis>/..._pinned_d<s>.json. Usage: SUBMIT=1 [WHAT="deploy transfer nonreason families"] bash ...
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/item6; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned NTHREADS=16 NBOOT=300"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=128 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
two() {  # $1 job prefix, $2 out subdir, $3 command (DRAW and RESULT_TAG set per draw)
  mkdir -p $D/$2
  for s in 0 2 4 6 8; do n=${1}_$s; c="$CD OUT_DIR=$D/$2"
    for t in $s $((s + 1)); do c="$c && DRAW=$t RESULT_TAG=_pinned_d$t $3 > $D/$2/${1}_d$t.txt 2>&1"; done
    echo -n "$n "; sub $n "$c; echo EXIT >> $D/$2/${1}_d$((s + 1)).txt"
  done
}
for w in ${WHAT:-deploy transfer nonreason families}; do case $w in
  deploy) for P in LCB Omni MMLU-Pro AIME APPS BCB CC SuperGPQA BBEH; do p=$(echo $P | tr -d '-' | tr A-Z a-z); two i6dep_$p deploy "$PY -u tmlr_strengthen.py deploy $P"; done ;;
  transfer) for P in LCB APPS MMLU-Pro AIME SuperGPQA BBEH Omni500; do p=$(echo $P | tr -d '-' | tr A-Z a-z); two i6xfer_$p transfer "$PY -u cost_generalization.py fulltransfer $P"; done ;;
  nonreason) for P in LCB Omni MMLU-Pro; do p=$(echo $P | tr -d '-' | tr A-Z a-z); two i6nr_$p nonreason "$PY -u nonreason_compare.py $P"; done ;;
  families) two i6fam families "$PY -u newfamily_eval.py LCB,Omni,MMLU-Pro" ;;
esac; done

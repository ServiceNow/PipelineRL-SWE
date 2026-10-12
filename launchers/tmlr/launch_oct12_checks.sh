#!/usr/bin/env bash
# 2026-10-12 (user-approved; all free): reviewer-proofing checks, NEW_PATH 4.A.88.
#  tr_<pool>        estimator suite with billed rates fitted on TRAINING calls only (billed.py RATE_FIT=train; SuperGPQA / BBEH: their
#                   own training calls, RATE_POOL), outputs pool_baselines_<P>_pinned_trainrate.json / fresh_baselines_pinned_trainrate.json
#  rtc_<pool>_<a>   draw-preserving re-splits (train + calibration reshuffled, paper test kept; seed -1 = paper split, same pipeline)
#  rsc_<pool>       matched control for the Oct 12 random re-splits: `resplit` on the paper split (seed -1), same pipeline
#  sd_<pool>        split-draw headroom: realized-cost oracle vs held-out-draw oracle on multi-draw test problems
# (The level-vs-differences and MMLU-Pro driver rows at billed prices took seconds and were run directly: mech_*_billed.txt.)
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; L=$R/reason_pinned_logs; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned OUT_DIR=$L NTHREADS=16 HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1 &&"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=0 GPU_MEM=0 CPU=$3 CPU_MEM=$4 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
TR="export RESULT_TAG=_pinned_trainrate RATE_FIT=train &&"
echo -n "tr_fresh "; sub tr_fresh "$CD $TR $PY -u fresh_baselines.py > $L/trainrate_fresh.txt 2>&1; echo EXIT \$? >> $L/trainrate_fresh.txt" 16 128
for P in LCB AIME APPS BCB CC SuperGPQA BBEH; do p=$(echo $P | tr A-Z a-z); RP=""; [[ $P == SuperGPQA || $P == BBEH ]] && RP="export RATE_POOL=$P &&"
  echo -n "tr_$p "; sub tr_$p "$CD $TR $RP $PY -u fresh_baselines.py --pool $P > $L/trainrate_$p.txt 2>&1; echo EXIT \$? >> $L/trainrate_$p.txt" 16 128
done
RS="export RESULT_TAG=_pinned NBOOT=100 &&"
for P in Omni MMLU-Pro; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
  for a in -1 0 5; do b=$((a < 0 ? 0 : a + 5)); n=${a#-}; [[ $a == -1 ]] && n=ctl
    echo -n "rtc_${p}_$n "; sub rtc_${p}_$n "$CD $RS $PY -u tmlr_strengthen.py resplit_tc $P $a $b > $L/strengthen_rtc_${p}_$n.txt 2>&1" 16 128; done
done
for P in LCB AIME APPS BCB CC SuperGPQA BBEH; do p=$(echo $P | tr A-Z a-z)
  echo -n "rsc_$p "; sub rsc_$p "$CD $RS $PY -u tmlr_strengthen.py resplit $P -1 0 > $L/strengthen_rsc_$p.txt 2>&1" 16 128
done
for P in LCB AIME BCB CC SuperGPQA BBEH Omni500; do p=$(echo $P | tr A-Z a-z)
  echo -n "sd_$p "; sub sd_$p "$CD export RESULT_TAG=_pinned NBOOT=100 REPS=20 && $PY -u tmlr_strengthen.py splitdraw $P > $L/strengthen_sd_$p.txt 2>&1" 16 128
done

#!/usr/bin/env bash
# 2026-10-29 (user-approved; free): SWE-Smith one-shot patches brought up to the main protocol (NEW_PATH 4.A.124): the same ten training
# sets and every arm of tab:allest / tab:pools as the nine datasets, plus the mix check and learning curves.
#   dr_swesmith_{0,5}   tmlr_strengthen.py draws (seeds -1..4, 5..9): headroom, median, mean, GBM, MixLLM-style, kNN, ZeroRouter pricing,
#                       difficulty bins, from success, ZeroRouter (4B)                                                        (CPU)
#   xe_swesmith_{0,5}   extra_estimators_draws.py: Route-To-Reason, SCX-style                                                (GPU)
#   zrf_swesmith_{0,5}  zr_faithful_draws.py: ZeroRouter as published (DistilBERT)                                            (GPU)
#   mx_swesmith_{0,5}   mix_check.py: every estimator vs a random mix of single routes                                         (GPU)
#   lc_swesmith_{0,5}   learning_curve.py                                                                                      (GPU)
# Usage: SUBMIT=1 bash launch_oct29_swesmith.sh
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; L=$R/reason_pinned_logs; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned RESULT_TAG=_pinned HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=$3 GPU_MEM=$4 CPU=$5 CPU_MEM=$6 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
for a in -1 5; do b=$([[ $a == -1 ]] && echo 5 || echo 10); k=$((a < 0 ? 0 : a)); D=$L/draws
  n=dr_swesmith_$k; echo -n "$n "; sub $n "$CD NTHREADS=16 NBOOT=300 OUT_DIR=$D && $PY -u tmlr_strengthen.py draws SWESmith $a $b > $D/draws_swesmith_$k.txt 2>&1; echo EXIT \$? >> $D/draws_swesmith_$k.txt" 0 0 16 128; echo
done
for s in 0 5; do
  D=$L/extraest; n=xe_swesmith_$s; echo -n "$n "; sub $n "$CD NTHREADS=8 NBOOT=300 OUT_DIR=$D && $PY -u extra_estimators_draws.py SWESmith $s $((s + 5)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt" 1 16 8 96; echo
  D=$L/zrfaithful; n=zrf_swesmith_$s; echo -n "$n "; sub $n "$CD NTHREADS=8 NBOOT=300 OUT_DIR=$D && $PY -u zr_faithful_draws.py SWESmith $s $((s + 5)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt" 1 16 8 96; echo
  D=$L/mixcheck; n=mx_swesmith_$s; echo -n "$n "; sub $n "$CD NTHREADS=8 NBOOT=200 OUT_DIR=$D && $PY -u mix_check.py SWESmith $s $((s + 5)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt" 1 16 8 96; echo
  D=$L/lcurve; n=lc_swesmith_$s; echo -n "$n "; sub $n "$CD NTHREADS=8 NBOOT=200 OUT_DIR=$D && $PY -u learning_curve.py SWESmith $s $((s + 5)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt" 1 16 8 96; echo
done

#!/usr/bin/env bash
# 2026-10-29 (user-approved; free, GPU): prefill-size scale table (NEW_PATH 4.A.122). Qwen3 hybrid 0.6B / 1.7B / 4B / 8B / 14B plus the
# paper's Qwen3-4B-2507, on LCB, Omni-MATH and MMLU-Pro; the 8B / 14B rows are a user-approved exception to the 4B cap, for this table only.
#   PHASE=extract   features: LCB for every hybrid size, Omni-MATH / MMLU-Pro for 14B (the others exist from 4.A.50), one GPU job per model
#   PHASE=latency   one GPU job: every model's batch-1 latency and batch-16 throughput on the same GPU (prefill_overhead.py)
#   PHASE=draws     complete routers on each prefill, 10 training sets (prefill_draws.py), one CPU job per (pool, tag, 5 sets);
#                   tags already run for tab:size (Omni-MATH / MMLU-Pro x 0.6B / 1.7B / 4B) are skipped
# Usage: SUBMIT=1 PHASE=extract|latency|draws bash launch_oct29_prefill_scale.sh
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; D=$R/reason_pinned_logs/prefillscale; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p $D
HERE=/home/toolkit/PipelineRL-SWE/analysis/cost_headroom
declare -A MODEL=([q06b]=Qwen/Qwen3-0.6B [q17b]=Qwen/Qwen3-1.7B [q4bhyb]=Qwen/Qwen3-4B [q8b]=Qwen/Qwen3-8B [q14b]=/mnt/llmd/base_models/Qwen3-14B)
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=$3 GPU_MEM=$4 CPU=$5 CPU_MEM=$6 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
case ${PHASE:?} in
extract)
  for T in q06b q17b q4bhyb q8b q14b; do DS=lcb; [[ $T == q14b ]] && DS="lcb omni500 mmlupro"
    n=pfs_ext_$T; echo -n "$n "; sub $n "bash $HERE/run_extract_scale.sh ${MODEL[$T]} $T '$DS' > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt" 1 80 8 96; echo; done ;;
latency)
  SPECS=paper=Qwen/Qwen3-4B-Instruct-2507; for T in q06b q17b q4bhyb q8b q14b; do SPECS="$SPECS,$T=${MODEL[$T]}"; done
  n=pfs_latency; echo -n "$n "; sub $n "cd $HERE && export HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1 POOLS_ONLY=LCB,Omni500,MMLU-Pro PREFILL_MODELS=$SPECS OUT_DIR=$D && $PY -u prefill_overhead.py > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt" 1 80 8 96; echo ;;
draws)
  CD="cd $HERE && export REASON_ROOT=$R/../reason_pinned NTHREADS=16 RESULT_TAG=_pinned NBOOT=300 OUT_DIR=$D"
  for P in LCB Omni MMLU-Pro; do p=$(echo $P | tr -d '-' | tr A-Z a-z)
    for T in q06b q17b q4bhyb q8b q14b; do [[ $P != LCB && $T != q8b && $T != q14b ]] && continue
      for s in 0 5; do n=pfs_${p}_${T}_$s; echo -n "$n "; sub $n "$CD && $PY -u prefill_draws.py $P $T $s $((s + 5)) > $D/$n.txt 2>&1; echo EXIT >> $D/$n.txt" 0 0 16 160; echo; done
    done
  done ;;
esac

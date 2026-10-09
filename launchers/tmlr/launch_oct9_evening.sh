#!/usr/bin/env bash
# 2026-10-09 evening (user-approved items 1, 2, 3, 5):
#  1 swev2_analysis   waits for SWE-Smith v2 Daytona labels, builds pinned tensors, readouts, full suite (free)
#  2 cgrobust_<pool>  robustness bootstrap: drop under transfer per arm, and drop - ours (free)
#  3 workedex         worked examples (long / short for their difficulty) for the paper (free)
#  5 refills (PAID, cents; $1 guards): Qwen3-235B non-reasoning training rows (concurrency 4: its pinned provider rate-limits),
#    BBEH / SuperGPQA gpt-oss API-error draws (concurrency 32; the originals ran at 384); each followed by its downstream rerun.
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; L=$R/reason_pinned_logs; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3
ENVS="export HF_HOME=/home/toolkit/.cache/huggingface HF_DATASETS_CACHE=/home/toolkit/.cache/huggingface/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
CD="cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom && export REASON_ROOT=$R/../reason_pinned RESULT_TAG=_pinned OUT_DIR=$L NTHREADS=16 &&"
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 GPU=0 GPU_MEM=0 CPU=$3 CPU_MEM=$4 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; }
cat > $R/math_pool_nonreason/run_refill_qw235.sh <<EOS
#!/usr/bin/env bash
cd /home/toolkit/PipelineRL-SWE; $ENVS
$PY pipelinerl/swe/scripts/math_pool/collect_math_pool.py --out-dir $R/math_pool_nonreason --pilot 0 --routes nr_qw235:1 --datasets omni500,mmlupro \
  --concurrency 4 --max-tokens 16000 --budget-usd 1 > $R/math_pool_nonreason/collect_refill_qw235.log 2>&1
cd analysis/cost_headroom; export REASON_ROOT=$R/../reason_pinned RESULT_TAG=_pinned OUT_DIR=$L
for P in Omni MMLU-Pro; do $PY -u nonreason_compare.py \$P > $L/nonreason_refill_\$(echo \$P | tr -d '-' | tr A-Z a-z).txt 2>&1; done
echo REFILL QW235 DONE >> $R/math_pool_nonreason/collect_refill_qw235.log
EOS
cat > $R/math_pool/run_refill_hetero.sh <<EOS
#!/usr/bin/env bash
cd /home/toolkit/PipelineRL-SWE; $ENVS
for DS in bbeh supergpqa; do
  $PY pipelinerl/swe/scripts/math_pool/collect_math_pool.py --out-dir $R/math_pool --pilot 0 --routes oss20lo:4,oss20md:3,oss120md:2,oss120hi:2 \
    --datasets \$DS --concurrency 32 --budget-usd 1 > $R/math_pool/collect_refill_\${DS}_oss.log 2>&1
  # the rebuild overwrites this pool's tensors / readouts: wait for the jobs that read them (robustness, worked examples)
  until grep -q "^DONE" $L/costgen_cgrobust_\$DS.txt 2>/dev/null && grep -q "^DONE" $L/worked_examples.txt 2>/dev/null; do sleep 120; done
  bash analysis/cost_headroom/hetero_suite.sh \$DS > $L/refill_\${DS}_hetero.txt 2>&1
done
echo REFILL HETERO DONE >> $R/math_pool/collect_refill_bbeh_oss.log
EOS
chmod +x $R/math_pool_nonreason/run_refill_qw235.sh $R/math_pool/run_refill_hetero.sh
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
echo -n "swev2_analysis "; sub swev2_analysis "bash $R/swesmith_reasoning_pool_v2/run_analysis.sh" 16 96
for P in LCB APPS BCB CC Omni500 MMLU-Pro AIME SuperGPQA BBEH; do p=$(echo $P | tr -d '-' | tr A-Z a-z); echo -n "cgrobust_$p "; sub cgrobust_$p "$CD $PY -u cost_generalization.py robust $P > $L/costgen_cgrobust_$p.txt 2>&1" 16 128; done
echo -n "workedex "; sub workedex "$CD $PY -u worked_examples.py > $L/worked_examples.txt 2>&1" 16 128
echo -n "refill_qw235 "; sub refill_qw235 "bash $R/math_pool_nonreason/run_refill_qw235.sh" 8 64
echo -n "refill_hetero "; sub refill_hetero "bash $R/math_pool/run_refill_hetero.sh" 16 128

#!/usr/bin/env bash
# NEW_PATH 4.A.59: recollect deepseek-v4-flash PINNED to one provider (default StreamLake; provider.only + no fallbacks via
# OPENROUTER_PIN_PROVIDER) on every pool, same prompts / settings / draw counts as the original collections. Outputs go to NEW dirs
# (originals untouched). APPS and the live run already have StreamLake-pinned draws. CPU eai jobs; commit + push first.
#   LCB  8 draws x 892   -> pool_v2_lcb_pinned/         (one job per draw)
#   BCB  8 draws x 1101  -> pool_v2_bcb_pinned/         (one job per draw)
#   CC   3 draws x 700   -> cc_pool/full_pinned/        (one job)
#   Omni-500 / MMLU-Pro-1000 / AIME-933, 3 draws -> math_pool_pinned/ (one job, $15 guard)
#   fresh MMLU-Pro 6,500 + fresh Omni 1,000, 1 draw -> math_expand_pinned_20261005/ (two jobs, $5 guards)
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; PIN=${PIN:-StreamLake}; C=${CONCURRENCY:-32}; WHAT=${WHAT:-all}
GAP=35; submit() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=${3:-8} CPU_MEM=${4:-32} SNAPSHOT=1 COMMAND="$2" 2>&1 | grep -E "^[0-9a-f]{8}-|rror" || true; sleep $GAP; }
if [[ $WHAT == all || $WHAT == lcb ]]; then PIN=$PIN ONLY=dsv4f CONCURRENCY=$C BASE=$R/pool_v2_lcb_pinned SUBMIT=${SUBMIT:-0} bash launchers/abstention/launch_pool_recollect.sh; fi
if [[ $WHAT == all || $WHAT == bcb ]]; then PIN=$PIN ONLY=dsv4f CONCURRENCY=$C BASE=$R/pool_v2_bcb_pinned SUBMIT=${SUBMIT:-0} bash launchers/abstention/launch_bcb_collect.sh; fi
O=$R/cc_pool/full_pinned; mkdir -p $O
cat > $O/run.sh <<EOF
#!/usr/bin/env bash
cd /home/toolkit/PipelineRL-SWE; export OPENROUTER_PIN_PROVIDER=$PIN
CC="-m pipelinerl.swe.scripts.codecontests.collect_cc_expert --tasks-file $R/cc_pool/cc_tasks.jsonl --output-dir $O --concurrency $C --grade-concurrency 4"
for d in 0 1 2; do python \$CC --route-label dsv4f --model deepseek/deepseek-v4-flash --temperature 0.7 --top-p 0.95 --max-tokens 128000 --reasoning-enabled --require-parameters --output-suffix _d\$d > $O/log_dsv4f_d\$d.txt 2>&1 & done
wait; echo ALL DONE > $O/done.txt
EOF
O2=$R/math_pool_pinned; mkdir -p $O2
cat > $O2/run.sh <<EOF
#!/usr/bin/env bash
cd /home/toolkit/PipelineRL-SWE; export OPENROUTER_PIN_PROVIDER=$PIN
export HF_HOME=/home/toolkit/.cache/huggingface HF_DATASETS_CACHE=/home/toolkit/.cache/huggingface/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python pipelinerl/swe/scripts/math_pool/collect_math_pool.py --out-dir $O2 --pilot 0 --routes dsv4f:3 --datasets omni500,mmlupro,aime \
  --concurrency 384 --budget-usd 15 > $O2/collect.log 2>&1
echo ALL DONE >> $O2/collect.log
EOF
O3=$R/math_expand_pinned_20261005; mkdir -p $O3
for ds in mmlupro omni500; do cat > $O3/run_$ds.sh <<EOF
#!/usr/bin/env bash
cd /home/toolkit/PipelineRL-SWE; export OPENROUTER_PIN_PROVIDER=$PIN
python -u pipelinerl/swe/scripts/math_pool/collect_expansion.py --plan analysis/cost_headroom/expansion_20261001/plan_pinned_dsv4f.json \
  --tasks analysis/cost_headroom/expansion_20261001/${ds}_tasks.jsonl --dataset $ds --out $O3 --budget-usd 5 --concurrency 128 > $O3/collect_$ds.log 2>&1
echo EXIT \$? >> $O3/collect_$ds.log
EOF
done
chmod +x $O/run.sh $O2/run.sh $O3/run_*.sh
if [[ ${SUBMIT:-0} != 1 ]]; then echo "dry run: CC $O/run.sh, math $O2/run.sh, fresh $O3/run_{mmlupro,omni500}.sh"; exit 0; fi
if [[ $WHAT == all || $WHAT == cc ]]; then submit pin_cc_$(date -u +%H%M%S) "bash $O/run.sh" 16 64; fi
if [[ $WHAT == all || $WHAT == math ]]; then submit pin_math_$(date -u +%H%M%S) "bash $O2/run.sh" 4 16; fi
if [[ $WHAT == all || $WHAT == fresh ]]; then for ds in mmlupro omni500; do submit pin_fresh_${ds}_$(date -u +%H%M%S) "bash $O3/run_$ds.sh" 4 16; done; fi

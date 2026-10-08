#!/usr/bin/env bash
# NEW_PATH 4.A.64: NON-REASONING pool on the same problems as the reasoning pool. Routes (each pinned to one provider, card sampling) are
# defined in collect_math_pool.ROUTES (nr_*). One draw per route, 16k output cap. Parts: math (Omni-500 + MMLU-Pro 1,000; $6 guard),
# fresh (Omni 1,000 test + MMLU-Pro 2,000 test subset; $4 guards), lcb (892 problems, one job per route). Commit + push first.
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; NR=${NR:-nr_ds4off nr_llama8 nr_qw30 nr_llama70 nr_qw235}; WHAT=${WHAT:-all}
CSV=$(echo $NR | tr ' ' '\n' | sed 's/$/:1/' | paste -sd,); GAP=35
submit() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=${3:-8} CPU_MEM=${4:-32} SNAPSHOT=1 COMMAND="$2" 2>&1 | grep -E "^[0-9a-f]{8}-|rror" | cut -c1-70 || true; sleep $GAP; }
ENVS="export HF_HOME=/home/toolkit/.cache/huggingface HF_DATASETS_CACHE=/home/toolkit/.cache/huggingface/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
O1=$R/math_pool_nonreason; mkdir -p $O1
cat > $O1/run.sh <<EOF
#!/usr/bin/env bash
cd /home/toolkit/PipelineRL-SWE; $ENVS
python pipelinerl/swe/scripts/math_pool/collect_math_pool.py --out-dir $O1 --pilot 0 --routes $CSV --datasets omni500,mmlupro --concurrency 96 \
  --max-tokens 16000 --budget-usd \${BUDGET:-6} > $O1/collect.log 2>&1
echo ALL DONE >> $O1/collect.log
EOF
O2=$R/math_expand_nonreason; mkdir -p $O2
for ds in omni500 mmlupro; do T=analysis/cost_headroom/expansion_20261001/${ds}_tasks.jsonl; [ $ds = mmlupro ] && T=analysis/cost_headroom/expansion_20261001/mmlupro_tasks_sub2000.jsonl
cat > $O2/run_$ds.sh <<EOF
#!/usr/bin/env bash
cd /home/toolkit/PipelineRL-SWE; $ENVS
python -u pipelinerl/swe/scripts/math_pool/collect_expansion.py --plan analysis/cost_headroom/expansion_20261001/plan_nonreason.json --tasks $T \
  --dataset $ds --out $O2 --budget-usd \${BUDGET:-4} --concurrency 96 --max-tokens 16000 > $O2/collect_$ds.log 2>&1
echo EXIT \$? >> $O2/collect_$ds.log
EOF
done
O3=$R/pool_v2_lcb_nonreason; mkdir -p $O3; SRC=$R/lcb_corrected_temporal_qwen_qwen3_4b_instruct_2507_1787205448
lcbspec() { case $1 in   # model | temperature | top_p | provider | reasoning-off
  nr_ds4off) echo "deepseek/deepseek-v4-flash 0.7 0.95 StreamLake 1";; nr_llama8) echo "meta-llama/llama-3.1-8b-instruct 0.6 0.9 DeepInfra 0";;
  nr_qw30) echo "qwen/qwen3-30b-a3b-instruct-2507 0.7 0.8 StreamLake 0";; nr_llama70) echo "meta-llama/llama-3.3-70b-instruct 0.6 0.9 Parasail 0";;
  nr_qw235) echo "qwen/qwen3-235b-a22b-2507 0.7 0.8 GMICloud 0";; nr_kimik2) echo "moonshotai/kimi-k2-0905 0.6 1.0 Novita 0";; esac; }
for r in $NR; do read M TT TP PV OFF <<< "$(lcbspec $r)"
cat > $O3/run_$r.sh <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd /home/toolkit/PipelineRL-SWE; export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 OPENROUTER_PIN_PROVIDER=$PV
[ "$OFF" = 1 ] && export OPENROUTER_REASONING_OFF=1
source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh
python pipelinerl/swe/scripts/livecodebench/collect_lcb_expert.py --source-collection-dir $SRC --output-dir $O3 --route-label $r \
  --model '$M' --splits train,eval --temperature $TT --top-p $TP --max-tokens 16000 --gen-timeout 1800 --max-invalid-frac 0.10 \
  --output-suffix _d0 --api-key-file /home/toolkit/.secrets/openrouter_api_key --concurrency 32 > $O3/log_$r.txt 2>&1
echo EXIT \$? >> $O3/log_$r.txt
EOF
done
chmod +x $O1/run.sh $O2/run_*.sh $O3/run_*.sh
if [[ ${SUBMIT:-0} != 1 ]]; then echo "dry run; routes: $CSV"; exit 0; fi
[[ $WHAT == all || $WHAT == math ]] && submit nonreason_math_$(date -u +%H%M%S) "bash $O1/run.sh" 4 16
[[ $WHAT == all || $WHAT == fresh ]] && for ds in omni500 mmlupro; do submit nonreason_${ds}_$(date -u +%H%M%S) "bash $O2/run_$ds.sh" 4 16; done
[[ $WHAT == all || $WHAT == lcb ]] && for r in $NR; do submit nrlcb_${r//_/}_$(date -u +%H%M%S) "bash $O3/run_$r.sh" 8 32; done
exit 0

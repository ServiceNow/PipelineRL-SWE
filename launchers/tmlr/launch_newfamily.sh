#!/usr/bin/env bash
# TMLR (NEW_PATH 4.A.76): new routed-model families on all three main sets, so cross-model generalisation (whole-family onboarding,
# transfer with unseen routes) is tested beyond one deepseek route. Approved 2026-10-09 (~$55 est., guards below sum to $80).
# Routes (collect_math_pool.ROUTES settings; reasoning on; one draw; each pinned to ONE provider, no fallbacks):
#   qw32    qwen/qwen3-32b                     SiliconFlow (as the 4.A.48 MMLU-Pro rows)   max 64k
#   glm47f  z-ai/glm-4.7-flash                 Cloudflare  (as the 4.A.48 MMLU-Pro rows)   max 64k
#   nemo120 nvidia/nemotron-3-super-120b-a12b  DeepInfra   (provider max output 16,384)     max 16k
# Problems: Omni-500 train+cal (350) + Omni 1,000 test; LCB 892 (train + eval); MMLU-Pro 2,700 for nemo120 only (qw32 / glm47f exist).
# Concurrency 12 per provider (rate limits); resumable; billed usage_cost spend guard per job. Usage: SUBMIT=1 bash launch_newfamily.sh
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; O=$R/second_family_20261009; OL=$R/pool_v2_lcb_second_family; mkdir -p $O $OL
SRC=$R/lcb_corrected_temporal_qwen_qwen3_4b_instruct_2507_1787205448; GAP=35
spec() { case $1 in qw32) echo "qwen/qwen3-32b SiliconFlow 64000";; glm47f) echo "z-ai/glm-4.7-flash Cloudflare 64000";; nemo120) echo "nvidia/nemotron-3-super-120b-a12b DeepInfra 16000";; esac; }
mathbud() { case $1 in qw32) echo 18;; glm47f) echo 14;; nemo120) echo 18;; esac; }
submit() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=${3:-4} CPU_MEM=${4:-16} SNAPSHOT=1 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; sleep $GAP; }
for r in qw32 glm47f nemo120; do read M PV MT <<< "$(spec $r)"; DS=omni500; [ $r = nemo120 ] && DS=omni500,mmlupro
cat > $O/run_$r.sh <<EOS
#!/usr/bin/env bash
cd /home/toolkit/PipelineRL-SWE; export HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1 OPENROUTER_PIN_PROVIDER=$PV
PYTHONPATH=. python -u pipelinerl/swe/scripts/math_pool/collect_second_family.py --out $O --datasets $DS --routes $r \
  --tasks-file $O/tasks_{ds}.jsonl --budget-usd $(mathbud $r) --concurrency 12 --max-tokens $MT > $O/collect_$r.log 2>&1
echo EXIT \$? >> $O/collect_$r.log
EOS
cat > $OL/run_$r.sh <<EOS
#!/usr/bin/env bash
set -uo pipefail
cd /home/toolkit/PipelineRL-SWE; export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 OPENROUTER_PIN_PROVIDER=$PV
source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh
python pipelinerl/swe/scripts/livecodebench/collect_lcb_expert.py --source-collection-dir $SRC --output-dir $OL --route-label $r \
  --model '$M' --reasoning-enabled --splits train,eval --temperature 0.6 --top-p 0.95 --max-tokens $MT --gen-timeout 3600 \
  --max-invalid-frac 0.10 --output-suffix _d0 --api-key-file /home/toolkit/.secrets/openrouter_api_key --concurrency 12 \
  --budget-usd 10 > $OL/log_$r.txt 2>&1
echo EXIT \$? >> $OL/log_$r.txt
EOS
done
chmod +x $O/run_*.sh $OL/run_*.sh
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run: scripts in $O and $OL"; exit 0; }
for r in qw32 glm47f nemo120; do
  echo -n "nfmath_$r "; submit nfmath_$r "bash $O/run_$r.sh" 4 16
  echo -n "nflcb_$r "; submit nflcb_$r "bash $OL/run_$r.sh" 8 32
done

#!/usr/bin/env bash
# NEW_PATH 4.A.70: SuperGPQA + BIG-Bench Extra Hard full pools (approved 2026-10-09, ~$13). Same 5 routes and draws as MMLU-Pro
# (4/3/3/2/2); oss20lo d0 = the existing screen draw (resumable collector reuses it). gpt-oss routes unpinned (provider changes price,
# not length); deepseek-v4-flash pinned to StreamLake in a separate process (OPENROUTER_PIN_PROVIDER is per-process). Commit + push first.
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; O=$R/math_pool
ENVS="export HF_HOME=/home/toolkit/.cache/huggingface HF_DATASETS_CACHE=/home/toolkit/.cache/huggingface/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
for ds in supergpqa bbeh; do G=5; GD=3; [ $ds = bbeh ] && G=8 && GD=4
cat > $O/run_${ds}_oss.sh <<EOF
#!/usr/bin/env bash
cd /home/toolkit/PipelineRL-SWE; $ENVS
python pipelinerl/swe/scripts/math_pool/collect_math_pool.py --out-dir $O --pilot 0 --routes oss20lo:4,oss20md:3,oss120md:2,oss120hi:2 \
  --datasets $ds --concurrency 384 --budget-usd $G > $O/collect_${ds}_oss.log 2>&1
echo ALL DONE >> $O/collect_${ds}_oss.log
EOF
cat > $O/run_${ds}_dsv4f.sh <<EOF
#!/usr/bin/env bash
cd /home/toolkit/PipelineRL-SWE; $ENVS; export OPENROUTER_PIN_PROVIDER=StreamLake
python pipelinerl/swe/scripts/math_pool/collect_math_pool.py --out-dir $O --pilot 0 --routes dsv4f:3 \
  --datasets $ds --concurrency 128 --budget-usd $GD > $O/collect_${ds}_dsv4f.log 2>&1
echo ALL DONE >> $O/collect_${ds}_dsv4f.log
EOF
chmod +x $O/run_${ds}_oss.sh $O/run_${ds}_dsv4f.sh; done
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run"; exit 0; }
for ds in supergpqa bbeh; do for part in oss dsv4f; do
  make job JOB_NAME="hetero_${ds}_${part}_$(date -u +%H%M%S)" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=4 CPU_MEM=16 SNAPSHOT=1 \
    COMMAND="bash $O/run_${ds}_${part}.sh" 2>&1 | grep -E "^[0-9a-f]{8}-|rror" | cut -c1-70; sleep 35; done; done

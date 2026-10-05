#!/usr/bin/env bash
# AIME 1983-2024 full pool (NEW_PATH 4.A.12 pre-registered NO GAIN call; screen probe R2 .13). Same five routes and draw counts as
# Omni (4/3/3/2/2); oss20lo d0 exists from the screen and is reused (resumable collector). Est. ~$25 billed; guard $30.
# CPU eai job; commit + push first.
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason
O=${R}/math_pool; NAME="aime_pool_$(date -u +%Y%m%d_%H%M%S)"
cat > ${O}/run_aime.sh <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd /home/toolkit/PipelineRL-SWE
export HF_HOME=/home/toolkit/.cache/huggingface HF_DATASETS_CACHE=/home/toolkit/.cache/huggingface/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python pipelinerl/swe/scripts/math_pool/collect_math_pool.py --out-dir ${O} --pilot 0 --routes oss20lo:4,oss20md:3,dsv4f:3,oss120md:2,oss120hi:2 \
  --datasets aime --concurrency 768 --budget-usd \${BUDGET:-30} > ${O}/collect_aime_full.log 2>&1
echo ALL DONE >> ${O}/collect_aime_full.log
EOF
chmod +x ${O}/run_aime.sh
if [ "${SUBMIT:-0}" != "1" ]; then echo "dry run:"; cat ${O}/run_aime.sh; exit 0; fi
make job JOB_NAME="${NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=4 CPU_MEM=16 SNAPSHOT=1 \
  COMMAND="bash ${O}/run_aime.sh"
echo "submitted ${NAME} -> ${O}/collect_aime_full.log"

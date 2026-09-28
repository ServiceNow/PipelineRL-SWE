#!/usr/bin/env bash
# Math pools for the cost-predictability rule (NEW_PATH 4.A.6). CPU-only eai job, OpenRouter, open models only.
#   PILOT=40 (default): 40 stratified problems each from MATH-500 and Omni-MATH-500, 1 draw x 5 routes -> ~$1;
#                       measures the accuracy spread (is the ladder flat?) and token counts for the full estimate.
#   PILOT=0 ROUTES=oss20lo:4,oss20md:3,dsv4f:3,oss120md:2,oss120hi:2 : the full pools (check in with the estimate first).
# Commit and push first: SNAPSHOT=1 runs the committed tree.
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason
PILOT=${PILOT:-40}
ROUTES=${ROUTES:-oss20lo:1,oss20md:1,dsv4f:1,oss120md:1,oss120hi:1}
DATASETS=${DATASETS:-math500,omni500}
O=${R}/math_pool$([ "${PILOT}" != "0" ] && echo "_pilot" || true)
NAME="math_pool$([ "${PILOT}" != "0" ] && echo "_pilot" || true)_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${O}"
cat > "${O}/run.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
export HF_HOME=/home/toolkit/.cache/huggingface HF_DATASETS_CACHE=/home/toolkit/.cache/huggingface/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python pipelinerl/swe/scripts/math_pool/collect_math_pool.py --out-dir ${O} --pilot ${PILOT} --routes ${ROUTES} --datasets ${DATASETS} \
  --concurrency ${CONCURRENCY:-256} > ${O}/collect.log 2>&1
echo ALL DONE >> ${O}/collect.log
EOF
chmod +x "${O}/run.sh"
if [ "${SUBMIT:-0}" != "1" ]; then echo "dry run (SUBMIT=1 to submit) ${NAME}:"; cat "${O}/run.sh"; exit 0; fi
make job JOB_NAME="${NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=4 CPU_MEM=16 SNAPSHOT=1 \
  COMMAND="bash ${O}/run.sh"
echo "submitted ${NAME} -> ${O}/collect.log"

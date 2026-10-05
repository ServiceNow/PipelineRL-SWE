#!/usr/bin/env bash
# Second coding pool for provider routing (NEW_PATH 4.A.51): deepseek-v4-flash pinned to StreamLake / GMICloud / DigitalOcean on all
# 700 CodeContests problems (other routes reused from cc_pool). Billed-cost guard $12 (est. ~$8-10). CPU eai job; commit + push first.
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason
O=${R}/provider_pilot_20261002; NAME="provider_cc_$(date -u +%Y%m%d_%H%M%S)"; mkdir -p ${O}
cat > ${O}/run_cc.sh <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd /home/toolkit/PipelineRL-SWE
PYTHONPATH=. python pipelinerl/swe/scripts/math_pool/collect_provider_pilot.py --out ${O} --datasets cc --budget-usd \${BUDGET:-12} > ${O}/collect_cc.log 2>&1
echo EXIT \$? >> ${O}/collect_cc.log
EOF
chmod +x ${O}/run_cc.sh
if [ "${SUBMIT:-0}" != "1" ]; then echo "dry run:"; cat ${O}/run_cc.sh; exit 0; fi
make job JOB_NAME="${NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=64 SNAPSHOT=1 \
  COMMAND="bash ${O}/run_cc.sh"
echo "submitted ${NAME} -> ${O}"

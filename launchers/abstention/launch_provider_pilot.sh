#!/usr/bin/env bash
# Provider-routing pilot (NEW_PATH 4.A.39): deepseek-v4-flash pinned to StreamLake / GMICloud / DigitalOcean on 2,000 fresh MMLU-Pro
# + all 1,000 APPS problems; billed-cost spend guard. CPU eai job; commit + push first.
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason
O=${R}/provider_pilot_20261002; NAME="provider_pilot_$(date -u +%Y%m%d_%H%M%S)"; mkdir -p ${O}
cat > ${O}/run.sh <<EOS
#!/usr/bin/env bash
set -uo pipefail
cd /home/toolkit/PipelineRL-SWE
PYTHONPATH=. python pipelinerl/swe/scripts/math_pool/collect_provider_pilot.py --out ${O} --budget-usd \${BUDGET:-18} > ${O}/collect.log 2>&1
echo EXIT \$? >> ${O}/collect.log
EOS
chmod +x ${O}/run.sh
if [ "${SUBMIT:-0}" != "1" ]; then echo "dry run:"; cat ${O}/run.sh; exit 0; fi
make job JOB_NAME="${NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=64 SNAPSHOT=1 \
  COMMAND="bash ${O}/run.sh"
echo "submitted ${NAME} -> ${O}"

#!/usr/bin/env bash
# Partial-generation cost predictor (NEW_PATH 4.A.6): first 512 output tokens of each route on the 700 CodeContests
# pool problems (reasoning prefix kept). CPU eai job, ~$0.75. Commit and push first.
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; O=${R}/cc_prefixes; NAME="cc_prefixes_$(date -u +%Y%m%d_%H%M%S)"; mkdir -p ${O}
cat > ${O}/run.sh <<EOS
#!/usr/bin/env bash
set -uo pipefail
python -m pipelinerl.swe.scripts.codecontests.collect_prefixes --problem-ids ${R}/cc_tensors/problem_ids.json --out-dir ${O} \
  --max-tokens 512 --concurrency 64 > ${O}/collect.log 2>&1
echo ALL DONE >> ${O}/collect.log
EOS
chmod +x ${O}/run.sh
if [ "${SUBMIT:-0}" != "1" ]; then echo "dry run:"; cat ${O}/run.sh; exit 0; fi
make job JOB_NAME="${NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=4 CPU_MEM=16 SNAPSHOT=1 \
  COMMAND="bash ${O}/run.sh"
echo "submitted ${NAME} -> ${O}/collect.log"

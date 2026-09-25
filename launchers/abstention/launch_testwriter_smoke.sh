#!/usr/bin/env bash
# Test-writer smoke test (NEW_PATH.md 2.7): spec-only unit-test suites, 5 writers, LCB test split.
# CPU-only eai job (OpenRouter calls only); ~341 problems x 5 writers ~= 1.7k cheap calls.
# After it lands, execution + analysis run locally:
#   ~/.conda/envs/pipeline-rl/bin/python3 pipelinerl/swe/scripts/livecodebench/run_test_suites.py \
#     --pool-dir $R/pool_v2_tensors_5rung --suites-dir $R/testwriter_smoke_lcb \
#     --out-dir $R/testwriter_smoke_lcb/verdicts --workers 16
set -euo pipefail

R=/mnt/llmd/results/exps/aristides/reason
NAME="testwriter_smoke_lcb_$(date -u +%Y%m%d_%H%M%S)_${RANDOM}"
DIR="${R}/${NAME}"
mkdir -p "${DIR}"

cat > "${DIR}/run.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd /home/toolkit/PipelineRL-SWE
python pipelinerl/swe/scripts/livecodebench/generate_test_suites.py \
  --pool-dir ${R}/pool_v2_tensors_5rung \
  --out-dir ${R}/testwriter_smoke_lcb \
  --splits test \
  --writers qwen4b,oss20lo,oss20md,dsv4f,oss120md \
  --concurrency 8 \
  --api-key-file /home/toolkit/.secrets/openrouter_api_key \
  2>&1 | tee ${DIR}/gen.log
EOF
chmod +x "${DIR}/run.sh"

if [ "${SUBMIT:-0}" != "1" ]; then
  echo "dry run (set SUBMIT=1 to submit): would submit ${NAME}, command:"
  cat "${DIR}/run.sh"
  exit 0
fi

make job JOB_NAME="${NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  GPU=0 CPU=4 CPU_MEM=16 SNAPSHOT=0 \
  COMMAND="bash ${DIR}/run.sh"
echo "submitted ${NAME} -> suites -> ${R}/testwriter_smoke_lcb"
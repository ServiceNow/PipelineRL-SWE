#!/usr/bin/env bash
# Regrade an existing corrected LCB collection and retry only failed oracle calls.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${LCB_COLLECTION_DIR:?Set LCB_COLLECTION_DIR to the corrected LCB collection}"

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
JOB_NAME=${JOB_NAME:-lcb_corrected_repair_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}
RELEASE_VERSION=${RELEASE_VERSION:-release_v6}
LCB_DATASET_REVISION=${LCB_DATASET_REVISION:-0fe84c3912ea0c4d4a78037083943e8f0c4dd505}
MIN_DATE=${MIN_DATE:-2023-09-01}
TEMPORAL_CUTOFF=${TEMPORAL_CUTOFF:-2024-10-01}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-10}
CONCURRENCY=${CONCURRENCY:-4}
SNAPSHOT=${SNAPSHOT:-1}
OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE:-/home/toolkit/.secrets/openrouter_api_key}

if [[ ! -s "${OPENROUTER_API_KEY_FILE}" ]]; then
  echo "Missing OpenRouter key at ${OPENROUTER_API_KEY_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
RUNNER="${OUTPUT_DIR}/run_repair.sh"
cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh

python pipelinerl/swe/scripts/livecodebench/repair_lcb_collection.py \\
  --errors-only \\
  --output-dir '${LCB_COLLECTION_DIR}' \\
  --release-version '${RELEASE_VERSION}' \\
  --dataset-revision '${LCB_DATASET_REVISION}' \\
  --min-date '${MIN_DATE}' \\
  --temporal-cutoff '${TEMPORAL_CUTOFF}' \\
  --eval-timeout ${EVAL_TIMEOUT} \\
  --scout-feedback-tests public \\
  2>&1 | tee '${OUTPUT_DIR}/regrade.log'

python pipelinerl/swe/scripts/livecodebench/collect_lcb_trajectories.py \\
  --phase oracle \\
  --output-dir '${LCB_COLLECTION_DIR}' \\
  --oracle-model 'openai/gpt-oss-120b' \\
  --api-key-file '${OPENROUTER_API_KEY_FILE}' \\
  --release-version '${RELEASE_VERSION}' \\
  --dataset-revision '${LCB_DATASET_REVISION}' \\
  --min-date '${MIN_DATE}' \\
  --temporal-cutoff '${TEMPORAL_CUTOFF}' \\
  --temperature 0.0 \\
  --concurrency ${CONCURRENCY} \\
  --eval-timeout ${EVAL_TIMEOUT} \\
  2>&1 | tee '${OUTPUT_DIR}/retry_oracle.log'

python pipelinerl/swe/scripts/livecodebench/repair_lcb_collection.py \\
  --validate-only \\
  --output-dir '${LCB_COLLECTION_DIR}' \\
  --release-version '${RELEASE_VERSION}' \\
  --dataset-revision '${LCB_DATASET_REVISION}' \\
  --min-date '${MIN_DATE}' \\
  --temporal-cutoff '${TEMPORAL_CUTOFF}' \\
  2>&1 | tee '${OUTPUT_DIR}/validate.log'
SCRIPT_EOF
chmod +x "${RUNNER}"

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" \
  NPROC=1 \
  GPU=0 \
  GPU_MEM=0 \
  CPU=16 \
  CPU_MEM=64 \
  COMMAND="bash ${RUNNER}"

echo "Job:        ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"

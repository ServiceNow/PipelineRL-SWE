#!/usr/bin/env bash
# Run Daytona eval on the scout patches collected for SWE-bench Verified.
#
# This feeds the test-execution feedback pipeline for Verified cross-domain
# scoring: Daytona results → augment_trajectories_with_test_feedback.py →
# score_cot_abstention_predictor.py.
#
# Required env vars:
#   VERIFIED_EVAL_DIR  -- output dir from launch_verified_abstention_eval.sh
#                         (must contain trajectories_verified.jsonl)
#
# Optional:
#   CONCURRENCY  -- Daytona sandbox concurrency (default: 16)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=/home/toolkit/.conda/envs/pipeline-rl/bin/python3

VERIFIED_EVAL_DIR=${VERIFIED_EVAL_DIR:?Need VERIFIED_EVAL_DIR (output of launch_verified_abstention_eval.sh)}
CONCURRENCY=${CONCURRENCY:-16}

TIMESTAMP=$(date +%s)
JOB_NAME=verified_scout_daytona_eval_${TIMESTAMP}
OUTPUT_DIR=/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}
mkdir -p "${OUTPUT_DIR}"

# Load DAYTONA_API_KEY from .env if not already set
if [[ -z "${DAYTONA_API_KEY:-}" ]]; then
  ENV_FILE="${REPO_ROOT}/.env"
  if [[ -f "${ENV_FILE}" ]]; then
    DAYTONA_API_KEY=$(grep -E '^DAYTONA_API_KEY=' "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"'"'")
  fi
fi
: "${DAYTONA_API_KEY:?Need DAYTONA_API_KEY — set it in .env or the environment}"

PREDS_DIR="${OUTPUT_DIR}/predictions"
mkdir -p "${PREDS_DIR}"
PREDS_FILE="${PREDS_DIR}/predictions_verified.jsonl"

# --- Step 1: extract patches from trajectories_verified.jsonl ---
echo "=== Extracting scout patches from ${VERIFIED_EVAL_DIR}/trajectories_verified.jsonl ==="
"${PYTHON}" - << PYEOF
import json
from pathlib import Path

src = Path("${VERIFIED_EVAL_DIR}/trajectories_verified.jsonl")
if not src.exists():
    raise SystemExit(f"Not found: {src}")

out = Path("${PREDS_FILE}")
n_ok = n_empty = 0
with open(src) as fin, open(out, "w") as fout:
    for line in fin:
        row = json.loads(line)
        iid = str(row.get("problem_id") or row.get("instance_id") or "").strip()
        patch = str(row.get("patch_text") or "").strip()
        if not patch:
            n_empty += 1
        else:
            n_ok += 1
        fout.write(json.dumps({
            "instance_id": iid,
            "model_patch": patch,
            "model": "qwen3-4b-thinking",
        }) + "\n")
print(f"  wrote {n_ok} patches, {n_empty} empty → ${PREDS_FILE}")
PYEOF
echo ""

# --- Step 2: submit Daytona eval job ---
RUN_ID="verified_scout_eval_${TIMESTAMP}"
RUNNER="${OUTPUT_DIR}/run_daytona_eval.sh"

cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ -z "\${DAYTONA_API_KEY:-}" ]]; then
  for _env_file in ${REPO_ROOT}/.env /home/toolkit/.env; do
    if [[ -f "\${_env_file}" ]]; then
      DAYTONA_API_KEY=\$(grep -E '^DAYTONA_API_KEY=' "\${_env_file}" | head -1 | cut -d'=' -f2- | tr -d '"' | tr -d "'")
      break
    fi
  done
fi
export DAYTONA_API_KEY
: "\${DAYTONA_API_KEY:?Need DAYTONA_API_KEY in .env}"
cd "${REPO_ROOT}"

echo "=== Daytona eval on SWE-bench Verified scout patches (concurrency=${CONCURRENCY}) ==="
python pipelinerl/swe/scripts/offline_router/run_swebench_eval_daytona.py \\
  --predictions-path "${PREDS_FILE}" \\
  --run-id           "${RUN_ID}" \\
  --concurrency      "${CONCURRENCY}" \\
  2>&1 | tee "${OUTPUT_DIR}/daytona_eval.log"

echo ""
echo "[done] Results: logs/run_evaluation/${RUN_ID}/"
echo ""
echo "Next step — augment trajectories with test feedback:"
echo "  python pipelinerl/swe/scripts/offline_router/augment_trajectories_with_test_feedback.py \\"
echo "    --trajectories-dir  ${VERIFIED_EVAL_DIR} \\"
echo "    --daytona-log-dir   logs/run_evaluation/${RUN_ID} \\"
echo "    --output-dir        ${OUTPUT_DIR}/trajectories_with_testfb \\"
echo "    --split eval"
SCRIPT_EOF
chmod +x "${RUNNER}"

echo "=== Submitting verified scout Daytona eval: ${JOB_NAME} ==="
make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=1 \
  GPU=0 \
  GPU_MEM=0 \
  CPU=8 \
  CPU_MEM=32 \
  COMMAND="bash ${RUNNER}"

echo ""
echo "Job:        ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Log:        ${OUTPUT_DIR}/daytona_eval.log"
echo "Results:    logs/run_evaluation/${RUN_ID}/"

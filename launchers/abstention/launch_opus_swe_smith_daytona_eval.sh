#!/usr/bin/env bash
# Evaluate Claude Opus 5 SWE-Smith predictions via Daytona.
#
# Steps:
#   1. Extract Opus patches from the collection parquet → predictions JSONL (local, fast)
#   2. Convert search/replace text → unified git diffs (local, fast)
#   3. Submit Daytona eval EAI job
#
# Optional env vars:
#   SOURCE_COLLECTION_DIR  -- Opus collection eval parquet dir (default below)
#   DATASET_PATH           -- SWE-Smith HF dataset path (default: ds_train)
#   CONCURRENCY            -- Daytona sandbox concurrency (default: 32)
#   ROUTE_IDX              -- route index in parquet for Opus outputs (default: 1)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=/home/toolkit/.conda/envs/pipeline-rl/bin/python3

TIMESTAMP=$(date +%s)
JOB_NAME=opus_swe_smith_daytona_eval_${TIMESTAMP}
OUTPUT_DIR=/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}

SOURCE_COLLECTION_DIR=${SOURCE_COLLECTION_DIR:-/mnt/llmd/results/exps/aristides/reason/swe_smith_collect_anthropic_claude_opus_5_openrouter_1786126553/collect/eval}
DATASET_PATH=${DATASET_PATH:-/mnt/llmd/data/swe_smith_bugged_context/ds_train}
CONCURRENCY=${CONCURRENCY:-32}
ROUTE_IDX=${ROUTE_IDX:-1}

# Load DAYTONA_API_KEY from .env if not already set
if [[ -z "${DAYTONA_API_KEY:-}" ]]; then
  ENV_FILE="${REPO_ROOT}/.env"
  if [[ -f "${ENV_FILE}" ]]; then
    DAYTONA_API_KEY=$(grep -E '^DAYTONA_API_KEY=' "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"'"'")
  fi
fi
: "${DAYTONA_API_KEY:?Need DAYTONA_API_KEY — set it in .env or the environment}"

mkdir -p "${OUTPUT_DIR}"
PREDS_DIR="${OUTPUT_DIR}/predictions"
mkdir -p "${PREDS_DIR}"
PREDS_FILE="${PREDS_DIR}/predictions_opus_eval.jsonl"

# --- Step 1: extract Opus patches from collection parquet (local) ---
echo "=== Extracting Opus patches from ${SOURCE_COLLECTION_DIR} (route ${ROUTE_IDX}) ==="
"${PYTHON}" - << PYEOF
import json, pandas as pd
from pathlib import Path

parquet_paths = sorted(Path("${SOURCE_COLLECTION_DIR}").glob("*.parquet"))
if not parquet_paths:
    raise FileNotFoundError("No parquets in ${SOURCE_COLLECTION_DIR}")
df = pd.concat([pd.read_parquet(p) for p in parquet_paths])
print(f"Loaded {len(df)} rows from {len(parquet_paths)} parquets")

n_ok = n_empty = 0
with open("${PREDS_FILE}", "w") as fh:
    for _, row in df.iterrows():
        iid = str(row.get("problem_id") or "").strip()
        outputs = row.get("route_outputs")
        if outputs is None or len(outputs) <= ${ROUTE_IDX}:
            raw = ""
        else:
            raw = str(outputs[${ROUTE_IDX}] or "").strip()
        if not raw:
            n_empty += 1
        else:
            n_ok += 1
        fh.write(json.dumps({
            "instance_id": iid,
            "model_patch": raw,
            "model": "anthropic/claude-opus-5",
        }) + "\n")

print(f"Wrote {n_ok} patches, {n_empty} empty → ${PREDS_FILE}")
PYEOF
echo ""

# --- Step 2: convert search/replace text → unified git diffs (local) ---
echo "=== Converting search/replace → git diffs ==="
"${PYTHON}" "${REPO_ROOT}/pipelinerl/swe/scripts/openrouter_sweep/convert_text_to_patches.py" \
  --predictions-dir "${PREDS_DIR}" \
  --dataset-path    "${DATASET_PATH}"
echo ""

# --- Step 3: build runner script ---
RUNNER="${OUTPUT_DIR}/run_daytona_eval.sh"
RUN_ID="opus_eval_${TIMESTAMP}"

cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
# Load DAYTONA_API_KEY from .env (avoids exposing it in the job command line)
if [[ -z "\${DAYTONA_API_KEY:-}" ]]; then
  for _env_file in /home/toolkit/PipelineRL-SWE/.env /home/toolkit/.env; do
    if [[ -f "\${_env_file}" ]]; then
      DAYTONA_API_KEY=\$(grep -E '^DAYTONA_API_KEY=' "\${_env_file}" | head -1 | cut -d'=' -f2- | tr -d '"' | tr -d "'")
      break
    fi
  done
fi
export DAYTONA_API_KEY
: "\${DAYTONA_API_KEY:?Need DAYTONA_API_KEY in .env}"
cd "${REPO_ROOT}"

echo "=== Evaluating Opus SWE-Smith predictions ==="
python "${REPO_ROOT}/pipelinerl/swe/scripts/offline_router/run_swesmith_eval_daytona.py" \
  --predictions_path "${PREDS_FILE}" \
  --run_id           "${RUN_ID}" \
  --concurrency      "${CONCURRENCY}" \
  2>&1 | tee "${OUTPUT_DIR}/daytona_eval.log"
SCRIPT_EOF
chmod +x "${RUNNER}"

# --- Step 4: submit EAI job ---
echo "=== Submitting Daytona eval job: ${JOB_NAME} ==="
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
echo "Output dir:  ${OUTPUT_DIR}"
echo "Daytona log: ${OUTPUT_DIR}/daytona_eval.log"
echo "Results:     logs/run_evaluation/${RUN_ID}/"

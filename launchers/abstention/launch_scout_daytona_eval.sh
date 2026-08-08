#!/usr/bin/env bash
# Evaluate 4B instruct scout patches via Daytona to get granular test failure info.
#
# This is the first stage of the test-feedback-guided abstention predictor:
# the structured failure signal (which tests failed, error types) from the scout
# eval is used as features to predict whether the oracle (oss-120b) will succeed.
#
# Steps (all local, then one EAI job):
#   1. Extract scout patches from trajectories JSONL → predictions JSONL
#   2. Convert search/replace text → unified git diffs
#   3. Submit Daytona eval EAI job (train + eval splits)
#
# Optional env vars:
#   SOURCE_TRAJECTORIES_DIR  -- dir containing trajectories_{train,eval}.jsonl
#   DATASET_PATH             -- SWE-Smith HF dataset (for file_contents lookup)
#   CONCURRENCY              -- Daytona sandbox concurrency (default: 8)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=/home/toolkit/.conda/envs/pipeline-rl/bin/python3

TIMESTAMP=$(date +%s)
JOB_NAME=scout_daytona_eval_${TIMESTAMP}
OUTPUT_DIR=/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}

SOURCE_TRAJECTORIES_DIR=${SOURCE_TRAJECTORIES_DIR:-/mnt/llmd/results/exps/aristides/reason/instruct_patches_trajectories_1785884242}
DATASET_PATH=${DATASET_PATH:-/mnt/llmd/data/swe_smith_bugged_context/ds_train}
CONCURRENCY=${CONCURRENCY:-8}

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

# --- Step 1: extract patches from trajectories JSONL (both splits) ---
echo "=== Extracting scout patches from ${SOURCE_TRAJECTORIES_DIR} ==="
"${PYTHON}" - << PYEOF
import json
from pathlib import Path

src = Path("${SOURCE_TRAJECTORIES_DIR}")
preds_dir = Path("${PREDS_DIR}")

for split in ("train", "eval"):
    traj_file = src / f"trajectories_{split}.jsonl"
    if not traj_file.exists():
        print(f"  [skip] {traj_file} not found")
        continue

    preds_file = preds_dir / f"predictions_{split}.jsonl"
    n_ok = n_empty = 0
    with open(traj_file) as fin, open(preds_file, "w") as fout:
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
                "model": "qwen3-4b-instruct",
            }) + "\n")
    print(f"  {split}: wrote {n_ok} patches, {n_empty} empty → {preds_file}")
PYEOF
echo ""

# --- Step 2: convert search/replace text → unified git diffs ---
echo "=== Converting search/replace → git diffs ==="
"${PYTHON}" "${REPO_ROOT}/pipelinerl/swe/scripts/openrouter_sweep/convert_text_to_patches.py" \
  --predictions-dir "${PREDS_DIR}" \
  --dataset-path    "${DATASET_PATH}"
echo ""

# --- Step 3: build runner script (runs both train + eval splits) ---
RUNNER="${OUTPUT_DIR}/run_daytona_eval.sh"
RUN_ID="scout_eval_${TIMESTAMP}"

cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
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

for split in train eval; do
  preds_file="${PREDS_DIR}/predictions_\${split}.jsonl"
  if [[ ! -f "\${preds_file}" ]]; then
    echo "[skip] \${preds_file} not found"
    continue
  fi
  echo "=== Daytona eval: \${split} split (concurrency=${CONCURRENCY}) ==="
  python "${REPO_ROOT}/pipelinerl/swe/scripts/offline_router/run_swesmith_eval_daytona.py" \
    --predictions_path "\${preds_file}" \
    --run_id           "${RUN_ID}_\${split}" \
    --concurrency      "${CONCURRENCY}" \
    2>&1 | tee "${OUTPUT_DIR}/daytona_eval_\${split}.log"
  echo ""
done

echo "[done] Results: logs/run_evaluation/${RUN_ID}_{train,eval}/"
SCRIPT_EOF
chmod +x "${RUNNER}"

# --- Step 4: submit EAI job ---
echo "=== Submitting scout Daytona eval job: ${JOB_NAME} ==="
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
echo "Logs:       ${OUTPUT_DIR}/daytona_eval_{train,eval}.log"
echo "Results:    logs/run_evaluation/${RUN_ID}_{train,eval}/"

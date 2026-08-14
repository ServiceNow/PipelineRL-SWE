#!/usr/bin/env bash
# Collect scout + oracle data for the 131 SWE-bench Verified instances that are
# in the full-500 dataset but NOT in the all_16k subset.
#
# Purpose: enable in-domain Verified predictor training with a proper 400/100
# train/eval split, rather than the 369-instance all_16k subset.
#
# What this does (3 jobs):
#   Job 1 — collect_scout:  Qwen3-4B-Instruct patches for 131 instances via OpenRouter
#   Job 2 — collect_oracle: gpt-oss-120b patches for 131 instances via OpenRouter
#   Job 3 — daytona_oracle: run 120b patches through Daytona → real oracle labels
#
# After all 3 complete, run augment_trajectories_with_test_feedback.py to get
# test feedback for the 131 scout patches, then combine with the existing 369
# to form a full-500 dataset.
#
# Required files (created by companion script):
#   MISSING_IDS_FILE  -- newline-separated list of 131 instance IDs
#
# Optional env vars:
#   SCOUT_MODEL         -- OpenRouter model (default: Qwen/Qwen3-4B-Instruct-2507)
#   ORACLE_MODEL        -- OpenRouter model (default: openai/gpt-oss-120b)
#   MAX_CONCURRENT      -- async concurrency (default: 8)
#   PHASE               -- scout | oracle | daytona | all (default: all)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=/home/toolkit/.conda/envs/pipeline-rl/bin/python3

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SCOUT_MODEL=${SCOUT_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
ORACLE_MODEL=${ORACLE_MODEL:-openai/gpt-oss-120b}
MAX_CONCURRENT=${MAX_CONCURRENT:-8}
CONCURRENCY=${CONCURRENCY:-8}   # Daytona sandbox concurrency
PHASE=${PHASE:-all}

FULL_DATASET_PATH=${FULL_DATASET_PATH:-/mnt/llmd/data/swebench_verified/full/ds}
MISSING_IDS_FILE=${MISSING_IDS_FILE:-/mnt/llmd/results/exps/aristides/reason/verified_131_missing_ids.txt}
OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE:-/home/toolkit/.secrets/openrouter_api_key}

JOB_NAME=${JOB_NAME:-verified_expand_to_500_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

if [[ ! -f "${MISSING_IDS_FILE}" ]]; then
  echo "Missing MISSING_IDS_FILE=${MISSING_IDS_FILE}" >&2
  echo "Run this first to create it:" >&2
  echo "  python3 -c \"" >&2
  echo "    from datasets import load_from_disk" >&2
  echo "    d1 = load_from_disk('/mnt/llmd/data/swebench_verified/full/ds')" >&2
  echo "    d2 = load_from_disk('/mnt/llmd/data/swebench_verified/all_16k/ds')" >&2
  echo "    missing = set(d1['id']) - set(d2['id'])" >&2
  echo "    open('${MISSING_IDS_FILE}','w').write('\n'.join(sorted(missing))+'\\n')\"" >&2
  exit 1
fi
if [[ ! -s "${OPENROUTER_API_KEY_FILE}" ]]; then
  echo "Missing OpenRouter key: ${OPENROUTER_API_KEY_FILE}" >&2
  exit 1
fi
if [[ ! -d "${FULL_DATASET_PATH}" ]]; then
  echo "Missing full dataset: ${FULL_DATASET_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Load DAYTONA_API_KEY from .env for the oracle Daytona job
if [[ -z "${DAYTONA_API_KEY:-}" ]]; then
  ENV_FILE="${REPO_ROOT}/.env"
  if [[ -f "${ENV_FILE}" ]]; then
    DAYTONA_API_KEY=$(grep -E '^DAYTONA_API_KEY=' "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"'"'")
  fi
fi
: "${DAYTONA_API_KEY:?Need DAYTONA_API_KEY — set it in .env or the environment}"

SCOUT_OUTPUT="${OUTPUT_DIR}/scout"
ORACLE_OUTPUT="${OUTPUT_DIR}/oracle"
PREDS_DIR="${OUTPUT_DIR}/oracle_preds"
RUN_ID="verified_expand_oracle_eval_${TIMESTAMP}"

# ── Job 1: Scout collection ─────────────────────────────────────────────────

if [[ "${PHASE}" == "scout" || "${PHASE}" == "all" ]]; then
  RUNNER_SCOUT="${OUTPUT_DIR}/run_scout.sh"
  cat > "${RUNNER_SCOUT}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
mkdir -p "${SCOUT_OUTPUT}"

echo "=== Scout collection: ${SCOUT_MODEL} on 131 missing Verified instances ==="

python - << 'PYEOF'
import asyncio, json, logging, os, re, time
from pathlib import Path
import aiohttp
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCOUT_MODEL = "${SCOUT_MODEL}"
API_KEY_FILE = "${OPENROUTER_API_KEY_FILE}"
MISSING_IDS_FILE = "${MISSING_IDS_FILE}"
FULL_DS_PATH = "${FULL_DATASET_PATH}"
OUTPUT_FILE = "${SCOUT_OUTPUT}/trajectories.jsonl"
MAX_CONCURRENT = ${MAX_CONCURRENT}

def get_api_key():
    k = os.environ.get("OPENROUTER_API_KEY", "")
    return k or Path(API_KEY_FILE).read_text().strip()

async def call_model(session, sem, pid, prompt, api_key):
    payload = {
        "model": SCOUT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000, "temperature": 0.7,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "X-Title": "PipelineRL-Verified-Expand"}
    async with sem:
        t0 = time.monotonic()
        async with session.post("https://openrouter.ai/api/v1/chat/completions",
                                headers=headers, json=payload,
                                timeout=aiohttp.ClientTimeout(total=300)) as resp:
            resp.raise_for_status()
            data = await resp.json()
        elapsed = time.monotonic() - t0
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        thinking = msg.get("reasoning") or ""
        usage = data.get("usage", {})
        log.info("  %s: %.1fs", pid, elapsed)
        return {"full_output": content, "thinking_text": thinking,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0)}

async def main():
    missing_ids = set(Path(MISSING_IDS_FILE).read_text().splitlines())
    missing_ids = {x.strip() for x in missing_ids if x.strip()}
    log.info("Missing IDs to collect: %d", len(missing_ids))

    ds = load_from_disk(FULL_DS_PATH)
    rows = {r["id"]: r for r in ds if r["id"] in missing_ids}
    log.info("Found in full dataset: %d", len(rows))

    done = set()
    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add(r["problem_id"])
    log.info("Already done: %d", len(done))

    api_key = get_api_key()
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async with aiohttp.ClientSession() as session:
        todo = [(pid, row) for pid, row in rows.items() if pid not in done]
        log.info("To collect: %d", len(todo))

        async def process(pid, row):
            problem = str(row.get("problem_statement") or "").strip()
            if row.get("gold_file_contents"):
                import json as _json
                fc = row["gold_file_contents"]
                if isinstance(fc, str):
                    try:
                        fc = _json.loads(fc)
                    except Exception:
                        fc = {}
                if fc:
                    file_ctx = "\n\n".join(f"<file path=\"{k}\">\n{v}\n</file>" for k, v in list(fc.items())[:5])
                    problem = f"{problem}\n\nRelevant files:\n{file_ctx}"
            prompt = (
                f"You are a software engineer. Fix the issue described below.\n\n"
                f"{problem}\n\n"
                f"Provide your fix using SEARCH/REPLACE blocks:\n"
                f"<<<<<<< SEARCH\n[original code]\n=======\n[new code]\n>>>>>>> REPLACE"
            )
            try:
                out = await call_model(session, sem, pid, prompt, api_key)
                patch = out["full_output"]
            except Exception as e:
                log.warning("Failed %s: %s", pid, e)
                patch = ""
            return {"problem_id": pid, "problem_statement": str(row.get("problem_statement","")),
                    "patch_text": patch, "thinking_text": out.get("thinking_text","") if patch else "",
                    "full_output": patch}

        tasks = [process(pid, row) for pid, row in todo]
        with open(OUTPUT_FILE, "a") as fh:
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                r = await coro
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                if (i+1) % 20 == 0:
                    log.info("Progress: %d/%d", i+1, len(tasks))

    log.info("Scout collection complete → %s", OUTPUT_FILE)

asyncio.run(main())
PYEOF
SCRIPT_EOF
  chmod +x "${RUNNER_SCOUT}"

  make -C "${REPO_ROOT}" job \
    JOB_NAME="${JOB_NAME}_scout" \
    ENV=pipeline-rl \
    CONDA_EXE=/opt/conda/bin/conda \
    SNAPSHOT=1 \
    NPROC=1 GPU=0 GPU_MEM=0 CPU=4 CPU_MEM=16 \
    COMMAND="bash ${RUNNER_SCOUT}"
  echo "Scout job submitted: ${JOB_NAME}_scout"
fi

# ── Job 2: Oracle collection ────────────────────────────────────────────────

if [[ "${PHASE}" == "oracle" || "${PHASE}" == "all" ]]; then
  RUNNER_ORACLE="${OUTPUT_DIR}/run_oracle.sh"
  cat > "${RUNNER_ORACLE}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
mkdir -p "${ORACLE_OUTPUT}"

echo "=== Oracle collection: ${ORACLE_MODEL} on 131 missing Verified instances ==="

python - << 'PYEOF'
import asyncio, json, logging, os, time
from pathlib import Path
import aiohttp
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ORACLE_MODEL = "${ORACLE_MODEL}"
API_KEY_FILE = "${OPENROUTER_API_KEY_FILE}"
MISSING_IDS_FILE = "${MISSING_IDS_FILE}"
FULL_DS_PATH = "${FULL_DATASET_PATH}"
OUTPUT_FILE = "${ORACLE_OUTPUT}/patches.jsonl"
MAX_CONCURRENT = ${MAX_CONCURRENT}

def get_api_key():
    k = os.environ.get("OPENROUTER_API_KEY", "")
    return k or Path(API_KEY_FILE).read_text().strip()

async def call_model(session, sem, pid, prompt, api_key):
    payload = {
        "model": ORACLE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000, "temperature": 0.7,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "X-Title": "PipelineRL-Verified-Expand-Oracle"}
    async with sem:
        t0 = time.monotonic()
        async with session.post("https://openrouter.ai/api/v1/chat/completions",
                                headers=headers, json=payload,
                                timeout=aiohttp.ClientTimeout(total=300)) as resp:
            resp.raise_for_status()
            data = await resp.json()
        log.info("  %s: %.1fs", pid, time.monotonic()-t0)
        return data["choices"][0]["message"].get("content","")

async def main():
    missing_ids = {x.strip() for x in Path(MISSING_IDS_FILE).read_text().splitlines() if x.strip()}
    ds = load_from_disk(FULL_DS_PATH)
    rows = {r["id"]: r for r in ds if r["id"] in missing_ids}

    done = set()
    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add(r["instance_id"])

    api_key = get_api_key()
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async with aiohttp.ClientSession() as session:
        todo = [(pid, row) for pid, row in rows.items() if pid not in done]
        log.info("Oracle to collect: %d", len(todo))

        async def process(pid, row):
            problem = str(row.get("problem_statement",""))
            if row.get("gold_file_contents"):
                import json as _json
                fc = row["gold_file_contents"]
                if isinstance(fc, str):
                    try:
                        fc = _json.loads(fc)
                    except Exception:
                        fc = {}
                if fc:
                    file_ctx = "\n\n".join(f"<file path=\"{k}\">\n{v}\n</file>" for k, v in list(fc.items())[:5])
                    problem = f"{problem}\n\nRelevant files:\n{file_ctx}"
            prompt = (
                f"You are a software engineer. Fix the issue described below.\n\n"
                f"{problem}\n\n"
                f"Provide your fix using SEARCH/REPLACE blocks:\n"
                f"<<<<<<< SEARCH\n[original code]\n=======\n[new code]\n>>>>>>> REPLACE"
            )
            try:
                patch = await call_model(session, sem, pid, prompt, api_key)
            except Exception as e:
                log.warning("Failed %s: %s", pid, e)
                patch = ""
            return {"instance_id": pid, "model_patch": patch, "model": ORACLE_MODEL}

        tasks = [process(pid, row) for pid, row in todo]
        with open(OUTPUT_FILE, "a") as fh:
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                r = await coro
                fh.write(json.dumps(r) + "\n")
                fh.flush()

    log.info("Oracle collection complete → %s", OUTPUT_FILE)

asyncio.run(main())
PYEOF
SCRIPT_EOF
  chmod +x "${RUNNER_ORACLE}"

  make -C "${REPO_ROOT}" job \
    JOB_NAME="${JOB_NAME}_oracle" \
    ENV=pipeline-rl \
    CONDA_EXE=/opt/conda/bin/conda \
    SNAPSHOT=1 \
    NPROC=1 GPU=0 GPU_MEM=0 CPU=4 CPU_MEM=16 \
    COMMAND="bash ${RUNNER_ORACLE}"
  echo "Oracle job submitted: ${JOB_NAME}_oracle"
fi

# ── Job 3: Oracle Daytona eval (run AFTER oracle collection completes) ────────
# Launch this manually once oracle patches are ready:
#   PHASE=daytona bash launchers/abstention/launch_verified_expand_to_500.sh
# Or set PHASE=daytona and run again.

if [[ "${PHASE}" == "daytona" ]]; then
  PREDS_FILE="${PREDS_DIR}/predictions_131.jsonl"
  mkdir -p "${PREDS_DIR}"

  # Step 1: convert oracle patches to predictions JSONL
  echo "=== Converting oracle patches → predictions JSONL ==="
  "${PYTHON}" - << PYEOF
import json
from pathlib import Path
preds = []
for line in Path("${ORACLE_OUTPUT}/patches.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        preds.append({"instance_id": r["instance_id"], "model_patch": r["model_patch"], "model": r["model"]})
with open("${PREDS_FILE}", "w") as f:
    for p in preds:
        f.write(json.dumps(p) + "\n")
print(f"Wrote {len(preds)} predictions → ${PREDS_FILE}")
PYEOF

  # Step 2: convert search/replace → git diffs
  echo "=== Converting search/replace → git diffs ==="
  "${PYTHON}" "${REPO_ROOT}/pipelinerl/swe/scripts/openrouter_sweep/convert_text_to_patches.py" \
    --predictions-dir "${PREDS_DIR}" \
    --dataset-path    "${FULL_DATASET_PATH}"

  # Step 3: Daytona eval
  RUNNER_DAYTONA="${OUTPUT_DIR}/run_daytona.sh"
  cat > "${RUNNER_DAYTONA}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ -z "\${DAYTONA_API_KEY:-}" ]]; then
  for _f in ${REPO_ROOT}/.env /home/toolkit/.env; do
    [[ -f "\${_f}" ]] && DAYTONA_API_KEY=\$(grep -E '^DAYTONA_API_KEY=' "\${_f}" | head -1 | cut -d'=' -f2- | tr -d '"'"'") && break
  done
fi
export DAYTONA_API_KEY
: "\${DAYTONA_API_KEY:?}"
cd "${REPO_ROOT}"
python pipelinerl/swe/scripts/offline_router/run_swebench_eval_daytona.py \\
  --predictions-path "${PREDS_FILE}" \\
  --run-id           "${RUN_ID}" \\
  --concurrency      "${CONCURRENCY}" \\
  2>&1 | tee "${OUTPUT_DIR}/daytona_eval.log"
echo "[done] logs/run_evaluation/${RUN_ID}/"
SCRIPT_EOF
  chmod +x "${RUNNER_DAYTONA}"

  make -C "${REPO_ROOT}" job \
    JOB_NAME="${JOB_NAME}_daytona" \
    ENV=pipeline-rl \
    CONDA_EXE=/opt/conda/bin/conda \
    SNAPSHOT=1 \
    NPROC=1 GPU=0 GPU_MEM=0 CPU=8 CPU_MEM=32 \
    COMMAND="bash ${RUNNER_DAYTONA}"
  echo "Daytona job submitted: ${JOB_NAME}_daytona"
  echo "Results will be in: logs/run_evaluation/${RUN_ID}/"
fi

echo ""
echo "=== Three-step plan for 131 missing Verified instances ==="
echo "  Step 1 — scout collection: ${JOB_NAME}_scout"
echo "  Step 2 — oracle collection: ${JOB_NAME}_oracle"
echo "  Step 3 — oracle Daytona eval (run with PHASE=daytona after step 2):"
echo "    OUTPUT_DIR=${OUTPUT_DIR} PHASE=daytona bash launchers/abstention/launch_verified_expand_to_500.sh"
echo ""
echo "  After step 3, run augment_trajectories_with_test_feedback.py on scout output"
echo "  to get test feedback, then combine with the existing 369-instance data."
echo ""
echo "  Output dir: ${OUTPUT_DIR}"

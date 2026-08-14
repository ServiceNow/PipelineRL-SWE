#!/usr/bin/env bash
# Collect Claude Opus 5 solve rates on the LCB eval set (100 problems).
#
# Runs Opus 5 via OpenRouter on each eval problem and evaluates using
# the LCB public+private test cases.  Output is a JSONL file with
# {problem_id, resolved, model} — same format as oracle_eval.jsonl from
# the main collection job.
#
# Requires: OPENROUTER_API_KEY_FILE (default below) with a valid key.
#
# After this completes, oracle_eval_opus5.jsonl is used to update
# cost_savings.html with S_OPS_LCB (Opus 5 LCB solve rate).
#
# Optional env vars:
#   SOURCE_COLLECTION_DIR  -- LCB collection dir containing trajectories_eval.jsonl
#                             (default: lcb_collect_qwen_qwen3_4b_instruct_2507)
#   OPUS_MODEL             -- OpenRouter model ID (default: anthropic/claude-opus-5)
#   MAX_CONCURRENT         -- async concurrency (default: 4, Opus is rate-limited)
#   MIN_DATE               -- filter problems on/after this date (default: 2023-09-01)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
OPUS_MODEL=${OPUS_MODEL:-anthropic/claude-opus-5}
MODEL_SLUG=$(echo "${OPUS_MODEL}" | tr '/.-' '___' | tr '[:upper:]' '[:lower:]')
MAX_CONCURRENT=${MAX_CONCURRENT:-4}
MIN_DATE=${MIN_DATE:-2023-09-01}

SOURCE_COLLECTION_DIR=${SOURCE_COLLECTION_DIR:-/mnt/llmd/results/exps/aristides/reason/lcb_collect_qwen_qwen3_4b_instruct_2507_1786218123}
OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE:-/home/toolkit/.secrets/openrouter_api_key}

JOB_NAME=${JOB_NAME:-lcb_opus5_eval_${MODEL_SLUG}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

if [[ ! -d "${SOURCE_COLLECTION_DIR}" ]]; then
  echo "Missing SOURCE_COLLECTION_DIR=${SOURCE_COLLECTION_DIR}" >&2
  exit 1
fi
if [[ ! -s "${OPENROUTER_API_KEY_FILE}" ]]; then
  echo "Missing OpenRouter key at OPENROUTER_API_KEY_FILE=${OPENROUTER_API_KEY_FILE}" >&2
  exit 1
fi

EVAL_TRAJ="${SOURCE_COLLECTION_DIR}/trajectories_eval.jsonl"
if [[ ! -f "${EVAL_TRAJ}" ]]; then
  echo "Missing eval trajectories: ${EVAL_TRAJ}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

RUNNER="${OUTPUT_DIR}/run_eval.sh"
cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "=== Opus 5 LCB eval: model=${OPUS_MODEL}, concurrency=${MAX_CONCURRENT} ==="

python - << 'PYEOF'
import asyncio, base64, json, logging, os, re, subprocess, tempfile, time, zlib
from collections import OrderedDict
from pathlib import Path
import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OPUS_MODEL = "${OPUS_MODEL}"
API_KEY_FILE = "${OPENROUTER_API_KEY_FILE}"
EVAL_TRAJ = "${EVAL_TRAJ}"
OUTPUT_FILE = "${OUTPUT_DIR}/oracle_eval_opus5.jsonl"
MAX_CONCURRENT = ${MAX_CONCURRENT}
MIN_DATE = "${MIN_DATE}"

CODE_RE = re.compile(r"\`\`\`(?:python)?\s*\n(.*?)\`\`\`", re.DOTALL)

def get_api_key():
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        key = Path(API_KEY_FILE).read_text().strip()
    return key

def decode_test_cases(encoded):
    if not encoded:
        return []
    try:
        raw = base64.b64decode(encoded + "==")
        return json.loads(zlib.decompress(raw))
    except Exception:
        try:
            return json.loads(encoded)
        except Exception:
            return []

def extract_code(output):
    m = CODE_RE.search(output)
    return m.group(1).strip() if m else output.strip()

def run_test(code, tc, timeout=10.0):
    if not code.strip():
        return False
    if tc.get("testtype", "stdin") != "stdin":
        return False
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        r = subprocess.run(["python3", tmp], input=tc.get("input",""),
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() == tc.get("output","").strip()
    except Exception:
        return False
    finally:
        os.unlink(tmp)

def evaluate_code(code, pub_tc, priv_tc):
    cases = decode_test_cases(pub_tc) + decode_test_cases(priv_tc)
    if not cases:
        return False
    return all(run_test(code, tc) for tc in cases)

async def call_opus(session, sem, problem_id, prompt, api_key):
    payload = {
        "model": OPUS_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert competitive programmer. Solve the problem and write a complete, correct Python solution. Output only Python code with no explanation."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 8192,
        "temperature": 0.7,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                "X-Title": "PipelineRL-LCB-Opus5"}
    async with sem:
        t0 = time.monotonic()
        async with session.post("https://openrouter.ai/api/v1/chat/completions",
                                headers=headers, json=payload,
                                timeout=aiohttp.ClientTimeout(total=180)) as resp:
            resp.raise_for_status()
            data = await resp.json()
        elapsed = time.monotonic() - t0
        content = data["choices"][0]["message"].get("content", "")
        log.info("  %s: %.1fs, %d chars", problem_id, elapsed, len(content))
        return content

async def main():
    from datasets import load_dataset
    ds = load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True)
    lcb_by_id = {}
    for row in ds:
        if row["contest_date"] < MIN_DATE:
            continue
        pid = f"{row['platform']}_{row['question_id']}"
        lcb_by_id[pid] = row

    # Load unique eval problem IDs from trajectories_eval.jsonl
    seen = OrderedDict()
    with open(EVAL_TRAJ) as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                pid = t["problem_id"]
                if pid not in seen:
                    seen[pid] = t
    eval_pids = list(seen.keys())
    log.info("Eval problems: %d", len(eval_pids))

    # Load already-done results
    done = {}
    out_path = Path(OUTPUT_FILE)
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["problem_id"]] = r
    log.info("Already done: %d", len(done))

    api_key = get_api_key()
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    results = dict(done)

    async with aiohttp.ClientSession() as session:
        todo = [pid for pid in eval_pids if pid not in done and pid in lcb_by_id]
        log.info("To collect: %d", len(todo))

        async def process(pid):
            row = lcb_by_id[pid]
            parts = [row["question_content"]]
            if row.get("starter_code"):
                parts.append(f"\nStarter code:\n\`\`\`python\n{row['starter_code']}\n\`\`\`")
            parts.append("\nWrite a complete Python solution. Read from stdin, write to stdout.")
            prompt = "\n".join(parts)
            try:
                output = await call_opus(session, sem, pid, prompt, api_key)
                code = extract_code(output)
                resolved = evaluate_code(code,
                    row.get("public_test_cases",""),
                    row.get("private_test_cases",""))
            except Exception as e:
                log.warning("Failed %s: %s", pid, e)
                resolved = False
            return {"problem_id": pid, "resolved": resolved, "model": OPUS_MODEL}

        tasks = [process(pid) for pid in todo]
        with open(OUTPUT_FILE, "a") as fh:
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                r = await coro
                results[r["problem_id"]] = r
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                if (i+1) % 10 == 0:
                    n_done = i+1
                    n_res = sum(x["resolved"] for x in results.values())
                    log.info("Progress: %d/%d done, %d resolved so far", n_done, len(tasks), n_res)

    total = [results[pid] for pid in eval_pids if pid in results]
    n_resolved = sum(r["resolved"] for r in total)
    log.info("=== FINAL: %d/%d resolved (%.1f%%) ===", n_resolved, len(total), 100*n_resolved/max(1,len(total)))
    print(f"OPUS5_LCB_SOLVED={n_resolved}")
    print(f"OPUS5_LCB_TOTAL={len(total)}")
    print(f"OPUS5_LCB_RATE={n_resolved/max(1,len(total)):.4f}")

asyncio.run(main())
PYEOF
SCRIPT_EOF
chmod +x "${RUNNER}"

echo "=== Submitting LCB Opus 5 eval job: ${JOB_NAME} ==="
make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT=1 \
  NPROC=1 \
  GPU=0 \
  GPU_MEM=0 \
  CPU=4 \
  CPU_MEM=16 \
  COMMAND="bash ${RUNNER}"

echo ""
echo "Output dir:    ${OUTPUT_DIR}"
echo "Results file:  ${OUTPUT_DIR}/oracle_eval_opus5.jsonl"
echo "Log:           ${OUTPUT_DIR}/run_eval.sh (run by EAI job)"

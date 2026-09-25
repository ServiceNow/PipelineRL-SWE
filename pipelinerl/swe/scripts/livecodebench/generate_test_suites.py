"""Spec-only unit-test-suite generation for the test-writer smoke test (NEW_PATH.md 2.7).

One suite per (problem, test-writer) on the LCB pool test split. Suites are written from the
problem statement ONLY (generator-blind): the test-writer must solve the problem itself to produce
expected outputs, which is what makes test-writer/generator error correlation measurable. Output
format is stdin/stdout test cases (the pool's candidates are stdin/stdout programs).

Resumable: a rerun skips (writer, problem_id) pairs already in the output file.
"""

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import aiohttp

SYSTEM = (
    "You are an expert competitive programmer. You write high-quality test cases for problems. "
    "You answer with strict JSON only."
)

INSTRUCTIONS = """\
You are given a competitive-programming problem statement. Write a small test suite for it.

A solution to this problem is a program that reads the input from stdin and writes the answer to \
stdout, exactly as specified in the statement's Input/Output sections.

Produce a JSON array of exactly 4 test cases. Each element must be an object with two string keys:
  "input": the exact text fed to stdin for this case (follow the input format in the statement \
precisely, including all lines and the trailing newline behavior),
  "expected_output": the exact expected stdout for that input.
Requirements:
- Case 1: the smallest valid example from the statement (use the statement's sample input if one \
is given, with its sample answer).
- Cases 2-3: typical cases you have solved yourself.
- Case 4: a meaningful edge case (minimum size, maximum constraints within reason, or a degenerate \
input).
- "expected_output" must be what a CORRECT program prints for that input. Solve the problem \
yourself before writing the expected outputs. Do not guess.
- No explanations, no markdown fences, no comments. Output ONLY the JSON array.

Problem statement:

"""


def extract_json_array(text: str):
    text = text.strip()
    # strip markdown fences if present
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        arr = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list) or not arr:
        return None
    cases = []
    for c in arr:
        if not isinstance(c, dict):
            continue
        inp, out = c.get("input"), c.get("expected_output")
        if isinstance(inp, str) and isinstance(out, str):
            cases.append({"input": inp, "expected_output": out})
    return cases or None


async def openrouter_call(
    session: aiohttp.ClientSession,
    model: str,
    user: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    reasoning_effort: str | None,
    reasoning_enabled: bool,
    gen_timeout: int,
    base_url: str = "https://openrouter.ai/api",
    title: str = "PipelineRL-testwriter-smoke",
):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    elif reasoning_enabled:
        payload["reasoning"] = {"enabled": True}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": title,
    }
    async with session.post(
        f"{base_url}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=300),
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content.strip() and msg.get("reasoning"):
        content = msg["reasoning"]
    usage = data.get("usage") or {}
    return content, usage


# slot -> (openrouter model, temperature, top_p, reasoning effort or "on")
WRITERS = {
    "qwen4b": ("qwen/qwen3-4b-instruct-2507", 0.7, None, None),
    "oss20lo": ("openai/gpt-oss-20b", 1.0, None, "low"),
    "oss20md": ("openai/gpt-oss-20b", 1.0, None, "medium"),
    "dsv4f": ("deepseek/deepseek-v4-flash", 0.7, 0.95, "on"),
    "oss120md": ("openai/gpt-oss-120b", 1.0, None, "medium"),
}


# Local-vLLM leg: the pool's Qwen3-4B scout is not on OpenRouter; it is served locally
# (see lcb_corrected_temporal_*/run_collect.sh). Model name as served by the local server.
LOCAL_MODEL = {"qwen4b": "Qwen/Qwen3-4B-Instruct-2507"}


async def gen_writer(
    writer: str,
    problems: list[dict],
    api_key: str,
    out_path: Path,
    concurrency: int,
    max_tokens: int = 4096,
    base_url: str = "https://openrouter.ai/api",
    max_retries: int = 2,
) -> None:
    model, temp, top_p, effort = WRITERS[writer]
    is_openrouter = base_url.startswith("https://openrouter")
    if not is_openrouter and writer in LOCAL_MODEL:
        model = LOCAL_MODEL[writer]
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["problem_id"])
                except Exception:
                    continue
    todo = [p for p in problems if p["problem_id"] not in done]
    print(f"[{writer}] {len(done)} existing, {len(todo)} to generate")
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    t0 = time.monotonic()

    async def one(session: aiohttp.ClientSession, prob: dict) -> None:
        user = INSTRUCTIONS + prob["problem_statement"]
        for attempt in range(max_retries + 1):
            try:
                async with sem:
                    content, usage = await openrouter_call(
                        session, model, user, api_key,
                        max_tokens=max_tokens, temperature=temp, top_p=top_p,
                        reasoning_effort=effort, reasoning_enabled=False, gen_timeout=240,
                        base_url=base_url,
                    )
                cases = extract_json_array(content)
                if cases is None:
                    continue
                row = {
                    "problem_id": prob["problem_id"],
                    "difficulty": prob.get("difficulty"),
                    "writer": writer,
                    "model": model,
                    "cases": cases,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "latency_s": round(time.monotonic() - t0, 2),
                    "attempts": attempt + 1,
                }
                async with lock:
                    with open(out_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                return
            except Exception as e:
                if attempt == max_retries:
                    print(f"[{writer}] {prob['problem_id']} FAILED: {type(e).__name__}: {e}")
                await asyncio.sleep(2 * (attempt + 1))

    conn = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=conn) as session:
        await asyncio.gather(*(one(session, p) for p in todo))
    print(f"[{writer}] done -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True, help="pool_v2_tensors_5rung dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", default="test", help="comma list from split_manifest")
    ap.add_argument("--writers", default="qwen4b,oss20lo,oss20md,dsv4f,oss120md")
    ap.add_argument("--api-key-file", default="/home/toolkit/.secrets/openrouter_api_key")
    ap.add_argument("--base-url", default="https://openrouter.ai/api",
                    help="local leg for qwen4b: http://localhost:8000 (vLLM server)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-cases", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    api_key = Path(args.api_key_file).read_text().strip()
    manifest = json.load(open(Path(args.pool_dir) / "split_manifest.json"))
    wanted_ids = set()
    for split in args.splits.split(","):
        wanted_ids.update(manifest[f"{split}_problem_ids"])
    problems = [
        json.loads(line)
        for line in open(Path(args.pool_dir) / "problems.jsonl")
        if json.loads(line)["problem_id"] in wanted_ids
    ]
    print(f"{len(problems)} problems from splits {args.splits}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for writer in args.writers.split(","):
        writer = writer.strip()
        if writer not in WRITERS:
            raise SystemExit(f"unknown writer {writer}; options: {sorted(WRITERS)}")
        out_path = Path(args.out_dir) / f"suites_{writer}.jsonl"
        asyncio.run(
            gen_writer(
                writer,
                problems,
                api_key,
                out_path,
                args.concurrency,
                max_tokens=args.max_tokens,
                base_url=args.base_url,
                max_retries=2,
            )
        )


if __name__ == "__main__":
    main()
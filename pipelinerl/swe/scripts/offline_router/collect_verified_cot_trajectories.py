#!/usr/bin/env python3
"""
Collect CoT trajectories from Qwen3-4B-Thinking-2507 on SWE-bench Verified
instances, reading problem statements from an existing parquet collection
(no Daytona needed — labels are already in the parquet).

Output:
  <output_dir>/trajectories_verified.jsonl  -- scout traces
  <output_dir>/labels_verified.parquet      -- 120B labels in route_successes format

Usage:
  python collect_verified_cot_trajectories.py \
    --parquet-dir /mnt/.../collect/eval \
    --label-route-idx 3 \
    --vllm-base-url http://localhost:8000 \
    --output-dir /mnt/.../verified_cot_XYZ
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

SCOUT_PARAMS = {
    "temperature": 0.6,
    "max_tokens": 8192,
    "top_p": 0.95,
}

SWE_PROMPT = (
    "You are an expert software engineer. You will be given a bug report and relevant "
    "source files. Reason carefully about the bug, then produce a patch that fixes it.\n\n"
    "{problem_statement}"
)


def _parse_thinking_and_patch(text: str) -> tuple[str, str]:
    m = THINK_RE.search(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return "", text.strip()


async def _call_vllm(
    session: aiohttp.ClientSession,
    base_url: str,
    model_name: str,
    problem_statement: str,
    sem: asyncio.Semaphore,
) -> tuple[str | None, dict]:
    async with sem:
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": SWE_PROMPT.format(
                problem_statement=problem_statement)}],
            **SCOUT_PARAMS,
        }
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("vLLM %d: %s", resp.status, body[:200])
                    return None, {}
                data = await resp.json()
        except Exception as exc:
            logger.warning("vLLM error: %s", exc)
            return None, {}
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        usage = data.get("usage", {})
        return text or None, usage


async def collect(args: argparse.Namespace) -> None:
    # Load existing parquet
    parquet_paths = sorted(Path(args.parquet_dir).glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files in {args.parquet_dir}")
    df = pd.concat([pd.read_parquet(p) for p in parquet_paths])
    logger.info("Loaded %d instances from parquet", len(df))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories_verified.jsonl"

    # Resume
    done: set[str] = set()
    if traj_path.exists():
        with traj_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("problem_id"):
                        done.add(r["problem_id"])
                except json.JSONDecodeError:
                    pass
    logger.info("Already done: %d  Remaining: %d", len(done), len(df) - len(done))

    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 4)

    async def process(row: dict) -> dict | None:
        pid = str(row.get("problem_id") or "").strip()
        if not pid or pid in done:
            return None
        # Use prompt_text if available (has full code context), else problem_statement
        problem = str(row.get("prompt_text") or row.get("problem_statement") or "").strip()
        if not problem:
            return None
        # Strip chat template wrapper if present
        if "<|im_start|>user" in problem:
            # extract just the user turn content
            m = re.search(r"<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|$)", problem, re.DOTALL)
            if m:
                problem = m.group(1).strip()
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            text, usage = await _call_vllm(session, args.vllm_base_url, args.scout_model,
                                           problem, sem)
        if text is None:
            return None
        thinking, patch = _parse_thinking_and_patch(text)
        return {
            "problem_id": pid,
            "problem_statement": str(row.get("problem_statement") or "").strip(),
            "thinking_text": thinking,
            "patch_text": patch,
            "full_output": text,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

    rows_to_process = [row for _, row in df.iterrows()
                       if str(row.get("problem_id") or "").strip() not in done]

    n_ok = n_err = 0
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [process(row) for row in rows_to_process]
        with traj_path.open("a") as fh:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is None:
                    n_err += 1
                    continue
                n_ok += 1
                fh.write(json.dumps(result) + "\n")
                fh.flush()
                if n_ok % 50 == 0:
                    logger.info("processed=%d errors=%d", n_ok, n_err)

    logger.info("Scout done: ok=%d errors=%d", n_ok, n_err)
    logger.info("Trajectories: %s", traj_path)

    # Write labels parquet reusing the existing route_successes column
    label_idx = args.label_route_idx
    rows_out = []
    traj_ids: set[str] = set()
    with traj_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("problem_id"):
                    traj_ids.add(r["problem_id"])
            except json.JSONDecodeError:
                pass

    for _, row in df.iterrows():
        pid = str(row.get("problem_id") or "").strip()
        if pid not in traj_ids:
            continue
        rs = list(row.get("route_successes") or [])
        if len(rs) <= label_idx:
            continue
        rows_out.append({
            "problem_id": pid,
            "problem_statement": str(row.get("problem_statement") or "").strip(),
            "route_successes": [bool(rs[label_idx])],
        })

    labels_df = pd.DataFrame(rows_out)
    labels_path = out_dir / "labels_verified.parquet"
    labels_df.to_parquet(labels_path, index=False)
    n_pos = sum(r["route_successes"][0] for r in rows_out)
    logger.info("Labels parquet: %d rows, %d positive (%.1f%%) → %s",
                len(rows_out), n_pos, 100 * n_pos / len(rows_out) if rows_out else 0, labels_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True,
                        help="Existing verified eval parquet dir (has route_successes)")
    parser.add_argument("--label-route-idx", type=int, default=3,
                        help="Index into route_successes for strong-model labels (default 3=120b)")
    parser.add_argument("--vllm-base-url", default="http://localhost:8000")
    parser.add_argument("--scout-model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    asyncio.run(collect(args))


if __name__ == "__main__":
    main()

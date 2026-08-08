#!/usr/bin/env python3
"""
Collect LiveCodeBench trajectories for abstention predictor training.

Two-phase collection:
  Phase 1 (scout): Run cheap scout model via OpenRouter, save thinking traces + code.
  Phase 2 (oracle): Run strong model (oss-120b) via OpenRouter, evaluate, save labels.

Output (compatible with train_cot_abstention_predictor.py + labels parquet):
  <output_dir>/trajectories_train.jsonl
  <output_dir>/trajectories_eval.jsonl
  <output_dir>/labels_train.parquet
  <output_dir>/labels_eval.parquet

Usage:
  # Phase 1: collect scout traces
  python collect_lcb_trajectories.py --phase scout \\
    --scout-model openai/gpt-oss-4b \\
    --output-dir /mnt/.../lcb_trajectories_XYZ

  # Phase 2: collect oracle labels
  python collect_lcb_trajectories.py --phase oracle \\
    --oracle-model openai/gpt-oss-120b \\
    --output-dir /mnt/.../lcb_trajectories_XYZ

  # Run eval only (score existing outputs)
  python collect_lcb_trajectories.py --phase eval \\
    --output-dir /mnt/.../lcb_trajectories_XYZ
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import random
import re
import subprocess
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

SCOUT_SYSTEM = (
    "You are an expert competitive programmer. "
    "Solve the problem and write a complete, correct Python solution. "
    "Output only Python code with no explanation."
)

ORACLE_SYSTEM = (
    "You are an expert competitive programmer. "
    "Solve the problem and write a complete, correct Python solution. "
    "Output only Python code with no explanation."
)


# ── Dataset helpers ──────────────────────────────────────────────────────────

def load_lcb(split: str = "test", max_samples: int = 0,
             min_date: str = "", max_date: str = "",
             difficulties: list[str] | None = None,
             seed: int = 42) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("livecodebench/code_generation_lite", split=split, trust_remote_code=True)
    rows = list(ds)
    if min_date:
        rows = [r for r in rows if r["contest_date"] >= min_date]
    if max_date:
        rows = [r for r in rows if r["contest_date"] <= max_date]
    if difficulties:
        rows = [r for r in rows if r["difficulty"] in difficulties]
    if max_samples and len(rows) > max_samples:
        rng = random.Random(seed)
        rows = rng.sample(rows, max_samples)
    logger.info("Loaded %d LCB problems (split=%s)", len(rows), split)
    return rows


def make_prompt(row: dict) -> str:
    parts = [row["question_content"]]
    if row.get("starter_code"):
        parts.append(f"\nStarter code:\n```python\n{row['starter_code']}\n```")
    parts.append("\nWrite a complete Python solution. Read from stdin, write to stdout.")
    return "\n".join(parts)


def problem_id(row: dict) -> str:
    return f"{row['platform']}_{row['question_id']}"


# ── Test case evaluation ─────────────────────────────────────────────────────

def decode_test_cases(encoded: str) -> list[dict]:
    """Decode base64+zlib compressed LCB test cases."""
    try:
        raw = base64.b64decode(encoded + "==")
        return json.loads(zlib.decompress(raw))
    except Exception:
        try:
            return json.loads(encoded)
        except Exception:
            return []


def extract_code(output: str) -> str:
    """Extract Python code from model output (handles ```python blocks)."""
    m = CODE_RE.search(output)
    if m:
        return m.group(1).strip()
    # If no fences, assume entire output is code (or try to find it)
    return output.strip()


def run_one_test(code: str, test_case: dict, timeout: float = 10.0) -> bool:
    if not code.strip():
        return False
    stdin_data = test_case.get("input", "")
    expected = test_case.get("output", "").strip()
    testtype = test_case.get("testtype", "stdin")
    if testtype != "stdin":
        # func_call style — skip for now (would need different eval)
        return False
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            ["python3", tmp],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        actual = result.stdout.strip()
        return actual == expected
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        os.unlink(tmp)


def evaluate_code(code: str, public_tc: str, private_tc: str,
                  use_private: bool = True) -> bool:
    """Return True if code passes all test cases."""
    cases = decode_test_cases(public_tc) if public_tc else []
    if use_private and private_tc:
        cases = cases + decode_test_cases(private_tc)
    if not cases:
        return False
    return all(run_one_test(code, tc) for tc in cases)


# ── OpenRouter async call ────────────────────────────────────────────────────

async def openrouter_call(
    session: aiohttp.ClientSession,
    model: str,
    system: str,
    user: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api",
    max_tokens: int = 4096,
    temperature: float = 0.7,
    title: str = "PipelineRL-LCB",
    semaphore: asyncio.Semaphore | None = None,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": title,
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async def _call():
        t0 = time.monotonic()
        async with session.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        latency = time.monotonic() - t0
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or ""
        usage = data.get("usage", {})
        return {
            "full_output": content,
            "thinking_text": reasoning,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "latency_s": latency,
        }

    if semaphore:
        async with semaphore:
            return await _call()
    return await _call()


# ── Phase 1: Scout ───────────────────────────────────────────────────────────

async def collect_scout(
    rows: list[dict],
    output_path: Path,
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    title: str,
) -> list[dict]:
    # Resume from existing — only count rows with actual output as done
    done: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("full_output", "").strip():
                    done[r["problem_id"]] = r
        logger.info("Resuming: %d already collected", len(done))

    todo = [r for r in rows if problem_id(r) not in done]
    logger.info("Scout: %d problems to collect (model=%s)", len(todo), model)

    sem = asyncio.Semaphore(concurrency)
    results = list(done.values())

    async with aiohttp.ClientSession() as session:
        async def process(row: dict) -> dict:
            pid = problem_id(row)
            prompt = make_prompt(row)
            try:
                out = await openrouter_call(
                    session, model, SCOUT_SYSTEM, prompt, api_key,
                    base_url=base_url, max_tokens=max_tokens,
                    temperature=temperature, title=title, semaphore=sem,
                )
            except Exception as e:
                logger.warning("Scout failed for %s: %s", pid, e)
                out = {"full_output": "", "thinking_text": "", "prompt_tokens": 0,
                       "completion_tokens": 0, "latency_s": 0.0}

            # Extract thinking from <think> tags if not in reasoning field
            full = out["full_output"]
            thinking = out["thinking_text"]
            if not thinking:
                m = THINK_RE.search(full)
                if m:
                    thinking = m.group(1).strip()
                    full = THINK_RE.sub("", full).strip()
                elif "</think>" in full:
                    parts = full.split("</think>", 1)
                    thinking = parts[0].strip()
                    full = parts[1].strip()

            code = extract_code(full)
            return {
                "problem_id": pid,
                "problem_statement": prompt,
                "question_title": row["question_title"],
                "platform": row["platform"],
                "difficulty": row["difficulty"],
                "contest_date": row["contest_date"],
                "thinking_text": thinking,
                "patch_text": code,
                "full_output": full,
                "model_name_or_path": model,
                "prompt_tokens": out["prompt_tokens"],
                "completion_tokens": out["completion_tokens"],
                "latency_s": out["latency_s"],
                # Store test case strings for eval
                "_public_tc": row.get("public_test_cases", ""),
                "_private_tc": row.get("private_test_cases", ""),
            }

        tasks = [process(r) for r in todo]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            with open(output_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            if (i + 1) % 50 == 0:
                logger.info("Scout: %d/%d done", i + 1, len(todo))

    logger.info("Scout complete: %d trajectories", len(results))
    return results


# ── Phase 2: Oracle + evaluation ─────────────────────────────────────────────

async def collect_oracle(
    rows: list[dict],
    output_path: Path,
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    title: str,
    use_private_tc: bool,
) -> list[dict]:
    done: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                done[r["problem_id"]] = r
        logger.info("Resuming oracle: %d already collected", len(done))

    todo = [r for r in rows if problem_id(r) not in done]
    logger.info("Oracle: %d problems to collect (model=%s)", len(todo), model)

    sem = asyncio.Semaphore(concurrency)
    results = list(done.values())

    async with aiohttp.ClientSession() as session:
        async def process(row: dict) -> dict:
            pid = problem_id(row)
            prompt = make_prompt(row)
            try:
                out = await openrouter_call(
                    session, model, ORACLE_SYSTEM, prompt, api_key,
                    base_url=base_url, max_tokens=max_tokens,
                    temperature=temperature, title=title, semaphore=sem,
                )
                code = extract_code(out["full_output"])
                resolved = evaluate_code(
                    code,
                    row.get("public_test_cases", ""),
                    row.get("private_test_cases", ""),
                    use_private=use_private_tc,
                )
            except Exception as e:
                logger.warning("Oracle failed for %s: %s", pid, e)
                resolved = False
            return {"problem_id": pid, "resolved": resolved, "model": model}

        tasks = [process(r) for r in todo]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            with open(output_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            if (i + 1) % 50 == 0:
                n_resolved = sum(r["resolved"] for r in results)
                logger.info("Oracle: %d/%d done, %d resolved", i + 1, len(todo), n_resolved)

    n_resolved = sum(r["resolved"] for r in results)
    logger.info("Oracle complete: %d/%d resolved (%.1f%%)",
                n_resolved, len(results), 100 * n_resolved / max(1, len(results)))
    return results


# ── Phase 3: Build parquet labels ────────────────────────────────────────────

def build_labels_parquet(oracle_results: list[dict], output_path: Path) -> None:
    """Build labels parquet compatible with train_cot_abstention_predictor.py."""
    rows = []
    for r in oracle_results:
        # route_successes[3] = oracle (oss-120b) success — mirrors SWE-Smith parquet schema
        route_successes = [False, False, False, r["resolved"]]
        route_rewards = [0.0, 0.0, 0.0, float(r["resolved"])]
        rows.append({
            "problem_id": r["problem_id"],
            "route_successes": route_successes,
            "route_rewards": route_rewards,
        })
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    n_pos = sum(r["resolved"] for r in oracle_results)
    logger.info("Wrote labels parquet: %d rows, %d positive (%.1f%%)",
                len(rows), n_pos, 100 * n_pos / max(1, len(rows)))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["scout", "oracle", "eval", "all"], default="all")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scout-model", default="openai/gpt-oss-4b")
    ap.add_argument("--oracle-model", default="openai/gpt-oss-120b")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--api-key-file", default="/home/toolkit/.secrets/openrouter_api_key")
    ap.add_argument("--base-url", default="https://openrouter.ai/api")
    ap.add_argument("--scout-base-url", default="",
                    help="Base URL for scout model (overrides --base-url for scout phase). "
                         "Use to point scout at a local vLLM server: http://localhost:8000")
    ap.add_argument("--scout-api-key", default="",
                    help="API key for scout base URL (if different from OpenRouter key). "
                         "For local vLLM: any non-empty string, e.g. 'local'")
    ap.add_argument("--title", default="PipelineRL-LCB")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Max problems to use (0 = all)")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--min-date", default="2023-09-01",
                    help="Only use problems on or after this date")
    ap.add_argument("--max-date", default="")
    ap.add_argument("--difficulties", nargs="+",
                    default=["easy", "medium", "hard"])
    ap.add_argument("--scout-max-tokens", type=int, default=4096)
    ap.add_argument("--oracle-max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--use-private-tc", action="store_true", default=False,
                    help="Include private test cases for evaluation (slower)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load API key (used for oracle and scout when no --scout-base-url given)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and Path(args.api_key_file).exists():
        api_key = Path(args.api_key_file).read_text().strip()
    if not api_key and args.phase in ("oracle", "all"):
        raise ValueError(f"No API key found in {args.api_key_env} or {args.api_key_file}")

    # Scout can use a separate endpoint (local vLLM)
    scout_base_url = args.scout_base_url or args.base_url
    scout_api_key = args.scout_api_key or api_key
    if not scout_api_key and args.phase in ("scout", "all"):
        raise ValueError(f"No API key for scout — pass --scout-api-key or ensure {args.api_key_file} exists")

    # Load dataset
    all_rows = load_lcb(
        split="test",
        max_samples=args.max_samples,
        min_date=args.min_date,
        max_date=args.max_date,
        difficulties=args.difficulties,
        seed=args.seed,
    )

    # Train/eval split
    rng = random.Random(args.seed)
    shuffled = list(all_rows)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * args.train_frac)
    train_rows = shuffled[:n_train]
    eval_rows = shuffled[n_train:]
    logger.info("Split: %d train / %d eval", len(train_rows), len(eval_rows))

    for split_name, split_rows in [("train", train_rows), ("eval", eval_rows)]:
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        scout_path = output_dir / f"scout_{split_name}.jsonl"
        oracle_path = output_dir / f"oracle_{split_name}.jsonl"
        traj_path = output_dir / f"trajectories_{split_name}.jsonl"
        labels_path = split_dir / "labels.parquet"

        if args.phase in ("scout", "all"):
            asyncio.run(collect_scout(
                split_rows, scout_path,
                model=args.scout_model,
                api_key=scout_api_key,
                base_url=scout_base_url,
                max_tokens=args.scout_max_tokens,
                temperature=args.temperature,
                concurrency=args.concurrency,
                title=args.title,
            ))

        if args.phase in ("oracle", "all"):
            asyncio.run(collect_oracle(
                split_rows, oracle_path,
                model=args.oracle_model,
                api_key=api_key,
                base_url=args.base_url,
                max_tokens=args.oracle_max_tokens,
                temperature=args.temperature,
                concurrency=args.concurrency,
                title=args.title,
                use_private_tc=args.use_private_tc,
            ))

        if args.phase in ("oracle", "all", "eval"):
            if oracle_path.exists():
                oracle_results = [json.loads(l) for l in open(oracle_path)]
                build_labels_parquet(oracle_results, labels_path)

        # Merge scout → trajectories (strip internal _* fields)
        if args.phase in ("scout", "all") and scout_path.exists():
            with open(traj_path, "w") as out_f:
                with open(scout_path) as in_f:
                    for line in in_f:
                        r = json.loads(line)
                        r.pop("_public_tc", None)
                        r.pop("_private_tc", None)
                        out_f.write(json.dumps(r) + "\n")
            logger.info("Wrote %s", traj_path)

    logger.info("Done. Output: %s", output_dir)


if __name__ == "__main__":
    main()

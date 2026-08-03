#!/usr/bin/env python3
"""
Collect MATH dataset CoT trajectories for abstention predictor training.

Two-phase collection:
  Phase 1: Run cheap scout (4B-Thinking via local vLLM) to get thinking traces.
  Phase 2: Run strong model (120B via OpenRouter) to get binary correct/incorrect labels.

Output files (compatible with train_cot_abstention_predictor.py):
  <output_dir>/trajectories_train.jsonl   -- scout traces for training split
  <output_dir>/trajectories_eval.jsonl    -- scout traces for eval split
  <output_dir>/labels_train.parquet       -- strong model labels (parquet)
  <output_dir>/labels_eval.parquet        -- strong model labels (parquet)

Labels parquet schema mirrors the SWE-Smith parquet used elsewhere:
  problem_id, route_successes (list[bool], index 0 = strong model correct)

Usage:
  # Phase 1: collect scout traces (requires vLLM running locally)
  python collect_math_cot_trajectories.py --phase scout \\
    --vllm-base-url http://localhost:8000 \\
    --scout-model Qwen/Qwen3-4B-Thinking-2507 \\
    --output-dir /mnt/.../math_cot_XYZ

  # Phase 2: collect strong model labels (requires OPENROUTER_API_KEY)
  python collect_math_cot_trajectories.py --phase labels \\
    --strong-model openai/gpt-oss-120b \\
    --output-dir /mnt/.../math_cot_XYZ
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import random
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MATH_TRAIN_PATH = (
    "/mnt/llmd/data/hendrycks___competition_math/default/1.0.0/"
    "52c6a268ae72ef772498d27551a3f682dac50cd8befddd0326d758cb6908b5f0/"
    "competition_math-train.arrow"
)
MATH_TEST_PATH = (
    "/mnt/llmd/data/hendrycks___competition_math/default/1.0.0/"
    "52c6a268ae72ef772498d27551a3f682dac50cd8befddd0326d758cb6908b5f0/"
    "competition_math-test.arrow"
)

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")

SCOUT_PARAMS: dict[str, Any] = {
    "temperature": 0.6,
    "max_tokens": 8192,
    "top_p": 0.95,
}

STRONG_SYSTEM = (
    "You are an expert mathematician. Solve the problem and put your final answer "
    "inside \\boxed{}. Be concise."
)


# ── Dataset loading ────────────────────────────────────────────────────────────

def _make_problem_id(problem: str, level: str, ptype: str) -> str:
    h = hashlib.md5(problem.encode()).hexdigest()[:8]
    safe_type = ptype.lower().replace(" ", "_").replace("&", "and")
    safe_level = level.lower().replace(" ", "").replace("?", "x")
    return f"math__{safe_type}__{safe_level}__{h}"


def load_math_split(
    arrow_path: str,
    max_samples: int | None,
    seed: int = 42,
    levels: list[str] | None = None,
) -> list[dict[str, Any]]:
    reader = ipc.open_stream(arrow_path)
    tbl = reader.read_all()
    problems = tbl["problem"].to_pylist()
    lvls     = tbl["level"].to_pylist()
    types    = tbl["type"].to_pylist()
    solutions = tbl["solution"].to_pylist()

    items = []
    for prob, lvl, typ, sol in zip(problems, lvls, types, solutions):
        if levels and lvl not in levels:
            continue
        gt_answer = _extract_boxed(sol)
        if not gt_answer:
            continue  # skip if no boxed answer in reference solution
        items.append({
            "problem_id": _make_problem_id(prob, lvl, typ),
            "problem": prob,
            "level": lvl,
            "type": typ,
            "solution": sol,
            "gt_answer": gt_answer,
        })

    rng = random.Random(seed)
    rng.shuffle(items)
    if max_samples:
        items = items[:max_samples]
    logger.info("Loaded %d problems from %s", len(items), arrow_path)
    return items


# ── Answer extraction & checking ───────────────────────────────────────────────

def _extract_boxed(text: str) -> str:
    """Return the last \\boxed{} content, or empty string."""
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else ""


def _normalise_answer(s: str) -> str:
    s = s.strip()
    # strip leading/trailing dollar signs and whitespace
    s = s.strip("$").strip()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # remove \left / \right
    s = re.sub(r"\\(?:left|right)", "", s)
    # normalise \dfrac → \frac
    s = s.replace(r"\dfrac", r"\frac")
    return s.strip()


def check_correct(model_output: str, gt_answer: str) -> bool:
    pred = _extract_boxed(model_output)
    if not pred:
        return False
    return _normalise_answer(pred) == _normalise_answer(gt_answer)


# ── Scout collection (local vLLM) ──────────────────────────────────────────────

def _build_scout_messages(problem: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Solve the following math problem. Show your reasoning, then put "
                "your final answer inside \\boxed{}.\n\n" + problem
            ),
        }
    ]


async def _chat_vllm(
    session: aiohttp.ClientSession,
    base_url: str,
    model_name: str,
    messages: list[dict],
    params: dict,
) -> tuple[str | None, dict, float]:
    import time
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {"model": model_name, "messages": messages, **params}
    t0 = time.monotonic()
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"vLLM {resp.status}: {body[:200]}")
        data = await resp.json()
    latency = time.monotonic() - t0
    text = data["choices"][0]["message"]["content"] or ""
    usage = data.get("usage", {})
    return text, usage, latency


def _extract_thinking_and_answer(full_text: str) -> tuple[str, str]:
    match = THINK_RE.search(full_text)
    if match:
        thinking = match.group(1).strip()
        answer_text = full_text[match.end():].strip()
    else:
        thinking = ""
        answer_text = full_text.strip()
    return thinking, answer_text


async def _scout_one(
    session: aiohttp.ClientSession,
    base_url: str,
    model_name: str,
    item: dict[str, Any],
    sem: asyncio.Semaphore,
) -> dict[str, Any] | None:
    async with sem:
        pid = item["problem_id"]
        messages = _build_scout_messages(item["problem"])
        try:
            text, usage, latency = await _chat_vllm(
                session, base_url, model_name, messages, SCOUT_PARAMS
            )
        except Exception as exc:
            logger.warning("scout problem=%s error=%s", pid, exc)
            return None
        if not text:
            return None
        thinking, answer_text = _extract_thinking_and_answer(text)
        scout_correct = check_correct(answer_text, item["gt_answer"])
        return {
            "problem_id": pid,
            "problem_statement": item["problem"],   # keep SWE-Smith naming for compat
            "level": item["level"],
            "type": item["type"],
            "thinking_text": thinking,
            "patch_text": answer_text,              # reuse "patch_text" field for compat
            "full_output": text,
            "gt_answer": item["gt_answer"],
            "scout_correct": scout_correct,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "latency_s": latency,
        }


async def run_scout_phase(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", MATH_TRAIN_PATH, args.train_max_samples,
         out_dir / "trajectories_train.jsonl"),
        ("eval",  MATH_TEST_PATH,  args.eval_max_samples,
         out_dir / "trajectories_eval.jsonl"),
    ]

    for split_name, arrow_path, max_s, traj_path in splits:
        items = load_math_split(arrow_path, max_s, seed=args.seed)

        # Resume support
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
        remaining = [it for it in items if it["problem_id"] not in done]
        logger.info("%s: done=%d remaining=%d", split_name, len(done), len(remaining))
        if not remaining:
            continue

        sem = asyncio.Semaphore(args.concurrency)
        timeout = aiohttp.ClientTimeout(total=300)
        connector = aiohttp.TCPConnector(limit=args.concurrency + 4)
        n_ok = n_err = n_correct = 0

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            tasks = [
                _scout_one(session, args.vllm_base_url, args.scout_model, it, sem)
                for it in remaining
            ]
            with traj_path.open("a") as fh:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    if result is None:
                        n_err += 1
                        continue
                    n_ok += 1
                    if result["scout_correct"]:
                        n_correct += 1
                    fh.write(json.dumps(result) + "\n")
                    fh.flush()
                    if n_ok % 50 == 0:
                        logger.info("%s: processed=%d correct=%d (%.1f%%) errors=%d",
                                    split_name, n_ok, n_correct,
                                    100*n_correct/n_ok if n_ok else 0, n_err)

        logger.info("%s done: correct=%d/%d (%.1f%%)", split_name, n_correct, n_ok,
                    100*n_correct/n_ok if n_ok else 0)
        logger.info("Trajectories: %s", traj_path)


# ── Strong model label collection (OpenRouter) ─────────────────────────────────

def _read_api_key(args: argparse.Namespace) -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key and args.api_key_file:
        key = Path(args.api_key_file).read_text().strip()
    if not key:
        env_file = Path(__file__).parents[4] / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("\"'")
                    break
    if not key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    return key


async def _strong_one(
    session: aiohttp.ClientSession,
    item: dict[str, Any],
    model: str,
    api_key: str,
    sem: asyncio.Semaphore,
) -> dict[str, Any] | None:
    async with sem:
        pid = item["problem_id"]
        messages = [
            {"role": "system", "content": STRONG_SYSTEM},
            {"role": "user",   "content": item["problem"]},
        ]
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.0,
        }
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "PipelineRL-SWE MATH strong labels",
        }
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("strong problem=%s status=%d body=%s",
                                   pid, resp.status, body[:200])
                    return None
                data = await resp.json()
        except Exception as exc:
            logger.warning("strong problem=%s error=%s", pid, exc)
            return None

        choices = data.get("choices", [])
        if not choices:
            return None
        text = ((choices[0].get("message") or {}).get("content") or "").strip()
        usage = data.get("usage", {})
        correct = check_correct(text, item["gt_answer"])
        return {
            "problem_id": pid,
            "strong_output": text,
            "strong_correct": correct,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        }


async def run_labels_phase(args: argparse.Namespace) -> None:
    api_key = _read_api_key(args)
    out_dir = Path(args.output_dir)

    splits = [
        ("train", out_dir / "trajectories_train.jsonl",
         out_dir / "labels_train.jsonl"),
        ("eval",  out_dir / "trajectories_eval.jsonl",
         out_dir / "labels_eval.jsonl"),
    ]

    for split_name, traj_path, labels_path in splits:
        if not traj_path.exists():
            logger.warning("No trajectories at %s — run --phase scout first", traj_path)
            continue

        # Load all problems from trajectories (they have gt_answer)
        items: list[dict] = []
        with traj_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("problem_id") and r.get("gt_answer"):
                        items.append({"problem_id": r["problem_id"],
                                      "problem": r["problem_statement"],
                                      "gt_answer": r["gt_answer"]})
                except json.JSONDecodeError:
                    pass

        # Resume
        done: set[str] = set()
        if labels_path.exists():
            with labels_path.open() as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        if r.get("problem_id"):
                            done.add(r["problem_id"])
                    except json.JSONDecodeError:
                        pass
        remaining = [it for it in items if it["problem_id"] not in done]
        logger.info("%s: done=%d remaining=%d", split_name, len(done), len(remaining))
        if not remaining:
            logger.info("%s labels already complete", split_name)
        else:
            sem = asyncio.Semaphore(args.concurrency)
            timeout = aiohttp.ClientTimeout(total=120)
            connector = aiohttp.TCPConnector(limit=args.concurrency + 4)
            n_ok = n_err = n_correct = 0

            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                tasks = [
                    _strong_one(session, it, args.strong_model, api_key, sem)
                    for it in remaining
                ]
                with labels_path.open("a") as fh:
                    for coro in asyncio.as_completed(tasks):
                        result = await coro
                        if result is None:
                            n_err += 1
                            continue
                        n_ok += 1
                        if result["strong_correct"]:
                            n_correct += 1
                        fh.write(json.dumps(result) + "\n")
                        fh.flush()
                        if n_ok % 50 == 0:
                            logger.info("%s: processed=%d correct=%d (%.1f%%) errors=%d",
                                        split_name, n_ok, n_correct,
                                        100*n_correct/n_ok if n_ok else 0, n_err)

            logger.info("%s strong model: correct=%d/%d (%.1f%%)",
                        split_name, n_correct, n_ok, 100*n_correct/n_ok if n_ok else 0)

        # Merge trajectories + labels → parquet (same schema as SWE-Smith parquet)
        _write_parquet(traj_path, labels_path, out_dir / f"labels_{split_name}.parquet")
        logger.info("Labels parquet: %s", out_dir / f"labels_{split_name}.parquet")


def _write_parquet(traj_path: Path, labels_path: Path, out_path: Path) -> None:
    """Merge trajectories and strong-model labels into a parquet compatible with
    train_cot_abstention_predictor.py (which reads route_successes[label_route_idx])."""
    trajs: dict[str, dict] = {}
    with traj_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("problem_id"):
                    trajs[r["problem_id"]] = r
            except json.JSONDecodeError:
                pass

    labels: dict[str, bool] = {}
    with labels_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("problem_id"):
                    labels[r["problem_id"]] = bool(r.get("strong_correct", False))
            except json.JSONDecodeError:
                pass

    rows = []
    for pid, traj in trajs.items():
        if pid not in labels:
            continue
        rows.append({
            "problem_id": pid,
            "problem_statement": traj["problem_statement"],
            "level": traj.get("level", ""),
            "type": traj.get("type", ""),
            "scout_correct": bool(traj.get("scout_correct", False)),
            # route_successes mirrors SWE-Smith format: index 0 = strong model
            "route_successes": [labels[pid]],
            "route_prompt_tokens": [int(traj.get("prompt_tokens", 0))],
            "route_output_tokens": [int(traj.get("completion_tokens", 0))],
        })

    if not rows:
        logger.warning("No merged rows for %s", out_path)
        return
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    n_correct = sum(labels[r["problem_id"]] for r in rows if r["problem_id"] in labels)
    logger.info("Wrote %d rows to %s (strong correct: %d / %.1f%%)",
                len(rows), out_path, n_correct, 100*n_correct/len(rows))


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect MATH CoT trajectories")
    parser.add_argument("--phase", choices=["scout", "labels", "both"], default="both",
                        help="Which phase to run (scout=vLLM traces, labels=strong model)")
    parser.add_argument("--output-dir", required=True)

    # Scout phase args
    parser.add_argument("--vllm-base-url", default="http://localhost:8000")
    parser.add_argument("--scout-model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--train-max-samples", type=int, default=1500)
    parser.add_argument("--eval-max-samples",  type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    # Labels phase args
    parser.add_argument("--strong-model", default="openai/gpt-oss-120b")
    parser.add_argument("--api-key-file", default="")

    args = parser.parse_args()

    if args.phase in ("scout", "both"):
        asyncio.run(run_scout_phase(args))
    if args.phase in ("labels", "both"):
        asyncio.run(run_labels_phase(args))


if __name__ == "__main__":
    main()

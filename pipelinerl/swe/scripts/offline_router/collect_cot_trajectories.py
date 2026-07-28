#!/usr/bin/env python3
"""
Collect repair trajectories with thinking traces from a local vLLM endpoint.

Uses Qwen3-4B-Thinking-2507 (or any thinking-capable model served locally).
Extracts the <think>...</think> block from each response and stores it
separately alongside the patch text.

Outputs two files per split:
  1. <output_dir>/predictions_<split>.jsonl  -- Daytona-compatible predictions
  2. <output_dir>/trajectories_<split>.jsonl -- Full outputs incl. thinking traces

The trajectories file is the input for train_autoregressive_verifier.py (after
Daytona labels are available).

Usage (assumes vLLM is already running):
  python collect_cot_trajectories.py \
    --vllm-base-url http://localhost:8000 \
    --model-name Qwen/Qwen3-4B-Thinking-2507 \
    --train-dataset-path /mnt/llmd/data/swe_smith_bugged_context/ds_train \
    --eval-dataset-path /mnt/llmd/data/swe_smith_bugged_context/ds_test \
    --output-dir /mnt/llmd/results/.../cot_trajectories \
    --train-max-samples 2000 \
    --eval-max-samples 500
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import aiohttp
from datasets import load_from_disk

from pipelinerl.swe.scripts.repair_eval_utils import (
    build_repair_messages,
    chat_completion,
    extract_search_replace_edits,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

DEFAULT_PARAMS: dict[str, Any] = {
    "temperature": 0.6,  # Qwen3 thinking recommended temperature
    "max_tokens": 32768,
    "top_p": 0.95,
}


def _parse_file_contents(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    return {}


def load_dataset_problems(
    dataset_path: str,
    max_samples: int | None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[dict[str, Any]]:
    ds = load_from_disk(dataset_path)
    import random
    items: list[dict[str, Any]] = []
    for item in ds:
        pid = str(
            item.get("issue_id") or item.get("instance_id") or item.get("id") or ""
        ).strip()
        if not pid:
            continue
        fc = _parse_file_contents(item.get("gold_file_contents", "{}"))
        stmt = str(item.get("problem_statement") or "").strip()
        if not fc or not stmt:
            continue
        items.append({
            "problem_id": pid,
            "problem_statement": stmt,
            "file_contents": fc,
            "oracle_patch": str(item.get("patch") or ""),
        })
    if shuffle:
        random.seed(seed)
        random.shuffle(items)
    if max_samples and len(items) > max_samples:
        items = items[:max_samples]
    logger.info("Loaded %d problems from %s", len(items), dataset_path)
    return items


def extract_thinking_and_patch(full_text: str) -> tuple[str, str]:
    """Split model output into thinking trace and patch text."""
    match = THINK_RE.search(full_text)
    if match:
        thinking = match.group(1).strip()
        # Patch is everything after </think>
        patch = full_text[match.end():].strip()
    else:
        thinking = ""
        patch = full_text.strip()
    return thinking, patch


def load_existing_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                pid = rec.get("instance_id") or rec.get("problem_id")
                if pid:
                    done.add(str(pid))
            except json.JSONDecodeError:
                pass
    return done


async def _infer_one(
    session: aiohttp.ClientSession,
    base_url: str,
    model_name: str,
    problem: dict[str, Any],
    sem: asyncio.Semaphore,
) -> dict[str, Any] | None:
    async with sem:
        pid = problem["problem_id"]
        messages, _ = build_repair_messages(
            problem["problem_statement"], problem["file_contents"]
        )
        try:
            text, usage, latency = await chat_completion(
                session=session,
                base_url=base_url,
                model_name=model_name,
                messages=messages,
                parameters=DEFAULT_PARAMS,
            )
        except Exception as exc:
            logger.warning("problem=%s error=%s", pid, exc)
            return None
        if text is None:
            return None
        thinking, patch_text = extract_thinking_and_patch(text)
        edits = extract_search_replace_edits(patch_text)
        return {
            "problem_id": pid,
            "problem_statement": problem["problem_statement"],
            "thinking_text": thinking,
            "patch_text": patch_text,
            "full_output": text,
            "n_edits": len(edits),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "latency_s": latency,
            # Daytona-compatible fields
            "instance_id": pid,
            "model_patch": patch_text,
            "model_name_or_path": model_name,
        }


async def collect_split(
    base_url: str,
    model_name: str,
    problems: list[dict[str, Any]],
    pred_path: Path,
    traj_path: Path,
    concurrency: int,
) -> None:
    done_ids = load_existing_ids(traj_path)
    remaining = [p for p in problems if p["problem_id"] not in done_ids]
    logger.info("done=%d  remaining=%d", len(done_ids), len(remaining))
    if not remaining:
        return

    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=concurrency + 4)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            _infer_one(session=session, base_url=base_url, model_name=model_name,
                       problem=p, sem=sem)
            for p in remaining
        ]
        with pred_path.open("a") as pfh, traj_path.open("a") as tfh:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is None:
                    continue
                # Daytona-compatible prediction record
                pred = {
                    "instance_id": result["instance_id"],
                    "model_patch": result["model_patch"],
                    "model_name_or_path": result["model_name_or_path"],
                }
                pfh.write(json.dumps(pred) + "\n")
                pfh.flush()
                tfh.write(json.dumps(result) + "\n")
                tfh.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CoT trajectories from local vLLM")
    parser.add_argument("--vllm-base-url", default="http://localhost:8000",
                        help="Base URL of the local vLLM server (default: http://localhost:8000)")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Thinking-2507",
                        help="Model name as served by vLLM")
    parser.add_argument("--train-dataset-path", required=True,
                        help="Path to HuggingFace train dataset on disk")
    parser.add_argument("--eval-dataset-path", default="",
                        help="Path to HuggingFace eval dataset on disk (optional)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write prediction and trajectory JSONL files")
    parser.add_argument("--train-max-samples", type=int, default=2000,
                        help="Max train problems to collect (default: 2000)")
    parser.add_argument("--eval-max-samples", type=int, default=500,
                        help="Max eval problems to collect (default: 500)")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Max concurrent requests to vLLM (default: 16)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits: list[tuple[str, str, int]] = []
    splits.append(("train", args.train_dataset_path, args.train_max_samples))
    if args.eval_dataset_path:
        splits.append(("eval", args.eval_dataset_path, args.eval_max_samples))

    for split_name, dataset_path, max_samples in splits:
        logger.info("=== Split: %s ===", split_name)
        problems = load_dataset_problems(
            dataset_path, max_samples=max_samples, seed=args.seed
        )
        pred_path = out_dir / f"predictions_{split_name}.jsonl"
        traj_path = out_dir / f"trajectories_{split_name}.jsonl"
        asyncio.run(collect_split(
            base_url=args.vllm_base_url,
            model_name=args.model_name,
            problems=problems,
            pred_path=pred_path,
            traj_path=traj_path,
            concurrency=args.concurrency,
        ))
        logger.info("Predictions: %s", pred_path)
        logger.info("Trajectories: %s", traj_path)

    logger.info("Done. Next steps:")
    logger.info("  1. Run Daytona eval on predictions_*.jsonl to get success labels")
    logger.info("  2. Run train_autoregressive_verifier.py with --trajectories-dir")


if __name__ == "__main__":
    main()

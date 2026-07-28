#!/usr/bin/env python3
"""
Collect repair predictions for the OpenRouter diversity sweep.

Runs 15 models from OpenRouter against the eval set, producing per-model
JSONL files compatible with run_swesmith_eval_daytona.py.

Usage:
  python collect_openrouter_sweep.py \
    --eval-parquet-dir /mnt/llmd/results/.../collect/eval \
    --dataset-path /mnt/llmd/data/swe_smith_bugged_context/ds_train \
    --output-dir /mnt/llmd/results/.../openrouter_sweep
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from datasets import load_from_disk

from pipelinerl.swe.scripts.repair_eval_utils import (
    build_repair_messages,
    chat_completion,
    extract_search_replace_edits,
)
from pipelinerl.swe.utils.repair_utils import apply_edits_to_files, generate_unified_diff

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OPENROUTER_BASE_URL = "https://openrouter.ai/api"
EXTRA_HEADERS = {"X-Title": "SWE-Router-Sweep"}

# 15 models spanning families and scale tiers, all with solid coding ability.
# MoE models annotated with active params since that drives inference cost.
SWEEP_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",        # 8B dense - Llama family baseline
    "ibm-granite/granite-4.1-8b",              # 8B dense - IBM coding
    "microsoft/phi-4",                          # 14B dense - strong reasoning
    "mistralai/mistral-small-3.2-24b-instruct", # 24B dense - Mistral family
    "qwen/qwen3-coder-30b-a3b-instruct",       # 30B total / 3.3B active MoE - cheap coder
    "google/gemma-4-31b-it",                    # 31B dense - Google family
    "qwen/qwen3-32b",                           # 32B dense - Qwen3 standard
    "meta-llama/llama-4-scout",                 # 109B total / 17B active MoE - Llama 4
    "meta-llama/llama-3.3-70b-instruct",        # 70B dense - Llama 3.3
    "deepseek/deepseek-v4-flash",               # MoE / ~7B active - fast DeepSeek
    "poolside/laguna-s-2.1",                    # Poolside code-specialized
    "deepseek/deepseek-chat-v3.1",              # MoE / ~37B active - top open chat
    "mistralai/codestral-2508",                 # Code-specialized Mistral
    "deepseek/deepseek-r1-0528",                # Reasoning model (always uses <think>)
    "mistralai/devstral-2512",                  # SWE-agent-tuned Mistral
]

DEFAULT_PARAMS: dict[str, Any] = {"temperature": 0.0, "max_tokens": 8192}
# Reasoning models need more token budget for their <think> blocks
MODEL_PARAM_OVERRIDES: dict[str, dict[str, Any]] = {
    "deepseek/deepseek-r1-0528": {"max_tokens": 32768},
}


def _slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(".", "_")


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


def load_eval_problems(parquet_dir: str) -> list[dict[str, Any]]:
    """Load problem_id and problem_statement from eval parquet shards."""
    paths = sorted(Path(parquet_dir).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")
    df = pd.concat([pd.read_parquet(p) for p in paths])
    problems = []
    for _, row in df.iterrows():
        pid = str(row.get("problem_id") or "").strip()
        stmt = str(row.get("problem_statement") or "").strip()
        if pid and stmt:
            problems.append({"problem_id": pid, "problem_statement": stmt})
    logger.info("Loaded %d eval problems from %s", len(problems), parquet_dir)
    return problems


def load_file_contents_lookup(dataset_path: str) -> dict[str, dict[str, str]]:
    """Build problem_id -> file_contents from a HuggingFace dataset on disk."""
    ds = load_from_disk(dataset_path)
    lookup: dict[str, dict[str, str]] = {}
    for item in ds:
        # issue_id in the raw dataset == problem_id in the parquet
        pid = str(
            item.get("issue_id") or item.get("instance_id") or item.get("id") or ""
        ).strip()
        if not pid:
            continue
        fc = _parse_file_contents(item.get("gold_file_contents", "{}"))
        if fc:
            lookup[pid] = fc
    logger.info("Loaded file_contents for %d problems from %s", len(lookup), dataset_path)
    return lookup


def load_existing_predictions(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                pid = rec.get("instance_id") or rec.get("problem_id")
                if pid:
                    done.add(str(pid))
            except json.JSONDecodeError:
                pass
    return done


def _edits_to_git_diff(file_contents: dict[str, str], edits: list[dict]) -> str:
    if not edits:
        return ""
    try:
        new_contents = apply_edits_to_files(file_contents, edits, silent=True)
    except Exception:
        return ""
    parts = []
    for path, new_code in new_contents.items():
        old_code = file_contents.get(path, "")
        if old_code == new_code:
            continue
        hunks = generate_unified_diff(old_code, new_code)
        if not hunks:
            continue
        parts.append(
            f"diff --git a/{path} b/{path}\n"
            f"--- a/{path}\n"
            f"+++ b/{path}\n"
            f"{hunks}"
        )
    return "\n".join(parts)


async def _infer_one(
    session: aiohttp.ClientSession,
    model_id: str,
    api_key: str,
    problem_id: str,
    problem_statement: str,
    file_contents: dict[str, str],
    params: dict[str, Any],
    sem: asyncio.Semaphore,
) -> dict[str, Any] | None:
    async with sem:
        if not file_contents:
            logger.warning("No file_contents for %s, skipping", problem_id)
            return None
        messages, _ = build_repair_messages(problem_statement, file_contents)
        try:
            text, usage, latency = await chat_completion(
                session=session,
                base_url=OPENROUTER_BASE_URL,
                model_name=model_id,
                messages=messages,
                parameters=params,
                api_key=api_key,
                extra_headers=EXTRA_HEADERS,
            )
        except Exception as exc:
            logger.warning("model=%s problem=%s error=%s", model_id, problem_id, exc)
            return None
        edits = extract_search_replace_edits(text or "")
        patch = _edits_to_git_diff(file_contents, edits)
        return {
            "instance_id": problem_id,
            "model_patch": patch,
            "model_name_or_path": model_id,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "latency_s": latency,
            "n_edits": len(edits),
        }


async def collect_model(
    model_id: str,
    api_key: str,
    problems: list[dict[str, Any]],
    file_contents_lookup: dict[str, dict[str, str]],
    out_path: Path,
    concurrency: int,
) -> None:
    done = load_existing_predictions(out_path)
    remaining = [p for p in problems if p["problem_id"] not in done]
    logger.info("model=%s  done=%d  remaining=%d", model_id, len(done), len(remaining))
    if not remaining:
        return

    params = {**DEFAULT_PARAMS, **MODEL_PARAM_OVERRIDES.get(model_id, {})}
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=concurrency + 4)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            _infer_one(
                session=session,
                model_id=model_id,
                api_key=api_key,
                problem_id=p["problem_id"],
                problem_statement=p["problem_statement"],
                file_contents=file_contents_lookup.get(p["problem_id"], {}),
                params=params,
                sem=sem,
            )
            for p in remaining
        ]
        with out_path.open("a") as fh:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is not None:
                    fh.write(json.dumps(result) + "\n")
                    fh.flush()

    total = len(load_existing_predictions(out_path))
    logger.info("model=%s  written=%d / %d", model_id, total, len(problems))


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenRouter diversity sweep collection")
    parser.add_argument("--eval-parquet-dir", required=True,
                        help="Directory of eval *.parquet shards")
    parser.add_argument("--dataset-path", required=True,
                        help="Path to HuggingFace dataset on disk (for file_contents)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write per-model prediction JSONL files")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Max concurrent API calls per model (default: 20)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of model IDs to run (default: all SWEEP_MODELS)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENROUTER_API_KEY environment variable not set")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = load_eval_problems(args.eval_parquet_dir)
    file_contents_lookup = load_file_contents_lookup(args.dataset_path)

    missing = [p["problem_id"] for p in problems if p["problem_id"] not in file_contents_lookup]
    if missing:
        logger.warning("%d problems have no file_contents (first 5: %s)", len(missing), missing[:5])

    models_to_run = args.models if args.models else SWEEP_MODELS
    for model_id in models_to_run:
        out_path = out_dir / f"{_slug(model_id)}.jsonl"
        logger.info("=== model: %s => %s ===", model_id, out_path.name)
        asyncio.run(collect_model(
            model_id=model_id,
            api_key=api_key,
            problems=problems,
            file_contents_lookup=file_contents_lookup,
            out_path=out_path,
            concurrency=args.concurrency,
        ))


if __name__ == "__main__":
    main()

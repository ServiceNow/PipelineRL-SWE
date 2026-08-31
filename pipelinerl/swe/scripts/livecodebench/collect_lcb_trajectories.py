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
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd

from pipelinerl.swe.scripts.livecodebench.mdp_utils import redact_sensitive_text

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

LCB_EVALUATOR_COMMIT = os.environ.get(
    "LCB_RUNNER_COMMIT", "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
)
LCB_DATASET_REPO = "livecodebench/code_generation_lite"
LCB_DATASET_REVISION = os.environ.get(
    "LCB_DATASET_REVISION",
    "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
)

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

def lcb_release_files(release_version: str) -> list[str]:
    """Return the raw files selected by the official LCB dataset script."""
    if release_version == "release_latest":
        return ["test.jsonl", *[f"test{i}.jsonl" for i in range(2, 7)]]

    cumulative = re.fullmatch(r"release_v([1-6])", release_version)
    if cumulative:
        end = int(cumulative.group(1))
        return ["test.jsonl", *[f"test{i}.jsonl" for i in range(2, end + 1)]]

    interval = re.fullmatch(r"v([1-6])(?:_v([1-6]))?", release_version)
    if interval:
        start = int(interval.group(1))
        end = int(interval.group(2) or start)
        if start <= end:
            return [
                "test.jsonl" if i == 1 else f"test{i}.jsonl"
                for i in range(start, end + 1)
            ]

    raise ValueError(f"Unsupported LiveCodeBench release tag: {release_version}")


def load_lcb(
    max_samples: int = 0,
    min_date: str = "",
    max_date: str = "",
    difficulties: list[str] | None = None,
    seed: int = 42,
    release_version: str = "release_latest",
    dataset_revision: str = LCB_DATASET_REVISION,
) -> list[dict]:
    try:
        from huggingface_hub import hf_hub_download
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    except ImportError as exc:
        raise RuntimeError(
            "Official LiveCodeBench runner unavailable. Source "
            "pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh first."
        ) from exc

    # datasets>=4 cannot execute the repository's legacy loading script. The
    # script only selects JSONL files, so reproduce that mapping against a
    # pinned Hub revision and keep the official runner's canonical schema.
    raw_problems = []
    release_files = lcb_release_files(release_version)
    for filename in release_files:
        dataset_path = hf_hub_download(
            repo_id=LCB_DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            revision=dataset_revision,
            token=False,
        )
        with open(dataset_path) as in_f:
            raw_problems.extend(
                json.loads(line) for line in in_f if line.strip()
            )
    logger.info(
        "Loaded %d LCB problems from %s@%s (%s)",
        len(raw_problems),
        LCB_DATASET_REPO,
        dataset_revision,
        ", ".join(release_files),
    )
    problems = [CodeGenerationProblem(**raw) for raw in raw_problems]
    rows = []
    for problem in problems:
        contest_date = problem.contest_date.date().isoformat()
        if min_date and contest_date < min_date:
            continue
        if max_date and contest_date > max_date:
            continue
        difficulty = problem.difficulty.value
        if difficulties and difficulty not in difficulties:
            continue
        rows.append({
            "question_content": problem.question_content,
            "question_title": problem.question_title,
            "question_id": str(problem.question_id),
            "platform": problem.platform.value,
            "difficulty": difficulty,
            "contest_date": contest_date,
            "starter_code": problem.starter_code,
            "metadata": problem.metadata,
            "_public_evaluation_sample": {
                "input_output": json.dumps({
                    "inputs": [test.input for test in problem.public_test_cases],
                    "outputs": [test.output for test in problem.public_test_cases],
                    "fn_name": problem.metadata.get("func_name"),
                }),
            },
            "_evaluation_sample": problem.get_evaluation_sample(),
            "_n_public_tests": len(problem.public_test_cases),
            "_n_private_tests": len(problem.private_test_cases),
            "_lcb_dataset_revision": dataset_revision,
        })
    if max_samples and len(rows) > max_samples:
        rng = random.Random(seed)
        rows = rng.sample(rows, max_samples)
    logger.info("Loaded %d LCB problems (release=%s)", len(rows), release_version)
    return rows


def make_prompt(row: dict) -> str:
    parts = [row["question_content"]]
    if row.get("starter_code"):
        parts.append(f"\nStarter code:\n```python\n{row['starter_code']}\n```")
    if (row.get("metadata") or {}).get("func_name"):
        parts.append(
            "\nComplete the requested Python function or class using the starter-code interface."
        )
    else:
        parts.append("\nWrite a complete Python solution. Read from stdin, write to stdout.")
    return "\n".join(parts)


def problem_id(row: dict) -> str:
    return f"{row['platform']}_{row['question_id']}"


def extract_code(output: str) -> str:
    """Extract Python code from model output (handles ```python blocks)."""
    m = CODE_RE.search(output)
    if m:
        return m.group(1).strip()
    # If no fences, assume entire output is code (or try to find it)
    return output.strip()


# ── Official test-case evaluation ───────────────────────────────────────────

_RESULT_NAMES = {
    -5: "test_runner_error",
    -4: "runtime_error",
    -3: "time_limit_exceeded",
    -2: "wrong_answer",
    -1: "global_timeout",
}


def evaluate_code(
    code: str,
    row: dict,
    timeout: int = 10,
    sample_key: str = "_evaluation_sample",
) -> dict[str, Any]:
    """Evaluate one generation with the pinned official LiveCodeBench runner."""
    if not code.strip():
        return {
            "resolved": False,
            "passing": [],
            "failing": ["NO_CODE"],
            "result_codes": [],
            "metadata": {"error_message": "EmptyGeneration"},
        }
    try:
        from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness

        results, metadata = check_correctness(
            row[sample_key], code, timeout=timeout, debug=False
        )
    except IndexError as exc:
        # The pinned runner constructs fallback -1 results after a global timeout
        # but still indexes an empty metadata list. Preserve its intended result.
        if str(exc) != "list index out of range":
            raise
        inputs = json.loads(row[sample_key]["input_output"]).get("inputs", [])
        result_codes = [-1 for _ in inputs]
        return {
            "resolved": False,
            "passing": [],
            "failing": [
                f"case_{idx:04d}:global_timeout"
                for idx in range(len(result_codes))
            ],
            "result_codes": result_codes,
            "metadata": {
                "error_message": "GlobalTimeout",
                "runner_exception": repr(exc),
            },
        }
    except Exception as exc:
        logger.warning("Official LCB evaluator failed for %s: %s", problem_id(row), exc)
        return {
            "resolved": False,
            "passing": [],
            "failing": ["TEST_RUNNER_ERROR"],
            "result_codes": [],
            "metadata": {"error_message": redact_sensitive_text(repr(exc))},
        }

    normalized = [
        bool(value) if isinstance(value, (bool, np.bool_)) else int(value)
        for value in results
    ]
    passing = [f"case_{idx:04d}" for idx, value in enumerate(normalized) if value is True]
    failing = [
        f"case_{idx:04d}:{_RESULT_NAMES.get(value, 'wrong_answer')}"
        for idx, value in enumerate(normalized)
        if value is not True
    ]
    return {
        "resolved": bool(normalized) and all(value is True for value in normalized),
        "passing": passing,
        "failing": failing,
        "result_codes": normalized,
        "metadata": json.loads(json.dumps(metadata, default=str)),
    }


def is_evaluator_infrastructure_error(
    result_codes: list[Any] | None,
    metadata: dict[str, Any] | None,
    failing: list[str] | None = None,
) -> bool:
    """Return whether grading failed outside the submitted program."""
    if "TEST_RUNNER_ERROR" in (failing or []):
        return True
    message = str((metadata or {}).get("error_message", ""))
    if message == "EmptyGeneration":
        return False
    return not result_codes and bool(message)


def format_test_feedback(report: dict[str, Any], suite_size: int) -> str:
    passing = report["passing"]
    failing = report["failing"]
    if report["resolved"]:
        header = f"Scout test execution: PASSED ({len(passing)}/{suite_size} tests)"
    else:
        header = (
            "Scout test execution: FAILED "
            f"after {len(passing)} passed tests ({suite_size} tests in suite)"
        )
    parts = [header]
    parts.extend(f"FAILED: {name}" for name in failing[:200])
    remaining = max(0, 200 - min(len(failing), 200))
    parts.extend(f"PASSED: {name}" for name in passing[:remaining])
    parts.append(
        f"Observed before stop: {len(passing)} passed, {len(failing)} failed; "
        f"suite size: {suite_size}"
    )
    return "\n".join(parts)


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
    gen_timeout: int = 120,
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
            timeout=aiohttp.ClientTimeout(total=gen_timeout),
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
    eval_timeout: int,
    feedback_tests: str,
    dataset_revision: str,
) -> list[dict]:
    # Old rows were graded by the broken local evaluator. Only resume rows that
    # carry the pinned official-evaluator marker and actual test feedback.
    done: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                if (
                    r.get("full_output", "").strip()
                    and r.get("_lcb_evaluator_commit") == LCB_EVALUATOR_COMMIT
                    and r.get("_lcb_dataset_revision") == dataset_revision
                    and r.get("_lcb_feedback_tests") == feedback_tests
                    and str(r.get("test_feedback") or "").strip()
                    and not is_evaluator_infrastructure_error(
                        r.get("_lcb_feedback_result_codes"),
                        r.get("_lcb_feedback_eval_metadata"),
                        r.get("_tf_failing"),
                    )
                    and not is_evaluator_infrastructure_error(
                        r.get("_lcb_result_codes"),
                        r.get("_lcb_eval_metadata"),
                    )
                ):
                    done[r["problem_id"]] = r
        logger.info("Resuming: %d already collected", len(done))

    todo = [r for r in rows if problem_id(r) not in done]
    logger.info("Scout: %d problems to collect (model=%s)", len(todo), model)

    sem = asyncio.Semaphore(concurrency)
    # The official runner creates a multiprocessing.Manager and child process
    # per call. Overlapping calls race during fork and corrupt labels.
    eval_sem = asyncio.Semaphore(1)
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
                logger.warning("Scout failed for %s: %s", pid, redact_sensitive_text(e))
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
            full_suite_size = row["_n_public_tests"] + row["_n_private_tests"]
            feedback_suite_size = (
                row["_n_public_tests"] if feedback_tests == "public" else full_suite_size
            )
            feedback_sample_key = (
                "_public_evaluation_sample"
                if feedback_tests == "public"
                else "_evaluation_sample"
            )
            async with eval_sem:
                feedback_report = await asyncio.to_thread(
                    evaluate_code, code, row, eval_timeout, feedback_sample_key
                )
                if feedback_tests == "public":
                    scout_report = await asyncio.to_thread(
                        evaluate_code, code, row, eval_timeout
                    )
                else:
                    scout_report = feedback_report
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
                "scout_correct": scout_report["resolved"],
                "test_feedback": format_test_feedback(
                    feedback_report, feedback_suite_size
                ),
                "_tf_failing": feedback_report["failing"],
                "_tf_passing": feedback_report["passing"],
                "_tf_resolved": feedback_report["resolved"],
                "_tf_patch_exists": bool(code.strip()),
                "_tf_total": feedback_suite_size,
                "_lcb_feedback_result_codes": feedback_report["result_codes"],
                "_lcb_feedback_eval_metadata": feedback_report["metadata"],
                "_lcb_result_codes": scout_report["result_codes"],
                "_lcb_eval_metadata": scout_report["metadata"],
                "_lcb_evaluator_commit": LCB_EVALUATOR_COMMIT,
                "_lcb_dataset_revision": dataset_revision,
                "_lcb_feedback_tests": feedback_tests,
                "_lcb_test_count": feedback_suite_size,
                "_lcb_full_test_count": full_suite_size,
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
    eval_timeout: int,
    dataset_revision: str,
) -> list[dict]:
    done: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                error_message = str(
                    (r.get("eval_metadata") or {}).get("error_message", "")
                )
                completed_generation = (
                    bool(r.get("full_output", "").strip())
                    or error_message == "EmptyGeneration"
                )
                if (
                    completed_generation
                    and r.get("_lcb_evaluator_commit") == LCB_EVALUATOR_COMMIT
                    and r.get("_lcb_dataset_revision") == dataset_revision
                    and isinstance(r.get("resolved"), bool)
                    and not is_evaluator_infrastructure_error(
                        r.get("result_codes"), r.get("eval_metadata")
                    )
                ):
                    done[r["problem_id"]] = r
        logger.info("Resuming oracle: %d already collected", len(done))

    todo = [r for r in rows if problem_id(r) not in done]
    logger.info("Oracle: %d problems to collect (model=%s)", len(todo), model)

    sem = asyncio.Semaphore(concurrency)
    # The official runner creates a multiprocessing.Manager and child process
    # per call. Overlapping calls race during fork and corrupt labels.
    eval_sem = asyncio.Semaphore(1)
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
                async with eval_sem:
                    report = await asyncio.to_thread(
                        evaluate_code, code, row, eval_timeout
                    )
            except Exception as e:
                logger.warning("Oracle failed for %s: %s", pid, redact_sensitive_text(e))
                out = {"full_output": "", "thinking_text": "", "prompt_tokens": 0,
                       "completion_tokens": 0, "latency_s": 0.0}
                code = ""
                report = {
                    "resolved": False,
                    "passing": [],
                    "failing": ["MODEL_CALL_ERROR"],
                    "result_codes": [],
                    "metadata": {"error_message": redact_sensitive_text(repr(e))},
                }
            return {
                "problem_id": pid,
                "resolved": report["resolved"],
                "model": model,
                "full_output": out["full_output"],
                "code": code,
                "prompt_tokens": out["prompt_tokens"],
                "completion_tokens": out["completion_tokens"],
                "latency_s": out["latency_s"],
                "result_codes": report["result_codes"],
                "eval_metadata": report["metadata"],
                "_lcb_evaluator_commit": LCB_EVALUATOR_COMMIT,
                "_lcb_dataset_revision": dataset_revision,
            }

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
    latest_by_id = {r["problem_id"]: r for r in oracle_results}
    rows = []
    for r in latest_by_id.values():
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
    n_pos = sum(r["resolved"] for r in latest_by_id.values())
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
    ap.add_argument(
        "--problems-file", default="",
        help=(
            "JSONL of pre-built problems in this script's own record shape, bypassing the "
            "LiveCodeBench download. Used to run the identical generation, extraction and "
            "grading path over TACO, whose input_output payload is already the format LCB "
            "normalises to. --min-date, --max-date, --difficulties and --release-version are "
            "ignored when set; filter when building the file instead."
        ),
    )
    ap.add_argument("--release-version", default="release_latest",
                    help="Official LiveCodeBench release tag")
    ap.add_argument("--dataset-revision", default=LCB_DATASET_REVISION,
                    help="Pinned Hugging Face dataset repository revision")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--temporal-cutoff", default="",
                    help="If set, train before this date and evaluate on/after it")
    ap.add_argument("--min-date", default="2023-09-01",
                    help="Only use problems on or after this date")
    ap.add_argument("--max-date", default="")
    ap.add_argument("--difficulties", nargs="+",
                    default=["easy", "medium", "hard"])
    ap.add_argument("--scout-max-tokens", type=int, default=4096)
    ap.add_argument("--oracle-max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--eval-timeout", type=int, default=10,
                    help="Per-test timeout passed to the official evaluator")
    ap.add_argument("--scout-feedback-tests", choices=["public", "all"], default="public",
                    help="Tests exposed as scout feedback. Use public for headline results; "
                         "all is a diagnostic upper bound only")
    ap.add_argument("--use-private-tc", action="store_true", default=False,
                    help="Deprecated compatibility flag; official evaluation always uses public+private tests")
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
    if args.problems_file:
        with open(args.problems_file) as handle:
            all_rows = [json.loads(line) for line in handle if line.strip()]
        if not all_rows:
            raise ValueError(f"No problems in {args.problems_file}")
        if args.max_samples and len(all_rows) > args.max_samples:
            all_rows = random.Random(args.seed).sample(all_rows, args.max_samples)
        logger.info("Loaded %d problems from %s", len(all_rows), args.problems_file)
    else:
        all_rows = load_lcb(
            max_samples=args.max_samples,
            min_date=args.min_date,
            max_date=args.max_date,
            difficulties=args.difficulties,
            seed=args.seed,
            release_version=args.release_version,
            dataset_revision=args.dataset_revision,
        )

    if args.temporal_cutoff:
        train_rows = [r for r in all_rows if r["contest_date"] < args.temporal_cutoff]
        eval_rows = [r for r in all_rows if r["contest_date"] >= args.temporal_cutoff]
        if not train_rows or not eval_rows:
            raise ValueError(
                f"Temporal cutoff {args.temporal_cutoff} produced an empty split: "
                f"train={len(train_rows)} eval={len(eval_rows)}"
            )
    else:
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
                eval_timeout=args.eval_timeout,
                feedback_tests=args.scout_feedback_tests,
                dataset_revision=args.dataset_revision,
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
                eval_timeout=args.eval_timeout,
                dataset_revision=args.dataset_revision,
            ))

        if args.phase in ("oracle", "all", "eval"):
            if oracle_path.exists():
                oracle_results = [json.loads(l) for l in open(oracle_path)]
                build_labels_parquet(oracle_results, labels_path)

        # Keep raw _tf_* fields: they are required for feedback ablations.
        if args.phase in ("scout", "all", "eval") and scout_path.exists():
            latest_scout: dict[str, dict] = {}
            with open(scout_path) as in_f:
                for line in in_f:
                    r = json.loads(line)
                    if (
                        r.get("_lcb_evaluator_commit") == LCB_EVALUATOR_COMMIT
                        and r.get("_lcb_dataset_revision") == args.dataset_revision
                    ):
                        latest_scout[r["problem_id"]] = r
            with open(traj_path, "w") as out_f:
                for r in latest_scout.values():
                    out_f.write(json.dumps(r) + "\n")
            logger.info("Wrote %s", traj_path)

    logger.info("Done. Output: %s", output_dir)


if __name__ == "__main__":
    main()

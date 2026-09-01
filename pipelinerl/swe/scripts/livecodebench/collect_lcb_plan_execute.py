#!/usr/bin/env python3
"""Plan-then-execute collection for LiveCodeBench.

Stage-1 test of the "large model plans, small model implements" protocol, before any
RL. Two phases, run as separate jobs:

  --phase plan     A planner model writes a short natural-language plan per problem
                   and NO code. Output: <output-dir>/<route>_<split>.jsonl with the plan
                   text and its real prompt/completion token counts (reasoning tokens
                   included, since that is what gets billed).

  --phase execute  An executor model receives the problem plus a plan and writes code.
                   Every generation is graded with the pinned official LCB runner on the
                   public suite and the full suite, exactly like collect_lcb_expert.py, so
                   the rows drop straight into the existing tensor builders as one more
                   route. Multi-draw via --output-suffix _d<k>.

The plan file is the only coupling between the phases, so the same plans can be executed
by several executors (scout, oss20, ...) and the same executor can run plans from several
planners (oss120, the scout itself as the COPE stage-1 control, a content-free scaffold).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

import aiohttp

from pipelinerl.swe.scripts.livecodebench.collect_lcb_expert import (
    _is_complete,
    _read_latest,
    _source_ids,
    validate_split,
)
from pipelinerl.swe.scripts.livecodebench.collect_lcb_trajectories import (
    LCB_DATASET_REVISION,
    LCB_EVALUATOR_COMMIT,
    evaluate_code,
    extract_code,
    load_lcb,
    make_prompt,
    openrouter_call,
    problem_id,
)
from pipelinerl.swe.scripts.livecodebench.mdp_utils import redact_sensitive_text

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PLAN_SYSTEM = (
    "You are an expert competitive programmer acting as a senior engineer who writes "
    "implementation plans for a junior colleague. You write plans, never code."
)

PLAN_INSTRUCTIONS = (
    "\n\nWrite a concise implementation plan for this problem. Include:\n"
    "1. The key observation that makes the problem tractable.\n"
    "2. The algorithm, step by step, precise enough to implement without further thought.\n"
    "3. The data structures to use.\n"
    "4. Time and space complexity, and why it fits the constraints.\n"
    "5. Edge cases and the exact input/output format.\n"
    "Keep it under 300 words. Do NOT write any code."
)

EXECUTE_SYSTEM = (
    "You are an expert competitive programmer. "
    "Implement the given plan as a complete, correct Python solution. "
    "Output only Python code with no explanation."
)

# The content-free scaffold from the 2604.01029 decomposition: same framing, no plan.
NULL_PLAN = (
    "1. Identify the key observation.\n"
    "2. Choose the algorithm and data structures.\n"
    "3. Check the complexity against the constraints.\n"
    "4. Handle edge cases and the exact input/output format."
)


def make_plan_prompt(row: dict[str, Any]) -> str:
    return make_prompt(row) + PLAN_INSTRUCTIONS


def make_execute_prompt(row: dict[str, Any], plan_text: str) -> str:
    return (
        make_prompt(row)
        + "\n\nA senior engineer wrote the following plan. Implement it faithfully.\n\n"
        + "<plan>\n"
        + plan_text.strip()
        + "\n</plan>\n\nOutput only Python code."
    )


def _plan_is_complete(row: dict[str, Any]) -> bool:
    return bool(str(row.get("plan_text") or "").strip())


def _read_plans(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing plan file: {path}")
    return {pid: row for pid, row in _read_latest(path).items() if _plan_is_complete(row)}


# ── phase: plan ─────────────────────────────────────────────────────────────


async def collect_plans(
    rows: list[dict[str, Any]],
    output_path: Path,
    model: str,
    route_label: str,
    api_key: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    title: str,
    gen_timeout: int,
    dataset_revision: str,
) -> None:
    done = {pid: r for pid, r in _read_latest(output_path).items() if _plan_is_complete(r)}
    todo = [row for row in rows if problem_id(row) not in done]
    logger.info("%s: %d/%d plans reusable, %d to collect", output_path.name, len(done), len(rows), len(todo))
    if not todo:
        return
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:

        async def process(row: dict[str, Any]) -> dict[str, Any]:
            pid = problem_id(row)
            try:
                out = await openrouter_call(
                    session, model, PLAN_SYSTEM, make_plan_prompt(row), api_key,
                    base_url=base_url, max_tokens=max_tokens, temperature=temperature,
                    title=title, semaphore=sem, gen_timeout=gen_timeout,
                )
                error = ""
            except Exception as exc:
                logger.warning("plan %s failed for %s: %s", route_label, pid, redact_sensitive_text(exc))
                out = {"full_output": "", "thinking_text": "", "prompt_tokens": 0,
                       "completion_tokens": 0, "latency_s": 0.0}
                error = redact_sensitive_text(repr(exc))
            plan_text = out["full_output"]
            return {
                "problem_id": pid,
                "route_label": route_label,
                "model": model,
                "plan_text": plan_text,
                "plan_words": len(plan_text.split()),
                # A plan that contains a fenced code block violated the instruction; keep
                # it (the executor can still use it) but flag it for the analysis.
                "plan_has_code_fence": "```" in plan_text,
                "thinking_text": out.get("thinking_text", ""),
                "thinking_chars": len(out.get("thinking_text", "") or ""),
                "prompt_tokens": out["prompt_tokens"],
                "completion_tokens": out["completion_tokens"],
                "latency_s": out["latency_s"],
                "error": error,
                "_lcb_dataset_revision": dataset_revision,
                "_generation_temperature": temperature,
            }

        for index, task in enumerate(asyncio.as_completed([process(r) for r in todo]), 1):
            result = await task
            with open(output_path, "a") as out_f:
                out_f.write(json.dumps(result) + "\n")
            if index % 25 == 0:
                logger.info("%s: %d/%d plans collected", route_label, index, len(todo))


def validate_plans(rows: list[dict[str, Any]], output_path: Path, max_invalid_frac: float) -> None:
    expected = {problem_id(r) for r in rows}
    have = set(_read_plans(output_path)) if output_path.exists() else set()
    missing = expected - have
    frac = len(missing) / max(1, len(expected))
    msg = f"{output_path.name}: {len(expected) - len(missing)}/{len(expected)} plans valid"
    if missing and frac > max_invalid_frac:
        raise ValueError(f"{msg} ({frac:.1%} missing > max_invalid_frac={max_invalid_frac})")
    print(msg, flush=True)


# ── phase: execute ──────────────────────────────────────────────────────────


async def collect_executions(
    rows: list[dict[str, Any]],
    plans: dict[str, dict[str, Any]] | None,
    output_path: Path,
    model: str,
    route_label: str,
    plan_route_label: str,
    api_key: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    title: str,
    eval_timeout: int,
    gen_timeout: int,
    dataset_revision: str,
    prompt_builder: Callable[[dict[str, Any], str], str] = make_execute_prompt,
) -> None:
    latest = _read_latest(output_path)
    done = {pid: r for pid, r in latest.items() if _is_complete(r, dataset_revision)}
    todo = [row for row in rows if problem_id(row) not in done]
    if plans is not None:
        skipped = [problem_id(r) for r in todo if problem_id(r) not in plans]
        if skipped:
            logger.warning("%s: %d problems have no plan and are skipped", output_path.name, len(skipped))
        todo = [r for r in todo if problem_id(r) in plans]
    logger.info("%s: %d/%d reusable, %d to collect", output_path.name, len(done), len(rows), len(todo))
    if not todo:
        return
    request_sem = asyncio.Semaphore(concurrency)
    eval_sem = asyncio.Semaphore(1)  # the official runner forks; concurrent forks corrupt labels

    async with aiohttp.ClientSession() as session:

        async def process(row: dict[str, Any]) -> dict[str, Any]:
            pid = problem_id(row)
            plan_row = plans[pid] if plans is not None else None
            plan_text = plan_row["plan_text"] if plan_row is not None else NULL_PLAN
            try:
                out = await openrouter_call(
                    session, model, EXECUTE_SYSTEM, prompt_builder(row, plan_text), api_key,
                    base_url=base_url, max_tokens=max_tokens, temperature=temperature,
                    title=title, semaphore=request_sem, gen_timeout=gen_timeout,
                )
                code = extract_code(out["full_output"])
                async with eval_sem:
                    public_report = await asyncio.to_thread(
                        evaluate_code, code, row, eval_timeout, "_public_evaluation_sample"
                    )
                    full_report = await asyncio.to_thread(evaluate_code, code, row, eval_timeout)
            except Exception as exc:
                logger.warning("%s failed for %s: %s", route_label, pid, redact_sensitive_text(exc))
                out = {"full_output": "", "thinking_text": "", "prompt_tokens": 0,
                       "completion_tokens": 0, "latency_s": 0.0}
                code = ""
                public_report = full_report = {
                    "resolved": False, "result_codes": [],
                    "metadata": {"error_message": redact_sensitive_text(repr(exc))},
                }
            return {
                "problem_id": pid,
                "route_label": route_label,
                "model": model,
                "full_output": out["full_output"],
                "code": code,
                "thinking_text": out.get("thinking_text", ""),
                "thinking_chars": len(out.get("thinking_text", "") or ""),
                "prompt_tokens": out["prompt_tokens"],
                "completion_tokens": out["completion_tokens"],
                "latency_s": out["latency_s"],
                "public_resolved": public_report["resolved"],
                "public_result_codes": public_report["result_codes"],
                "public_eval_metadata": public_report["metadata"],
                "resolved": full_report["resolved"],
                "result_codes": full_report["result_codes"],
                "eval_metadata": full_report["metadata"],
                # Plan provenance, so the cost of the composite can be reconstructed.
                "plan_route_label": plan_route_label,
                "plan_model": plan_row["model"] if plan_row is not None else None,
                "plan_prompt_tokens": plan_row["prompt_tokens"] if plan_row is not None else 0,
                "plan_completion_tokens": plan_row["completion_tokens"] if plan_row is not None else 0,
                "plan_words": plan_row["plan_words"] if plan_row is not None else len(NULL_PLAN.split()),
                "_lcb_evaluator_commit": LCB_EVALUATOR_COMMIT,
                "_lcb_dataset_revision": dataset_revision,
                "_generation_temperature": temperature,
            }

        for index, task in enumerate(asyncio.as_completed([process(r) for r in todo]), 1):
            result = await task
            with open(output_path, "a") as out_f:
                out_f.write(json.dumps(result) + "\n")
            if index % 25 == 0:
                logger.info("%s: %d/%d collected", route_label, index, len(todo))


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["plan", "execute"], required=True)
    parser.add_argument("--source-collection-dir", required=True,
                        help="Existing scout collection; supplies and checks the problem ids")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--route-label", required=True,
                        help="plan: name of the planner route (e.g. plan120). "
                             "execute: name of the composite route (e.g. plan120_scout)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--base-url", default="https://openrouter.ai/api")
    parser.add_argument("--plan-file", default="",
                        help="execute: plan jsonl with '{split}' placeholder, e.g. .../plan120_{split}.jsonl. "
                             "Omit to run the content-free NULL scaffold control.")
    parser.add_argument("--plan-route-label", default="",
                        help="execute: recorded planner route label (default: derived from --plan-file)")
    parser.add_argument("--release-version", default="release_v6")
    parser.add_argument("--dataset-revision", default=LCB_DATASET_REVISION)
    parser.add_argument("--min-date", default="2023-09-01")
    parser.add_argument("--temporal-cutoff", default="2024-10-01")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--eval-timeout", type=int, default=10)
    parser.add_argument("--gen-timeout", type=int, default=600)
    parser.add_argument("--title", default="PipelineRL-LCB-plan-execute")
    parser.add_argument("--output-suffix", default="", help="execute: e.g. _d3 for draw 3")
    parser.add_argument("--splits", default="eval")
    parser.add_argument("--max-invalid-frac", type=float, default=0.05)
    parser.add_argument("--max-problems", type=int, default=0,
                        help="Screen mode: first N problems per split by sorted id")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_collection_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_lcb(min_date=args.min_date, release_version=args.release_version,
                    dataset_revision=args.dataset_revision)
    splits = {
        "train": [r for r in rows if r["contest_date"] < args.temporal_cutoff],
        "eval": [r for r in rows if r["contest_date"] >= args.temporal_cutoff],
    }
    wanted = {s.strip() for s in args.splits.split(",") if s.strip()}
    splits = {k: v for k, v in splits.items() if k in wanted}
    for split, split_rows in splits.items():
        if _source_ids(source_dir, split) != {problem_id(r) for r in split_rows}:
            raise ValueError(f"{split}: source ids do not match dataset ids")
    if args.max_problems:
        splits = {k: sorted(v, key=problem_id)[: args.max_problems] for k, v in splits.items()}
        logger.info("screen mode: %s", {k: len(v) for k, v in splits.items()})

    if args.phase == "plan":
        out_name = lambda split: output_dir / f"{args.route_label}_{split}.jsonl"  # noqa: E731
        if args.validate_only:
            for split, split_rows in splits.items():
                validate_plans(split_rows, out_name(split), args.max_invalid_frac)
            return
        api_key = os.environ.get("OPENROUTER_API_KEY", "") or Path(args.api_key_file).read_text().strip()
        if not api_key:
            raise ValueError("No API key available")
        for split, split_rows in splits.items():
            asyncio.run(collect_plans(
                split_rows, out_name(split), args.model, args.route_label, api_key, args.base_url,
                args.max_tokens, args.temperature, args.concurrency, args.title, args.gen_timeout,
                args.dataset_revision,
            ))
            validate_plans(split_rows, out_name(split), args.max_invalid_frac)
        return

    # execute
    out_name = lambda split: output_dir / f"{args.route_label}_{split}{args.output_suffix}.jsonl"  # noqa: E731
    if args.validate_only:
        for split, split_rows in splits.items():
            validate_split(split_rows, out_name(split), args.dataset_revision,
                           max_invalid_frac=args.max_invalid_frac)
        return
    api_key = os.environ.get("OPENROUTER_API_KEY", "") or Path(args.api_key_file).read_text().strip()
    if not api_key:
        raise ValueError("No API key available")
    plan_route_label = args.plan_route_label or (
        Path(args.plan_file).name.split("_{split}")[0] if args.plan_file else "null_plan"
    )
    for split, split_rows in splits.items():
        plans = _read_plans(Path(args.plan_file.format(split=split))) if args.plan_file else None
        asyncio.run(collect_executions(
            split_rows, plans, out_name(split), args.model, args.route_label, plan_route_label,
            api_key, args.base_url, args.max_tokens, args.temperature, args.concurrency, args.title,
            args.eval_timeout, args.gen_timeout, args.dataset_revision,
        ))
        # Problems without a plan were skipped above, not failed; validate what was attempted.
        planned_rows = [r for r in split_rows if plans is None or problem_id(r) in plans]
        validate_split(planned_rows, out_name(split), args.dataset_revision,
                       max_invalid_frac=args.max_invalid_frac)


if __name__ == "__main__":
    main()

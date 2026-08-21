#!/usr/bin/env python3
"""Collect a corrected LiveCodeBench expert tier on fixed scout problem IDs."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import aiohttp

from pipelinerl.swe.scripts.livecodebench.collect_lcb_trajectories import (
    LCB_DATASET_REVISION,
    LCB_EVALUATOR_COMMIT,
    ORACLE_SYSTEM,
    evaluate_code,
    extract_code,
    is_evaluator_infrastructure_error,
    load_lcb,
    make_prompt,
    openrouter_call,
    problem_id,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _read_latest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        with open(path) as in_f:
            for line in in_f:
                if line.strip():
                    row = json.loads(line)
                    rows[str(row["problem_id"])] = row
    return rows


def _source_ids(source_dir: Path, split: str) -> set[str]:
    path = source_dir / f"scout_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing scout collection: {path}")
    return set(_read_latest(path))


def _is_complete(row: dict[str, Any], dataset_revision: str) -> bool:
    message = str((row.get("eval_metadata") or {}).get("error_message", ""))
    completed_generation = bool(str(row.get("full_output") or "").strip()) or (
        message == "EmptyGeneration"
    )
    return (
        completed_generation
        and row.get("_lcb_evaluator_commit") == LCB_EVALUATOR_COMMIT
        and row.get("_lcb_dataset_revision") == dataset_revision
        and isinstance(row.get("resolved"), bool)
        and isinstance(row.get("public_resolved"), bool)
        and not is_evaluator_infrastructure_error(
            row.get("result_codes"), row.get("eval_metadata")
        )
        and not is_evaluator_infrastructure_error(
            row.get("public_result_codes"), row.get("public_eval_metadata")
        )
    )


async def collect_split(
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
    eval_timeout: int,
    dataset_revision: str,
    gen_timeout: int = 120,
) -> None:
    latest = _read_latest(output_path)
    done = {pid: row for pid, row in latest.items() if _is_complete(row, dataset_revision)}
    todo = [row for row in rows if problem_id(row) not in done]
    logger.info(
        "%s: %d/%d reusable, %d to collect", output_path.name, len(done), len(rows), len(todo)
    )
    if not todo:
        return

    request_sem = asyncio.Semaphore(concurrency)
    # The official runner forks a worker per call; concurrent forks corrupt labels.
    eval_sem = asyncio.Semaphore(1)
    results = list(done.values())

    async with aiohttp.ClientSession() as session:
        async def process(row: dict[str, Any]) -> dict[str, Any]:
            pid = problem_id(row)
            try:
                out = await openrouter_call(
                    session,
                    model,
                    ORACLE_SYSTEM,
                    make_prompt(row),
                    api_key,
                    base_url=base_url,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    title=title,
                    semaphore=request_sem,
                    gen_timeout=gen_timeout,
                )
                code = extract_code(out["full_output"])
                async with eval_sem:
                    public_report = await asyncio.to_thread(
                        evaluate_code,
                        code,
                        row,
                        eval_timeout,
                        "_public_evaluation_sample",
                    )
                    full_report = await asyncio.to_thread(
                        evaluate_code, code, row, eval_timeout
                    )
            except Exception as exc:
                logger.warning("%s failed for %s: %s", route_label, pid, exc)
                out = {
                    "full_output": "",
                    "thinking_text": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_s": 0.0,
                }
                code = ""
                public_report = full_report = {
                    "resolved": False,
                    "result_codes": [],
                    "metadata": {"error_message": repr(exc)},
                }
            return {
                "problem_id": pid,
                "route_label": route_label,
                "model": model,
                "full_output": out["full_output"],
                "code": code,
                "prompt_tokens": out["prompt_tokens"],
                "completion_tokens": out["completion_tokens"],
                "latency_s": out["latency_s"],
                "public_resolved": public_report["resolved"],
                "public_result_codes": public_report["result_codes"],
                "public_eval_metadata": public_report["metadata"],
                "resolved": full_report["resolved"],
                "result_codes": full_report["result_codes"],
                "eval_metadata": full_report["metadata"],
                "_lcb_evaluator_commit": LCB_EVALUATOR_COMMIT,
                "_lcb_dataset_revision": dataset_revision,
            }

        for index, task in enumerate(asyncio.as_completed([process(row) for row in todo]), 1):
            result = await task
            results.append(result)
            with open(output_path, "a") as out_f:
                out_f.write(json.dumps(result) + "\n")
            if index % 25 == 0:
                logger.info("%s: %d/%d collected", route_label, index, len(todo))


def validate_split(rows: list[dict[str, Any]], output_path: Path, dataset_revision: str) -> None:
    expected_ids = {problem_id(row) for row in rows}
    latest = _read_latest(output_path)
    missing = expected_ids - set(latest)
    invalid = [
        pid
        for pid in expected_ids & set(latest)
        if not _is_complete(latest[pid], dataset_revision)
    ]
    if missing or invalid:
        raise ValueError(
            f"{output_path}: missing={len(missing)} invalid={len(invalid)}"
        )
    print(f"{output_path.name}: valid {len(expected_ids)} rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-collection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--route-label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--base-url", default="https://openrouter.ai/api")
    parser.add_argument("--release-version", default="release_v6")
    parser.add_argument("--dataset-revision", default=LCB_DATASET_REVISION)
    parser.add_argument("--min-date", default="2023-09-01")
    parser.add_argument("--temporal-cutoff", default="2024-10-01")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--eval-timeout", type=int, default=10)
    parser.add_argument("--gen-timeout", type=int, default=120,
                        help="HTTP timeout in seconds for a single OpenRouter generation")
    parser.add_argument("--title", default="PipelineRL-LCB-routing")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_collection_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_lcb(
        min_date=args.min_date,
        release_version=args.release_version,
        dataset_revision=args.dataset_revision,
    )
    splits = {
        "train": [row for row in rows if row["contest_date"] < args.temporal_cutoff],
        "eval": [row for row in rows if row["contest_date"] >= args.temporal_cutoff],
    }
    for split, split_rows in splits.items():
        source_ids = _source_ids(source_dir, split)
        actual_ids = {problem_id(row) for row in split_rows}
        if source_ids != actual_ids:
            raise ValueError(
                f"{split} source IDs {len(source_ids)} do not match dataset IDs {len(actual_ids)}"
            )

    if args.validate_only:
        for split, split_rows in splits.items():
            validate_split(
                split_rows, output_dir / f"{args.route_label}_{split}.jsonl", args.dataset_revision
            )
        return

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        api_key = Path(args.api_key_file).read_text().strip()
    if not api_key:
        raise ValueError("No OpenRouter API key available")
    for split, split_rows in splits.items():
        asyncio.run(
            collect_split(
                split_rows,
                output_dir / f"{args.route_label}_{split}.jsonl",
                args.model,
                args.route_label,
                api_key,
                args.base_url,
                args.max_tokens,
                args.temperature,
                args.concurrency,
                args.title,
                args.eval_timeout,
                args.dataset_revision,
                gen_timeout=args.gen_timeout,
            )
        )
        validate_split(
            split_rows, output_dir / f"{args.route_label}_{split}.jsonl", args.dataset_revision
        )


if __name__ == "__main__":
    main()

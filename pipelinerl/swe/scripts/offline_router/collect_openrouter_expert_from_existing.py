#!/usr/bin/env python
"""Collect an OpenRouter expert route while reusing an existing primary route."""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq

from pipelinerl.swe.load_datasets import load_local_swe_dataset
from pipelinerl.swe.scripts.offline_router.common import (
    infer_language_from_problem,
    problem_id_from_item,
    sanitize_for_json,
    write_json,
)
from pipelinerl.swe.scripts.repair_eval_utils import (
    build_repair_messages,
    chat_completion,
    extract_search_replace_edits,
)
from pipelinerl.swe.utils.repair_utils import calculate_precise_reward

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-collection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="google/gemini-3-flash-preview")
    parser.add_argument("--expert-label", default=None)
    parser.add_argument("--base-url", default="https://openrouter.ai/api")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--train-dataset-names", default="swe_bench_train")
    parser.add_argument("--train-dataset-path", default="/mnt/llmd/data/swebench/ds_train")
    parser.add_argument("--train-dataset-label", default=None)
    parser.add_argument("--train-max-samples", type=int, default=4096)
    parser.add_argument("--eval-dataset-names", default="swebench_lite")
    parser.add_argument("--eval-dataset-path", default="/mnt/llmd/data/swebench_lite/ds")
    parser.add_argument("--eval-dataset-label", default=None)
    parser.add_argument("--eval-max-samples", type=int, default=500)
    parser.add_argument("--collect-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collect-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=15000)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--connector-limit", type=int, default=64)
    parser.add_argument("--max-concurrent-problems", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--progress-log-every", type=int, default=25)
    parser.add_argument("--openrouter-title", default="PipelineRL-SWE offline router")
    parser.add_argument("--openrouter-referer", default=None)
    return parser.parse_args()


def _split_names(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _load_dataset_for_split(args: argparse.Namespace, split_name: str) -> list[dict[str, Any]]:
    if split_name == "train":
        dataset_names = _split_names(args.train_dataset_names)
        dataset_path = args.train_dataset_path
        dataset_label = args.train_dataset_label
        max_samples = args.train_max_samples
    elif split_name == "eval":
        dataset_names = _split_names(args.eval_dataset_names)
        dataset_path = args.eval_dataset_path
        dataset_label = args.eval_dataset_label
        max_samples = args.eval_max_samples
    else:
        raise ValueError(f"Unsupported split: {split_name}")

    dataset = load_local_swe_dataset(
        dataset_names=dataset_names,
        dataset_path=dataset_path,
        shuffle=bool(args.shuffle),
        seed=int(args.seed),
        dataset_label=dataset_label,
        max_samples=max_samples,
    )
    return sorted(dataset, key=problem_id_from_item)


def _read_api_key(args: argparse.Namespace) -> str:
    key = os.environ.get(args.api_key_env)
    if key:
        return key.strip()
    if args.api_key_file:
        path = Path(args.api_key_file)
        if path.exists():
            return path.read_text().strip()
    raise ValueError(
        f"Missing OpenRouter API key. Set {args.api_key_env} or pass --api-key-file pointing to a readable file."
    )


def _load_source_rows(split_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not split_dir.exists():
        return rows
    for shard in sorted(split_dir.glob("*.parquet")):
        table = pq.read_table(shard)
        for row in table.to_pylist():
            problem_id = row.get("problem_id")
            if isinstance(problem_id, str) and problem_id:
                rows[problem_id] = row
    return rows


def _load_existing_problem_ids(split_dir: Path) -> tuple[set[str], int]:
    existing: set[str] = set()
    max_index = -1
    if not split_dir.exists():
        return existing, 0
    for shard in sorted(split_dir.glob("*.parquet")):
        try:
            max_index = max(max_index, int(shard.stem.split("-")[-1]))
        except ValueError:
            pass
        try:
            table = pq.read_table(shard, columns=["problem_id"])
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to read existing shard %s: %s", shard, exc)
            continue
        for value in table.column("problem_id").to_pylist():
            if isinstance(value, str) and value:
                existing.add(value)
    return existing, max_index + 1


def _row_count_in_shard(path: Path) -> int:
    try:
        return int(pq.read_metadata(path).num_rows)
    except Exception:  # pylint: disable=broad-except
        return 0


def _write_parquet_shard(split_dir: Path, shard_index: int, rows: list[dict[str, Any]]) -> Path:
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"{split_dir.name}-{shard_index:05d}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _list_value(row: dict[str, Any], key: str, index: int, default: Any) -> Any:
    values = row.get(key)
    if isinstance(values, list) and len(values) > index:
        return values[index]
    return default


def _derive_failure_type(success: bool, reward_metadata: dict[str, Any], request_error: str | None) -> str:
    if request_error:
        return "request_error"
    if bool(reward_metadata.get("format_error")):
        return "format"
    if success:
        return "none"
    return "semantic"


async def _run_openrouter_expert(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    api_key: str,
    repair_messages: list[dict[str, str]],
    file_contents: dict[str, str],
    oracle_patch: str,
) -> dict[str, Any]:
    request_error: str | None = None
    usage: dict[str, Any] = {}
    repair_text = ""
    start = time.time()
    parameters: dict[str, Any] = {
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
    }
    extra_headers: dict[str, str] = {}
    if args.openrouter_title:
        extra_headers["X-Title"] = str(args.openrouter_title)
    if args.openrouter_referer:
        extra_headers["HTTP-Referer"] = str(args.openrouter_referer)

    try:
        # chat_completion handles the OpenAI-compatible response shape and usage accounting.
        repair_text, usage, latency = await chat_completion(
            session=session,
            base_url=args.base_url,
            model_name=args.model,
            messages=repair_messages,
            parameters=parameters,
            api_key=api_key,
            extra_headers=extra_headers,
        )
    except Exception as exc:  # pylint: disable=broad-except
        latency = time.time() - start
        request_error = str(exc)

    if request_error:
        reward = 0.0
        reward_metadata: dict[str, Any] = {"request_error": True, "error": request_error}
    else:
        edits = extract_search_replace_edits(repair_text)
        reward, reward_metadata = calculate_precise_reward(file_contents, oracle_patch, edits)

    success = bool(reward and reward > float(args.success_threshold))
    return {
        "output_text": repair_text,
        "reward": float(reward or 0.0),
        "success": success,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
        "latency_s": float(latency),
        "failure_type": _derive_failure_type(success, reward_metadata, request_error),
        "reward_metadata": reward_metadata,
    }


async def _collect_problem(
    problem: dict[str, Any],
    source_row: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
    session: aiohttp.ClientSession,
    split_name: str,
) -> dict[str, Any]:
    problem_id = problem_id_from_item(problem)
    file_contents = problem.get("file_contents") or {}
    if not isinstance(file_contents, dict) or not file_contents:
        raise ValueError(f"Problem {problem_id} missing file_contents")
    problem_statement = problem.get("problem_statement")
    if not isinstance(problem_statement, str) or not problem_statement.strip():
        raise ValueError(f"Problem {problem_id} missing problem_statement")

    repair_messages, _stage_input = build_repair_messages(problem_statement, file_contents)
    expert = await _run_openrouter_expert(
        session=session,
        args=args,
        api_key=api_key,
        repair_messages=repair_messages,
        file_contents=file_contents,
        oracle_patch=str(problem.get("patch") or ""),
    )

    primary_output = _list_value(
        source_row,
        "route_outputs",
        0,
        source_row.get("primary_output_text") or "",
    )
    primary_reward = float(_list_value(source_row, "route_rewards", 0, 0.0) or 0.0)
    primary_success = bool(_list_value(source_row, "route_successes", 0, primary_reward > float(args.success_threshold)))

    return {
        "problem_id": problem_id,
        "dataset": str(problem.get("dataset") or source_row.get("dataset") or ""),
        "split": split_name,
        "repo": str(problem.get("repo") or source_row.get("repo") or ""),
        "base_commit": str(problem.get("base_commit") or source_row.get("base_commit") or ""),
        "language": str(source_row.get("language") or infer_language_from_problem(problem)),
        "problem_statement": problem_statement,
        "prompt_text": str(source_row.get("prompt_text") or ""),
        "primary_output_text": str(primary_output or ""),
        "performance_targets": [primary_reward, float(expert["reward"])],
        "route_rewards": [primary_reward, float(expert["reward"])],
        "route_successes": [primary_success, bool(expert["success"])],
        "route_prompt_tokens": [
            int(_list_value(source_row, "route_prompt_tokens", 0, 0) or 0),
            int(expert["prompt_tokens"]),
        ],
        "route_output_tokens": [
            int(_list_value(source_row, "route_output_tokens", 0, 0) or 0),
            int(expert["output_tokens"]),
        ],
        "route_latencies_s": [
            float(_list_value(source_row, "route_latencies_s", 0, 0.0) or 0.0),
            float(expert["latency_s"]),
        ],
        "route_outputs": [str(primary_output or ""), str(expert["output_text"] or "")],
        "route_failure_types": [
            str(_list_value(source_row, "route_failure_types", 0, "unknown") or "unknown"),
            str(expert["failure_type"]),
        ],
    }


async def _collect_split(args: argparse.Namespace, split_name: str, api_key: str) -> dict[str, Any]:
    source_rows = _load_source_rows(Path(args.source_collection_dir) / split_name)
    dataset = _load_dataset_for_split(args, split_name)
    split_dir = Path(args.output_dir) / split_name
    existing_ids, next_shard_index = _load_existing_problem_ids(split_dir)

    pending: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    skipped_existing = 0
    skipped_missing_source = 0
    for problem in dataset:
        problem_id = problem_id_from_item(problem)
        if problem_id in existing_ids:
            skipped_existing += 1
            continue
        source_row = source_rows.get(problem_id)
        if source_row is None:
            skipped_missing_source += 1
            continue
        pending.append((len(pending), problem, source_row))

    logger.info(
        "Starting %s OpenRouter expert collection: dataset_rows=%d source_rows=%d "
        "skipped_existing=%d skipped_missing_source=%d pending=%d shard_size=%d",
        split_name,
        len(dataset),
        len(source_rows),
        skipped_existing,
        skipped_missing_source,
        len(pending),
        int(args.shard_size),
    )

    connector = aiohttp.TCPConnector(limit=int(args.connector_limit))
    timeout = aiohttp.ClientTimeout(total=float(args.request_timeout))
    semaphore = asyncio.Semaphore(int(args.max_concurrent_problems))
    max_in_flight = max(1, int(args.max_concurrent_problems) * 4)

    written = 0
    failed = 0
    buffered_rows: list[dict[str, Any]] = []
    ready_rows: dict[int, dict[str, Any] | None] = {}
    next_ready_idx = 0
    shard_row_counts: list[int] = []

    async def _task_wrapper(
        order_idx: int,
        problem: dict[str, Any],
        source_row: dict[str, Any],
        session: aiohttp.ClientSession,
    ) -> tuple[int, dict[str, Any] | None]:
        async with semaphore:
            try:
                row = await _collect_problem(
                    problem=problem,
                    source_row=source_row,
                    args=args,
                    api_key=api_key,
                    session=session,
                    split_name=split_name,
                )
                return order_idx, row
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Failed to collect %s problem %s: %s", split_name, problem_id_from_item(problem), exc)
                return order_idx, None

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        pending_iter = iter(pending)
        in_flight: set[asyncio.Task] = set()
        while True:
            while len(in_flight) < max_in_flight:
                try:
                    order_idx, problem, source_row = next(pending_iter)
                except StopIteration:
                    break
                in_flight.add(asyncio.create_task(_task_wrapper(order_idx, problem, source_row, session)))
            if not in_flight:
                break

            done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                order_idx, row = await future
                ready_rows[order_idx] = row
                while next_ready_idx in ready_rows:
                    ready = ready_rows.pop(next_ready_idx)
                    next_ready_idx += 1
                    if ready is None:
                        failed += 1
                        continue
                    buffered_rows.append(ready)
                    written += 1
                    if len(buffered_rows) >= int(args.shard_size):
                        shard_path = _write_parquet_shard(split_dir, next_shard_index, buffered_rows)
                        shard_row_counts.append(_row_count_in_shard(shard_path))
                        logger.info(
                            "%s progress: processed=%d/%d written=%d failed=%d shards=%d last_shard=%s",
                            split_name,
                            written + failed,
                            len(pending),
                            written,
                            failed,
                            len(shard_row_counts),
                            shard_path.name,
                        )
                        next_shard_index += 1
                        buffered_rows = []
                    elif (written + failed) % max(1, int(args.progress_log_every)) == 0:
                        logger.info(
                            "%s progress: processed=%d/%d written=%d failed=%d buffered=%d in_flight=%d",
                            split_name,
                            written + failed,
                            len(pending),
                            written,
                            failed,
                            len(buffered_rows),
                            len(in_flight),
                        )

    if buffered_rows:
        shard_path = _write_parquet_shard(split_dir, next_shard_index, buffered_rows)
        shard_row_counts.append(_row_count_in_shard(shard_path))

    logger.info(
        "Finished %s collection: pending=%d written=%d failed=%d shards=%d",
        split_name,
        len(pending),
        written,
        failed,
        len(shard_row_counts),
    )
    return {
        "split": split_name,
        "n_dataset_rows": len(dataset),
        "n_source_rows": len(source_rows),
        "n_skipped_existing": skipped_existing,
        "n_skipped_missing_source": skipped_missing_source,
        "n_pending": len(pending),
        "n_written": written,
        "n_failed": failed,
        "shard_row_counts": shard_row_counts,
    }


def _safe_config(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    payload["api_key_env"] = str(args.api_key_env)
    if payload.get("api_key_file"):
        payload["api_key_file"] = str(payload["api_key_file"])
    return sanitize_for_json(payload)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = _read_api_key(args)

    write_json(output_dir / "collection_config.json", _safe_config(args))
    started_at = time.time()
    split_summaries: list[dict[str, Any]] = []
    if args.collect_train:
        split_summaries.append(asyncio.run(_collect_split(args, "train", api_key)))
    if args.collect_eval:
        split_summaries.append(asyncio.run(_collect_split(args, "eval", api_key)))

    metadata = {
        "schema_version": 1,
        "collection_started_at_unix": started_at,
        "collection_completed_at_unix": time.time(),
        "source_collection_dir": str(args.source_collection_dir),
        "route_labels": ["primary_model", args.expert_label or f"expert_0:{args.model}"],
        "route_model_names": ["primary_model", args.model],
        "prompt_builder": "pipelinerl.swe.scripts.repair_eval_utils.build_repair_messages",
        "reward_function": "pipelinerl.swe.utils.repair_utils.calculate_precise_reward",
        "primary_route_source": "copied_from_source_collection_route_0",
        "expert_route_source": "openrouter_chat_completions",
        "split_summaries": split_summaries,
    }
    write_json(output_dir / "metadata.json", metadata)
    logger.info("OpenRouter expert collection complete: output_dir=%s", output_dir)


if __name__ == "__main__":
    main()

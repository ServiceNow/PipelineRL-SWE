#!/usr/bin/env python
import asyncio
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import aiohttp
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

from pipelinerl.swe.scripts.offline_router.common import (
    RouteSpec,
    infer_language_from_problem,
    problem_id_from_item,
    render_prompt_text,
    route_label_for_expert,
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


def _row_count_in_shard(path: Path) -> int:
    try:
        metadata = pq.read_metadata(path)
        return int(metadata.num_rows)
    except Exception:
        return 0


def _load_existing_problem_ids(split_dir: Path) -> tuple[set[str], int]:
    existing: set[str] = set()
    max_index = -1
    if not split_dir.exists():
        return existing, 0
    for shard in sorted(split_dir.glob("*.parquet")):
        stem = shard.stem
        try:
            max_index = max(max_index, int(stem.split("-")[-1]))
        except ValueError:
            pass
        try:
            table = pq.read_table(shard, columns=["problem_id"])
        except Exception as exc:
            logger.warning("Failed to read existing shard %s: %s", shard, exc)
            continue
        for value in table.column("problem_id").to_pylist():
            if isinstance(value, str) and value:
                existing.add(value)
    return existing, max_index + 1


def _write_parquet_shard(split_dir: Path, shard_index: int, rows: list[dict[str, Any]]) -> Path:
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"{split_dir.name}-{shard_index:05d}.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return path


def _resolve_primary_route(cfg: DictConfig) -> RouteSpec:
    primary_cfg = cfg.offline_router.primary_model
    if not primary_cfg.get("base_url"):
        raise ValueError("offline_router.primary_model.base_url must be set for offline collection")
    model_name = str(primary_cfg.get("model_name") or primary_cfg.get("model_path"))
    return RouteSpec(
        label=str(primary_cfg.get("label") or "primary_model"),
        model_name=model_name,
        base_url=str(primary_cfg.base_url),
        parameters=dict(OmegaConf.to_container(primary_cfg.get("parameters", {}), resolve=True) or {}),
        api_key=primary_cfg.get("api_key"),
    )


def _resolve_expert_routes(cfg: DictConfig) -> list[RouteSpec]:
    expert_cfgs = sorted(list(cfg.offline_router.get("experts", [])), key=lambda item: int(item.get("expert_rank", 0)))
    if not expert_cfgs:
        return []

    base_urls = list(cfg.offline_router.get("expert_base_urls", []))
    if not base_urls:
        base_urls = [str(item.get("base_url")) for item in expert_cfgs if item.get("base_url")]
        if len(base_urls) != len(expert_cfgs):
            raise ValueError(
                "Need offline_router.expert_base_urls or base_url set on each offline_router.experts entry"
            )

    if len(base_urls) != len(expert_cfgs):
        raise ValueError(
            f"expert_base_urls length mismatch: expected {len(expert_cfgs)} urls, got {len(base_urls)}"
        )

    routes: list[RouteSpec] = []
    for expert_cfg, base_url in zip(expert_cfgs, base_urls):
        expert_rank = int(expert_cfg.get("expert_rank", 0))
        model_name = str(expert_cfg.get("model_name") or expert_cfg.get("model_path"))
        routes.append(
            RouteSpec(
                label=str(expert_cfg.get("label") or route_label_for_expert(expert_rank, model_name)),
                model_name=model_name,
                base_url=str(base_url),
                parameters=dict(OmegaConf.to_container(expert_cfg.get("parameters", {}), resolve=True) or {}),
                api_key=expert_cfg.get("api_key"),
            )
        )
    return routes


def _load_split_dataset(cfg: DictConfig, split_name: str) -> list[dict[str, Any]]:
    dataset_cfg = cfg.offline_router.dataset
    dataset_loader = get_method(dataset_cfg.loader)
    params = {
        "shuffle": bool(cfg.offline_router.collection.get("shuffle", False)),
        "seed": int(cfg.get("seed", 42)),
        "max_samples": dataset_cfg.get(f"{split_name}_max_samples"),
    }

    if split_name == "train":
        dataset_names = list(dataset_cfg.train_dataset_names)
        params["dataset_path"] = str(dataset_cfg.train_dataset_path)
        dataset_label = dataset_cfg.get("train_dataset_label")
    elif split_name == "eval":
        dataset_names = list(dataset_cfg.eval_dataset_names)
        params["dataset_path"] = str(dataset_cfg.eval_dataset_path)
        dataset_label = dataset_cfg.get("eval_dataset_label")
    else:
        raise ValueError(f"Unsupported split name: {split_name}")

    if dataset_label:
        params["dataset_label"] = str(dataset_label)
    dataset = dataset_loader(dataset_names, **params)
    dataset = sorted(dataset, key=problem_id_from_item)
    return dataset


def _derive_failure_type(success: bool, reward_metadata: dict[str, Any], request_error: str | None) -> str:
    if request_error:
        return "request_error"
    if bool(reward_metadata.get("format_error")):
        return "format"
    if success:
        return "none"
    return "semantic"


async def _run_route(
    session: aiohttp.ClientSession,
    route: RouteSpec,
    repair_messages: list[dict[str, str]],
    file_contents: dict[str, str],
    oracle_patch: str,
    success_threshold: float,
) -> dict[str, Any]:
    start = time.time()
    request_error: str | None = None
    usage: dict[str, Any] = {}
    repair_text = ""
    try:
        repair_text, usage, latency = await chat_completion(
            session=session,
            base_url=route.base_url,
            model_name=route.model_name,
            messages=repair_messages,
            parameters=route.parameters,
            api_key=route.api_key,
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

    success = bool(reward and reward > success_threshold)
    return {
        "label": route.label,
        "model_name": route.model_name,
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
    route_specs: list[RouteSpec],
    prompt_tokenizer: Any,
    session: aiohttp.ClientSession,
    split_name: str,
    success_threshold: float,
) -> dict[str, Any]:
    problem_id = problem_id_from_item(problem)
    file_contents = problem.get("file_contents") or {}
    if not isinstance(file_contents, dict) or not file_contents:
        raise ValueError(f"Problem {problem_id} missing file_contents")
    problem_statement = problem.get("problem_statement")
    if not isinstance(problem_statement, str) or not problem_statement.strip():
        raise ValueError(f"Problem {problem_id} missing problem_statement")

    repair_messages, file_context = build_repair_messages(problem_statement, file_contents)
    prompt_text = render_prompt_text(prompt_tokenizer, repair_messages)
    route_results = await asyncio.gather(
        *[
            _run_route(
                session=session,
                route=route,
                repair_messages=repair_messages,
                file_contents=file_contents,
                oracle_patch=str(problem.get("patch") or ""),
                success_threshold=success_threshold,
            )
            for route in route_specs
        ]
    )

    primary_output_text = route_results[0]["output_text"]
    return {
        "problem_id": problem_id,
        "dataset": str(problem.get("dataset") or ""),
        "split": split_name,
        "repo": str(problem.get("repo") or ""),
        "base_commit": str(problem.get("base_commit") or ""),
        "language": infer_language_from_problem(problem),
        "problem_statement": problem_statement,
        "file_context": file_context,
        "file_paths": sorted(str(path) for path in file_contents),
        "prompt_text": prompt_text,
        "primary_output_text": primary_output_text,
        "performance_targets": [result["reward"] for result in route_results],
        "route_rewards": [result["reward"] for result in route_results],
        "route_successes": [bool(result["success"]) for result in route_results],
        "route_prompt_tokens": [int(result["prompt_tokens"]) for result in route_results],
        "route_output_tokens": [int(result["output_tokens"]) for result in route_results],
        "route_latencies_s": [float(result["latency_s"]) for result in route_results],
        "route_outputs": [str(result["output_text"]) for result in route_results],
        "route_failure_types": [str(result["failure_type"]) for result in route_results],
    }


async def _collect_split(
    cfg: DictConfig,
    split_name: str,
    split_dir: Path,
    route_specs: list[RouteSpec],
    prompt_tokenizer: Any,
) -> dict[str, Any]:
    dataset = _load_split_dataset(cfg, split_name)
    existing_ids, next_shard_index = _load_existing_problem_ids(split_dir)
    pending: list[tuple[int, dict[str, Any]]] = []
    skipped_existing = 0
    for problem in dataset:
        problem_id = problem_id_from_item(problem)
        if problem_id in existing_ids:
            skipped_existing += 1
            continue
        pending.append((len(pending), problem))

    collection_cfg = cfg.offline_router.collection
    success_threshold = float(cfg.offline_router.collection.get("success_threshold", 0.8))
    connector = aiohttp.TCPConnector(limit=int(collection_cfg.get("connector_limit", 128)))
    timeout = aiohttp.ClientTimeout(total=float(collection_cfg.get("request_timeout", 1800)))
    semaphore = asyncio.Semaphore(int(collection_cfg.get("max_concurrent_problems", 32)))
    max_in_flight = max(1, int(collection_cfg.get("max_concurrent_problems", 32)) * 4)
    progress_log_every = max(1, int(collection_cfg.get("progress_log_every", 25)))

    written = 0
    failed = 0
    buffered_rows: list[dict[str, Any]] = []
    ready_rows: dict[int, dict[str, Any] | None] = {}
    next_ready_idx = 0
    shard_row_counts: list[int] = []

    logger.info(
        "Starting %s collection: source_rows=%d skipped_existing=%d pending=%d shard_size=%d max_in_flight=%d",
        split_name,
        len(dataset),
        skipped_existing,
        len(pending),
        int(collection_cfg.get("shard_size", 64)),
        max_in_flight,
    )

    async def _task_wrapper(order_idx: int, problem: dict[str, Any], session: aiohttp.ClientSession):
        async with semaphore:
            try:
                row = await _collect_problem(
                    problem=problem,
                    route_specs=route_specs,
                    prompt_tokenizer=prompt_tokenizer,
                    session=session,
                    split_name=split_name,
                    success_threshold=success_threshold,
                )
                return order_idx, row
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Failed to collect %s split problem %s: %s", split_name, problem_id_from_item(problem), exc)
                return order_idx, None

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        progress = tqdm(total=len(pending), desc=f"Collect {split_name}", unit="problem")
        pending_iter = iter(pending)
        in_flight: set[asyncio.Task] = set()
        try:
            while True:
                while len(in_flight) < max_in_flight:
                    try:
                        order_idx, problem = next(pending_iter)
                    except StopIteration:
                        break
                    in_flight.add(asyncio.create_task(_task_wrapper(order_idx, problem, session)))
                if not in_flight:
                    break

                done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                for future in done:
                    order_idx, row = await future
                    progress.update(1)
                    ready_rows[order_idx] = row
                    while next_ready_idx in ready_rows:
                        ready = ready_rows.pop(next_ready_idx)
                        next_ready_idx += 1
                        if ready is None:
                            failed += 1
                            continue
                        buffered_rows.append(ready)
                        written += 1
                        if len(buffered_rows) >= int(collection_cfg.get("shard_size", 64)):
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
                        elif (written + failed) % progress_log_every == 0:
                            logger.info(
                                "%s progress: processed=%d/%d written=%d failed=%d buffered=%d shards=%d in_flight=%d",
                                split_name,
                                written + failed,
                                len(pending),
                                written,
                                failed,
                                len(buffered_rows),
                                len(shard_row_counts),
                                len(in_flight),
                            )
        finally:
            progress.close()

    if buffered_rows:
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

    logger.info(
        "Finished %s collection: source_rows=%d skipped_existing=%d pending=%d written=%d failed=%d shards=%d",
        split_name,
        len(dataset),
        skipped_existing,
        len(pending),
        written,
        failed,
        len(shard_row_counts),
    )

    return {
        "split": split_name,
        "n_source_rows": len(dataset),
        "n_skipped_existing": skipped_existing,
        "n_pending": len(pending),
        "n_written": written,
        "n_failed": failed,
        "shard_row_counts": shard_row_counts,
    }


def run_collection(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_specs = [_resolve_primary_route(cfg)] + _resolve_expert_routes(cfg)
    route_labels = [route.label for route in route_specs]
    route_model_names = [route.model_name for route in route_specs]
    prompt_tokenizer_name = str(
        cfg.offline_router.primary_model.get("tokenizer_name")
        or cfg.offline_router.primary_model.get("model_path")
    )
    prompt_tokenizer = AutoTokenizer.from_pretrained(prompt_tokenizer_name, use_fast=True)

    write_json(output_dir / "collection_config.json", sanitize_for_json(OmegaConf.to_container(cfg, resolve=True)))

    split_summaries: list[dict[str, Any]] = []
    started_at = time.time()
    for split_name in ("train", "eval"):
        if not bool(cfg.offline_router.collection.get(f"collect_{split_name}", True)):
            continue
        split_dir = output_dir / split_name
        summary = asyncio.run(
            _collect_split(
                cfg=cfg,
                split_name=split_name,
                split_dir=split_dir,
                route_specs=route_specs,
                prompt_tokenizer=prompt_tokenizer,
            )
        )
        split_summaries.append(summary)

    metadata = {
        "schema_version": 1,
        "collection_started_at_unix": started_at,
        "collection_completed_at_unix": time.time(),
        "route_labels": route_labels,
        "route_model_names": route_model_names,
        "prompt_builder": "pipelinerl.swe.scripts.repair_eval_utils.build_repair_messages",
        "reward_function": "pipelinerl.swe.utils.repair_utils.calculate_precise_reward",
        "primary_model_prompt_tokenizer_name": prompt_tokenizer_name,
        "split_summaries": split_summaries,
    }
    write_json(output_dir / "metadata.json", metadata)
    logger.info("Offline router collection complete: output_dir=%s", output_dir)


@hydra.main(config_path="../../../../conf", config_name="offline_router_collect", version_base=None)
def main(cfg: DictConfig) -> None:
    run_collection(cfg)


if __name__ == "__main__":
    main()

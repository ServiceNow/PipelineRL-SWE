#!/usr/bin/env python
"""Materialize an offline-router dataset from aligned trace outputs.

This is for targets that do not require real pass/fail labels, especially output
cost prediction. It aligns per-model outputs.jsonl files by instance id and
writes the parquet schema consumed by train_qwen_embedding_router_baseline.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer

from pipelinerl.swe.load_datasets import load_local_swe_dataset
from pipelinerl.swe.scripts.offline_router.common import (
    infer_language_from_problem,
    problem_id_from_item,
    render_prompt_text,
)
from pipelinerl.swe.scripts.repair_eval_utils import build_repair_messages


DEFAULT_OUTPUT_DIR = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_trace_cost_4route_1780461385/collect"
)
DEFAULT_TRAIN_DATASET_PATH = "/mnt/llmd/data/swe_smith_bugged_context_strict/ds_train"
DEFAULT_EVAL_DATASET_PATH = "/mnt/llmd/data/swe_smith_bugged_context_strict/ds_test"
DEFAULT_TRAIN_DATASET_NAME = "swe_smith_train_bugged_context"
DEFAULT_EVAL_DATASET_NAME = "swe_smith_test_bugged_context"
DEFAULT_PROMPT_TOKENIZER_NAME = "Qwen/Qwen3-4B-Instruct-2507"

DEFAULT_TRAIN_ROUTES = [
    (
        "qwen3_4b_instruct_2507",
        "scout:Qwen/Qwen3-4B-Instruct-2507",
        "/mnt/llmd/results/exps/aristides/reason/"
        "offline_router_swe_smith_real_train1500_collect_qwen3_4b_instruct_2507_1780461385/"
        "collect/models/Qwen_Qwen3-4B-Instruct-2507/outputs.jsonl",
    ),
    (
        "gpt_oss_20b",
        "solver:openai/gpt-oss-20b",
        "/mnt/llmd/results/exps/aristides/reason/"
        "offline_router_swe_smith_real_train1500_collect_gpt_oss_20b_1780461385/"
        "collect/models/openai_gpt-oss-20b/outputs.jsonl",
    ),
    (
        "qwen3_coder_30b_a3b",
        "solver:Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "/mnt/llmd/results/exps/aristides/reason/"
        "offline_router_swe_smith_real_train1500_collect_qwen3_coder_30b_a3b_1780461385/"
        "collect/models/Qwen_Qwen3-Coder-30B-A3B-Instruct/outputs.jsonl",
    ),
    (
        "gpt_oss_120b",
        "solver:openai/gpt-oss-120b",
        "/mnt/llmd/results/exps/aristides/reason/"
        "offline_router_swe_smith_real_train1500_collect_gpt_oss_120b_1780461385/"
        "collect/models/openai_gpt-oss-120b/outputs.jsonl",
    ),
]
DEFAULT_EVAL_ROUTES = [
    (
        "qwen3_4b_instruct_2507",
        "scout:Qwen/Qwen3-4B-Instruct-2507",
        "/mnt/llmd/results/exps/aristides/reason/"
        "offline_router_swe_smith_real_eval500_collect_qwen3_4b_instruct_2507_1780461385/"
        "collect/models/Qwen_Qwen3-4B-Instruct-2507/outputs.jsonl",
    ),
    (
        "gpt_oss_20b",
        "solver:openai/gpt-oss-20b",
        "/mnt/llmd/results/exps/aristides/reason/"
        "offline_router_swe_smith_real_eval500_collect_gpt_oss_20b_1780461385/"
        "collect/models/openai_gpt-oss-20b/outputs.jsonl",
    ),
    (
        "qwen3_coder_30b_a3b",
        "solver:Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "/mnt/llmd/results/exps/aristides/reason/"
        "offline_router_swe_smith_real_eval500_collect_qwen3_coder_30b_a3b_1780461385/"
        "collect/models/Qwen_Qwen3-Coder-30B-A3B-Instruct/outputs.jsonl",
    ),
    (
        "gpt_oss_120b",
        "solver:openai/gpt-oss-120b",
        "/mnt/llmd/results/exps/aristides/reason/"
        "offline_router_swe_smith_real_eval500_collect_gpt_oss_120b_1780461385/"
        "collect/models/openai_gpt-oss-120b/outputs.jsonl",
    ),
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_route(raw: str) -> tuple[str, str, Path]:
    parts = raw.split("=", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid route {raw!r}; expected run_id=route_label=/path/to/outputs.jsonl")
    run_id, label, outputs_path = parts
    if not run_id.strip() or not label.strip() or not outputs_path.strip():
        raise ValueError(f"Invalid route {raw!r}; empty component")
    return run_id.strip(), label.strip(), Path(outputs_path.strip())


def _load_outputs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instance_id = str(row.get("instance_id") or row.get("problem_id") or row.get("id") or "").strip()
            if instance_id:
                rows[instance_id] = row
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _load_reference_split_ids(dataset_dir: Path, split: str) -> set[str]:
    split_dir = dataset_dir / split
    files = sorted(split_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards found for reference split={split} in {split_dir}")
    ids: set[str] = set()
    for file_path in files:
        table = pq.read_table(file_path)
        for row in table.to_pylist():
            ids.add(problem_id_from_item(row))
    if not ids:
        raise ValueError(f"No problem ids found for reference split={split} in {split_dir}")
    return ids


def _load_dataset_by_id(dataset_path: Path, dataset_name: str, seed: int) -> dict[str, dict[str, Any]]:
    rows = load_local_swe_dataset(
        dataset_names=[dataset_name] if dataset_name else [],
        dataset_path=str(dataset_path),
        shuffle=False,
        seed=int(seed),
        dataset_label=dataset_name or None,
        max_samples=None,
    )
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        problem_id = problem_id_from_item(row)
        if problem_id not in by_id:
            by_id[problem_id] = row
    if not by_id:
        raise ValueError(f"No dataset rows loaded from {dataset_path}")
    return by_id


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _row_values(output_row: dict[str, Any]) -> dict[str, Any]:
    proxy_reward = _safe_float(output_row.get("proxy_reward"), 0.0)
    proxy_success_raw = output_row.get("proxy_success")
    proxy_success = bool(proxy_success_raw) if proxy_success_raw is not None else proxy_reward > 0.0
    return {
        "performance_targets": proxy_reward,
        "route_rewards": proxy_reward,
        "route_successes": proxy_success,
        "route_prompt_tokens": _safe_int(output_row.get("prompt_tokens"), 0),
        "route_output_tokens": _safe_int(output_row.get("output_tokens"), 0),
        "route_latencies_s": _safe_float(output_row.get("latency_s"), 0.0),
        "route_outputs": str(output_row.get("output_text") or ""),
        "route_model_patches": str(output_row.get("model_patch") or ""),
        "route_failure_types": str(output_row.get("failure_type") or "unknown"),
    }


def _make_router_row(
    *,
    problem: dict[str, Any],
    problem_id: str,
    outputs_by_route: list[dict[str, dict[str, Any]]],
    route_labels: list[str],
    tokenizer: Any,
    split: str,
    dataset_name: str,
) -> dict[str, Any]:
    file_contents = problem.get("file_contents") or {}
    problem_statement = problem.get("problem_statement")
    if not isinstance(file_contents, dict) or not file_contents:
        raise ValueError(f"{problem_id} missing file_contents")
    if not isinstance(problem_statement, str) or not problem_statement.strip():
        raise ValueError(f"{problem_id} missing problem_statement")
    repair_messages, _stage_input = build_repair_messages(problem_statement, file_contents)
    prompt_text = render_prompt_text(tokenizer, repair_messages)

    vector_values: dict[str, list[Any]] = {
        "performance_targets": [],
        "route_rewards": [],
        "route_successes": [],
        "route_prompt_tokens": [],
        "route_output_tokens": [],
        "route_latencies_s": [],
        "route_outputs": [],
        "route_model_patches": [],
        "route_failure_types": [],
    }
    for output_map in outputs_by_route:
        values = _row_values(output_map[problem_id])
        for key, value in values.items():
            vector_values[key].append(value)

    return {
        "problem_id": problem_id,
        "dataset": str(problem.get("dataset") or dataset_name),
        "split": split,
        "repo": str(problem.get("repo") or outputs_by_route[0][problem_id].get("repo") or ""),
        "base_commit": str(problem.get("base_commit") or outputs_by_route[0][problem_id].get("base_commit") or ""),
        "language": infer_language_from_problem(problem),
        "problem_statement": problem_statement,
        "prompt_text": prompt_text,
        "primary_output_text": str(vector_values["route_outputs"][0] or ""),
        "route_labels": list(route_labels),
        **vector_values,
    }


def _write_shards(rows: list[dict[str, Any]], split_dir: Path, split: str, shard_size: int) -> list[int]:
    split_dir.mkdir(parents=True, exist_ok=True)
    counts: list[int] = []
    for shard_idx, start in enumerate(range(0, len(rows), int(shard_size))):
        shard_rows = rows[start : start + int(shard_size)]
        pq.write_table(pa.Table.from_pylist(shard_rows), split_dir / f"{split}-{shard_idx:05d}.parquet")
        counts.append(len(shard_rows))
    return counts


def _materialize_split(
    *,
    split: str,
    dataset_path: Path,
    dataset_name: str,
    route_specs: list[tuple[str, str, Path]],
    tokenizer: Any,
    seed: int,
    max_rows: int,
    include_ids: set[str] | None = None,
    exclude_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route_labels = [label for _run_id, label, _path in route_specs]
    outputs_by_route = [_load_outputs(path) for _run_id, _label, path in route_specs]
    dataset_by_id = _load_dataset_by_id(dataset_path, dataset_name, seed)

    aligned_ids = set(dataset_by_id)
    for output_map in outputs_by_route:
        aligned_ids &= set(output_map)
    n_aligned_before_filters = len(aligned_ids)
    missing_include_ids: set[str] = set()
    excluded_present_ids: set[str] = set()
    if include_ids is not None:
        missing_include_ids = set(include_ids) - aligned_ids
        aligned_ids &= set(include_ids)
    if exclude_ids is not None:
        excluded_present_ids = aligned_ids & set(exclude_ids)
        aligned_ids -= set(exclude_ids)
    ordered_ids = sorted(aligned_ids)
    if int(max_rows) > 0:
        ordered_ids = ordered_ids[: int(max_rows)]

    rows: list[dict[str, Any]] = []
    skipped = 0
    for problem_id in tqdm(ordered_ids, desc=f"materialize trace-cost {split}"):
        try:
            rows.append(
                _make_router_row(
                    problem=dataset_by_id[problem_id],
                    problem_id=problem_id,
                    outputs_by_route=outputs_by_route,
                    route_labels=route_labels,
                    tokenizer=tokenizer,
                    split=split,
                    dataset_name=dataset_name,
                )
            )
        except Exception:
            skipped += 1
    output_token_means: dict[str, float | None] = {}
    for route_idx, route_label in enumerate(route_labels):
        values = [float(row["route_output_tokens"][route_idx]) for row in rows]
        output_token_means[route_label] = (sum(values) / len(values)) if values else None
    summary = {
        "split": split,
        "dataset_path": str(dataset_path),
        "dataset_name": dataset_name,
        "route_run_ids": [run_id for run_id, _label, _path in route_specs],
        "route_labels": route_labels,
        "route_output_paths": [str(path) for _run_id, _label, path in route_specs],
        "n_dataset_rows": len(dataset_by_id),
        "n_aligned_before_filters": int(n_aligned_before_filters),
        "n_aligned_available": len(aligned_ids),
        "n_requested_include_ids": len(include_ids) if include_ids is not None else None,
        "n_missing_include_ids": int(len(missing_include_ids)),
        "n_excluded_ids_present": int(len(excluded_present_ids)),
        "n_written": len(rows),
        "n_skipped_make_row": int(skipped),
        "max_rows": int(max_rows),
        "mean_output_tokens_by_route": output_token_means,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-dataset-path", default=DEFAULT_TRAIN_DATASET_PATH)
    parser.add_argument("--eval-dataset-path", default=DEFAULT_EVAL_DATASET_PATH)
    parser.add_argument("--train-dataset-name", default=DEFAULT_TRAIN_DATASET_NAME)
    parser.add_argument("--eval-dataset-name", default=DEFAULT_EVAL_DATASET_NAME)
    parser.add_argument("--train-route", action="append", default=[])
    parser.add_argument("--eval-route", action="append", default=[])
    parser.add_argument("--prompt-tokenizer-name", default=DEFAULT_PROMPT_TOKENIZER_NAME)
    parser.add_argument(
        "--split-reference-dataset-dir",
        default="",
        help=(
            "Optional existing router dataset whose train/eval problem_id splits should be reused. "
            "When set, eval is materialized from the train trace routes using the reference eval ids."
        ),
    )
    parser.add_argument("--reference-train-split", default="train")
    parser.add_argument("--reference-eval-split", default="eval")
    parser.add_argument(
        "--reference-train-mode",
        choices=["all_non_eval", "reference_train"],
        default="all_non_eval",
        help="With a reference split, train on all aligned non-eval trace ids or exactly the reference train ids.",
    )
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    train_route_specs = [_parse_route(raw) for raw in args.train_route]
    eval_route_specs = [_parse_route(raw) for raw in args.eval_route]
    if not train_route_specs:
        train_route_specs = [(run_id, label, Path(path)) for run_id, label, path in DEFAULT_TRAIN_ROUTES]

    split_reference_dataset_dir = Path(args.split_reference_dataset_dir) if args.split_reference_dataset_dir else None
    reference_train_ids: set[str] | None = None
    reference_eval_ids: set[str] | None = None
    train_include_ids: set[str] | None = None
    train_exclude_ids: set[str] | None = None
    eval_include_ids: set[str] | None = None
    eval_exclude_ids: set[str] | None = None
    eval_dataset_path = Path(args.eval_dataset_path)
    eval_dataset_name = str(args.eval_dataset_name)
    if split_reference_dataset_dir is not None:
        reference_train_ids = _load_reference_split_ids(split_reference_dataset_dir, str(args.reference_train_split))
        reference_eval_ids = _load_reference_split_ids(split_reference_dataset_dir, str(args.reference_eval_split))
        if reference_train_ids & reference_eval_ids:
            raise ValueError("Reference train/eval ids overlap")
        train_exclude_ids = set(reference_eval_ids)
        if str(args.reference_train_mode) == "reference_train":
            train_include_ids = set(reference_train_ids)
        eval_include_ids = set(reference_eval_ids)
        eval_route_specs = list(train_route_specs)
        eval_dataset_path = Path(args.train_dataset_path)
        eval_dataset_name = str(args.train_dataset_name)
    elif not eval_route_specs:
        eval_route_specs = [(run_id, label, Path(path)) for run_id, label, path in DEFAULT_EVAL_ROUTES]

    train_labels = [label for _run_id, label, _path in train_route_specs]
    eval_labels = [label for _run_id, label, _path in eval_route_specs]
    if train_labels != eval_labels:
        raise ValueError(f"Train/eval route labels differ: train={train_labels}, eval={eval_labels}")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.prompt_tokenizer_name))
    train_rows, train_summary = _materialize_split(
        split="train",
        dataset_path=Path(args.train_dataset_path),
        dataset_name=str(args.train_dataset_name),
        route_specs=train_route_specs,
        tokenizer=tokenizer,
        seed=int(args.seed),
        max_rows=int(args.max_train_rows),
        include_ids=train_include_ids,
        exclude_ids=train_exclude_ids,
    )
    eval_rows, eval_summary = _materialize_split(
        split="eval",
        dataset_path=eval_dataset_path,
        dataset_name=eval_dataset_name,
        route_specs=eval_route_specs,
        tokenizer=tokenizer,
        seed=int(args.seed) + 1,
        max_rows=int(args.max_eval_rows),
        include_ids=eval_include_ids,
        exclude_ids=eval_exclude_ids,
    )
    if not train_rows or not eval_rows:
        raise ValueError(f"Materialized empty split: train={len(train_rows)} eval={len(eval_rows)}")

    train_counts = _write_shards(train_rows, output_dir / "train", "train", int(args.shard_size))
    eval_counts = _write_shards(eval_rows, output_dir / "eval", "eval", int(args.shard_size))

    metadata = {
        "dataset_kind": "swe_smith_trace_cost_from_model_outputs",
        "route_labels": train_labels,
        "route_run_ids": [run_id for run_id, _label, _path in train_route_specs],
        "prompt_tokenizer_name": str(args.prompt_tokenizer_name),
        "split_reference_dataset_dir": str(split_reference_dataset_dir) if split_reference_dataset_dir is not None else None,
        "reference_train_split": str(args.reference_train_split),
        "reference_eval_split": str(args.reference_eval_split),
        "reference_train_mode": str(args.reference_train_mode),
        "n_reference_train_ids": len(reference_train_ids) if reference_train_ids is not None else None,
        "n_reference_eval_ids": len(reference_eval_ids) if reference_eval_ids is not None else None,
        "seed": int(args.seed),
        "shard_size": int(args.shard_size),
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "train_shard_row_counts": train_counts,
        "eval_shard_row_counts": eval_counts,
        "split_summaries": [train_summary, eval_summary],
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "collection_config.json", metadata)
    _write_json(output_dir / "trace_cost_materialization_summary.json", metadata)
    print(json.dumps({"output_dir": str(output_dir), **metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

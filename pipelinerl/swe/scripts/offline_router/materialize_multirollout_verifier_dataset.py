#!/usr/bin/env python
"""Materialize multi-rollout attempts for verifier rescoring.

The cascade/verifier scorer expects router-style rows containing a repair prompt
plus one generated attempt per route. For multi-rollout analysis, we create one
row per `(instance, rollout)` and preserve the original instance id and rollout
id so predicted verifier scores can be joined back to real pass/fail reports.
"""

from __future__ import annotations

import argparse
import json
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


DEFAULT_TRACE_ROOT = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_multi_rollout_trace_collect_1781382734/eval150"
)
DEFAULT_DATASET_PATH = "/mnt/llmd/data/swe_smith_bugged_context/ds_test"
DEFAULT_DATASET_NAME = "swe_smith_test_bugged_context"
DEFAULT_OUTPUT_DIR = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_multirollout_eval150_verifier_rescore_dataset/collect"
)
DEFAULT_PROMPT_TOKENIZER_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ROUTE_SPECS = [
    (
        "qwen3_4b_instruct_2507",
        "scout:Qwen/Qwen3-4B-Instruct-2507",
        "Qwen_Qwen3-4B-Instruct-2507",
    ),
    (
        "gpt_oss_20b",
        "solver:openai/gpt-oss-20b",
        "openai_gpt-oss-20b",
    ),
    (
        "qwen3_coder_30b_a3b",
        "solver:Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen_Qwen3-Coder-30B-A3B-Instruct",
    ),
    (
        "gpt_oss_120b",
        "solver:openai/gpt-oss-120b",
        "openai_gpt-oss-120b",
    ),
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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


def _default_route_path(trace_root: Path, route_slug: str, rollout_idx: int, model_slug: str) -> Path:
    return (
        trace_root
        / route_slug
        / f"rollout_{int(rollout_idx)}"
        / "collect"
        / "models"
        / model_slug
        / "outputs.jsonl"
    )


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
        "route_model_patches": str(output_row.get("model_patch") or output_row.get("patch") or ""),
        "route_failure_types": str(output_row.get("failure_type") or "unknown"),
    }


def _make_row(
    *,
    problem: dict[str, Any],
    original_problem_id: str,
    rollout_idx: int,
    route_labels: list[str],
    output_maps: list[dict[str, dict[str, Any]]],
    tokenizer: Any,
    dataset_name: str,
) -> dict[str, Any]:
    file_contents = problem.get("file_contents") or {}
    problem_statement = problem.get("problem_statement")
    if not isinstance(file_contents, dict) or not file_contents:
        raise ValueError(f"{original_problem_id} missing file_contents")
    if not isinstance(problem_statement, str) or not problem_statement.strip():
        raise ValueError(f"{original_problem_id} missing problem_statement")
    repair_messages, file_context = build_repair_messages(problem_statement, file_contents)
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
    for output_map in output_maps:
        values = _row_values(output_map[original_problem_id])
        for key, value in values.items():
            vector_values[key].append(value)

    unique_problem_id = f"{original_problem_id}::rollout_{int(rollout_idx)}"
    return {
        "problem_id": unique_problem_id,
        "original_problem_id": original_problem_id,
        "rollout_idx": int(rollout_idx),
        "dataset": dataset_name,
        "split": "eval",
        "repo": str(problem.get("repo") or ""),
        "base_commit": str(problem.get("base_commit") or ""),
        "language": infer_language_from_problem(problem),
        "problem_statement": problem_statement,
        "file_context": file_context,
        "prompt_text": prompt_text,
        "primary_output_text": str(vector_values["route_outputs"][0]) if vector_values["route_outputs"] else "",
        "route_labels": list(route_labels),
        **vector_values,
    }


def _write_split(path: Path, rows: list[dict[str, Any]], shard_size: int) -> list[int]:
    path.mkdir(parents=True, exist_ok=True)
    counts: list[int] = []
    for start in range(0, len(rows), int(shard_size)):
        shard = rows[start : start + int(shard_size)]
        pq.write_table(pa.Table.from_pylist(shard), path / f"eval-{len(counts):05d}.parquet")
        counts.append(len(shard))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-tokenizer-name", default=DEFAULT_PROMPT_TOKENIZER_NAME)
    parser.add_argument("--rollouts", default="0,1,2")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shard-size", type=int, default=64)
    args = parser.parse_args()

    trace_root = Path(args.trace_root)
    output_dir = Path(args.output_dir)
    rollout_indices = [int(part.strip()) for part in str(args.rollouts).split(",") if part.strip()]
    route_run_ids = [spec[0] for spec in DEFAULT_ROUTE_SPECS]
    route_labels = [spec[1] for spec in DEFAULT_ROUTE_SPECS]
    tokenizer = AutoTokenizer.from_pretrained(args.prompt_tokenizer_name)
    problems_by_id = _load_dataset_by_id(Path(args.dataset_path), str(args.dataset_name), int(args.seed))

    rows: list[dict[str, Any]] = []
    split_summaries: list[dict[str, Any]] = []
    for rollout_idx in rollout_indices:
        output_maps: list[dict[str, dict[str, Any]]] = []
        route_output_paths: list[str] = []
        for route_slug, _route_label, model_slug in DEFAULT_ROUTE_SPECS:
            path = _default_route_path(trace_root, route_slug, int(rollout_idx), model_slug)
            route_output_paths.append(str(path))
            output_maps.append(_load_outputs(path))
        common_ids = set(output_maps[0])
        for output_map in output_maps[1:]:
            common_ids &= set(output_map)
        common_ids &= set(problems_by_id)
        written = 0
        for problem_id in tqdm(sorted(common_ids), desc=f"materialize rollout {rollout_idx}"):
            try:
                rows.append(
                    _make_row(
                        problem=problems_by_id[problem_id],
                        original_problem_id=problem_id,
                        rollout_idx=int(rollout_idx),
                        route_labels=route_labels,
                        output_maps=output_maps,
                        tokenizer=tokenizer,
                        dataset_name=str(args.dataset_name),
                    )
                )
                written += 1
            except Exception as exc:
                print(f"Skipping {problem_id} rollout={rollout_idx}: {exc}", flush=True)
        split_summaries.append(
            {
                "rollout_idx": int(rollout_idx),
                "n_aligned_available": len(common_ids),
                "n_written": int(written),
                "route_output_paths": route_output_paths,
            }
        )

    if not rows:
        raise ValueError("No rows materialized")
    if output_dir.exists():
        for old_file in output_dir.rglob("*.parquet"):
            old_file.unlink()
    shard_counts = _write_split(output_dir / "eval", rows, int(args.shard_size))
    metadata = {
        "dataset_kind": "swe_smith_multirollout_verifier_rescore",
        "n_eval": len(rows),
        "n_train": 0,
        "eval_shard_row_counts": shard_counts,
        "route_labels": route_labels,
        "route_run_ids": route_run_ids,
        "rollouts": rollout_indices,
        "trace_root": str(trace_root),
        "dataset_path": str(args.dataset_path),
        "dataset_name": str(args.dataset_name),
        "prompt_tokenizer_name": str(args.prompt_tokenizer_name),
        "split_summaries": split_summaries,
        "seed": int(args.seed),
        "shard_size": int(args.shard_size),
    }
    _write_json(output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

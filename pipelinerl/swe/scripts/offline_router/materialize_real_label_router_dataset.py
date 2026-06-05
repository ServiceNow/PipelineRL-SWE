#!/usr/bin/env python
"""Materialize an offline-router parquet dataset from real SWE-Smith pass/fail labels.

The input labels are SWE-Smith/SWE-bench-style root report.json files with
ids_resolved / ids_unresolved, usually packed in an AWS results tarball. The
input traces are per-model outputs.jsonl files from collect_model_discovery_candidates.
The output is the existing offline-router parquet schema consumed by
train_qwen_embedding_router_baseline.py.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tarfile
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


DEFAULT_RESULTS_TARBALL = (
    "router_analysis/aws_eval_packages/swe_smith_train1500_real_eval_results_1780639659.tar.gz"
)
DEFAULT_DATASET_PATH = "/mnt/llmd/data/swe_smith_bugged_context_strict/ds_train"
DEFAULT_DATASET_NAME = "swe_smith_train_bugged_context"
DEFAULT_OUTPUT_DIR = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect"
)
DEFAULT_ROUTES = [
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_route(raw: str) -> tuple[str, str, Path]:
    parts = raw.split("=", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid --route {raw!r}; expected run_id=route_label=/path/to/outputs.jsonl")
    run_id, label, outputs_path = parts
    if not run_id.strip() or not label.strip() or not outputs_path.strip():
        raise ValueError(f"Invalid --route {raw!r}; empty component")
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
            instance_id = str(row.get("instance_id") or "")
            if instance_id:
                rows[instance_id] = row
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _load_reports_from_tarball(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    reports: dict[str, dict[str, Any]] = {}
    with tarfile.open(path, "r:gz") as tf:
        for member in tf.getmembers():
            parts = member.name.split("/")
            if len(parts) >= 5 and parts[-1] == "report.json" and parts[-3] == "run_evaluation":
                run_name = parts[-2]
                fileobj = tf.extractfile(member)
                if fileobj is None:
                    continue
                reports[run_name] = json.load(fileobj)
    if not reports:
        raise ValueError(f"No root report.json files found in {path}")
    return reports


def _labels_for_route(reports: dict[str, dict[str, Any]], split_prefix: str, run_id: str) -> dict[str, int]:
    report_name = f"{split_prefix}_{run_id}"
    report = reports.get(report_name)
    if report is None:
        available = ", ".join(sorted(reports)[:20])
        raise KeyError(f"Missing report {report_name!r}; available starts with: {available}")
    labels: dict[str, int] = {}
    for instance_id in report.get("ids_unresolved", []) or []:
        labels[str(instance_id)] = 0
    for instance_id in report.get("ids_resolved", []) or []:
        labels[str(instance_id)] = 1
    return labels


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


def _row_values(output_row: dict[str, Any], label: int) -> dict[str, Any]:
    failure_type = str(output_row.get("failure_type") or "unknown")
    if int(label) == 1:
        success = True
    else:
        success = False
    return {
        "performance_targets": float(label),
        "route_rewards": float(label),
        "route_successes": bool(success),
        "route_prompt_tokens": int(output_row.get("prompt_tokens") or 0),
        "route_output_tokens": int(output_row.get("output_tokens") or 0),
        "route_latencies_s": float(output_row.get("latency_s") or 0.0),
        "route_outputs": str(output_row.get("output_text") or ""),
        "route_failure_types": failure_type,
    }


def _make_router_row(
    *,
    problem: dict[str, Any],
    problem_id: str,
    labels_by_route: list[dict[str, int]],
    outputs_by_route: list[dict[str, dict[str, Any]]],
    route_labels: list[str],
    tokenizer: Any,
    split: str,
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
        "route_failure_types": [],
    }
    for label_map, output_map in zip(labels_by_route, outputs_by_route, strict=True):
        values = _row_values(output_map[problem_id], label_map[problem_id])
        for key, value in values.items():
            vector_values[key].append(value)

    return {
        "problem_id": problem_id,
        "dataset": str(problem.get("dataset") or DEFAULT_DATASET_NAME),
        "split": split,
        "repo": str(problem.get("repo") or ""),
        "base_commit": str(problem.get("base_commit") or ""),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-tarball", default=DEFAULT_RESULTS_TARBALL)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-prefix", default="swe_smith_train1500")
    parser.add_argument("--route", action="append", default=[])
    parser.add_argument("--prompt-tokenizer-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--eval-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    route_specs = [_parse_route(raw) for raw in args.route]
    if not route_specs:
        route_specs = [(run_id, label, Path(outputs_path)) for run_id, label, outputs_path in DEFAULT_ROUTES]
    route_labels = [label for _run_id, label, _outputs in route_specs]

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = _load_reports_from_tarball(Path(args.results_tarball))
    labels_by_route = [_labels_for_route(reports, str(args.split_prefix), run_id) for run_id, _label, _path in route_specs]
    outputs_by_route = [_load_outputs(path) for _run_id, _label, path in route_specs]
    dataset_by_id = _load_dataset_by_id(Path(args.dataset_path), str(args.dataset_name), int(args.seed))

    aligned_ids = set(dataset_by_id)
    for label_map, output_map in zip(labels_by_route, outputs_by_route, strict=True):
        aligned_ids &= set(label_map)
        aligned_ids &= set(output_map)
    ordered_ids = sorted(aligned_ids)
    rng = random.Random(int(args.seed))
    rng.shuffle(ordered_ids)
    if int(args.eval_size) > 0:
        eval_size = min(int(args.eval_size), len(ordered_ids) - 1)
    else:
        eval_size = int(round(len(ordered_ids) * float(args.eval_fraction)))
        eval_size = min(max(1, eval_size), len(ordered_ids) - 1)
    eval_ids = set(ordered_ids[:eval_size])
    train_ids = [problem_id for problem_id in ordered_ids if problem_id not in eval_ids]
    eval_id_list = [problem_id for problem_id in ordered_ids if problem_id in eval_ids]

    tokenizer = AutoTokenizer.from_pretrained(str(args.prompt_tokenizer_name))
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    for split, ids in (("train", train_ids), ("eval", eval_id_list)):
        for problem_id in tqdm(ids, desc=f"materialize {split}"):
            split_rows[split].append(
                _make_router_row(
                    problem=dataset_by_id[problem_id],
                    problem_id=problem_id,
                    labels_by_route=labels_by_route,
                    outputs_by_route=outputs_by_route,
                    route_labels=route_labels,
                    tokenizer=tokenizer,
                    split=split,
                )
            )

    split_summaries = []
    for split in ("train", "eval"):
        counts = _write_shards(split_rows[split], output_dir / split, split, int(args.shard_size))
        split_summaries.append(
            {
                "split": split,
                "n_written": len(split_rows[split]),
                "shard_row_counts": counts,
            }
        )

    route_pass_counts = [sum(labels[problem_id] for problem_id in aligned_ids) for labels in labels_by_route]
    metadata = {
        "dataset_kind": "swe_smith_real_labels_from_train_split",
        "source_results_tarball": str(args.results_tarball),
        "source_dataset_path": str(args.dataset_path),
        "source_split_prefix": str(args.split_prefix),
        "route_labels": route_labels,
        "route_run_ids": [run_id for run_id, _label, _path in route_specs],
        "route_output_paths": [str(path) for _run_id, _label, path in route_specs],
        "n_aligned": len(aligned_ids),
        "n_train": len(split_rows["train"]),
        "n_eval": len(split_rows["eval"]),
        "eval_fraction": float(args.eval_fraction),
        "eval_size_requested": int(args.eval_size),
        "seed": int(args.seed),
        "route_pass_counts_on_aligned": {
            route_labels[idx]: int(route_pass_counts[idx]) for idx in range(len(route_labels))
        },
        "route_pass_rates_on_aligned": {
            route_labels[idx]: (float(route_pass_counts[idx]) / len(aligned_ids) if aligned_ids else None)
            for idx in range(len(route_labels))
        },
        "split_summaries": split_summaries,
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "collection_config.json", metadata)
    _write_json(output_dir / "real_label_materialization_summary.json", metadata)
    print(json.dumps({"output_dir": str(output_dir), **metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

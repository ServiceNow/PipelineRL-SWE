#!/usr/bin/env python
"""Materialize random or active real-label subsets for verifier finetuning."""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset



DEFAULT_SOURCE_DATASET_DIR = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect"
)
DEFAULT_PROXY_PREDICTIONS = [
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_trace_expanded_4route_qwen3_embedding_8b_lora_proxy_verifier_soft_bce_r32_qkvo_mlp_5epoch_1781684950/"
    "train_qwen3_embedding_8b_lora_proxy_verifier_soft_bce_r32_qkvo_mlp_5epoch/train_attempt_predictions.jsonl",
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_trace_expanded_4route_qwen3_embedding_8b_lora_proxy_verifier_soft_bce_r32_qkvo_mlp_5epoch_1781684950/"
    "train_qwen3_embedding_8b_lora_proxy_verifier_soft_bce_r32_qkvo_mlp_5epoch/eval_attempt_predictions.jsonl",
]


def _problem_id_from_row(row: dict[str, Any]) -> str:
    for key in ("problem_id", "instance_id", "id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Row is missing a problem id field: keys={sorted(row.keys())}")


def _load_split(dataset_dir: Path, split_name: str) -> list[dict[str, Any]]:
    files = sorted(glob.glob(str(dataset_dir / split_name / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet shards found for split={split_name} in {dataset_dir / split_name}")
    return list(load_dataset("parquet", data_files={split_name: files})[split_name])


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_split(rows: list[dict[str, Any]], output_dir: Path, split_name: str, shard_size: int) -> list[int]:
    split_dir = output_dir / split_name
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    counts: list[int] = []
    for start in range(0, len(rows), int(shard_size)):
        shard = rows[start : start + int(shard_size)]
        table = pa.Table.from_pylist(shard)
        shard_path = split_dir / f"{split_name}-{len(counts):05d}.parquet"
        pq.write_table(table, shard_path)
        counts.append(len(shard))
    if not counts:
        table = pa.Table.from_pylist([])
        pq.write_table(table, split_dir / f"{split_name}-00000.parquet")
        counts.append(0)
    return counts


def _load_proxy_scores(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pid = str(row.get("problem_id") or row.get("original_problem_id") or "")
                if not pid:
                    continue
                by_problem[pid].append(row)
    return by_problem


def _score_row(row: dict[str, Any], proxy_scores: dict[str, list[dict[str, Any]]], strategy: str) -> float:
    pid = _problem_id_from_row(row)
    preds = [float(item.get("pred_score") or 0.0) for item in proxy_scores.get(pid, [])]
    if not preds:
        return math.inf
    arr = np.asarray(preds, dtype=float)
    if strategy == "uncertainty":
        return float(np.min(np.abs(arr - 0.5)))
    if strategy == "mean_uncertainty":
        return float(np.mean(np.abs(arr - 0.5)))
    if strategy == "top2_margin":
        ranked = np.sort(arr)[::-1]
        if len(ranked) < 2:
            return 0.0
        return float(ranked[0] - ranked[1])
    if strategy == "high_variance":
        return -float(np.std(arr))
    if strategy == "high_score":
        return -float(np.max(arr))
    if strategy == "random":
        return 0.0
    raise ValueError(f"Unknown strategy={strategy!r}")


def _select_rows(
    train_rows: list[dict[str, Any]],
    proxy_scores: dict[str, list[dict[str, Any]]],
    strategy: str,
    budget: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(int(seed))
    rows = list(train_rows)
    budget = min(int(budget), len(rows))
    if strategy == "random":
        selected = rng.sample(rows, budget)
    else:
        scored = [(_score_row(row, proxy_scores, strategy), rng.random(), row) for row in rows]
        finite = [item for item in scored if math.isfinite(float(item[0]))]
        missing = [item for item in scored if not math.isfinite(float(item[0]))]
        ranked = sorted(finite, key=lambda item: (float(item[0]), float(item[1])))
        selected = [item[2] for item in ranked[:budget]]
        if len(selected) < budget:
            selected.extend([item[2] for item in missing[: budget - len(selected)]])
    selected_ids = {_problem_id_from_row(row) for row in selected}
    selected_scores = [float(_score_row(row, proxy_scores, strategy)) for row in selected if strategy != "random"]
    return selected, {
        "strategy": strategy,
        "budget_requested": int(budget),
        "n_selected": len(selected),
        "n_source_train": len(train_rows),
        "n_with_proxy_scores": sum(1 for row in train_rows if _problem_id_from_row(row) in proxy_scores),
        "selected_ids": sorted(selected_ids),
        "selected_score_mean": float(np.mean(selected_scores)) if selected_scores else None,
        "selected_score_min": float(np.min(selected_scores)) if selected_scores else None,
        "selected_score_max": float(np.max(selected_scores)) if selected_scores else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset-dir", default=DEFAULT_SOURCE_DATASET_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strategy", choices=["random", "uncertainty", "mean_uncertainty", "top2_margin", "high_variance", "high_score"], required=True)
    parser.add_argument("--budget-instances", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--proxy-predictions", action="append", default=[])
    parser.add_argument("--shard-size", type=int, default=64)
    args = parser.parse_args()

    source_dir = Path(args.source_dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _load_split(source_dir, "train")
    eval_rows = _load_split(source_dir, "eval")
    proxy_paths = [Path(path) for path in (args.proxy_predictions or DEFAULT_PROXY_PREDICTIONS)]
    proxy_scores = _load_proxy_scores(proxy_paths)
    selected_train_rows, selection_summary = _select_rows(
        train_rows,
        proxy_scores,
        str(args.strategy),
        int(args.budget_instances),
        int(args.seed),
    )

    train_counts = _write_split(selected_train_rows, output_dir, "train", int(args.shard_size))
    eval_counts = _write_split(eval_rows, output_dir, "eval", int(args.shard_size))
    metadata = {}
    source_metadata = source_dir / "metadata.json"
    if source_metadata.exists():
        metadata = json.loads(source_metadata.read_text())
    metadata.update(
        {
            "dataset_kind": "active_real_label_verifier_subset",
            "source_dataset_dir": str(source_dir),
            "source_n_train": len(train_rows),
            "source_n_eval": len(eval_rows),
            "n_train": len(selected_train_rows),
            "n_eval": len(eval_rows),
            "active_selection": selection_summary,
            "proxy_prediction_paths": [str(path) for path in proxy_paths],
            "split_summaries": [
                {"split": "train", "n_written": len(selected_train_rows), "shard_row_counts": train_counts},
                {"split": "eval", "n_written": len(eval_rows), "shard_row_counts": eval_counts},
            ],
        }
    )
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "selection_summary.json", selection_summary)
    print(json.dumps({"output_dir": str(output_dir), **selection_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

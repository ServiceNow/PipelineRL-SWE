#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

from pipelinerl.swe.scripts.offline_router.common import problem_id_from_item, write_json


def _load_split(dataset_dir: Path, split_name: str):
    files = sorted((dataset_dir / split_name).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards found for split={split_name} in {dataset_dir / split_name}")
    return load_dataset("parquet", data_files={split_name: [str(path) for path in files]})[split_name]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _prediction_problem_key(dataset: Any, problem_id: Any) -> str:
    return f"{dataset}::{problem_id}"


def _parse_lambdas(raw_values: list[str] | None) -> list[float]:
    if not raw_values:
        return [0.0, 1.0e-5, 5.0e-5, 1.0e-4]
    values: list[float] = []
    for raw in raw_values:
        for piece in str(raw).split(","):
            piece = piece.strip()
            if piece:
                values.append(float(piece))
    return values or [0.0, 1.0e-5, 5.0e-5, 1.0e-4]


def _build_examples(dataset_dir: Path, predictions_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    predictions = _read_jsonl(predictions_path)
    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    route_labels = list(metadata.get("route_labels") or [])
    if len(route_labels) != 2:
        raise ValueError(
            f"Delta bucket report currently requires exactly 2 routes; found {len(route_labels)} in {dataset_dir / 'metadata.json'}"
        )
    eval_dataset = _load_split(dataset_dir, "eval")
    eval_lookup: dict[str, dict[str, Any]] = {}
    for row in eval_dataset:
        key = _prediction_problem_key(row.get("dataset"), problem_id_from_item(row))
        if key not in eval_lookup:
            eval_lookup[key] = row

    examples: list[dict[str, Any]] = []
    for row in predictions:
        pred_rewards = row.get("pred_rewards")
        if not isinstance(pred_rewards, list) or len(pred_rewards) != 2:
            continue
        key = _prediction_problem_key(row.get("dataset"), row.get("problem_id"))
        source_row = eval_lookup.get(key)
        if source_row is None:
            continue
        true_rewards = source_row.get("performance_targets")
        prompt_tokens = source_row.get("route_prompt_tokens")
        output_tokens = source_row.get("route_output_tokens")
        if (
            not isinstance(true_rewards, list)
            or not isinstance(prompt_tokens, list)
            or not isinstance(output_tokens, list)
            or len(true_rewards) != 2
            or len(prompt_tokens) != 2
            or len(output_tokens) != 2
        ):
            continue
        try:
            pred_rewards = [float(value) for value in pred_rewards]
            true_rewards = [float(value) for value in true_rewards]
            prompt_tokens = [float(value) for value in prompt_tokens]
            output_tokens = [float(value) for value in output_tokens]
        except (TypeError, ValueError):
            continue
        examples.append(
            {
                "problem_id": row.get("problem_id"),
                "dataset": row.get("dataset"),
                "pred_primary": pred_rewards[0],
                "pred_expert": pred_rewards[1],
                "pred_delta": pred_rewards[0] - pred_rewards[1],
                "true_primary": true_rewards[0],
                "true_expert": true_rewards[1],
                "true_delta": true_rewards[0] - true_rewards[1],
                "primary_wins": true_rewards[0] > true_rewards[1],
                "expert_wins": true_rewards[1] > true_rewards[0],
                "output_primary": output_tokens[0],
                "output_expert": output_tokens[1],
                "total_primary": prompt_tokens[0] + output_tokens[0],
                "total_expert": prompt_tokens[1] + output_tokens[1],
            }
        )
    if not examples:
        raise ValueError("No usable eval examples found")
    return route_labels, examples


def _quantile_bucket_ranges(sorted_examples: list[dict[str, Any]], bucket_count: int) -> list[tuple[int, int]]:
    n = len(sorted_examples)
    ranges: list[tuple[int, int]] = []
    for bucket_idx in range(bucket_count):
        start = int(np.floor(bucket_idx * n / bucket_count))
        end = int(np.floor((bucket_idx + 1) * n / bucket_count))
        if start >= end:
            continue
        ranges.append((start, end))
    return ranges


def _bucket_summary(
    bucket_examples: list[dict[str, Any]],
    bucket_idx: int,
    bucket_count: int,
    lambdas: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n = len(bucket_examples)
    pred_deltas = np.asarray([row["pred_delta"] for row in bucket_examples], dtype=np.float64)
    true_deltas = np.asarray([row["true_delta"] for row in bucket_examples], dtype=np.float64)
    primary_rewards = np.asarray([row["true_primary"] for row in bucket_examples], dtype=np.float64)
    expert_rewards = np.asarray([row["true_expert"] for row in bucket_examples], dtype=np.float64)
    output_primary = np.asarray([row["output_primary"] for row in bucket_examples], dtype=np.float64)
    output_expert = np.asarray([row["output_expert"] for row in bucket_examples], dtype=np.float64)
    total_primary = np.asarray([row["total_primary"] for row in bucket_examples], dtype=np.float64)
    total_expert = np.asarray([row["total_expert"] for row in bucket_examples], dtype=np.float64)

    bucket_row = {
        "bucket_idx": int(bucket_idx),
        "bucket_count": int(bucket_count),
        "n": int(n),
        "pred_delta_min": float(np.min(pred_deltas)),
        "pred_delta_max": float(np.max(pred_deltas)),
        "pred_delta_mean": float(np.mean(pred_deltas)),
        "pred_delta_median": float(np.median(pred_deltas)),
        "true_delta_mean": float(np.mean(true_deltas)),
        "true_delta_median": float(np.median(true_deltas)),
        "primary_win_rate": float(np.mean([row["primary_wins"] for row in bucket_examples])),
        "expert_win_rate": float(np.mean([row["expert_wins"] for row in bucket_examples])),
        "mean_true_primary_reward": float(np.mean(primary_rewards)),
        "mean_true_expert_reward": float(np.mean(expert_rewards)),
        "mean_reward_gain_primary_minus_expert": float(np.mean(primary_rewards - expert_rewards)),
        "mean_output_token_gain_primary_minus_expert": float(np.mean(output_primary - output_expert)),
        "mean_total_token_gain_primary_minus_expert": float(np.mean(total_primary - total_expert)),
    }

    utility_rows: list[dict[str, Any]] = []
    for lambda_value in lambdas:
        lambda_value = float(lambda_value)
        utility_rows.append(
            {
                "bucket_idx": int(bucket_idx),
                "bucket_count": int(bucket_count),
                "lambda": lambda_value,
                "cost_metric": "output_tokens",
                "primary_mean_utility": float(np.mean(primary_rewards - lambda_value * output_primary)),
                "expert_mean_utility": float(np.mean(expert_rewards - lambda_value * output_expert)),
                "mean_utility_gain_primary_minus_expert": float(
                    np.mean((primary_rewards - lambda_value * output_primary) - (expert_rewards - lambda_value * output_expert))
                ),
            }
        )
        utility_rows.append(
            {
                "bucket_idx": int(bucket_idx),
                "bucket_count": int(bucket_count),
                "lambda": lambda_value,
                "cost_metric": "total_tokens",
                "primary_mean_utility": float(np.mean(primary_rewards - lambda_value * total_primary)),
                "expert_mean_utility": float(np.mean(expert_rewards - lambda_value * total_expert)),
                "mean_utility_gain_primary_minus_expert": float(
                    np.mean((primary_rewards - lambda_value * total_primary) - (expert_rewards - lambda_value * total_expert))
                ),
            }
        )
    return bucket_row, utility_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Bucket predicted reward deltas and report realized win/utility rates")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Collected offline-router dataset dir")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to eval_predictions.jsonl")
    parser.add_argument("--output-json", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--output-buckets-csv", type=Path, default=None, help="Output CSV path for bucket summaries")
    parser.add_argument("--output-utility-csv", type=Path, default=None, help="Output CSV path for bucket utility rows")
    parser.add_argument("--bucket-count", type=int, default=10, help="Equal-count bucket count over predicted delta")
    parser.add_argument("--lambdas", nargs="*", default=None, help="Utility lambdas")
    args = parser.parse_args()

    if args.bucket_count <= 0:
        raise ValueError("--bucket-count must be positive")
    lambdas = _parse_lambdas(args.lambdas)
    route_labels, examples = _build_examples(args.dataset_dir, args.predictions)
    sorted_examples = sorted(examples, key=lambda row: row["pred_delta"])
    bucket_ranges = _quantile_bucket_ranges(sorted_examples, args.bucket_count)

    bucket_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    for bucket_idx, (start, end) in enumerate(bucket_ranges):
        bucket_row, bucket_utility_rows = _bucket_summary(
            sorted_examples[start:end],
            bucket_idx=bucket_idx,
            bucket_count=len(bucket_ranges),
            lambdas=lambdas,
        )
        bucket_row["start_rank"] = int(start)
        bucket_row["end_rank_exclusive"] = int(end)
        bucket_rows.append(bucket_row)
        utility_rows.extend(bucket_utility_rows)

    report = {
        "dataset_dir": str(args.dataset_dir),
        "predictions": str(args.predictions),
        "route_labels": route_labels,
        "bucket_count_requested": int(args.bucket_count),
        "bucket_count_actual": int(len(bucket_rows)),
        "n_examples": int(len(examples)),
        "lambda_values": [float(value) for value in lambdas],
        "overall": {
            "pred_delta_min": float(min(row["pred_delta"] for row in examples)),
            "pred_delta_max": float(max(row["pred_delta"] for row in examples)),
            "pred_delta_mean": float(np.mean([row["pred_delta"] for row in examples])),
            "true_delta_mean": float(np.mean([row["true_delta"] for row in examples])),
            "primary_win_rate": float(np.mean([row["primary_wins"] for row in examples])),
            "expert_win_rate": float(np.mean([row["expert_wins"] for row in examples])),
        },
        "buckets": bucket_rows,
        "bucket_utilities": utility_rows,
    }

    output_json = args.output_json or args.predictions.with_name("delta_bucket_calibration.json")
    output_buckets_csv = args.output_buckets_csv or args.predictions.with_name("delta_bucket_calibration_buckets.csv")
    output_utility_csv = args.output_utility_csv or args.predictions.with_name("delta_bucket_calibration_utility.csv")
    write_json(output_json, report)
    _write_csv(
        output_buckets_csv,
        bucket_rows,
        [
            "bucket_idx",
            "bucket_count",
            "start_rank",
            "end_rank_exclusive",
            "n",
            "pred_delta_min",
            "pred_delta_max",
            "pred_delta_mean",
            "pred_delta_median",
            "true_delta_mean",
            "true_delta_median",
            "primary_win_rate",
            "expert_win_rate",
            "mean_true_primary_reward",
            "mean_true_expert_reward",
            "mean_reward_gain_primary_minus_expert",
            "mean_output_token_gain_primary_minus_expert",
            "mean_total_token_gain_primary_minus_expert",
        ],
    )
    _write_csv(
        output_utility_csv,
        utility_rows,
        [
            "bucket_idx",
            "bucket_count",
            "lambda",
            "cost_metric",
            "primary_mean_utility",
            "expert_mean_utility",
            "mean_utility_gain_primary_minus_expert",
        ],
    )
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_buckets_csv": str(output_buckets_csv),
                "output_utility_csv": str(output_utility_csv),
                "n_examples": len(examples),
                "bucket_count": len(bucket_rows),
            }
        )
    )


if __name__ == "__main__":
    main()

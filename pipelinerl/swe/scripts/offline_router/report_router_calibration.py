#!/usr/bin/env python
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from pipelinerl.swe.scripts.offline_router.common import (
    compute_pairwise_metrics,
    compute_per_route_metrics,
    csv_headers_for_pairwise_metrics,
    csv_headers_for_route_metrics,
    write_json,
)


def _read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


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


def _sanitize_run_name(value: str) -> str:
    value = value.rstrip("/")
    return Path(value).name or value.replace("/", "_")


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _extract_route_labels(summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    route_labels = summary.get("route_labels")
    if isinstance(route_labels, list) and all(isinstance(label, str) for label in route_labels):
        return list(route_labels)
    for row in rows:
        row_labels = row.get("route_labels")
        if isinstance(row_labels, list) and all(isinstance(label, str) for label in row_labels):
            return list(row_labels)
    raise ValueError("Could not infer route_labels from summary.json or eval_predictions.jsonl")


def _extract_reward_arrays(rows: list[dict[str, Any]], route_count: int) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[list[float]] = []
    y_pred: list[list[float]] = []
    for row in rows:
        true_rewards = row.get("true_rewards")
        pred_rewards = row.get("pred_rewards")
        if not isinstance(true_rewards, list) or not isinstance(pred_rewards, list):
            continue
        if len(true_rewards) != route_count or len(pred_rewards) != route_count:
            continue
        true_values = [_finite_float(value) for value in true_rewards]
        pred_values = [_finite_float(value) for value in pred_rewards]
        if any(value is None for value in true_values) or any(value is None for value in pred_values):
            continue
        y_true.append([float(value) for value in true_values])
        y_pred.append([float(value) for value in pred_values])
    if not y_true:
        raise ValueError("No valid prediction rows with true_rewards/pred_rewards")
    return np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)


def _extract_costs(row: dict[str, Any], route_count: int, prefixes: tuple[str, ...]) -> list[float] | None:
    for key in prefixes:
        value = row.get(key)
        if isinstance(value, list) and len(value) == route_count:
            costs = [_finite_float(item) for item in value]
            if all(cost is not None for cost in costs):
                return [float(cost) for cost in costs]

    route_predictions = row.get("route_predictions")
    if isinstance(route_predictions, list) and len(route_predictions) == route_count:
        for key in ("pred_cost", "true_cost", "cost", "total_tokens", "output_tokens"):
            costs = []
            for route_row in route_predictions:
                if not isinstance(route_row, dict):
                    costs = []
                    break
                cost = _finite_float(route_row.get(key))
                if cost is None:
                    costs = []
                    break
                costs.append(float(cost))
            if len(costs) == route_count:
                return costs
    return None


def _load_optional_cost_arrays(rows: list[dict[str, Any]], route_count: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    true_costs: list[list[float]] = []
    pred_costs: list[list[float]] = []
    for row in rows:
        true_row_costs = _extract_costs(row, route_count, ("true_costs", "route_costs", "costs"))
        pred_row_costs = _extract_costs(row, route_count, ("pred_costs", "route_costs", "costs"))
        if true_row_costs is None or pred_row_costs is None:
            return None, None
        true_costs.append(true_row_costs)
        pred_costs.append(pred_row_costs)
    if not true_costs:
        return None, None
    return np.asarray(true_costs, dtype=np.float64), np.asarray(pred_costs, dtype=np.float64)


def _top_rounded(values: np.ndarray, limit: int = 10) -> list[dict[str, Any]]:
    counts = Counter(round(float(value), 2) for value in values.tolist())
    return [
        {"value": float(value), "count": int(count)}
        for value, count in counts.most_common(limit)
    ]


def _route_collapse_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    route_labels: list[str],
    near_zero_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_eval = int(y_true.shape[0])
    for idx, route_label in enumerate(route_labels):
        true_vals = y_true[:, idx]
        pred_vals = y_pred[:, idx]
        pred_zero = int(np.sum(pred_vals == 0.0))
        true_zero = int(np.sum(true_vals == 0.0))
        pred_near_zero = int(np.sum(pred_vals <= near_zero_threshold))
        true_near_zero = int(np.sum(true_vals <= near_zero_threshold))
        rows.append(
            {
                "route_idx": int(idx),
                "route_label": route_label,
                "n_eval": n_eval,
                "bias": float(np.mean(pred_vals - true_vals)),
                "mean_ratio_pred_over_true": (
                    float(np.mean(pred_vals) / np.mean(true_vals)) if abs(float(np.mean(true_vals))) > 1e-12 else None
                ),
                "pred_exact_zero_count": pred_zero,
                "pred_exact_zero_rate": float(pred_zero / n_eval),
                "true_exact_zero_count": true_zero,
                "true_exact_zero_rate": float(true_zero / n_eval),
                "pred_near_zero_count": pred_near_zero,
                "pred_near_zero_rate": float(pred_near_zero / n_eval),
                "true_near_zero_count": true_near_zero,
                "true_near_zero_rate": float(true_near_zero / n_eval),
                "top_rounded_pred": _top_rounded(pred_vals),
                "top_rounded_true": _top_rounded(true_vals),
            }
        )
    return rows


def _bucket_edges(bucket_count: int) -> np.ndarray:
    if bucket_count <= 0:
        raise ValueError("--bucket-count must be positive")
    return np.linspace(0.0, 1.0, bucket_count + 1)


def _reliability_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    route_labels: list[str],
    bucket_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    edges = _bucket_edges(bucket_count)
    for route_idx, route_label in enumerate(route_labels):
        true_vals = y_true[:, route_idx]
        pred_vals = y_pred[:, route_idx]
        bucket_ids = np.digitize(pred_vals, edges[1:-1], right=False)
        for bucket_idx in range(bucket_count):
            mask = bucket_ids == bucket_idx
            n_bucket = int(np.sum(mask))
            low = float(edges[bucket_idx])
            high = float(edges[bucket_idx + 1])
            if n_bucket == 0:
                rows.append(
                    {
                        "route_idx": route_idx,
                        "route_label": route_label,
                        "bucket_idx": bucket_idx,
                        "bucket_low": low,
                        "bucket_high": high,
                        "n": 0,
                        "mean_pred": None,
                        "mean_true": None,
                        "bias": None,
                        "mae": None,
                        "rmse": None,
                    }
                )
                continue
            pred_bucket = pred_vals[mask]
            true_bucket = true_vals[mask]
            mse = float(np.mean((pred_bucket - true_bucket) ** 2))
            rows.append(
                {
                    "route_idx": route_idx,
                    "route_label": route_label,
                    "bucket_idx": bucket_idx,
                    "bucket_low": low,
                    "bucket_high": high,
                    "n": n_bucket,
                    "mean_pred": float(np.mean(pred_bucket)),
                    "mean_true": float(np.mean(true_bucket)),
                    "bias": float(np.mean(pred_bucket - true_bucket)),
                    "mae": float(np.mean(np.abs(pred_bucket - true_bucket))),
                    "rmse": float(math.sqrt(mse)),
                }
            )
    return rows


def _select_route(scores: np.ndarray, costs: np.ndarray | None = None) -> int:
    max_score = float(np.max(scores))
    candidate_indices = np.flatnonzero(np.abs(scores - max_score) <= 1e-12)
    if candidate_indices.size == 1 or costs is None:
        return int(candidate_indices[0])
    candidate_costs = costs[candidate_indices]
    return int(candidate_indices[int(np.argmin(candidate_costs))])


def _decision_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    route_labels: list[str],
    lambda_values: list[float],
    true_costs: np.ndarray | None,
    pred_costs: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_eval, route_count = y_true.shape
    has_costs = true_costs is not None and pred_costs is not None
    for lambda_value in lambda_values:
        if lambda_value != 0.0 and not has_costs:
            continue
        chosen_indices: list[int] = []
        oracle_indices: list[int] = []
        chosen_true_rewards: list[float] = []
        chosen_true_utilities: list[float] = []
        oracle_true_utilities: list[float] = []
        for row_idx in range(n_eval):
            pred_row_costs = pred_costs[row_idx] if has_costs else None
            true_row_costs = true_costs[row_idx] if has_costs else None
            pred_scores = y_pred[row_idx].copy()
            true_scores = y_true[row_idx].copy()
            if has_costs:
                pred_scores = pred_scores - lambda_value * pred_row_costs
                true_scores = true_scores - lambda_value * true_row_costs
            chosen_idx = _select_route(pred_scores, pred_row_costs)
            oracle_idx = _select_route(true_scores, true_row_costs)
            chosen_indices.append(chosen_idx)
            oracle_indices.append(oracle_idx)
            chosen_true_rewards.append(float(y_true[row_idx, chosen_idx]))
            if has_costs:
                chosen_true_utilities.append(
                    float(y_true[row_idx, chosen_idx] - lambda_value * true_row_costs[chosen_idx])
                )
            else:
                chosen_true_utilities.append(float(y_true[row_idx, chosen_idx]))
            oracle_true_utilities.append(float(true_scores[oracle_idx]))

        chosen_array = np.asarray(chosen_indices, dtype=np.int64)
        oracle_array = np.asarray(oracle_indices, dtype=np.int64)
        base_route_rows: dict[str, Any] = {}
        for route_idx, route_label in enumerate(route_labels):
            base_route_rows[f"select_rate_route_{route_idx}"] = float(np.mean(chosen_array == route_idx))
            base_route_rows[f"always_route_{route_idx}_mean_true_reward"] = float(np.mean(y_true[:, route_idx]))
            base_route_rows[f"always_route_{route_idx}_label"] = route_label

        rows.append(
            {
                "lambda": float(lambda_value),
                "has_costs": bool(has_costs),
                "n_eval": int(n_eval),
                "chosen_mean_true_reward": float(np.mean(chosen_true_rewards)),
                "chosen_mean_true_utility": float(np.mean(chosen_true_utilities)),
                "oracle_mean_true_utility": float(np.mean(oracle_true_utilities)),
                "oracle_capture": (
                    float(np.mean(chosen_true_utilities) / np.mean(oracle_true_utilities))
                    if abs(float(np.mean(oracle_true_utilities))) > 1e-12
                    else None
                ),
                "oracle_regret": float(np.mean(np.asarray(oracle_true_utilities) - np.asarray(chosen_true_utilities))),
                "oracle_match_rate": float(np.mean(chosen_array == oracle_array)),
                **base_route_rows,
            }
        )
    return rows


def _parse_lambdas(value: str) -> list[float]:
    result: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            result.append(float(part))
    return result or [0.0]


def _summarize_run(
    run_name: str,
    run_dir: Path,
    bucket_count: int,
    near_zero_threshold: float,
    lambda_values: list[float],
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    predictions_path = run_dir / "eval_predictions.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing eval_predictions.jsonl: {predictions_path}")

    summary = _read_json(summary_path)
    prediction_rows = _read_jsonl(predictions_path)
    route_labels = _extract_route_labels(summary, prediction_rows)
    y_true, y_pred = _extract_reward_arrays(prediction_rows, route_count=len(route_labels))
    true_costs, pred_costs = _load_optional_cost_arrays(prediction_rows, route_count=len(route_labels))
    route_metrics = compute_per_route_metrics(y_true, y_pred, route_labels)
    pairwise_metrics = compute_pairwise_metrics(y_true, y_pred, route_labels)
    collapse_stats = _route_collapse_stats(
        y_true,
        y_pred,
        route_labels,
        near_zero_threshold=near_zero_threshold,
    )
    reliability = _reliability_rows(y_true, y_pred, route_labels, bucket_count=bucket_count)
    decisions = _decision_rows(
        y_true,
        y_pred,
        route_labels,
        lambda_values=lambda_values,
        true_costs=true_costs,
        pred_costs=pred_costs,
    )

    run_summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "seed": summary.get("seed"),
        "supervision_mode": summary.get("supervision_mode"),
        "target_precision": (summary.get("text_reward") or {}).get("target_precision"),
        "target_grid_count": (summary.get("text_reward") or {}).get("target_grid_count"),
        "bin_count": (summary.get("text_reward") or {}).get("bin_count"),
        "bin_value_order": (summary.get("text_reward") or {}).get("bin_value_order"),
        "n_eval": int(y_true.shape[0]),
        "route_count": int(y_true.shape[1]),
        "route_labels": route_labels,
        "best_eval_loss": summary.get("best_eval_loss"),
        "train_loss": (summary.get("history") or [{}])[-1].get("train_loss") if summary.get("history") else None,
        "parse_failure_rate": (summary.get("text_reward") or {}).get("best_eval_parse_failure_rate"),
        "has_costs": true_costs is not None and pred_costs is not None,
    }
    if pairwise_metrics:
        first_pair = pairwise_metrics[0]
        for key in ("roc_auc", "ranking_accuracy_sign", "pearson_delta", "spearman_delta", "delta_mae"):
            run_summary[key] = first_pair.get(key)

    return {
        "run_summary": run_summary,
        "route_metrics": route_metrics,
        "pairwise_metrics": pairwise_metrics,
        "collapse_stats": collapse_stats,
        "reliability": reliability,
        "decisions": decisions,
    }


def _flatten_route_rows(run_name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"run_name": run_name, **row} for row in rows]


def _flatten_collapse_rows(run_name: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flat_rows: list[dict[str, Any]] = []
    top_values: dict[str, Any] = {}
    for row in rows:
        row_copy = dict(row)
        key = f"{run_name}::route_{row['route_idx']}"
        top_values[key] = {
            "run_name": run_name,
            "route_idx": row["route_idx"],
            "route_label": row["route_label"],
            "top_rounded_pred": row_copy.pop("top_rounded_pred"),
            "top_rounded_true": row_copy.pop("top_rounded_true"),
        }
        flat_rows.append({"run_name": run_name, **row_copy})
    return flat_rows, top_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare offline-router reward calibration diagnostics.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec as NAME=/path/to/train_output_dir or just /path/to/train_output_dir.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write report files.")
    parser.add_argument("--bucket-count", type=int, default=10, help="Reliability bucket count over predicted reward.")
    parser.add_argument("--near-zero-threshold", type=float, default=0.05)
    parser.add_argument("--lambdas", default="0.0", help="Comma-separated lambda values for decision summaries.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lambda_values = _parse_lambdas(args.lambdas)

    all_payload: dict[str, Any] = {
        "bucket_count": int(args.bucket_count),
        "near_zero_threshold": float(args.near_zero_threshold),
        "lambda_values": lambda_values,
        "runs": {},
        "top_rounded_values": {},
    }
    run_summary_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    collapse_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    for run_spec in args.run:
        if "=" in run_spec:
            run_name, run_path = run_spec.split("=", 1)
        else:
            run_path = run_spec
            run_name = _sanitize_run_name(run_path)
        run_dir = Path(run_path)
        payload = _summarize_run(
            run_name=run_name,
            run_dir=run_dir,
            bucket_count=int(args.bucket_count),
            near_zero_threshold=float(args.near_zero_threshold),
            lambda_values=lambda_values,
        )
        all_payload["runs"][run_name] = payload
        run_summary_rows.append(payload["run_summary"])
        route_rows.extend(_flatten_route_rows(run_name, payload["route_metrics"]))
        pairwise_rows.extend(_flatten_route_rows(run_name, payload["pairwise_metrics"]))
        collapse_flat_rows, top_values = _flatten_collapse_rows(run_name, payload["collapse_stats"])
        collapse_rows.extend(collapse_flat_rows)
        all_payload["top_rounded_values"].update(top_values)
        reliability_rows.extend(_flatten_route_rows(run_name, payload["reliability"]))
        decision_rows.extend(_flatten_route_rows(run_name, payload["decisions"]))

    write_json(output_dir / "calibration_report.json", all_payload)
    _write_csv(
        output_dir / "run_summary.csv",
        run_summary_rows,
        [
            "run_name",
            "run_dir",
            "seed",
            "supervision_mode",
            "target_precision",
            "target_grid_count",
            "bin_count",
            "bin_value_order",
            "n_eval",
            "route_count",
            "best_eval_loss",
            "train_loss",
            "parse_failure_rate",
            "has_costs",
            "roc_auc",
            "ranking_accuracy_sign",
            "pearson_delta",
            "spearman_delta",
            "delta_mae",
        ],
    )
    _write_csv(output_dir / "route_metrics.csv", route_rows, ["run_name", *csv_headers_for_route_metrics()])
    _write_csv(output_dir / "pairwise_metrics.csv", pairwise_rows, ["run_name", *csv_headers_for_pairwise_metrics()])
    _write_csv(
        output_dir / "collapse_stats.csv",
        collapse_rows,
        [
            "run_name",
            "route_idx",
            "route_label",
            "n_eval",
            "bias",
            "mean_ratio_pred_over_true",
            "pred_exact_zero_count",
            "pred_exact_zero_rate",
            "true_exact_zero_count",
            "true_exact_zero_rate",
            "pred_near_zero_count",
            "pred_near_zero_rate",
            "true_near_zero_count",
            "true_near_zero_rate",
        ],
    )
    _write_csv(
        output_dir / "reliability_buckets.csv",
        reliability_rows,
        [
            "run_name",
            "route_idx",
            "route_label",
            "bucket_idx",
            "bucket_low",
            "bucket_high",
            "n",
            "mean_pred",
            "mean_true",
            "bias",
            "mae",
            "rmse",
        ],
    )

    decision_headers = [
        "run_name",
        "lambda",
        "has_costs",
        "n_eval",
        "chosen_mean_true_reward",
        "chosen_mean_true_utility",
        "oracle_mean_true_utility",
        "oracle_capture",
        "oracle_regret",
        "oracle_match_rate",
    ]
    max_route_count = max((int(row.get("route_count", 0)) for row in run_summary_rows), default=0)
    for route_idx in range(max_route_count):
        decision_headers.extend(
            [
                f"select_rate_route_{route_idx}",
                f"always_route_{route_idx}_mean_true_reward",
                f"always_route_{route_idx}_label",
            ]
        )
    _write_csv(output_dir / "decision_summary.csv", decision_rows, decision_headers)

    print(f"Wrote calibration report to {output_dir}")
    print(f"Runs: {', '.join(row['run_name'] for row in run_summary_rows)}")


if __name__ == "__main__":
    main()

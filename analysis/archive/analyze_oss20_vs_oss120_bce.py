#!/usr/bin/env python3
"""Analyze the OSS-20B vs OSS-120B reward-BCE comparative.

This reporter is intentionally narrow: it evaluates a two-route reward predictor
over original routes 1 and 3 from the SWE-Smith real-label dataset, using
deployable predicted output costs for action selection and realized output costs
for reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


DEFAULT_DATASET_DIR = Path(
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect"
)
DEFAULT_COST_PREDICTIONS = Path(
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_trace_expanded_cost_model_old286_eval_score_1781113205/"
    "eval_predictions.jsonl"
)
DEFAULT_ROUTE_IDXS = [1, 3]
DEFAULT_COST_WEIGHTS = [2.78e-7, 1.299e-6, 4.64e-6, 1.113e-5]
DEFAULT_LAMBDAS = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    12.5,
    15.0,
    20.0,
    25.0,
    30.0,
    40.0,
    50.0,
    75.0,
    100.0,
    150.0,
    200.0,
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("Expected at least one route index")
    return out


def parse_float_list(value: str) -> list[float]:
    out = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("Expected at least one float")
    return out


def resolve_prediction_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / "eval_predictions.jsonl",
        path / "train_qwen3_embedding_8b_lora_reward_bce_oss20_vs_oss120_10epoch" / "eval_predictions.jsonl",
        path / "train_qwen3_embedding_8b_lora_reward_bce_oss20_vs_oss120_r32_qkvo_mlp_10epoch" / "eval_predictions.jsonl",
        path / "train_qwen3_embedding_8b_lora_reward_bce_10epoch" / "eval_predictions.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(path.glob("**/eval_predictions.jsonl"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve unique eval_predictions.jsonl under {path}")


def maybe_summary_path(predictions_path: Path) -> Path | None:
    summary = predictions_path.with_name("summary.json")
    return summary if summary.is_file() else None


def load_full_route_labels(dataset_dir: Path) -> list[str]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    labels = metadata.get("route_labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Missing route_labels in {metadata_path}")
    return [str(label) for label in labels]


def read_split_rows(dataset_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    split_dir = dataset_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(split_dir)
    columns = [
        "dataset",
        "problem_id",
        "performance_targets",
        "route_prompt_tokens",
        "route_output_tokens",
    ]
    rows: dict[str, dict[str, Any]] = {}
    for shard in sorted(split_dir.glob("*.parquet")):
        table = pq.read_table(shard, columns=columns)
        for row in table.to_pylist():
            rows[str(row["problem_id"])] = row
    if not rows:
        raise ValueError(f"No rows read from {split_dir}")
    return rows


def infer_prediction_source_idxs(
    *,
    predictions_path: Path,
    prediction_rows: list[dict[str, Any]],
    full_route_labels: list[str],
    requested_route_idxs: list[int],
) -> list[int]:
    summary_path = maybe_summary_path(predictions_path)
    if summary_path is not None:
        summary = json.loads(summary_path.read_text())
        config = summary.get("config", {})
        target_route_idxs = config.get("target_route_idxs")
        if isinstance(target_route_idxs, list) and target_route_idxs:
            return [int(idx) for idx in target_route_idxs]

    if not prediction_rows:
        raise ValueError("Empty prediction file")
    row_labels = prediction_rows[0].get("route_labels")
    if not isinstance(row_labels, list):
        return list(requested_route_idxs)
    label_to_idx = {label: idx for idx, label in enumerate(full_route_labels)}
    source_idxs = [label_to_idx[str(label)] for label in row_labels if str(label) in label_to_idx]
    if len(source_idxs) == len(row_labels):
        return source_idxs
    return list(requested_route_idxs)


def cost_prediction_map(cost_row: dict[str, Any], route_count: int) -> dict[int, float]:
    values = cost_row.get("pred_output_tokens")
    if not isinstance(values, list):
        return {}
    preds = [max(0.0, float(value)) for value in values]
    route_idxs = cost_row.get("cost_route_idxs") or cost_row.get("predicted_cost_route_idxs")
    if isinstance(route_idxs, list) and len(route_idxs) == len(preds):
        mapped: dict[int, float] = {}
        for raw_idx, pred in zip(route_idxs, preds):
            try:
                route_idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= route_idx < route_count:
                mapped[route_idx] = pred
        if mapped:
            return mapped
    if len(preds) == route_count:
        return {idx: preds[idx] for idx in range(route_count)}
    if len(preds) == route_count - 1:
        return {idx: preds[idx - 1] for idx in range(1, route_count)}
    return {idx: preds[idx] for idx in range(min(route_count, len(preds)))}


def direct_random_at_cost(always_points: list[dict[str, float | str]], cap: float) -> dict[str, float | str] | None:
    candidates: list[dict[str, float | str]] = []
    for point in always_points:
        if float(point["cost"]) <= cap + 1.0e-15:
            candidates.append({"reward": float(point["reward"]), "cost": float(point["cost"]), "label": str(point["label"])})
    for a, b in combinations(always_points, 2):
        if abs(float(a["cost"]) - float(b["cost"])) < 1.0e-15:
            continue
        low, high = (a, b) if float(a["cost"]) < float(b["cost"]) else (b, a)
        low_cost = float(low["cost"])
        high_cost = float(high["cost"])
        if cap < low_cost - 1.0e-15 or cap > high_cost + 1.0e-15:
            continue
        high_weight = (cap - low_cost) / (high_cost - low_cost)
        high_weight = min(1.0, max(0.0, high_weight))
        low_weight = 1.0 - high_weight
        candidates.append(
            {
                "reward": low_weight * float(low["reward"]) + high_weight * float(high["reward"]),
                "cost": cap,
                "label": f"{low_weight:.4f}*{low['label']} + {high_weight:.4f}*{high['label']}",
            }
        )
    if not candidates:
        return None
    return max(candidates, key=lambda row: (float(row["reward"]), -float(row["cost"])))


def build_examples(
    *,
    reward_predictions: Path,
    cost_predictions: Path,
    dataset_dir: Path,
    route_idxs: list[int],
    cost_weights: list[float],
    split: str,
    decision_cost_mode: str,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    full_route_labels = load_full_route_labels(dataset_dir)
    eval_lookup = read_split_rows(dataset_dir, split)
    train_lookup = read_split_rows(dataset_dir, "train") if decision_cost_mode == "fixed_train_mean" else {}
    reward_rows = read_jsonl(reward_predictions)
    cost_lookup = {str(row.get("problem_id")): row for row in read_jsonl(cost_predictions)}
    source_idxs = infer_prediction_source_idxs(
        predictions_path=reward_predictions,
        prediction_rows=reward_rows,
        full_route_labels=full_route_labels,
        requested_route_idxs=route_idxs,
    )
    if set(route_idxs) - set(source_idxs):
        raise ValueError(f"Requested route idxs {route_idxs} are not all in reward prediction source idxs {source_idxs}")
    source_to_local = {source_idx: local_idx for local_idx, source_idx in enumerate(source_idxs)}
    local_idxs = [source_to_local[source_idx] for source_idx in route_idxs]

    fixed_train_mean_costs: dict[int, float] = {}
    if decision_cost_mode == "fixed_train_mean":
        for route_idx in route_idxs:
            costs = []
            for row in train_lookup.values():
                output_tokens = row.get("route_output_tokens")
                if isinstance(output_tokens, list) and len(output_tokens) > route_idx:
                    costs.append(float(output_tokens[route_idx]) * float(cost_weights[route_idx]))
            fixed_train_mean_costs[route_idx] = float(np.mean(costs)) if costs else math.nan

    skipped = {"missing_eval": 0, "missing_cost": 0, "invalid": 0}
    examples: list[dict[str, Any]] = []
    for reward_row in reward_rows:
        problem_id = str(reward_row.get("problem_id"))
        eval_row = eval_lookup.get(problem_id)
        cost_row = cost_lookup.get(problem_id)
        if eval_row is None:
            skipped["missing_eval"] += 1
            continue
        if cost_row is None:
            skipped["missing_cost"] += 1
            continue

        pred_rewards_all = reward_row.get("pred_rewards")
        true_rewards_all = eval_row.get("performance_targets")
        output_tokens = eval_row.get("route_output_tokens")
        if (
            not isinstance(pred_rewards_all, list)
            or not isinstance(true_rewards_all, list)
            or not isinstance(output_tokens, list)
            or len(cost_weights) <= max(route_idxs)
            or len(true_rewards_all) <= max(route_idxs)
            or len(output_tokens) <= max(route_idxs)
            or len(pred_rewards_all) <= max(local_idxs)
        ):
            skipped["invalid"] += 1
            continue

        pred_output_by_source = cost_prediction_map(cost_row, len(full_route_labels))
        pred_rewards = [float(pred_rewards_all[local_idx]) for local_idx in local_idxs]
        true_rewards = [float(true_rewards_all[route_idx]) for route_idx in route_idxs]
        true_costs = [float(output_tokens[route_idx]) * float(cost_weights[route_idx]) for route_idx in route_idxs]
        if decision_cost_mode == "oracle":
            decision_costs = list(true_costs)
        elif decision_cost_mode == "fixed_train_mean":
            decision_costs = [float(fixed_train_mean_costs[route_idx]) for route_idx in route_idxs]
        elif decision_cost_mode == "predicted":
            decision_costs = []
            for route_idx in route_idxs:
                if route_idx not in pred_output_by_source:
                    skipped["invalid"] += 1
                    break
                decision_costs.append(float(pred_output_by_source[route_idx]) * float(cost_weights[route_idx]))
            if len(decision_costs) != len(route_idxs):
                continue
        else:
            raise ValueError(f"Unsupported decision cost mode: {decision_cost_mode}")

        examples.append(
            {
                "problem_id": problem_id,
                "pred_rewards": pred_rewards,
                "true_rewards": true_rewards,
                "true_costs": true_costs,
                "decision_costs": decision_costs,
            }
        )

    route_labels = [full_route_labels[idx] for idx in route_idxs]
    if not examples:
        raise ValueError(f"No joined examples. skipped={skipped}")
    return examples, route_labels, skipped


def route_counts_json(choices: np.ndarray, route_labels: list[str]) -> str:
    return json.dumps(
        {route_labels[idx]: int(np.sum(choices == idx)) for idx in range(len(route_labels))},
        sort_keys=True,
    )


def analyze(
    *,
    examples: list[dict[str, Any]],
    route_labels: list[str],
    lambdas: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pred_rewards = np.asarray([row["pred_rewards"] for row in examples], dtype=np.float64)
    true_rewards = np.asarray([row["true_rewards"] for row in examples], dtype=np.float64)
    true_costs = np.asarray([row["true_costs"] for row in examples], dtype=np.float64)
    decision_costs = np.asarray([row["decision_costs"] for row in examples], dtype=np.float64)
    n = true_rewards.shape[0]
    row_idx = np.arange(n)

    route_mean_rewards = np.mean(true_rewards, axis=0)
    route_mean_costs = np.mean(true_costs, axis=0)
    always_points = [
        {"label": route_labels[idx], "reward": float(route_mean_rewards[idx]), "cost": float(route_mean_costs[idx])}
        for idx in range(len(route_labels))
    ]
    always_utility_rows: list[dict[str, Any]] = []
    for lambda_value in lambdas:
        for idx, label in enumerate(route_labels):
            always_utility_rows.append(
                {
                    "lambda": lambda_value,
                    "policy": f"always::{label}",
                    "mean_reward": float(route_mean_rewards[idx]),
                    "mean_cost": float(route_mean_costs[idx]),
                    "mean_utility": float(route_mean_rewards[idx] - lambda_value * route_mean_costs[idx]),
                }
            )

    points: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    route_20_idx = 0
    route_120_idx = 1
    global_120_marginal = np.logical_and(true_rewards[:, route_120_idx] >= 0.5, true_rewards[:, route_20_idx] < 0.5)
    global_20_marginal = np.logical_and(true_rewards[:, route_20_idx] >= 0.5, true_rewards[:, route_120_idx] < 0.5)
    for lambda_value in lambdas:
        scores = pred_rewards - float(lambda_value) * decision_costs
        choices = np.argmax(scores, axis=1)
        selected_rewards = true_rewards[row_idx, choices]
        selected_costs = true_costs[row_idx, choices]
        mean_reward = float(np.mean(selected_rewards))
        mean_cost = float(np.mean(selected_costs))
        mean_utility = float(mean_reward - float(lambda_value) * mean_cost)
        random = direct_random_at_cost(always_points, mean_cost)
        random_reward = math.nan if random is None else float(random["reward"])
        random_label = "" if random is None else str(random["label"])
        always_best_utility = max(
            float(route_mean_rewards[idx] - float(lambda_value) * route_mean_costs[idx])
            for idx in range(len(route_labels))
        )
        oracle_choices = np.argmax(true_rewards - float(lambda_value) * true_costs, axis=1)
        points.append(
            {
                "lambda": float(lambda_value),
                "mean_reward": mean_reward,
                "mean_cost": mean_cost,
                "mean_utility": mean_utility,
                "mean_decision_cost": float(np.mean(decision_costs[row_idx, choices])),
                "direct_random_reward_at_same_cost": random_reward,
                "gain_vs_direct_random_same_cost": mean_reward - random_reward
                if math.isfinite(random_reward)
                else math.nan,
                "direct_random_mix": random_label,
                "best_always_utility": always_best_utility,
                "utility_gain_vs_best_always": mean_utility - always_best_utility,
                "oracle_choice_acc": float(np.mean(choices == oracle_choices)),
                "choice_counts_by_route": route_counts_json(choices, route_labels),
                "choose_120b_frac": float(np.mean(choices == route_120_idx)),
            }
        )

        selected_120 = choices == route_120_idx
        selected_20 = choices == route_20_idx
        quality_rows.append(
            {
                "lambda": float(lambda_value),
                "selected_120b_count": int(np.sum(selected_120)),
                "selected_120b_frac": float(np.mean(selected_120)),
                "selected_120b_reward": float(np.mean(true_rewards[selected_120, route_120_idx]))
                if np.any(selected_120)
                else math.nan,
                "selected_120b_marginal_win_rate": float(np.mean(global_120_marginal[selected_120]))
                if np.any(selected_120)
                else math.nan,
                "global_120b_marginal_win_rate": float(np.mean(global_120_marginal)),
                "selected_120b_marginal_lift": float(np.mean(global_120_marginal[selected_120]) - np.mean(global_120_marginal))
                if np.any(selected_120)
                else math.nan,
                "selected_20b_count": int(np.sum(selected_20)),
                "selected_20b_reward": float(np.mean(true_rewards[selected_20, route_20_idx]))
                if np.any(selected_20)
                else math.nan,
                "selected_20b_marginal_win_rate": float(np.mean(global_20_marginal[selected_20]))
                if np.any(selected_20)
                else math.nan,
                "global_20b_marginal_win_rate": float(np.mean(global_20_marginal)),
                "mean_pred_margin_120b_minus_20b": float(np.mean(pred_rewards[:, route_120_idx] - pred_rewards[:, route_20_idx])),
                "selected_120b_pred_margin_mean": float(
                    np.mean((pred_rewards[:, route_120_idx] - pred_rewards[:, route_20_idx])[selected_120])
                )
                if np.any(selected_120)
                else math.nan,
            }
        )
    return points, quality_rows, always_utility_rows


def summarize(points: list[dict[str, Any]], route_labels: list[str], skipped: dict[str, int], n_examples: int) -> list[dict[str, Any]]:
    finite_gain_points = [row for row in points if math.isfinite(float(row["gain_vs_direct_random_same_cost"]))]
    best_gain = max(finite_gain_points, key=lambda row: float(row["gain_vs_direct_random_same_cost"]))
    best_utility = max(points, key=lambda row: float(row["utility_gain_vs_best_always"]))
    positive = [row for row in finite_gain_points if float(row["gain_vs_direct_random_same_cost"]) > 0.0]
    return [
        {
            "n_examples": n_examples,
            "route_labels": json.dumps(route_labels),
            "skipped": json.dumps(skipped, sort_keys=True),
            "best_same_cost_gain": float(best_gain["gain_vs_direct_random_same_cost"]),
            "best_same_cost_lambda": float(best_gain["lambda"]),
            "best_same_cost_reward": float(best_gain["mean_reward"]),
            "best_same_cost_cost": float(best_gain["mean_cost"]),
            "best_same_cost_random_reward": float(best_gain["direct_random_reward_at_same_cost"]),
            "positive_same_cost_gain_lambdas": len(positive),
            "mean_positive_same_cost_gain": float(np.mean([float(row["gain_vs_direct_random_same_cost"]) for row in positive]))
            if positive
            else 0.0,
            "best_utility_gain_vs_always": float(best_utility["utility_gain_vs_best_always"]),
            "best_utility_lambda": float(best_utility["lambda"]),
            "best_utility": float(best_utility["mean_utility"]),
        }
    ]


def maybe_plot(points: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    xs = [float(row["mean_cost"]) for row in points]
    ys = [float(row["mean_reward"]) for row in points]
    random_x = [float(row["mean_cost"]) for row in points if math.isfinite(float(row["direct_random_reward_at_same_cost"]))]
    random_y = [
        float(row["direct_random_reward_at_same_cost"])
        for row in points
        if math.isfinite(float(row["direct_random_reward_at_same_cost"]))
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, marker="o", label="20-vs-120 BCE router")
    if random_x:
        order = np.argsort(np.asarray(random_x))
        ax.plot(np.asarray(random_x)[order], np.asarray(random_y)[order], linestyle="--", label="direct-random frontier")
    ax.set_xlabel("Mean realized weighted output cost")
    ax.set_ylabel("Mean real pass rate")
    ax.set_title("OSS-20B vs OSS-120B")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reward_vs_cost.png", dpi=180)
    fig.savefig(output_dir / "reward_vs_cost.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-run", type=Path, required=True, help="Run dir, train output dir, or eval_predictions.jsonl")
    parser.add_argument("--cost-predictions", type=Path, default=DEFAULT_COST_PREDICTIONS)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-idxs", default=",".join(str(idx) for idx in DEFAULT_ROUTE_IDXS))
    parser.add_argument("--route-cost-weights", default=",".join(str(value) for value in DEFAULT_COST_WEIGHTS))
    parser.add_argument("--lambdas", default=",".join(str(value) for value in DEFAULT_LAMBDAS))
    parser.add_argument("--split", default="eval", choices=["train", "eval"])
    parser.add_argument("--decision-cost-mode", choices=["predicted", "fixed_train_mean", "oracle"], default="predicted")
    args = parser.parse_args()

    reward_predictions = resolve_prediction_path(args.reward_run)
    route_idxs = parse_int_list(args.route_idxs)
    cost_weights = parse_float_list(args.route_cost_weights)
    lambdas = parse_float_list(args.lambdas)
    examples, route_labels, skipped = build_examples(
        reward_predictions=reward_predictions,
        cost_predictions=args.cost_predictions,
        dataset_dir=args.dataset_dir,
        route_idxs=route_idxs,
        cost_weights=cost_weights,
        split=args.split,
        decision_cost_mode=str(args.decision_cost_mode),
    )
    points, quality_rows, always_utility_rows = analyze(
        examples=examples,
        route_labels=route_labels,
        lambdas=lambdas,
    )
    summary_rows = summarize(points, route_labels, skipped, len(examples))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "points.csv", points)
    write_csv(args.output_dir / "selection_quality.csv", quality_rows)
    write_csv(args.output_dir / "always_utility.csv", always_utility_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_json(
        args.output_dir / "summary.json",
        {
            "reward_predictions": str(reward_predictions),
            "cost_predictions": str(args.cost_predictions),
            "dataset_dir": str(args.dataset_dir),
            "split": args.split,
            "decision_cost_mode": args.decision_cost_mode,
            "route_idxs": route_idxs,
            "route_labels": route_labels,
            "route_cost_weights": cost_weights,
            "lambdas": lambdas,
            "n_examples": len(examples),
            "skipped": skipped,
            "summary": summary_rows[0],
        },
    )
    maybe_plot(points, args.output_dir)

    row = summary_rows[0]
    print(
        "Best same-cost gain: "
        f"{float(row['best_same_cost_gain']):+.4f} at lambda={row['best_same_cost_lambda']} "
        f"(reward={float(row['best_same_cost_reward']):.4f}, "
        f"random={float(row['best_same_cost_random_reward']):.4f})"
    )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()

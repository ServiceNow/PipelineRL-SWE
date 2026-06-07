#!/usr/bin/env python
import argparse
import csv
import json
import math
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_LAMBDAS = [0.0, 1.0e-5, 2.0e-5, 5.0e-5, 1.0e-4, 2.0e-4]
DEFAULT_ROUTE_LABELS = [
    "primary_model",
    "expert_0:openai/gpt-oss-120b",
    "expert_0:google/gemini-3-flash-preview",
]
DEFAULT_ROUTE_COST_WEIGHTS = [1.0, 3.0, 20.0]


def _key(row: dict[str, Any]) -> str:
    return f"{row.get('dataset')}::{row.get('problem_id')}"


def _parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    return values or list(DEFAULT_LAMBDAS)


def _parse_cost_metrics(value: str) -> list[str]:
    metrics = [part.strip() for part in value.split(",") if part.strip()]
    for metric in metrics:
        if metric not in {"output_tokens", "total_tokens"}:
            raise ValueError(f"Unsupported cost metric: {metric}")
    return metrics or ["output_tokens"]


def _parse_cost_weights(value: str | None, route_labels: list[str]) -> list[float]:
    if value is None or not value.strip():
        defaults = dict(zip(DEFAULT_ROUTE_LABELS, DEFAULT_ROUTE_COST_WEIGHTS, strict=True))
        return [float(defaults.get(label, 1.0)) for label in route_labels]

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if all("=" not in part for part in parts):
        weights = [float(part) for part in parts]
        if len(weights) != len(route_labels):
            raise ValueError(f"Expected {len(route_labels)} positional weights, got {len(weights)}")
        return weights

    mapping: dict[str, float] = {}
    for part in parts:
        label, weight = part.split("=", 1)
        mapping[label.strip()] = float(weight.strip())
    defaults = dict(zip(DEFAULT_ROUTE_LABELS, DEFAULT_ROUTE_COST_WEIGHTS, strict=True))
    return [float(mapping.get(label, defaults.get(label, 1.0))) for label in route_labels]


def _read_attempt_scores(path: Path, target_dim: int) -> dict[str, list[float]]:
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped[_key(row)][int(row["route_idx"])] = float(row["pred_score"])
    return {
        key: [float(route_scores[idx]) for idx in range(target_dim)]
        for key, route_scores in grouped.items()
        if all(idx in route_scores for idx in range(target_dim))
    }


def _read_source_rows(dataset_dir: Path, split: str) -> list[dict[str, Any]]:
    split_dir = dataset_dir / split
    paths = sorted(split_dir.glob("*.parquet"))
    if not paths:
        raise ValueError(f"No parquet files under {split_dir}")
    rows: list[dict[str, Any]] = []
    columns = [
        "dataset",
        "problem_id",
        "performance_targets",
        "route_prompt_tokens",
        "route_output_tokens",
    ]
    for path in paths:
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    return rows


def _build_arrays(
    *,
    dataset_dir: Path,
    split: str,
    pred_by_key: dict[str, list[float]],
    target_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preds: list[list[float]] = []
    rewards: list[list[float]] = []
    prompt_tokens: list[list[float]] = []
    output_tokens: list[list[float]] = []
    for row in _read_source_rows(dataset_dir, split):
        pred_scores = pred_by_key.get(_key(row))
        route_rewards = row.get("performance_targets")
        route_prompt_tokens = row.get("route_prompt_tokens")
        route_output_tokens = row.get("route_output_tokens")
        if (
            pred_scores is None
            or not isinstance(route_rewards, list)
            or not isinstance(route_prompt_tokens, list)
            or not isinstance(route_output_tokens, list)
            or len(pred_scores) != target_dim
            or len(route_rewards) != target_dim
            or len(route_prompt_tokens) != target_dim
            or len(route_output_tokens) != target_dim
        ):
            continue
        preds.append([float(value) for value in pred_scores])
        rewards.append([float(value) for value in route_rewards])
        prompt_tokens.append([float(value) for value in route_prompt_tokens])
        output_tokens.append([float(value) for value in route_output_tokens])
    if not preds:
        raise ValueError(f"No joined {split} examples")
    return (
        np.asarray(preds, dtype=np.float64),
        np.asarray(rewards, dtype=np.float64),
        np.asarray(prompt_tokens, dtype=np.float64),
        np.asarray(output_tokens, dtype=np.float64),
    )


def _candidate_thresholds(scores: np.ndarray, max_candidates: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, int(max_candidates))
    values = np.quantile(scores.astype(np.float64), quantiles)
    eps = max(1.0e-6, float(np.std(scores)) * 1.0e-6)
    return np.asarray(
        sorted(set([float(np.min(scores) - eps), *[float(value) for value in values], float(np.max(scores) + eps)])),
        dtype=np.float64,
    )


def _route_costs(
    prompt_tokens: np.ndarray,
    output_tokens: np.ndarray,
    cost_metric: str,
    route_cost_weights: np.ndarray,
) -> np.ndarray:
    token_counts = output_tokens.astype(np.float64).copy()
    if cost_metric == "total_tokens":
        token_counts = token_counts + prompt_tokens
    return token_counts * route_cost_weights[None, :]


def _evaluate_thresholds(
    pred_scores: np.ndarray,
    rewards: np.ndarray,
    route_costs: np.ndarray,
    thresholds: list[float],
    order: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(thresholds) != len(order) - 1:
        raise ValueError(f"Expected {len(order) - 1} thresholds for order length {len(order)}")

    choices = np.full(pred_scores.shape[0], int(order[-1]), dtype=np.int64)
    called = np.zeros((pred_scores.shape[0], pred_scores.shape[1]), dtype=bool)
    still_running = np.ones(pred_scores.shape[0], dtype=bool)
    for route_idx, threshold in zip(order[:-1], thresholds, strict=True):
        route_idx = int(route_idx)
        called[still_running, route_idx] = True
        should_stop = still_running & (pred_scores[:, route_idx] >= float(threshold))
        choices[should_stop] = route_idx
        still_running = still_running & (~should_stop)
    called[still_running, int(order[-1])] = True
    chosen_rewards = rewards[np.arange(rewards.shape[0]), choices]
    total_costs = np.sum(called.astype(np.float64) * route_costs, axis=1)
    return choices, called, chosen_rewards, total_costs


def _optimize_thresholds(
    *,
    pred_scores: np.ndarray,
    rewards: np.ndarray,
    route_costs: np.ndarray,
    lambda_value: float,
    max_threshold_candidates: int,
    order: list[int],
) -> tuple[list[float], float]:
    if len(order) == 4:
        return _optimize_thresholds_four_route(
            pred_scores=pred_scores,
            rewards=rewards,
            route_costs=route_costs,
            lambda_value=lambda_value,
            max_threshold_candidates=max_threshold_candidates,
            order=order,
        )
    if len(order) != 3:
        return _optimize_thresholds_bruteforce(
            pred_scores=pred_scores,
            rewards=rewards,
            route_costs=route_costs,
            lambda_value=lambda_value,
            max_threshold_candidates=max_threshold_candidates,
            order=order,
        )
    first_idx, second_idx, third_idx = [int(idx) for idx in order]
    candidates_0 = _candidate_thresholds(pred_scores[:, first_idx], max_threshold_candidates)
    candidates_1 = _candidate_thresholds(pred_scores[:, second_idx], max_threshold_candidates)
    best_thresholds = [float(candidates_0[0]), float(candidates_1[0])]
    best_utility = -float("inf")

    reward_first = rewards[:, first_idx]
    reward_second = rewards[:, second_idx]
    reward_third = rewards[:, third_idx]
    cost_first = route_costs[:, first_idx]
    cost_second = route_costs[:, second_idx]
    cost_third = route_costs[:, third_idx]
    pred_first = pred_scores[:, first_idx]
    pred_second = pred_scores[:, second_idx]

    for threshold_0 in candidates_0:
        stop_at_first = pred_first >= float(threshold_0)
        base_rewards = np.where(stop_at_first, reward_first, reward_third)[:, None]
        base_costs = np.where(stop_at_first, cost_first, cost_first + cost_second + cost_third)[:, None]
        stop_at_second = (~stop_at_first)[:, None] & (pred_second[:, None] >= candidates_1[None, :])
        chosen_rewards = np.where(stop_at_second, reward_second[:, None], base_rewards)
        chosen_costs = np.where(stop_at_second, (cost_first + cost_second)[:, None], base_costs)
        utilities = np.mean(chosen_rewards - (float(lambda_value) * chosen_costs), axis=0)
        best_idx = int(np.argmax(utilities))
        if float(utilities[best_idx]) > best_utility:
            best_utility = float(utilities[best_idx])
            best_thresholds = [float(threshold_0), float(candidates_1[best_idx])]
    return best_thresholds, best_utility


def _optimize_thresholds_four_route(
    *,
    pred_scores: np.ndarray,
    rewards: np.ndarray,
    route_costs: np.ndarray,
    lambda_value: float,
    max_threshold_candidates: int,
    order: list[int],
) -> tuple[list[float], float]:
    first_idx, second_idx, third_idx, fourth_idx = [int(idx) for idx in order]
    candidates_0 = _candidate_thresholds(pred_scores[:, first_idx], max_threshold_candidates)
    candidates_1 = _candidate_thresholds(pred_scores[:, second_idx], max_threshold_candidates)
    candidates_2 = _candidate_thresholds(pred_scores[:, third_idx], max_threshold_candidates)
    best_thresholds = [float(candidates_0[0]), float(candidates_1[0]), float(candidates_2[0])]
    best_utility = -float("inf")

    reward_first = rewards[:, first_idx]
    reward_second = rewards[:, second_idx]
    reward_third = rewards[:, third_idx]
    reward_fourth = rewards[:, fourth_idx]
    cost_first = route_costs[:, first_idx]
    cost_second = route_costs[:, second_idx]
    cost_third = route_costs[:, third_idx]
    cost_fourth = route_costs[:, fourth_idx]
    pred_first = pred_scores[:, first_idx]
    pred_second = pred_scores[:, second_idx]
    pred_third = pred_scores[:, third_idx]

    for threshold_0 in candidates_0:
        stop_at_first = pred_first >= float(threshold_0)
        reached_second = ~stop_at_first
        for threshold_1 in candidates_1:
            stop_at_second = reached_second & (pred_second >= float(threshold_1))
            reached_third = reached_second & (~stop_at_second)
            base_rewards = np.where(stop_at_first, reward_first, np.where(stop_at_second, reward_second, reward_fourth))[:, None]
            base_costs = np.where(
                stop_at_first,
                cost_first,
                np.where(stop_at_second, cost_first + cost_second, cost_first + cost_second + cost_third + cost_fourth),
            )[:, None]
            stop_at_third = reached_third[:, None] & (pred_third[:, None] >= candidates_2[None, :])
            chosen_rewards = np.where(stop_at_third, reward_third[:, None], base_rewards)
            chosen_costs = np.where(stop_at_third, (cost_first + cost_second + cost_third)[:, None], base_costs)
            utilities = np.mean(chosen_rewards - (float(lambda_value) * chosen_costs), axis=0)
            best_idx = int(np.argmax(utilities))
            if float(utilities[best_idx]) > best_utility:
                best_utility = float(utilities[best_idx])
                best_thresholds = [float(threshold_0), float(threshold_1), float(candidates_2[best_idx])]
    return best_thresholds, best_utility


def _optimize_thresholds_bruteforce(
    *,
    pred_scores: np.ndarray,
    rewards: np.ndarray,
    route_costs: np.ndarray,
    lambda_value: float,
    max_threshold_candidates: int,
    order: list[int],
) -> tuple[list[float], float]:
    if len(order) < 2:
        raise ValueError("Cascade order must contain at least two routes")
    candidate_lists = [
        _candidate_thresholds(pred_scores[:, int(route_idx)], max_threshold_candidates)
        for route_idx in order[:-1]
    ]
    best_thresholds = [float(candidates[0]) for candidates in candidate_lists]
    best_utility = -float("inf")
    for thresholds in itertools.product(*candidate_lists):
        _, _, chosen_rewards, total_costs = _evaluate_thresholds(
            pred_scores,
            rewards,
            route_costs,
            [float(value) for value in thresholds],
            order,
        )
        utility = float(np.mean(chosen_rewards - (float(lambda_value) * total_costs)))
        if utility > best_utility:
            best_utility = utility
            best_thresholds = [float(value) for value in thresholds]
    return best_thresholds, best_utility


def _summarize_cascade(
    *,
    pred_scores: np.ndarray,
    rewards: np.ndarray,
    route_costs: np.ndarray,
    route_labels: list[str],
    lambda_value: float,
    cost_metric: str,
    thresholds: list[float],
    order: list[int],
    train_utility: float,
) -> dict[str, Any]:
    choices, called, chosen_rewards, total_costs = _evaluate_thresholds(
        pred_scores,
        rewards,
        route_costs,
        thresholds,
        order,
    )
    mean_reward = float(np.mean(chosen_rewards))
    mean_cost = float(np.mean(total_costs))
    return {
        "policy": "cascade_train_tau_weighted",
        "policy_type": "cascade_train_thresholds",
        "lambda": float(lambda_value),
        "cost_metric": cost_metric,
        "cost_accounting": "sequential_cascade",
        "mean_true_reward": mean_reward,
        "mean_true_cost": mean_cost,
        "mean_true_utility": mean_reward - (float(lambda_value) * mean_cost),
        "choice_counts_by_route": json.dumps(
            {route_labels[idx]: int(np.sum(choices == idx)) for idx in range(len(route_labels))},
            sort_keys=True,
        ),
        "called_counts_by_route": json.dumps(
            {route_labels[idx]: int(np.sum(called[:, idx])) for idx in range(len(route_labels))},
            sort_keys=True,
        ),
        "thresholds": json.dumps([float(value) for value in thresholds]),
        "cascade_order": json.dumps([int(idx) for idx in order]),
        "cascade_order_labels": json.dumps([route_labels[int(idx)] for idx in order]),
        "train_utility": float(train_utility),
    }


def _candidate_orders(target_dim: int, search_orders: bool) -> list[list[int]]:
    if not search_orders:
        return [list(range(target_dim))]
    return [list(order) for order in itertools.permutations(range(target_dim))]


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _load_direct_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            policy = str(row["policy"])
            if row["cost_metric"] != "output_tokens":
                continue
            if not (
                policy.startswith("direct_router")
                or policy.startswith("reward_only_router")
                or policy.startswith("always_direct::")
                or policy.startswith("oracle_")
            ):
                continue
            rows.append(
                {
                    "policy": row["policy"],
                    "policy_type": row["policy_type"],
                    "lambda": float(row["lambda"]),
                    "cost_metric": row["cost_metric"],
                    "cost_accounting": row["cost_accounting"],
                    "mean_true_reward": float(row["mean_true_reward"]),
                    "mean_true_cost": float(row["mean_true_cost"]),
                    "mean_true_utility": float(row["mean_true_utility"]),
                    "choice_counts_by_route": row["choice_counts_by_route"],
                    "called_counts_by_route": "",
                }
            )
    return rows


def _write_combined_plot(combined_rows: list[dict[str, Any]], output_dir: Path) -> None:
    plot_policies = {
        "direct_router_after_primary_pred_reward_pred_cost_weighted": "Direct router",
        "cascade_train_tau_weighted": "Cascade",
        "always_direct::primary_model": "Always primary",
        "always_direct::expert_0:openai/gpt-oss-120b": "Always OSS",
        "always_direct::expert_0:google/gemini-3-flash-preview": "Always Gemini",
        "oracle_after_primary_utility_weighted": "Oracle after primary",
        "oracle_direct_utility_weighted": "Oracle direct",
    }
    colors = {
        "Direct router": "#1f77b4",
        "Cascade": "#e377c2",
        "Always primary": "#2ca02c",
        "Always OSS": "#ff7f0e",
        "Always Gemini": "#d62728",
        "Oracle after primary": "#9467bd",
        "Oracle direct": "#7f7f7f",
    }
    lambdas = sorted({float(row["lambda"]) for row in combined_rows})
    x_positions = list(range(len(lambdas)))
    x_labels = ["0" if value == 0.0 else f"{value:g}" for value in lambdas]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), constrained_layout=True)

    ax = axes[0]
    for policy, label in plot_policies.items():
        rows_by_lambda = {
            float(row["lambda"]): row for row in combined_rows if row["policy"] == policy
        }
        if not rows_by_lambda:
            continue
        y_values = [float(rows_by_lambda[value]["mean_true_utility"]) for value in lambdas]
        ax.plot(
            x_positions,
            y_values,
            marker="o",
            linewidth=3.0 if label in {"Direct router", "Cascade"} else 1.8,
            linestyle="--" if label.startswith("Oracle") else "-",
            label=label,
            color=colors[label],
        )
    ax.set_title("Realized utility, weighted output-token costs")
    ax.set_xlabel("lambda")
    ax.set_ylabel("mean true utility")
    ax.set_xticks(x_positions, x_labels)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    selected_lambda = 2.0e-5
    for policy, label in plot_policies.items():
        if label == "Oracle direct":
            continue
        matches = [
            row
            for row in combined_rows
            if row["policy"] == policy and abs(float(row["lambda"]) - selected_lambda) < 1.0e-12
        ]
        if not matches:
            continue
        row = matches[0]
        ax.scatter(float(row["mean_true_cost"]), float(row["mean_true_reward"]), s=90, color=colors[label])
        ax.annotate(
            label,
            (float(row["mean_true_cost"]), float(row["mean_true_reward"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("Reward/cost position at lambda=2e-5")
    ax.set_xlabel("mean true weighted output cost")
    ax.set_ylabel("mean true reward")
    ax.grid(True, alpha=0.25)
    fig.suptitle(
        "3-way direct router vs sequential cascade, weighted output-token costs",
        fontsize=12,
    )
    fig.savefig(output_dir / "direct_vs_cascade_weighted_output_utility.png", dpi=180)
    fig.savefig(output_dir / "direct_vs_cascade_weighted_output_utility.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cascade-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--direct-utility-csv", type=Path, default=None)
    parser.add_argument("--route-labels", default=",".join(DEFAULT_ROUTE_LABELS))
    parser.add_argument("--route-cost-weights", default=None)
    parser.add_argument("--lambdas", default=",".join(str(value) for value in DEFAULT_LAMBDAS))
    parser.add_argument("--cost-metrics", default="output_tokens,total_tokens")
    parser.add_argument("--max-threshold-candidates", type=int, default=51)
    parser.add_argument(
        "--search-orders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search over all route permutations and select the best train utility order for each lambda/cost metric.",
    )
    args = parser.parse_args()

    route_labels = [part.strip() for part in str(args.route_labels).split(",") if part.strip()]
    route_cost_weights = np.asarray(_parse_cost_weights(args.route_cost_weights, route_labels), dtype=np.float64)
    lambdas = _parse_float_list(str(args.lambdas))
    cost_metrics = _parse_cost_metrics(str(args.cost_metrics))
    target_dim = len(route_labels)

    train_pred, train_rewards, train_prompt_tokens, train_output_tokens = _build_arrays(
        dataset_dir=args.dataset_dir,
        split="train",
        pred_by_key=_read_attempt_scores(args.cascade_dir / "train_attempt_predictions.jsonl", target_dim),
        target_dim=target_dim,
    )
    eval_pred, eval_rewards, eval_prompt_tokens, eval_output_tokens = _build_arrays(
        dataset_dir=args.dataset_dir,
        split="eval",
        pred_by_key=_read_attempt_scores(args.cascade_dir / "eval_attempt_predictions.jsonl", target_dim),
        target_dim=target_dim,
    )

    rows: list[dict[str, Any]] = []
    learned_thresholds: dict[str, Any] = {}
    candidate_orders = _candidate_orders(target_dim, bool(args.search_orders))
    for lambda_value in lambdas:
        for cost_metric in cost_metrics:
            train_costs = _route_costs(train_prompt_tokens, train_output_tokens, cost_metric, route_cost_weights)
            eval_costs = _route_costs(eval_prompt_tokens, eval_output_tokens, cost_metric, route_cost_weights)
            best_order: list[int] | None = None
            best_thresholds: list[float] | None = None
            best_train_utility = -float("inf")
            order_results: list[dict[str, Any]] = []
            for order in candidate_orders:
                thresholds, train_utility = _optimize_thresholds(
                    pred_scores=train_pred,
                    rewards=train_rewards,
                    route_costs=train_costs,
                    lambda_value=float(lambda_value),
                    max_threshold_candidates=int(args.max_threshold_candidates),
                    order=order,
                )
                order_results.append(
                    {
                        "order": [int(idx) for idx in order],
                        "order_labels": [route_labels[int(idx)] for idx in order],
                        "thresholds": thresholds,
                        "train_utility": train_utility,
                    }
                )
                if train_utility > best_train_utility:
                    best_train_utility = train_utility
                    best_thresholds = thresholds
                    best_order = [int(idx) for idx in order]
            if best_order is None or best_thresholds is None:
                raise ValueError("No cascade order was evaluated")
            rows.append(
                _summarize_cascade(
                    pred_scores=eval_pred,
                    rewards=eval_rewards,
                    route_costs=eval_costs,
                    route_labels=route_labels,
                    lambda_value=float(lambda_value),
                    cost_metric=cost_metric,
                    thresholds=best_thresholds,
                    order=best_order,
                    train_utility=best_train_utility,
                )
            )
            learned_thresholds[f"{cost_metric}::{float(lambda_value):g}"] = {
                "order": best_order,
                "order_labels": [route_labels[int(idx)] for idx in best_order],
                "thresholds": best_thresholds,
                "train_utility": best_train_utility,
                "lambda": float(lambda_value),
                "cost_metric": cost_metric,
                "searched_orders": order_results,
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cascade_headers = [
        "policy",
        "policy_type",
        "lambda",
        "cost_metric",
        "cost_accounting",
        "mean_true_reward",
        "mean_true_cost",
        "mean_true_utility",
        "choice_counts_by_route",
        "called_counts_by_route",
        "thresholds",
        "cascade_order",
        "cascade_order_labels",
        "train_utility",
    ]
    _write_csv(args.output_dir / "cascade_weighted_cost_utility.csv", rows, cascade_headers)
    (args.output_dir / "cascade_weighted_cost_utility.json").write_text(
        json.dumps(
            {
                "cascade_dir": str(args.cascade_dir),
                "dataset_dir": str(args.dataset_dir),
                "n_train": int(train_pred.shape[0]),
                "n_eval": int(eval_pred.shape[0]),
                "route_labels": route_labels,
                "route_cost_weights": {
                    label: float(weight) for label, weight in zip(route_labels, route_cost_weights, strict=True)
                },
                "search_orders": bool(args.search_orders),
                "max_threshold_candidates": int(args.max_threshold_candidates),
                "learned_thresholds": learned_thresholds,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )

    if args.direct_utility_csv is not None:
        combined_rows = _load_direct_rows(args.direct_utility_csv)
        combined_rows.extend(
            {
                "policy": row["policy"],
                "policy_type": row["policy_type"],
                "lambda": float(row["lambda"]),
                "cost_metric": row["cost_metric"],
                "cost_accounting": row["cost_accounting"],
                "mean_true_reward": float(row["mean_true_reward"]),
                "mean_true_cost": float(row["mean_true_cost"]),
                "mean_true_utility": float(row["mean_true_utility"]),
                "choice_counts_by_route": row["choice_counts_by_route"],
                "called_counts_by_route": row["called_counts_by_route"],
                "cascade_order": row["cascade_order"],
                "cascade_order_labels": row["cascade_order_labels"],
            }
            for row in rows
            if row["cost_metric"] == "output_tokens"
        )
        combined_rows = sorted(combined_rows, key=lambda row: (float(row["lambda"]), str(row["policy"])))
        combined_headers = [
            "policy",
            "policy_type",
            "lambda",
            "cost_metric",
            "cost_accounting",
            "mean_true_reward",
            "mean_true_cost",
            "mean_true_utility",
            "choice_counts_by_route",
            "called_counts_by_route",
            "cascade_order",
            "cascade_order_labels",
        ]
        _write_csv(args.output_dir / "combined_direct_cascade_weighted_output_utility.csv", combined_rows, combined_headers)
        _write_combined_plot(combined_rows, args.output_dir)

    print(args.output_dir / "cascade_weighted_cost_utility.csv")
    if args.direct_utility_csv is not None:
        print(args.output_dir / "combined_direct_cascade_weighted_output_utility.csv")
        print(args.output_dir / "direct_vs_cascade_weighted_output_utility.png")
        print(args.output_dir / "direct_vs_cascade_weighted_output_utility.pdf")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure whether a Q router spends budget on better instances than random.

This analysis uses deployable direct-Q decisions: predicted expert costs plus a
fixed scout cost for action selection, and realized costs/rewards for reporting.
It compares Q to two instance-agnostic random baselines:

1. Same-route-mix random: preserve Q's route selection counts, but randomize
   which instances get each route. This isolates instance selection quality.
2. Direct-random frontier: the best randomized direct policy under the same
   mean realized cost cap. This is the stronger budget frontier baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_q_budget_frontier import Point, _best_random_direct_at_cap
from diagnose_cost_oracle_gap import (
    DEFAULT_COST_RUN,
    DEFAULT_DATASET_DIR,
    DEFAULT_Q_RUN,
    align_split,
    load_predicted_costs,
    read_bare_state_rows,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_lambdas(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def load_eval_arrays(
    *,
    q_run: Path,
    cost_run: Path,
    dataset_dir: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    q_summary = json.loads((q_run / "summary.json").read_text())
    config = q_summary["config"]
    route_labels = list(config["route_labels"])
    fixed_costs = np.asarray(config["fixed_train_route_costs"], dtype=np.float64)
    route_cost_weights = np.asarray(config["route_output_cost_weights"], dtype=np.float64)

    pred_rows = read_bare_state_rows(q_run / "eval_state_predictions.jsonl")
    q_probs = np.asarray([row["pred_success_probs"] for row in pred_rows], dtype=np.float64)
    rewards, realized_costs, problem_ids = align_split(
        pred_rows=pred_rows,
        dataset_dir=dataset_dir,
        split="eval",
        route_cost_weights=route_cost_weights,
    )
    dumped_rewards = np.asarray([row["true_rewards"] for row in pred_rows], dtype=np.float64)
    if not np.allclose(rewards, dumped_rewards):
        raise ValueError("Q eval dump rewards do not match real-label dataset rewards")

    predicted_costs = load_predicted_costs(
        cost_run=cost_run,
        split="eval",
        problem_ids=problem_ids,
        route_count=len(route_labels),
        route_cost_weights=route_cost_weights,
        scout_cost=float(fixed_costs[0]),
    )
    if predicted_costs is None:
        raise FileNotFoundError(f"Missing eval cost predictions under {cost_run}")
    return route_labels, q_probs, rewards, realized_costs, problem_ids, predicted_costs


def route_counts_json(choices: np.ndarray, route_labels: list[str]) -> str:
    return json.dumps(
        {route_labels[idx]: int(np.sum(choices == idx)) for idx in range(len(route_labels))},
        sort_keys=True,
    )


def permutation_reward_null(
    *,
    rewards: np.ndarray,
    choices: np.ndarray,
    observed_reward: float,
    n_perm: int,
    seed: int,
) -> tuple[float, float, float]:
    if n_perm <= 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    values = np.empty(n_perm, dtype=np.float64)
    n = len(choices)
    row_idx = np.arange(n)
    for perm_idx in range(n_perm):
        shuffled = choices[rng.permutation(n)]
        values[perm_idx] = float(np.mean(rewards[row_idx, shuffled]))
    std = float(np.std(values))
    z = math.nan if std == 0.0 else float((observed_reward - np.mean(values)) / std)
    # One-sided: how often same-route-mix random is at least as good as Q.
    p_ge = float((1.0 + np.sum(values >= observed_reward)) / (n_perm + 1.0))
    return float(np.mean(values)), std, p_ge if math.isfinite(z) else math.nan


def analyze(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    route_labels, q_probs, rewards, realized_costs, problem_ids, predicted_costs = load_eval_arrays(
        q_run=args.q_run,
        cost_run=args.cost_run,
        dataset_dir=args.dataset_dir,
    )
    del problem_ids

    route_mean_rewards = np.mean(rewards, axis=0)
    route_mean_costs = np.mean(realized_costs, axis=0)
    scout_reward = rewards[:, 0]
    route_delta_vs_scout = rewards - scout_reward[:, None]
    route_mean_delta_vs_scout = np.mean(route_delta_vs_scout, axis=0)
    route_points = [
        Point("Always direct", float(route_mean_rewards[idx]), float(route_mean_costs[idx]), "eval_means", label, 0.0)
        for idx, label in enumerate(route_labels)
    ]
    cost_20b = float(route_mean_costs[1])
    cost_120b = float(route_mean_costs[3])
    n = rewards.shape[0]
    row_idx = np.arange(n)

    summary_rows: list[dict[str, Any]] = []
    per_route_rows: list[dict[str, Any]] = []
    for lambda_value in parse_lambdas(args.lambdas):
        choices = np.argmax(q_probs - lambda_value * predicted_costs, axis=1)
        selected_rewards = rewards[row_idx, choices]
        selected_costs = realized_costs[row_idx, choices]
        selected_delta = route_delta_vs_scout[row_idx, choices]
        q_reward = float(np.mean(selected_rewards))
        q_cost = float(np.mean(selected_costs))

        mix_probs = np.bincount(choices, minlength=len(route_labels)).astype(np.float64) / n
        same_mix_reward = float(np.dot(mix_probs, route_mean_rewards))
        same_mix_cost = float(np.dot(mix_probs, route_mean_costs))
        same_mix_delta = float(np.dot(mix_probs, route_mean_delta_vs_scout))
        q_delta = float(np.mean(selected_delta))

        random_front = _best_random_direct_at_cap(route_points, q_cost)
        random_front_reward = math.nan if random_front is None else float(random_front.reward)
        random_front_cost = math.nan if random_front is None else float(random_front.cost)
        random_front_label = "" if random_front is None else random_front.label

        null_mean, null_std, null_p_ge = permutation_reward_null(
            rewards=rewards,
            choices=choices,
            observed_reward=q_reward,
            n_perm=args.permutations,
            seed=args.seed + int(lambda_value * 1000),
        )
        null_z = math.nan if null_std == 0.0 else float((q_reward - null_mean) / null_std)

        non_scout = choices != 0
        if np.any(non_scout):
            spend_delta_mean = float(np.mean(selected_delta[non_scout]))
            spend_cost_mean = float(np.mean((selected_costs - realized_costs[:, 0])[non_scout]))
        else:
            spend_delta_mean = math.nan
            spend_cost_mean = math.nan

        summary_rows.append(
            {
                "lambda": lambda_value,
                "q_mean_reward": q_reward,
                "q_mean_cost": q_cost,
                "q_cost_pct_of_20b": 100.0 * q_cost / cost_20b,
                "q_cost_pct_of_120b": 100.0 * q_cost / cost_120b,
                "same_route_mix_random_reward": same_mix_reward,
                "same_route_mix_random_cost": same_mix_cost,
                "q_reward_lift_vs_same_route_mix_random": q_reward - same_mix_reward,
                "q_delta_vs_scout": q_delta,
                "same_route_mix_delta_vs_scout": same_mix_delta,
                "q_delta_lift_vs_same_route_mix_random": q_delta - same_mix_delta,
                "random_frontier_reward_at_q_cost": random_front_reward,
                "random_frontier_cost_at_q_cost": random_front_cost,
                "q_reward_lift_vs_direct_random_frontier": q_reward - random_front_reward,
                "same_route_mix_permutation_reward_mean": null_mean,
                "same_route_mix_permutation_reward_std": null_std,
                "same_route_mix_permutation_reward_z": null_z,
                "same_route_mix_permutation_p_ge_q_reward": null_p_ge,
                "non_scout_call_rate": float(np.mean(non_scout)),
                "non_scout_selected_delta_vs_scout_mean": spend_delta_mean,
                "non_scout_selected_incremental_cost_mean": spend_cost_mean,
                "choice_counts_by_route": route_counts_json(choices, route_labels),
                "direct_random_frontier_label": random_front_label,
            }
        )

        for route_idx, route_label in enumerate(route_labels):
            mask = choices == route_idx
            route_count = int(np.sum(mask))
            row: dict[str, Any] = {
                "lambda": lambda_value,
                "route_idx": route_idx,
                "route_label": route_label,
                "selected_count": route_count,
                "selected_frac": route_count / n,
                "global_route_reward_mean": float(route_mean_rewards[route_idx]),
                "global_delta_vs_scout_mean": float(route_mean_delta_vs_scout[route_idx]),
                "global_route_cost_mean": float(route_mean_costs[route_idx]),
            }
            if route_count:
                row.update(
                    {
                        "selected_route_reward_mean": float(np.mean(rewards[mask, route_idx])),
                        "selected_delta_vs_scout_mean": float(np.mean(route_delta_vs_scout[mask, route_idx])),
                        "selected_route_cost_mean": float(np.mean(realized_costs[mask, route_idx])),
                        "selected_reward_lift_vs_global_route_mean": float(
                            np.mean(rewards[mask, route_idx]) - route_mean_rewards[route_idx]
                        ),
                        "selected_delta_lift_vs_global_route_mean": float(
                            np.mean(route_delta_vs_scout[mask, route_idx]) - route_mean_delta_vs_scout[route_idx]
                        ),
                    }
                )
            per_route_rows.append(row)

    return summary_rows, per_route_rows


def plot_summary(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(summary_rows, key=lambda row: row["q_cost_pct_of_20b"])
    x = [row["q_cost_pct_of_20b"] for row in rows]
    q = [row["q_mean_reward"] for row in rows]
    same = [row["same_route_mix_random_reward"] for row in rows]
    frontier = [row["random_frontier_reward_at_q_cost"] for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(x, q, marker="o", linewidth=2.0, label="Q direct")
    ax.plot(x, same, marker="o", linewidth=1.8, label="Random with Q route mix")
    ax.plot(x, frontier, marker="o", linewidth=1.8, label="Direct-random frontier")
    ax.set_xlabel("Realized mean cost (% of always-20B)")
    ax.set_ylabel("Mean real reward / pass rate")
    ax.set_xlim(0, max(105.0, max(x) * 1.05))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "q_spend_quality_vs_random_20b.png", dpi=180)
    fig.savefig(out_dir / "q_spend_quality_vs_random_20b.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    ax.plot(
        x,
        [row["q_reward_lift_vs_same_route_mix_random"] for row in rows],
        marker="o",
        linewidth=2.0,
        label="Q lift vs same-route-mix random",
    )
    ax.plot(
        x,
        [row["q_reward_lift_vs_direct_random_frontier"] for row in rows],
        marker="o",
        linewidth=2.0,
        label="Q lift vs direct-random frontier",
    )
    ax.set_xlabel("Realized mean cost (% of always-20B)")
    ax.set_ylabel("Mean reward lift")
    ax.set_xlim(0, max(105.0, max(x) * 1.05))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "q_spend_quality_lift_20b.png", dpi=180)
    fig.savefig(out_dir / "q_spend_quality_lift_20b.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-run", type=Path, default=DEFAULT_Q_RUN)
    parser.add_argument("--cost-run", type=Path, default=DEFAULT_COST_RUN)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("router_analysis/swe_smith_train1500_real_4route_q_spend_quality_178108"),
    )
    parser.add_argument("--lambdas", default="0,5,10,15,20,25,30,35,40,45,50,55,60,75,100,150,200")
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, per_route_rows = analyze(args)
    write_csv(args.output_dir / "spend_quality_summary.csv", summary_rows)
    write_csv(args.output_dir / "spend_quality_per_route.csv", per_route_rows)
    plot_summary(summary_rows, args.output_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "q_run": str(args.q_run),
                "cost_run": str(args.cost_run),
                "dataset_dir": str(args.dataset_dir),
                "outputs": {
                    "summary": str(args.output_dir / "spend_quality_summary.csv"),
                    "per_route": str(args.output_dir / "spend_quality_per_route.csv"),
                    "plot_reward": str(args.output_dir / "q_spend_quality_vs_random_20b.png"),
                    "plot_lift": str(args.output_dir / "q_spend_quality_lift_20b.png"),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Wrote {args.output_dir}")
    for row in summary_rows:
        if row["lambda"] in {25.0, 35.0, 50.0, 75.0, 100.0}:
            print(
                row["lambda"],
                {
                    "cost_pct_20b": f"{row['q_cost_pct_of_20b']:.1f}",
                    "q_reward": f"{row['q_mean_reward']:.3f}",
                    "same_mix_random": f"{row['same_route_mix_random_reward']:.3f}",
                    "direct_random_frontier": f"{row['random_frontier_reward_at_q_cost']:.3f}",
                    "lift_same_mix": f"{row['q_reward_lift_vs_same_route_mix_random']:.3f}",
                    "lift_frontier": f"{row['q_reward_lift_vs_direct_random_frontier']:.3f}",
                    "perm_p_ge": f"{row['same_route_mix_permutation_p_ge_q_reward']:.4f}",
                },
            )


if __name__ == "__main__":
    main()

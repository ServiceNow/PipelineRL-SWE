#!/usr/bin/env python3
"""Compare a state-policy utility CSV against a direct-random frontier.

The random baseline is instance-agnostic: mix always-route policies to satisfy
an average cost cap, then take the best expected reward on that convex hull.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _direct_random_at_cost(always_points: list[dict[str, float | str]], cap: float) -> dict[str, float | str] | None:
    candidates: list[dict[str, float | str]] = []
    for point in always_points:
        if float(point["cost"]) <= cap + 1.0e-15:
            candidates.append(
                {
                    "reward": float(point["reward"]),
                    "cost": float(point["cost"]),
                    "label": str(point["label"]),
                }
            )
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


def _always_points(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("policy_type") != "always_direct":
            continue
        policy = row["policy"]
        if policy in seen:
            continue
        seen.add(policy)
        out.append(
            {
                "label": policy.replace("always::", ""),
                "reward": float(row["mean_reward"]),
                "cost": float(row["mean_cost"]),
            }
        )
    return sorted(out, key=lambda row: float(row["cost"]))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-policy-utility", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--budget-fracs",
        default="0.25,0.3,0.4,0.5,0.75,1.0",
        help="Comma-separated budget caps as fractions of the most expensive always-route cost.",
    )
    args = parser.parse_args()

    rows = _read_csv(args.state_policy_utility)
    always = _always_points(rows)
    if len(always) < 2:
        raise ValueError("Need at least two always_direct rows")
    baseline_cost = max(float(row["cost"]) for row in always)
    baseline_reward = max(always, key=lambda row: float(row["cost"]))["reward"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    point_rows: list[dict[str, object]] = []
    learned_rows = [
        row
        for row in rows
        if row.get("policy_type") not in {"always_direct", "oracle_direct", "oracle_direct_fixed_cost_selection", "empty"}
    ]
    for row in learned_rows:
        cost = float(row["mean_cost"])
        random = _direct_random_at_cost(always, cost)
        if random is None:
            continue
        reward = float(row["mean_reward"])
        point_rows.append(
            {
                "policy": row["policy"],
                "policy_type": row["policy_type"],
                "lambda": row["lambda"],
                "mean_reward": reward,
                "mean_cost": cost,
                "cost_frac_of_expensive": cost / baseline_cost,
                "oracle_match_rate": row.get("oracle_match_rate", ""),
                "direct_random_reward_at_same_cost": random["reward"],
                "gain_vs_direct_random_same_cost": reward - float(random["reward"]),
                "direct_random_mix": random["label"],
                "choice_counts_by_route": row.get("choice_counts_by_route", ""),
            }
        )
    _write_csv(args.output_dir / "state_policy_vs_direct_random_points.csv", point_rows)

    budget_rows: list[dict[str, object]] = []
    budget_fracs = [float(value) for value in str(args.budget_fracs).split(",") if value.strip()]
    for frac in budget_fracs:
        cap = float(frac) * baseline_cost
        feasible = [row for row in point_rows if float(row["mean_cost"]) <= cap + 1.0e-15]
        random = _direct_random_at_cost(always, cap)
        if random is None:
            continue
        if feasible:
            best = max(feasible, key=lambda row: (float(row["mean_reward"]), -float(row["mean_cost"])))
            budget_rows.append(
                {
                    "budget_frac_of_expensive": frac,
                    "cap_cost": cap,
                    "best_policy": best["policy"],
                    "best_lambda": best["lambda"],
                    "best_reward": best["mean_reward"],
                    "best_cost": best["mean_cost"],
                    "best_cost_frac_of_expensive": best["cost_frac_of_expensive"],
                    "direct_random_reward": random["reward"],
                    "gain_vs_direct_random": float(best["mean_reward"]) - float(random["reward"]),
                    "direct_random_mix": random["label"],
                }
            )
        else:
            budget_rows.append(
                {
                    "budget_frac_of_expensive": frac,
                    "cap_cost": cap,
                    "best_policy": "",
                    "best_lambda": "",
                    "best_reward": "",
                    "best_cost": "",
                    "best_cost_frac_of_expensive": "",
                    "direct_random_reward": random["reward"],
                    "gain_vs_direct_random": "",
                    "direct_random_mix": random["label"],
                }
            )
    _write_csv(args.output_dir / "state_policy_vs_direct_random_budget_frontier.csv", budget_rows)

    print("Always routes:")
    for row in always:
        print(f"  {row['label']}: reward={float(row['reward']):.4f}, cost={float(row['cost']):.6g}")
    print("\nBest learned points by gain vs direct-random at same cost:")
    for row in sorted(point_rows, key=lambda row: float(row["gain_vs_direct_random_same_cost"]), reverse=True)[:10]:
        print(
            f"  gain={float(row['gain_vs_direct_random_same_cost']):+.4f} "
            f"reward={float(row['mean_reward']):.4f} "
            f"cost_frac={float(row['cost_frac_of_expensive']):.3f} "
            f"lambda={row['lambda']} policy={row['policy']}"
        )
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Budget-cap frontiers for the SWE-Smith real-label Q-policy runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Point:
    method: str
    reward: float
    cost: float
    source: str
    label: str
    lambda_value: float | None


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _dedup(points: Iterable[Point]) -> list[Point]:
    seen: set[tuple[str, float, float, str, float | None]] = set()
    out: list[Point] = []
    for p in points:
        key = (p.method, p.reward, p.cost, p.label, p.lambda_value)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _best_under_budget(points: list[Point], cap: float) -> Point | None:
    feasible = [p for p in points if p.cost <= cap]
    if not feasible:
        return None
    return max(feasible, key=lambda p: (p.reward, -p.cost))


def _best_random_direct_at_cap(always_points: list[Point], cap: float) -> Point | None:
    """Best instance-agnostic randomized direct policy under an average cost cap."""
    candidates: list[Point] = []
    for point in always_points:
        if point.cost <= cap + 1e-15:
            candidates.append(
                Point(
                    "Direct random",
                    point.reward,
                    point.cost,
                    "computed_convex_hull",
                    point.label,
                    None,
                )
            )

    for a, b in combinations(always_points, 2):
        if abs(a.cost - b.cost) < 1e-15:
            continue
        low, high = (a, b) if a.cost < b.cost else (b, a)
        if cap < low.cost - 1e-15:
            continue
        if cap > high.cost + 1e-15:
            continue
        high_weight = (cap - low.cost) / (high.cost - low.cost)
        high_weight = min(1.0, max(0.0, high_weight))
        low_weight = 1.0 - high_weight
        reward = low_weight * low.reward + high_weight * high.reward
        cost = low_weight * low.cost + high_weight * high.cost
        candidates.append(
            Point(
                "Direct random",
                reward,
                cost,
                "computed_convex_hull",
                f"{low_weight:.4f}*{low.label} + {high_weight:.4f}*{high.label}",
                None,
            )
        )

    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.reward, -p.cost))


def _add_direct_random_points(
    points: list[Point],
    *,
    baseline_cost: float,
    budget_pcts: list[float],
    plot_max_pcts: list[float],
) -> list[Point]:
    always_points = [p for p in points if p.method == "Always direct"]
    cap_pcts = set(budget_pcts)
    for max_pct in plot_max_pcts:
        cap_pcts.update(i * max_pct / 400.0 for i in range(401))
    random_points = []
    for pct in sorted(cap_pcts):
        cap = baseline_cost * pct / 100.0
        point = _best_random_direct_at_cap(always_points, cap)
        if point is not None:
            random_points.append(point)
    return _dedup([*points, *random_points])


def _load_points(args: argparse.Namespace) -> list[Point]:
    points: list[Point] = []

    q_rows = _read_csv(args.q_predicted_rows)
    for row in q_rows:
        policy = row["policy"]
        lam = _float_or_none(row.get("lambda"))
        reward = float(row["mean_reward"])
        cost = float(row["mean_cost"])
        if row["policy_type"] == "always_direct" and lam == 0.0:
            label = policy.replace("always::", "")
            points.append(Point("Always direct", reward, cost, "q_predicted_rows", label, lam))
        elif policy == "state_policy::flexible_bare_max4":
            points.append(Point("Q chain flexible (pred cost)", reward, cost, "q_predicted_rows", policy, lam))
        elif policy == "state_policy::forced_scout_0_max4":
            points.append(Point("Q chain forced-scout (pred cost)", reward, cost, "q_predicted_rows", policy, lam))

    diag_rows = _read_csv(args.q_diag_rows)
    cost_mode_to_method = {
        "predicted_expert_fixed_scout": "Q direct (pred cost)",
        "fixed_train_mean": "Q direct (fixed train cost)",
        "oracle_actual_cost": "Q direct (oracle cost diagnostic)",
    }
    for row in diag_rows:
        if row["split"] != "eval":
            continue
        reward = float(row["mean_reward"])
        cost = float(row["mean_cost"])
        lam = _float_or_none(row.get("lambda"))
        if row["policy"] == "q_bare_direct" and row["cost_mode"] in cost_mode_to_method:
            points.append(
                Point(
                    cost_mode_to_method[row["cost_mode"]],
                    reward,
                    cost,
                    "q_diag_rows",
                    f"lambda={lam:g}",
                    lam,
                )
            )
        elif row["policy"] == "oracle_true_success_and_cost":
            points.append(
                Point(
                    "Oracle route upper bound",
                    reward,
                    cost,
                    "q_diag_rows",
                    f"lambda={lam:g}",
                    lam,
                )
            )

    cascade_rows = _read_csv(args.cascade_rows)
    for row in cascade_rows:
        if row.get("policy") != "cascade_train_tau_weighted":
            continue
        lam = _float_or_none(row.get("lambda"))
        points.append(
            Point(
                "Cascade",
                float(row["mean_true_reward"]),
                float(row["mean_true_cost"]),
                "cascade_rows",
                f"lambda={lam:g}",
                lam,
            )
        )

    points = _dedup(points)
    aggregate_methods = {
        "Q direct (pred cost)",
        "Q chain flexible (pred cost)",
        "Q chain forced-scout (pred cost)",
    }
    points.extend(
        Point(
            "Best Q policy (pred cost)",
            p.reward,
            p.cost,
            p.source,
            f"{p.method}: {p.label}",
            p.lambda_value,
        )
        for p in points
        if p.method in aggregate_methods
    )
    return _dedup(points)


def _write_long_table(
    out_path: Path,
    points_by_method: dict[str, list[Point]],
    budget_pcts: list[float],
    baseline_cost: float,
    baseline_reward: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for budget_pct in budget_pcts:
        cap = baseline_cost * budget_pct / 100.0
        always = _best_under_budget(points_by_method["Always direct"], cap)
        random = _best_under_budget(points_by_method.get("Direct random", []), cap)
        cascade = _best_under_budget(points_by_method["Cascade"], cap)
        for method, points in points_by_method.items():
            best = _best_under_budget(points, cap)
            if best is None:
                continue
            rows.append(
                {
                    "budget_pct_of_120b_cost": f"{budget_pct:.3f}",
                    "cap_cost": f"{cap:.12g}",
                    "method": method,
                    "best_reward": f"{best.reward:.9f}",
                    "best_cost": f"{best.cost:.12g}",
                    "best_cost_pct_of_120b": f"{100.0 * best.cost / baseline_cost:.3f}",
                    "reward_pct_of_120b_reward": f"{100.0 * best.reward / baseline_reward:.3f}",
                    "gain_vs_always_at_cap": "" if always is None else f"{best.reward - always.reward:.9f}",
                    "gain_vs_direct_random_at_cap": ""
                    if random is None
                    else f"{best.reward - random.reward:.9f}",
                    "gain_vs_cascade_at_cap": "" if cascade is None else f"{best.reward - cascade.reward:.9f}",
                    "chosen_point": best.label,
                    "lambda": "" if best.lambda_value is None else f"{best.lambda_value:g}",
                }
            )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_wide_table(
    out_path: Path,
    points_by_method: dict[str, list[Point]],
    budget_pcts: list[float],
    baseline_cost: float,
    baseline_reward: float,
) -> list[dict[str, str]]:
    methods = [
        "Always direct",
        "Direct random",
        "Cascade",
        "Best Q policy (pred cost)",
        "Q direct (pred cost)",
        "Q direct (oracle cost diagnostic)",
        "Q chain flexible (pred cost)",
        "Q chain forced-scout (pred cost)",
        "Oracle route upper bound",
    ]
    rows: list[dict[str, str]] = []
    for budget_pct in budget_pcts:
        cap = baseline_cost * budget_pct / 100.0
        row: dict[str, str] = {
            "budget_pct_of_120b_cost": f"{budget_pct:.3f}",
            "cap_cost": f"{cap:.12g}",
        }
        best_by_method: dict[str, Point | None] = {
            method: _best_under_budget(points_by_method.get(method, []), cap) for method in methods
        }
        always = best_by_method["Always direct"]
        cascade = best_by_method["Cascade"]
        for method in methods:
            best = best_by_method[method]
            prefix = method.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            if best is None:
                row[f"{prefix}_reward"] = ""
                row[f"{prefix}_cost_pct"] = ""
                row[f"{prefix}_point"] = ""
                continue
            row[f"{prefix}_reward"] = f"{best.reward:.6f}"
            row[f"{prefix}_cost_pct"] = f"{100.0 * best.cost / baseline_cost:.2f}"
            row[f"{prefix}_point"] = best.label
        if best_by_method["Q direct (pred cost)"] and always:
            row["q_direct_pred_gain_vs_always"] = f"{best_by_method['Q direct (pred cost)'].reward - always.reward:.6f}"
        else:
            row["q_direct_pred_gain_vs_always"] = ""
        if best_by_method["Q direct (pred cost)"] and cascade:
            row["q_direct_pred_gain_vs_cascade"] = f"{best_by_method['Q direct (pred cost)'].reward - cascade.reward:.6f}"
        else:
            row["q_direct_pred_gain_vs_cascade"] = ""
        random = best_by_method["Direct random"]
        if best_by_method["Q direct (pred cost)"] and random:
            row["q_direct_pred_gain_vs_direct_random"] = (
                f"{best_by_method['Q direct (pred cost)'].reward - random.reward:.6f}"
            )
        else:
            row["q_direct_pred_gain_vs_direct_random"] = ""
        if best_by_method["Best Q policy (pred cost)"] and random:
            row["best_q_pred_gain_vs_direct_random"] = (
                f"{best_by_method['Best Q policy (pred cost)'].reward - random.reward:.6f}"
            )
        else:
            row["best_q_pred_gain_vs_direct_random"] = ""
        row["q_direct_pred_pct_of_120b_reward"] = (
            ""
            if best_by_method["Q direct (pred cost)"] is None
            else f"{100.0 * best_by_method['Q direct (pred cost)'].reward / baseline_reward:.2f}"
        )
        rows.append(row)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_points(out_path: Path, points: list[Point], baseline_cost: float, baseline_reward: float) -> None:
    rows = [
        {
            "method": p.method,
            "reward": f"{p.reward:.9f}",
            "cost": f"{p.cost:.12g}",
            "cost_pct_of_120b": f"{100.0 * p.cost / baseline_cost:.3f}",
            "reward_pct_of_120b": f"{100.0 * p.reward / baseline_reward:.3f}",
            "label": p.label,
            "lambda": "" if p.lambda_value is None else f"{p.lambda_value:g}",
            "source": p.source,
        }
        for p in sorted(points, key=lambda p: (p.method, p.cost, p.reward))
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_frontiers(
    out_path: Path,
    points_by_method: dict[str, list[Point]],
    baseline_cost: float,
    max_budget_pct: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [
        "Always direct",
        "Direct random",
        "Cascade",
        "Best Q policy (pred cost)",
        "Q direct (pred cost)",
        "Q direct (oracle cost diagnostic)",
        "Q chain flexible (pred cost)",
    ]
    caps = [i * max_budget_pct / 400.0 for i in range(401)]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for method in methods:
        if method not in points_by_method:
            continue
        xs: list[float] = []
        ys: list[float] = []
        for cap_pct in caps:
            cap = baseline_cost * cap_pct / 100.0
            best = _best_under_budget(points_by_method[method], cap)
            if best is not None:
                xs.append(cap_pct)
                ys.append(best.reward)
        if xs:
            ax.step(xs, ys, where="post", linewidth=2.0, label=method)

    ax.set_xlabel("Budget cap (% of always-120B mean cost)")
    ax.set_ylabel("Mean real reward / pass rate")
    ax.set_xlim(0, max_budget_pct)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--q-predicted-rows",
        type=Path,
        default=Path(
            "router_analysis/swe_smith_train1500_real_4route_q_predicted_cost_rescore_178099/"
            "predicted_cost_q_vs_cascade_rows.csv"
        ),
    )
    parser.add_argument(
        "--q-diag-rows",
        type=Path,
        default=Path(
            "router_analysis/swe_smith_train1500_real_4route_cost_oracle_gap_diag_178108/"
            "cost_oracle_gap_policy_rows.csv"
        ),
    )
    parser.add_argument(
        "--cascade-rows",
        type=Path,
        default=Path(
            "router_analysis/swe_smith_train1500_real_4route_cascade_aws_rescore_178099/"
            "cascade_weighted_cost_utility.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("router_analysis/swe_smith_train1500_real_4route_q_budget_frontier_178108"),
    )
    parser.add_argument(
        "--budget-pcts",
        type=float,
        nargs="*",
        default=[2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 30, 32, 35, 40, 50, 75, 100],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = _load_points(args)
    points_by_method: dict[str, list[Point]] = {}
    for point in points:
        points_by_method.setdefault(point.method, []).append(point)

    always_120b = [
        p
        for p in points_by_method["Always direct"]
        if p.label == "solver:openai/gpt-oss-120b"
    ][0]
    baseline_cost = always_120b.cost
    baseline_reward = always_120b.reward
    points = _add_direct_random_points(
        points,
        baseline_cost=baseline_cost,
        budget_pcts=args.budget_pcts,
        plot_max_pcts=[35.0, 100.0],
    )
    points_by_method = {}
    for point in points:
        points_by_method.setdefault(point.method, []).append(point)

    _write_points(args.output_dir / "frontier_points.csv", points, baseline_cost, baseline_reward)
    long_rows = _write_long_table(
        args.output_dir / "budget_frontier_long.csv",
        points_by_method,
        args.budget_pcts,
        baseline_cost,
        baseline_reward,
    )
    wide_rows = _write_wide_table(
        args.output_dir / "budget_frontier_wide.csv",
        points_by_method,
        args.budget_pcts,
        baseline_cost,
        baseline_reward,
    )
    _plot_frontiers(args.output_dir / "budget_frontier_zoom_0_35.png", points_by_method, baseline_cost, 35.0)
    _plot_frontiers(args.output_dir / "budget_frontier_full_0_100.png", points_by_method, baseline_cost, 100.0)

    summary = {
        "baseline_120b_cost": baseline_cost,
        "baseline_120b_reward": baseline_reward,
        "methods": sorted(points_by_method),
        "n_points": len(points),
        "outputs": {
            "frontier_points": str(args.output_dir / "frontier_points.csv"),
            "budget_frontier_long": str(args.output_dir / "budget_frontier_long.csv"),
            "budget_frontier_wide": str(args.output_dir / "budget_frontier_wide.csv"),
            "budget_frontier_zoom_0_35": str(args.output_dir / "budget_frontier_zoom_0_35.png"),
            "budget_frontier_full_0_100": str(args.output_dir / "budget_frontier_full_0_100.png"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print("\nSelected budget-cap rows:")
    for row in wide_rows:
        budget = float(row["budget_pct_of_120b_cost"])
        if budget in {10.0, 20.0, 22.0, 25.0, 30.0, 35.0}:
            print(
                budget,
                {
                    "always": row.get("always_direct_reward"),
                    "cascade": row.get("cascade_reward"),
                    "q_pred": row.get("q_direct_pred_cost_reward"),
                    "q_pred_cost_pct": row.get("q_direct_pred_cost_cost_pct"),
                    "q_gain_always": row.get("q_direct_pred_gain_vs_always"),
                    "q_gain_cascade": row.get("q_direct_pred_gain_vs_cascade"),
                },
            )


if __name__ == "__main__":
    main()

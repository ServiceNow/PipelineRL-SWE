#!/usr/bin/env python3
"""Generate figures from a schema-v2 full-execution replay result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STYLES = {
    "counts": ("#777777", "o", "Counts"),
    "counts_abstain": ("#ff7f0e", "^", "Counts + abstain"),
    "content": ("#1f77b4", "s", "Static content"),
    "content_abstain": ("#9467bd", "D", "Static content + abstain"),
    "sequential": ("#2ca02c", "P", "Failure-history policy"),
    "sequential_abstain": ("#d62728", "X", "Failure-history + abstain"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.replay_results).read_text())
    if data.get("protocol") != "scout_first_full_execution_failure_region":
        raise ValueError("Refusing to plot a non-v2 or weak-verifier replay")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for policy, (color, marker, label) in STYLES.items():
        points = sorted(
            [row for row in data["results"] if row["policy"] == policy],
            key=lambda row: row["mean_realized_cost"],
        )
        if points:
            ax.plot(
                [row["mean_realized_cost"] for row in points],
                [100 * row["correctness"] for row in points],
                color=color,
                marker=marker,
                label=label,
                linewidth=1.8,
                markersize=5,
            )
    for name, label, marker in (
        ("single_scout", "Single scout", "v"),
        ("single_oss20", "Single oss-20b", "<"),
        ("single_oss120", "Single oss-120b", ">"),
        ("single_pass_cascade", "Scout → 20B → 120B", "*"),
    ):
        point = next((row for row in data["results"] if row["policy"] == name), None)
        if point:
            ax.scatter(
                [point["mean_realized_cost"]],
                [100 * point["correctness"]],
                marker=marker,
                s=85,
                edgecolors="black",
                linewidths=0.5,
                label=label,
                zorder=5,
            )
    unit = "estimated USD" if data.get("cost_mode") == "usd" else "weighted calls"
    ax.set_xlabel(f"Mean realized inference cost ({unit})")
    ax.set_ylabel("Resolve rate (%)")
    ax.set_title("Scout-first allocation after full-execution failures")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_mdp_full_execution_frontier.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for policy, (color, marker, label) in STYLES.items():
        points = sorted(
            [row for row in data["results"] if row["policy"] == policy],
            key=lambda row: row["mean_realized_cost"],
        )
        points = [row for row in points if row.get("conditional_correctness_after_scout_failure") is not None]
        if points:
            ax.plot(
                [row["mean_realized_cost"] for row in points],
                [100 * row["conditional_correctness_after_scout_failure"] for row in points],
                color=color,
                marker=marker,
                label=label,
            )
    ax.set_xlabel(f"Mean realized inference cost ({unit})")
    ax.set_ylabel("Resolve rate after scout failure (%)")
    ax.set_title("Decision-relevant performance in the reachable failure region")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_mdp_failure_region_frontier.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

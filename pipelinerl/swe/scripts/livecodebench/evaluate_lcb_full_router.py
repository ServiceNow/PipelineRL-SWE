#!/usr/bin/env python3
"""Evaluate direct LCB routing against sequential and matched-cost baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read(path: Path) -> list[dict]:
    with open(path) as in_f:
        return [json.loads(line) for line in in_f if line.strip()]


def _summary(rows: list[dict], choices: np.ndarray, weights: np.ndarray) -> dict:
    rewards = np.array([row["route_successes"] for row in rows], dtype=float)
    tokens = np.array([row["route_completion_tokens"] for row in rows], dtype=float)
    weighted = tokens * weights
    total_cost = weighted[:, 0] + np.where(
        choices == 0, 0.0, weighted[np.arange(len(rows)), choices]
    )
    return {
        "correctness": float(rewards[np.arange(len(rows)), choices].mean()),
        "mean_weighted_completion_tokens": float(total_cost.mean()),
        "route_counts": {
            str(i): int((choices == i).sum()) for i in range(rewards.shape[1])
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--cost-weights", default="1,5,30",
                        help="Relative per-output-token costs for scout, mid, high")
    parser.add_argument("--lambdas", default="0,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5")
    args = parser.parse_args()

    rows = _read(Path(args.predictions))
    labels = rows[0]["route_labels"]
    weights = np.array([float(value) for value in args.cost_weights.split(",")])
    if len(weights) != len(labels):
        raise ValueError("One cost weight is required per route")
    lambdas = [float(value) for value in args.lambdas.split(",")]
    rewards = np.array([row["route_successes"] for row in rows], dtype=float)
    public = np.array(
        [[value is True for value in row["route_public_successes"]] for row in rows],
        dtype=bool,
    )
    tokens = np.array([row["route_completion_tokens"] for row in rows], dtype=float)
    weighted = tokens * weights
    incremental = weighted.copy()
    incremental[:, 0] = 0.0

    policies: list[dict] = []
    for route_idx, label in enumerate(labels):
        policies.append({"policy": f"always_{label}", **_summary(rows, np.full(len(rows), route_idx), weights)})

    sequential = np.where(public[:, 0], 0, np.where(public[:, 1], 1, 2))
    policies.append({"policy": "sequential_public_feedback", **_summary(rows, sequential, weights)})

    oracle = np.zeros(len(rows), dtype=int)
    for i in range(len(rows)):
        successes = np.flatnonzero(rewards[i] > 0.5)
        oracle[i] = int(successes[np.argmin(incremental[i, successes])]) if len(successes) else 2
    policies.append({"policy": "oracle_cheapest_success", **_summary(rows, oracle, weights)})

    probabilities = np.array([row["p_successes"] for row in rows], dtype=float)
    for lam in lambdas:
        choices = np.argmax(probabilities - lam * incremental, axis=1)
        direct = _summary(rows, choices, weights)
        route_mix = np.bincount(choices, minlength=len(labels)) / len(rows)
        random_correct = float((rewards * route_mix).sum(axis=1).mean())
        random_cost = float((weighted[:, 0] + (weighted * route_mix).sum(axis=1) - weighted[:, 0] * route_mix[0]).mean())
        policies.append({"policy": "direct_router", "lambda": lam, **direct})
        policies.append(
            {
                "policy": "random_matched_route_mix",
                "lambda": lam,
                "correctness": random_correct,
                "mean_weighted_completion_tokens": random_cost,
                "route_mix": route_mix.tolist(),
            }
        )

    payload = {
        "route_labels": labels,
        "cost_weights": weights.tolist(),
        "n_eval": len(rows),
        "policies": policies,
    }
    Path(args.output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Analyze verifier scores against real multi-rollout SWE-Smith pass/fail labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROUTE_SLUGS = {
    0: "qwen3_4b_instruct_2507",
    1: "gpt_oss_20b",
    2: "qwen3_coder_30b_a3b",
    3: "gpt_oss_120b",
}
ROUTE_LABELS = {
    0: "Qwen3-4B",
    1: "OSS-20B",
    2: "Qwen3-Coder-30B-A3B",
    3: "OSS-120B",
}
DEFAULT_COST_WEIGHTS = [2.78e-7, 1.299e-6, 4.64e-6, 1.113e-5]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return math.nan
    return float(np.corrcoef(x.astype(float), y.astype(float))[0, 1])


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = scores.astype(float)
    labels = labels.astype(int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    values = np.concatenate([pos, neg])
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # Average ranks for ties.
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = float(np.mean(np.arange(start + 1, end + 1, dtype=float)))
        ranks[order[start:end]] = avg_rank
        start = end
    pos_ranks = ranks[: len(pos)]
    return float((np.sum(pos_ranks) - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _load_real_labels(report_root: Path, route_idx: int, rollout_idx: int) -> dict[str, int]:
    run_dir = report_root / f"swe_smith_eval150_{ROUTE_SLUGS[int(route_idx)]}_r{int(rollout_idx)}"
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    labels: dict[str, int] = {}
    for report_path in run_dir.glob("*/report.json"):
        instance_id = report_path.parent.name
        report = json.loads(report_path.read_text())
        labels[instance_id] = 1 if bool(report.get("resolved")) else 0
    return labels


def _build_attempt_table(
    score_rows: list[dict[str, Any]],
    report_root: Path,
    route_cost_weights: list[float],
) -> list[dict[str, Any]]:
    labels_cache: dict[tuple[int, int], dict[str, int]] = {}
    attempts: list[dict[str, Any]] = []
    for row in score_rows:
        route_idx = int(row["source_route_idx"])
        rollout_idx = int(row["rollout_idx"])
        key = (route_idx, rollout_idx)
        if key not in labels_cache:
            labels_cache[key] = _load_real_labels(report_root, route_idx, rollout_idx)
        original_problem_id = str(row["original_problem_id"])
        if original_problem_id not in labels_cache[key]:
            continue
        output_tokens = float(row.get("output_tokens") or 0.0)
        attempts.append(
            {
                "original_problem_id": original_problem_id,
                "route_idx": route_idx,
                "rollout_idx": rollout_idx,
                "route": ROUTE_LABELS.get(route_idx, str(route_idx)),
                "pred_score": float(row["pred_score"]),
                "proxy_score": float(row.get("true_proxy_score", row.get("proxy_reward", 0.0)) or 0.0),
                "real_success": int(labels_cache[key][original_problem_id]),
                "output_tokens": output_tokens,
                "weighted_output_cost": output_tokens * float(route_cost_weights[route_idx]),
            }
        )
    return attempts


def _attempt_lookup(attempts: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    return {
        (str(row["original_problem_id"]), int(row["route_idx"]), int(row["rollout_idx"])): row
        for row in attempts
    }


def _metrics_by_route(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route_idx in sorted({int(row["route_idx"]) for row in attempts}):
        subset = [row for row in attempts if int(row["route_idx"]) == route_idx]
        scores = np.asarray([row["pred_score"] for row in subset], dtype=float)
        proxy = np.asarray([row["proxy_score"] for row in subset], dtype=float)
        real = np.asarray([row["real_success"] for row in subset], dtype=int)
        rows.append(
            {
                "route_idx": route_idx,
                "route": ROUTE_LABELS.get(route_idx, str(route_idx)),
                "n": len(subset),
                "real_pass_rate": float(np.mean(real)) if len(real) else math.nan,
                "pred_vs_real_pearson": _safe_corr(scores, real.astype(float)),
                "pred_vs_real_auc": _roc_auc(scores, real),
                "pred_vs_proxy_pearson": _safe_corr(scores, proxy),
                "proxy_vs_real_pearson": _safe_corr(proxy, real.astype(float)),
                "proxy_vs_real_auc": _roc_auc(proxy, real),
                "mean_pred_score": float(np.mean(scores)) if len(scores) else math.nan,
                "mean_proxy_score": float(np.mean(proxy)) if len(proxy) else math.nan,
            }
        )
    return rows


def _policy_baselines(lookup: dict[tuple[str, int, int], dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policies = [
        ("20B_r0", [(1, 0)]),
        ("20B_r1", [(1, 1)]),
        ("20B_r2", [(1, 2)]),
        ("120B_r0", [(3, 0)]),
        ("20B_x3_pred_best", [(1, 0), (1, 1), (1, 2)]),
        ("20B_x3_oracle_best", [(1, 0), (1, 1), (1, 2)]),
    ]
    for name, choices in policies:
        rewards: list[float] = []
        costs: list[float] = []
        for pid in ids:
            candidates = [lookup[(pid, route_idx, rollout_idx)] for route_idx, rollout_idx in choices]
            if name.endswith("pred_best"):
                selected = max(candidates, key=lambda row: float(row["pred_score"]))
                rewards.append(float(selected["real_success"]))
            elif name.endswith("oracle_best"):
                rewards.append(float(max(row["real_success"] for row in candidates)))
            else:
                rewards.append(float(candidates[0]["real_success"]))
            costs.append(float(sum(row["weighted_output_cost"] for row in candidates)))
        rows.append(
            {
                "policy": name,
                "n": len(ids),
                "mean_reward": float(np.mean(rewards)),
                "mean_cost": float(np.mean(costs)),
            }
        )
    return rows


def _branching_grid(
    lookup: dict[tuple[str, int, int], dict[str, Any]],
    ids: list[str],
    max_thresholds: int,
) -> list[dict[str, Any]]:
    r0_scores = np.asarray([lookup[(pid, 1, 0)]["pred_score"] for pid in ids], dtype=float)
    quantiles = np.linspace(0.0, 1.0, int(max_thresholds))
    candidates = sorted(set(float(value) for value in np.quantile(r0_scores, quantiles)))
    eps = max(1.0e-6, float(np.std(r0_scores)) * 1.0e-6)
    candidates = [float(np.min(r0_scores) - eps)] + candidates + [float(np.max(r0_scores) + eps)]
    rows: list[dict[str, Any]] = []
    for resample_tau in candidates:
        for submit_tau in candidates:
            if float(resample_tau) > float(submit_tau):
                continue
            rewards: list[float] = []
            costs: list[float] = []
            actions = {"submit_20b_r0": 0, "resample_20b": 0, "escalate_120b": 0}
            for pid in ids:
                first = lookup[(pid, 1, 0)]
                if float(first["pred_score"]) >= float(submit_tau):
                    actions["submit_20b_r0"] += 1
                    rewards.append(float(first["real_success"]))
                    costs.append(float(first["weighted_output_cost"]))
                elif float(first["pred_score"]) >= float(resample_tau):
                    actions["resample_20b"] += 1
                    candidates_20b = [lookup[(pid, 1, rollout_idx)] for rollout_idx in (0, 1, 2)]
                    selected = max(candidates_20b, key=lambda row: float(row["pred_score"]))
                    rewards.append(float(selected["real_success"]))
                    costs.append(float(sum(row["weighted_output_cost"] for row in candidates_20b)))
                else:
                    actions["escalate_120b"] += 1
                    escalated = lookup[(pid, 3, 0)]
                    rewards.append(float(escalated["real_success"]))
                    costs.append(float(first["weighted_output_cost"]) + float(escalated["weighted_output_cost"]))
            rows.append(
                {
                    "resample_tau": float(resample_tau),
                    "submit_tau": float(submit_tau),
                    "n": len(ids),
                    "mean_reward": float(np.mean(rewards)),
                    "mean_cost": float(np.mean(costs)),
                    **actions,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-jsonl", required=True)
    parser.add_argument(
        "--report-root",
        default="router_analysis/uploaded_eval_full_20260617/swe_smith_multirollout_eval150_1781382734/logs/run_evaluation",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--route-cost-weights", default=",".join(str(value) for value in DEFAULT_COST_WEIGHTS))
    parser.add_argument("--max-thresholds", type=int, default=41)
    args = parser.parse_args()

    route_cost_weights = [float(part.strip()) for part in str(args.route_cost_weights).split(",") if part.strip()]
    if len(route_cost_weights) <= max(ROUTE_SLUGS):
        raise ValueError("--route-cost-weights must include weights for all route indices")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attempts = _build_attempt_table(_read_jsonl(Path(args.scores_jsonl)), Path(args.report_root), route_cost_weights)
    lookup = _attempt_lookup(attempts)
    ids = sorted(
        {
            pid
            for pid in {str(row["original_problem_id"]) for row in attempts}
            if all((pid, route_idx, rollout_idx) in lookup for route_idx in (1, 3) for rollout_idx in (0, 1, 2))
        }
    )
    if not ids:
        raise ValueError("No complete ids for OSS-20B/OSS-120B rollouts")
    metrics = _metrics_by_route(attempts)
    baselines = _policy_baselines(lookup, ids)
    grid = _branching_grid(lookup, ids, int(args.max_thresholds))
    frontier: list[dict[str, Any]] = []
    for row in sorted(grid, key=lambda item: (float(item["mean_cost"]), -float(item["mean_reward"]))):
        if not frontier or float(row["mean_reward"]) > max(float(prev["mean_reward"]) for prev in frontier):
            frontier.append(row)
    best_by_reward = max(grid, key=lambda row: float(row["mean_reward"]))
    best_by_utility_rows = []
    for lambda_value in [0, 1, 2, 5, 10, 20, 50, 100]:
        best = max(grid, key=lambda row: float(row["mean_reward"]) - float(lambda_value) * float(row["mean_cost"]))
        best_by_utility_rows.append(
            {
                "lambda": float(lambda_value),
                "mean_utility": float(best["mean_reward"]) - float(lambda_value) * float(best["mean_cost"]),
                **best,
            }
        )

    _write_csv(output_dir / "attempt_metrics_by_route.csv", metrics)
    _write_csv(output_dir / "policy_baselines.csv", baselines)
    _write_csv(output_dir / "branching_policy_grid.csv", grid)
    _write_csv(output_dir / "branching_policy_frontier.csv", frontier)
    _write_csv(output_dir / "branching_policy_best_by_utility.csv", best_by_utility_rows)
    summary = {
        "n_attempts": len(attempts),
        "n_complete_20b_120b_ids": len(ids),
        "route_cost_weights": route_cost_weights,
        "attempt_metrics_by_route": metrics,
        "policy_baselines": baselines,
        "best_branching_by_reward": best_by_reward,
        "best_branching_by_utility": best_by_utility_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

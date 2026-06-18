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


def _sequential_resample_grid(
    lookup: dict[tuple[str, int, int], dict[str, Any]],
    ids: list[str],
    max_thresholds: int,
) -> list[dict[str, Any]]:
    all_scores = np.asarray(
        [lookup[(pid, 1, rollout_idx)]["pred_score"] for pid in ids for rollout_idx in (0, 1, 2)],
        dtype=float,
    )
    quantiles = np.linspace(0.0, 1.0, int(max_thresholds))
    candidates = sorted(set(float(value) for value in np.quantile(all_scores, quantiles)))
    eps = max(1.0e-6, float(np.std(all_scores)) * 1.0e-6)
    candidates = [float(np.min(all_scores) - eps)] + candidates + [float(np.max(all_scores) + eps)]
    rows: list[dict[str, Any]] = []
    for escalate_tau in candidates:
        for submit_tau in candidates:
            if float(escalate_tau) > float(submit_tau):
                continue
            rewards: list[float] = []
            costs: list[float] = []
            submit_counts = {1: 0, 2: 0, 3: 0}
            escalate_counts = {1: 0, 2: 0, 3: 0}
            rollout_counts: list[int] = []
            for pid in ids:
                seen: list[dict[str, Any]] = []
                cost = 0.0
                selected: dict[str, Any] | None = None
                for rollout_idx in (0, 1, 2):
                    attempt = lookup[(pid, 1, rollout_idx)]
                    seen.append(attempt)
                    cost += float(attempt["weighted_output_cost"])
                    best_seen = max(seen, key=lambda row: float(row["pred_score"]))
                    best_score = float(best_seen["pred_score"])
                    n_seen = len(seen)
                    if best_score >= float(submit_tau):
                        selected = best_seen
                        submit_counts[n_seen] += 1
                        break
                    if best_score < float(escalate_tau):
                        selected = lookup[(pid, 3, 0)]
                        cost += float(selected["weighted_output_cost"])
                        escalate_counts[n_seen] += 1
                        break
                    if rollout_idx == 2:
                        selected = best_seen
                        submit_counts[n_seen] += 1
                        break
                if selected is None:
                    raise RuntimeError("Sequential policy failed to select an attempt")
                rewards.append(float(selected["real_success"]))
                costs.append(cost)
                rollout_counts.append(len(seen))
            rows.append(
                {
                    "escalate_tau": float(escalate_tau),
                    "submit_tau": float(submit_tau),
                    "n": len(ids),
                    "mean_reward": float(np.mean(rewards)),
                    "mean_cost": float(np.mean(costs)),
                    "mean_20b_rollouts": float(np.mean(rollout_counts)),
                    "submit_20b_after_1": int(submit_counts[1]),
                    "submit_20b_after_2": int(submit_counts[2]),
                    "submit_20b_after_3": int(submit_counts[3]),
                    "escalate_120b_after_1": int(escalate_counts[1]),
                    "escalate_120b_after_2": int(escalate_counts[2]),
                    "escalate_120b_after_3": int(escalate_counts[3]),
                    "escalate_120b": int(sum(escalate_counts.values())),
                }
            )
    return rows


def _threshold_pairs(candidates: list[float]) -> list[tuple[float, float]]:
    return [
        (float(escalate_tau), float(submit_tau))
        for escalate_tau in candidates
        for submit_tau in candidates
        if float(escalate_tau) <= float(submit_tau)
    ]


def _flexible_state_policy_grid(
    lookup: dict[tuple[str, int, int], dict[str, Any]],
    ids: list[str],
    max_thresholds: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    threshold_source: list[float] = []
    for pid in ids:
        attempts = [lookup[(pid, 1, rollout_idx)] for rollout_idx in (0, 1, 2)]
        best_after_1 = max(attempts[:1], key=lambda row: float(row["pred_score"]))
        best_after_2 = max(attempts[:2], key=lambda row: float(row["pred_score"]))
        best_after_3 = max(attempts[:3], key=lambda row: float(row["pred_score"]))
        s1 = float(best_after_1["pred_score"])
        s2 = float(best_after_2["pred_score"])
        s3 = float(best_after_3["pred_score"])
        c1 = float(attempts[0]["weighted_output_cost"])
        c2 = c1 + float(attempts[1]["weighted_output_cost"])
        c3 = c2 + float(attempts[2]["weighted_output_cost"])
        escalated = lookup[(pid, 3, 0)]
        c120 = float(escalated["weighted_output_cost"])
        examples.append(
            {
                "s1": s1,
                "s2": s2,
                "s3": s3,
                "r1": float(best_after_1["real_success"]),
                "r2": float(best_after_2["real_success"]),
                "r3": float(best_after_3["real_success"]),
                "r120": float(escalated["real_success"]),
                "c1": c1,
                "c2": c2,
                "c3": c3,
                "c1_120": c1 + c120,
                "c2_120": c2 + c120,
                "c3_120": c3 + c120,
            }
        )
        threshold_source.extend([s1, s2, s3])
    scores = np.asarray(threshold_source, dtype=float)
    quantiles = np.linspace(0.0, 1.0, int(max_thresholds))
    candidates = sorted(set(float(value) for value in np.quantile(scores, quantiles)))
    eps = max(1.0e-6, float(np.std(scores)) * 1.0e-6)
    candidates = [float(np.min(scores) - eps)] + candidates + [float(np.max(scores) + eps)]
    pairs = _threshold_pairs(candidates)

    rows: list[dict[str, Any]] = []
    for escalate_tau_1, submit_tau_1 in pairs:
        for escalate_tau_2, submit_tau_2 in pairs:
            for final_escalate_tau in candidates:
                rewards: list[float] = []
                costs: list[float] = []
                actions = {
                    "submit_20b_after_1": 0,
                    "submit_20b_after_2": 0,
                    "submit_20b_after_3": 0,
                    "escalate_120b_after_1": 0,
                    "escalate_120b_after_2": 0,
                    "escalate_120b_after_3": 0,
                }
                rollout_counts: list[int] = []
                for ex in examples:
                    if float(ex["s1"]) >= submit_tau_1:
                        rewards.append(float(ex["r1"]))
                        costs.append(float(ex["c1"]))
                        rollout_counts.append(1)
                        actions["submit_20b_after_1"] += 1
                    elif float(ex["s1"]) < escalate_tau_1:
                        rewards.append(float(ex["r120"]))
                        costs.append(float(ex["c1_120"]))
                        rollout_counts.append(1)
                        actions["escalate_120b_after_1"] += 1
                    elif float(ex["s2"]) >= submit_tau_2:
                        rewards.append(float(ex["r2"]))
                        costs.append(float(ex["c2"]))
                        rollout_counts.append(2)
                        actions["submit_20b_after_2"] += 1
                    elif float(ex["s2"]) < escalate_tau_2:
                        rewards.append(float(ex["r120"]))
                        costs.append(float(ex["c2_120"]))
                        rollout_counts.append(2)
                        actions["escalate_120b_after_2"] += 1
                    elif float(ex["s3"]) < final_escalate_tau:
                        rewards.append(float(ex["r120"]))
                        costs.append(float(ex["c3_120"]))
                        rollout_counts.append(3)
                        actions["escalate_120b_after_3"] += 1
                    else:
                        rewards.append(float(ex["r3"]))
                        costs.append(float(ex["c3"]))
                        rollout_counts.append(3)
                        actions["submit_20b_after_3"] += 1
                rows.append(
                    {
                        "escalate_tau_1": float(escalate_tau_1),
                        "submit_tau_1": float(submit_tau_1),
                        "escalate_tau_2": float(escalate_tau_2),
                        "submit_tau_2": float(submit_tau_2),
                        "final_escalate_tau": float(final_escalate_tau),
                        "n": len(ids),
                        "mean_reward": float(np.mean(rewards)),
                        "mean_cost": float(np.mean(costs)),
                        "mean_20b_rollouts": float(np.mean(rollout_counts)),
                        "escalate_120b": int(
                            actions["escalate_120b_after_1"]
                            + actions["escalate_120b_after_2"]
                            + actions["escalate_120b_after_3"]
                        ),
                        **actions,
                    }
                )
    return rows


def _frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (float(item["mean_cost"]), -float(item["mean_reward"]))):
        if not frontier or float(row["mean_reward"]) > max(float(prev["mean_reward"]) for prev in frontier):
            frontier.append(row)
    return frontier


def _evaluate_policy_on_ids(
    lookup: dict[tuple[str, int, int], dict[str, Any]],
    ids: list[str],
    policy: dict[str, Any],
    policy_type: str,
) -> dict[str, Any]:
    rewards: list[float] = []
    costs: list[float] = []
    actions: dict[str, int] = {}
    rollout_counts: list[int] = []

    def inc(key: str) -> None:
        actions[key] = int(actions.get(key, 0)) + 1

    for pid in ids:
        if policy_type == "batched":
            first = lookup[(pid, 1, 0)]
            if float(first["pred_score"]) >= float(policy["submit_tau"]):
                selected = first
                cost = float(first["weighted_output_cost"])
                inc("submit_20b_r0")
            elif float(first["pred_score"]) >= float(policy["resample_tau"]):
                candidates_20b = [lookup[(pid, 1, rollout_idx)] for rollout_idx in (0, 1, 2)]
                selected = max(candidates_20b, key=lambda row: float(row["pred_score"]))
                cost = float(sum(row["weighted_output_cost"] for row in candidates_20b))
                inc("resample_20b")
            else:
                selected = lookup[(pid, 3, 0)]
                cost = float(first["weighted_output_cost"]) + float(selected["weighted_output_cost"])
                inc("escalate_120b")
            rewards.append(float(selected["real_success"]))
            costs.append(cost)
            continue

        if policy_type == "sequential":
            seen: list[dict[str, Any]] = []
            cost = 0.0
            selected: dict[str, Any] | None = None
            for rollout_idx in (0, 1, 2):
                attempt = lookup[(pid, 1, rollout_idx)]
                seen.append(attempt)
                cost += float(attempt["weighted_output_cost"])
                best_seen = max(seen, key=lambda row: float(row["pred_score"]))
                best_score = float(best_seen["pred_score"])
                n_seen = len(seen)
                if best_score >= float(policy["submit_tau"]):
                    selected = best_seen
                    inc(f"submit_20b_after_{n_seen}")
                    break
                if best_score < float(policy["escalate_tau"]):
                    selected = lookup[(pid, 3, 0)]
                    cost += float(selected["weighted_output_cost"])
                    inc(f"escalate_120b_after_{n_seen}")
                    break
                if rollout_idx == 2:
                    selected = best_seen
                    inc("submit_20b_after_3")
                    break
            if selected is None:
                raise RuntimeError("Sequential policy failed to select")
            rewards.append(float(selected["real_success"]))
            costs.append(cost)
            rollout_counts.append(len(seen))
            continue

        if policy_type == "flexible":
            attempts = [lookup[(pid, 1, rollout_idx)] for rollout_idx in (0, 1, 2)]
            best_after_1 = max(attempts[:1], key=lambda row: float(row["pred_score"]))
            best_after_2 = max(attempts[:2], key=lambda row: float(row["pred_score"]))
            best_after_3 = max(attempts[:3], key=lambda row: float(row["pred_score"]))
            s1 = float(best_after_1["pred_score"])
            s2 = float(best_after_2["pred_score"])
            s3 = float(best_after_3["pred_score"])
            c1 = float(attempts[0]["weighted_output_cost"])
            c2 = c1 + float(attempts[1]["weighted_output_cost"])
            c3 = c2 + float(attempts[2]["weighted_output_cost"])
            escalated = lookup[(pid, 3, 0)]
            c120 = float(escalated["weighted_output_cost"])
            if s1 >= float(policy["submit_tau_1"]):
                rewards.append(float(best_after_1["real_success"]))
                costs.append(c1)
                rollout_counts.append(1)
                inc("submit_20b_after_1")
            elif s1 < float(policy["escalate_tau_1"]):
                rewards.append(float(escalated["real_success"]))
                costs.append(c1 + c120)
                rollout_counts.append(1)
                inc("escalate_120b_after_1")
            elif s2 >= float(policy["submit_tau_2"]):
                rewards.append(float(best_after_2["real_success"]))
                costs.append(c2)
                rollout_counts.append(2)
                inc("submit_20b_after_2")
            elif s2 < float(policy["escalate_tau_2"]):
                rewards.append(float(escalated["real_success"]))
                costs.append(c2 + c120)
                rollout_counts.append(2)
                inc("escalate_120b_after_2")
            elif s3 < float(policy["final_escalate_tau"]):
                rewards.append(float(escalated["real_success"]))
                costs.append(c3 + c120)
                rollout_counts.append(3)
                inc("escalate_120b_after_3")
            else:
                rewards.append(float(best_after_3["real_success"]))
                costs.append(c3)
                rollout_counts.append(3)
                inc("submit_20b_after_3")
            continue

        raise ValueError(f"Unknown policy_type={policy_type}")

    row = {
        "n": len(ids),
        "mean_reward": float(np.mean(rewards)) if rewards else math.nan,
        "mean_cost": float(np.mean(costs)) if costs else math.nan,
    }
    if rollout_counts:
        row["mean_20b_rollouts"] = float(np.mean(rollout_counts))
    row.update(actions)
    return row


def _filter_grid_by_ids(rows: list[dict[str, Any]], lookup: dict[tuple[str, int, int], dict[str, Any]], ids: list[str], policy_type: str) -> list[dict[str, Any]]:
    return [{**row, **_evaluate_policy_on_ids(lookup, ids, row, policy_type)} for row in rows]


def _split_threshold_eval(
    lookup: dict[tuple[str, int, int], dict[str, Any]],
    ids: list[str],
    batched_grid: list[dict[str, Any]],
    sequential_grid: list[dict[str, Any]],
    flexible_grid: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    lambdas: list[float],
    n_splits: int,
    seed: int,
) -> list[dict[str, Any]]:
    import random

    rng = random.Random(int(seed))
    rows: list[dict[str, Any]] = []
    _ = baselines
    for split_idx in range(int(n_splits)):
        shuffled = list(ids)
        rng.shuffle(shuffled)
        mid = len(shuffled) // 2
        train_ids = sorted(shuffled[:mid])
        eval_ids = sorted(shuffled[mid:])
        train_grids = {
            "batched": _filter_grid_by_ids(batched_grid, lookup, train_ids, "batched"),
            "sequential": _filter_grid_by_ids(sequential_grid, lookup, train_ids, "sequential"),
            "flexible": _filter_grid_by_ids(flexible_grid, lookup, train_ids, "flexible"),
        }
        source_grids = {
            "batched": train_grids["batched"],
            "sequential": train_grids["sequential"],
            "flexible": train_grids["flexible"],
        }
        eval_deployable_baselines = [row for row in _policy_baselines(lookup, eval_ids) if "oracle" not in str(row["policy"])]
        for lambda_value in lambdas:
            best_baseline = max(
                eval_deployable_baselines,
                key=lambda row: float(row["mean_reward"]) - float(lambda_value) * float(row["mean_cost"]),
            )
            best_baseline_utility = float(best_baseline["mean_reward"]) - float(lambda_value) * float(best_baseline["mean_cost"])
            for policy_type, grid_rows in source_grids.items():
                selected_train = max(
                    grid_rows,
                    key=lambda row: float(row["mean_reward"]) - float(lambda_value) * float(row["mean_cost"]),
                )
                eval_row = _evaluate_policy_on_ids(lookup, eval_ids, selected_train, policy_type)
                eval_utility = float(eval_row["mean_reward"]) - float(lambda_value) * float(eval_row["mean_cost"])
                rows.append(
                    {
                        "split_idx": split_idx,
                        "lambda": float(lambda_value),
                        "policy_type": policy_type,
                        "train_utility": float(selected_train["mean_reward"]) - float(lambda_value) * float(selected_train["mean_cost"]),
                        "train_reward": float(selected_train["mean_reward"]),
                        "train_cost": float(selected_train["mean_cost"]),
                        "eval_utility": eval_utility,
                        "eval_reward": float(eval_row["mean_reward"]),
                        "eval_cost": float(eval_row["mean_cost"]),
                        "best_deployable_baseline": str(best_baseline["policy"]),
                        "best_deployable_baseline_utility": best_baseline_utility,
                        "eval_utility_minus_baseline": eval_utility - best_baseline_utility,
                        **{f"eval_{key}": value for key, value in eval_row.items() if key not in {"n", "mean_reward", "mean_cost"}},
                    }
                )
    return rows


def _best_by_utility(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_utility_rows = []
    for lambda_value in [0, 1, 2, 5, 10, 20, 50, 100]:
        best = max(rows, key=lambda row: float(row["mean_reward"]) - float(lambda_value) * float(row["mean_cost"]))
        best_by_utility_rows.append(
            {
                "lambda": float(lambda_value),
                "mean_utility": float(best["mean_reward"]) - float(lambda_value) * float(best["mean_cost"]),
                **best,
            }
        )
    return best_by_utility_rows


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
    parser.add_argument("--max-flex-thresholds", type=int, default=13)
    parser.add_argument("--split-eval-repeats", type=int, default=0)
    parser.add_argument("--split-eval-seed", type=int, default=17)
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
    frontier = _frontier(grid)
    best_by_reward = max(grid, key=lambda row: float(row["mean_reward"]))
    best_by_utility_rows = _best_by_utility(grid)
    sequential_grid = _sequential_resample_grid(lookup, ids, int(args.max_thresholds))
    sequential_frontier = _frontier(sequential_grid)
    sequential_best_by_reward = max(sequential_grid, key=lambda row: float(row["mean_reward"]))
    sequential_best_by_utility_rows = _best_by_utility(sequential_grid)
    flexible_grid = _flexible_state_policy_grid(lookup, ids, int(args.max_flex_thresholds))
    flexible_frontier = _frontier(flexible_grid)
    flexible_best_by_reward = max(flexible_grid, key=lambda row: float(row["mean_reward"]))
    flexible_best_by_utility_rows = _best_by_utility(flexible_grid)
    split_eval_rows: list[dict[str, Any]] = []
    split_eval_summary: list[dict[str, Any]] = []
    if int(args.split_eval_repeats) > 0:
        split_eval_rows = _split_threshold_eval(
            lookup,
            ids,
            grid,
            sequential_grid,
            flexible_grid,
            baselines,
            [0.0, 1.0, 2.0, 5.0, 10.0, 20.0],
            int(args.split_eval_repeats),
            int(args.split_eval_seed),
        )
        for lambda_value in [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]:
            for policy_type in ["batched", "sequential", "flexible"]:
                subset = [row for row in split_eval_rows if row["lambda"] == lambda_value and row["policy_type"] == policy_type]
                if not subset:
                    continue
                split_eval_summary.append(
                    {
                        "lambda": lambda_value,
                        "policy_type": policy_type,
                        "n_splits": len(subset),
                        "mean_eval_reward": float(np.mean([row["eval_reward"] for row in subset])),
                        "mean_eval_cost": float(np.mean([row["eval_cost"] for row in subset])),
                        "mean_eval_utility": float(np.mean([row["eval_utility"] for row in subset])),
                        "mean_eval_utility_minus_baseline": float(np.mean([row["eval_utility_minus_baseline"] for row in subset])),
                        "std_eval_utility_minus_baseline": float(np.std([row["eval_utility_minus_baseline"] for row in subset])),
                        "win_rate_vs_baseline": float(np.mean([row["eval_utility_minus_baseline"] > 0.0 for row in subset])),
                    }
                )

    _write_csv(output_dir / "attempt_metrics_by_route.csv", metrics)
    _write_csv(output_dir / "policy_baselines.csv", baselines)
    _write_csv(output_dir / "branching_policy_grid.csv", grid)
    _write_csv(output_dir / "branching_policy_frontier.csv", frontier)
    _write_csv(output_dir / "branching_policy_best_by_utility.csv", best_by_utility_rows)
    _write_csv(output_dir / "sequential_resample_policy_grid.csv", sequential_grid)
    _write_csv(output_dir / "sequential_resample_policy_frontier.csv", sequential_frontier)
    _write_csv(output_dir / "sequential_resample_policy_best_by_utility.csv", sequential_best_by_utility_rows)
    _write_csv(output_dir / "flexible_state_policy_grid.csv", flexible_grid)
    _write_csv(output_dir / "flexible_state_policy_frontier.csv", flexible_frontier)
    _write_csv(output_dir / "flexible_state_policy_best_by_utility.csv", flexible_best_by_utility_rows)
    if split_eval_rows:
        _write_csv(output_dir / "split_threshold_eval_rows.csv", split_eval_rows)
        _write_csv(output_dir / "split_threshold_eval_summary.csv", split_eval_summary)
    summary = {
        "n_attempts": len(attempts),
        "n_complete_20b_120b_ids": len(ids),
        "route_cost_weights": route_cost_weights,
        "attempt_metrics_by_route": metrics,
        "policy_baselines": baselines,
        "best_branching_by_reward": best_by_reward,
        "best_branching_by_utility": best_by_utility_rows,
        "best_sequential_resample_by_reward": sequential_best_by_reward,
        "best_sequential_resample_by_utility": sequential_best_by_utility_rows,
        "best_flexible_state_policy_by_reward": flexible_best_by_reward,
        "best_flexible_state_policy_by_utility": flexible_best_by_utility_rows,
        "max_flex_thresholds": int(args.max_flex_thresholds),
        "split_threshold_eval_summary": split_eval_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

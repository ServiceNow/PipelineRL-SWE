#!/usr/bin/env python
"""Diagnose whether verifier scores can become calibrated P(success).

This script is intentionally post-hoc. It consumes already-scored rollout
attempts, joins them with real pass/fail labels, fits tiny calibration maps on
held-in problem IDs, and evaluates on heldout problem IDs.

The goal is to separate two failure modes:

1. The verifier score cannot be calibrated into useful absolute P(success).
2. P(success) is usable, but the stop/resample/escalate utility problem still
   needs a better action-value model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_multirollout_verifier_scores import (
    DEFAULT_COST_WEIGHTS,
    ROUTE_LABELS,
    _attempt_lookup,
    _build_attempt_table,
    _read_jsonl,
    _roc_auc,
    _safe_corr,
)


DEFAULT_REPORT_ROOT = (
    "router_analysis/uploaded_eval_full_20260617/"
    "swe_smith_multirollout_eval150_1781382734/logs/run_evaluation"
)
DEFAULT_LAMBDAS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]


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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    p = min(max(float(p), 1.0e-6), 1.0 - 1.0e-6)
    return float(math.log(p / (1.0 - p)))


def _clip_prob(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1.0e-6, 1.0 - 1.0e-6)


def _brier(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(probs) == 0:
        return math.nan
    return float(np.mean((_clip_prob(probs) - labels.astype(float)) ** 2))


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(probs) == 0:
        return math.nan
    p = _clip_prob(probs)
    y = labels.astype(float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    if len(probs) == 0:
        return math.nan
    p = _clip_prob(probs)
    y = labels.astype(float)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    bucket_ids = np.digitize(p, edges[1:-1], right=False)
    total = float(len(p))
    error = 0.0
    for bucket_idx in range(int(bins)):
        mask = bucket_ids == bucket_idx
        if not np.any(mask):
            continue
        error += float(np.sum(mask)) / total * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return float(error)


def _reliability_rows(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    model: str,
    split_idx: int,
    level: str,
    bins: int = 10,
) -> list[dict[str, Any]]:
    p = _clip_prob(probs)
    y = labels.astype(float)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    bucket_ids = np.digitize(p, edges[1:-1], right=False)
    rows: list[dict[str, Any]] = []
    for bucket_idx in range(int(bins)):
        mask = bucket_ids == bucket_idx
        row: dict[str, Any] = {
            "split_idx": int(split_idx),
            "level": level,
            "model": model,
            "bucket_idx": int(bucket_idx),
            "bucket_low": float(edges[bucket_idx]),
            "bucket_high": float(edges[bucket_idx + 1]),
            "n": int(np.sum(mask)),
        }
        if np.any(mask):
            row.update(
                {
                    "mean_pred": float(np.mean(p[mask])),
                    "mean_true": float(np.mean(y[mask])),
                    "abs_gap": abs(float(np.mean(p[mask])) - float(np.mean(y[mask]))),
                }
            )
        else:
            row.update({"mean_pred": None, "mean_true": None, "abs_gap": None})
        rows.append(row)
    return rows


def _metric_row(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    model: str,
    split_idx: int,
    level: str,
    route_idx: int | None = None,
) -> dict[str, Any]:
    p = _clip_prob(probs)
    y = labels.astype(int)
    return {
        "split_idx": int(split_idx),
        "level": level,
        "model": model,
        "route_idx": route_idx,
        "route": ROUTE_LABELS.get(route_idx, str(route_idx)) if route_idx is not None else None,
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else math.nan,
        "auc": _roc_auc(p, y) if len(y) else math.nan,
        "pearson": _safe_corr(p, y.astype(float)) if len(y) else math.nan,
        "brier": _brier(p, y),
        "nll": _nll(p, y),
        "ece": _ece(p, y),
        "mean_pred": float(np.mean(p)) if len(p) else math.nan,
    }


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, l2: float, steps: int, lr: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 0:
        prior = (float(np.sum(y)) + 0.5) / (len(y) + 1.0)
        return np.asarray([_logit(prior)], dtype=float)
    design = np.concatenate([np.ones((x.shape[0], 1), dtype=float), x], axis=1)
    prior = (float(np.sum(y)) + 0.5) / (len(y) + 1.0)
    weights = np.zeros(design.shape[1], dtype=float)
    weights[0] = _logit(prior)
    reg_mask = np.ones_like(weights)
    reg_mask[0] = 0.0
    if len(set(y.astype(int).tolist())) < 2:
        return np.asarray([_logit(prior)] + [0.0] * x.shape[1], dtype=float)
    for _ in range(int(steps)):
        probs = _sigmoid(design @ weights)
        grad = (design.T @ (probs - y)) / float(len(y))
        grad += float(l2) * reg_mask * weights / float(len(y))
        weights -= float(lr) * grad
    return weights


def _predict_logistic(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if len(weights) == 1:
        return _sigmoid(np.full(x.shape[0], float(weights[0]), dtype=float))
    design = np.concatenate([np.ones((x.shape[0], 1), dtype=float), x], axis=1)
    return _sigmoid(design @ weights)


def _attempt_features(rows: list[dict[str, Any]], mode: str, route_values: list[int]) -> np.ndarray:
    score = np.asarray([float(row["pred_score"]) for row in rows], dtype=float)
    log_tokens = np.asarray([math.log1p(float(row.get("output_tokens") or 0.0)) / 10.0 for row in rows], dtype=float)
    if mode == "global":
        return score[:, None]
    if mode == "global_len":
        return np.column_stack([score, log_tokens])
    if mode == "route_intercept":
        columns = [score]
        routes = [int(row["route_idx"]) for row in rows]
        for route_idx in route_values[1:]:
            columns.append(np.asarray([1.0 if route == route_idx else 0.0 for route in routes], dtype=float))
        return np.column_stack(columns)
    if mode == "route_intercept_len":
        columns = [score, log_tokens]
        routes = [int(row["route_idx"]) for row in rows]
        for route_idx in route_values[1:]:
            columns.append(np.asarray([1.0 if route == route_idx else 0.0 for route in routes], dtype=float))
        return np.column_stack(columns)
    raise ValueError(f"Unknown attempt calibration mode: {mode}")


def _state_features(rows: list[dict[str, Any]], mode: str) -> np.ndarray:
    score = np.asarray([float(row["best_score"]) for row in rows], dtype=float)
    n_seen = np.asarray([float(row["n_seen"]) for row in rows], dtype=float)
    margin = np.asarray([float(row["margin"]) for row in rows], dtype=float)
    mean_score = np.asarray([float(row["mean_score"]) for row in rows], dtype=float)
    std_score = np.asarray([float(row["std_score"]) for row in rows], dtype=float)
    log_tokens = np.asarray([math.log1p(float(row.get("best_output_tokens") or 0.0)) / 10.0 for row in rows], dtype=float)
    if mode == "state_score":
        return score[:, None]
    if mode == "state_score_n":
        return np.column_stack([score, n_seen / 3.0])
    if mode == "state_full":
        return np.column_stack([score, n_seen / 3.0, margin, mean_score, std_score, log_tokens])
    raise ValueError(f"Unknown state calibration mode: {mode}")


def _complete_ids(lookup: dict[tuple[str, int, int], dict[str, Any]]) -> list[str]:
    ids = sorted({pid for pid, _, _ in lookup})
    return [
        pid
        for pid in ids
        if all((pid, 1, rollout_idx) in lookup for rollout_idx in (0, 1, 2)) and (pid, 3, 0) in lookup
    ]


def _attempt_rows_for_ids(
    lookup: dict[tuple[str, int, int], dict[str, Any]],
    ids: list[str],
    route_rollouts: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid in ids:
        for route_idx, rollout_idx in route_rollouts:
            key = (pid, int(route_idx), int(rollout_idx))
            if key in lookup:
                rows.append(lookup[key])
    return rows


def _state_rows_for_ids(lookup: dict[tuple[str, int, int], dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid in ids:
        seen: list[dict[str, Any]] = []
        for rollout_idx in (0, 1, 2):
            attempt = lookup[(pid, 1, rollout_idx)]
            seen.append(attempt)
            ranked = sorted(seen, key=lambda row: float(row["pred_score"]), reverse=True)
            best = ranked[0]
            second = ranked[1] if len(ranked) > 1 else ranked[0]
            scores = np.asarray([float(row["pred_score"]) for row in seen], dtype=float)
            rows.append(
                {
                    "original_problem_id": pid,
                    "n_seen": len(seen),
                    "best_route_idx": int(best["route_idx"]),
                    "best_rollout_idx": int(best["rollout_idx"]),
                    "best_score": float(best["pred_score"]),
                    "margin": float(best["pred_score"]) - float(second["pred_score"]),
                    "mean_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "best_output_tokens": float(best.get("output_tokens") or 0.0),
                    "real_success": int(best["real_success"]),
                    "cost_so_far": float(sum(float(row["weighted_output_cost"]) for row in seen)),
                    "r120": int(lookup[(pid, 3, 0)]["real_success"]),
                    "c120": float(lookup[(pid, 3, 0)]["weighted_output_cost"]),
                }
            )
    return rows


def _score_state_map(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    mode: str,
    *,
    l2: float,
    steps: int,
    lr: float,
) -> dict[tuple[str, int], float]:
    y_train = np.asarray([int(row["real_success"]) for row in train_rows], dtype=int)
    weights = _fit_logistic(_state_features(train_rows, mode), y_train, l2=l2, steps=steps, lr=lr)
    probs = _predict_logistic(weights, _state_features(eval_rows, mode))
    return {(str(row["original_problem_id"]), int(row["n_seen"])): float(prob) for row, prob in zip(eval_rows, probs)}


def _state_examples(
    lookup: dict[tuple[str, int, int], dict[str, Any]],
    ids: list[str],
    state_scores: dict[tuple[str, int], float] | None = None,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for pid in ids:
        attempts = [lookup[(pid, 1, rollout_idx)] for rollout_idx in (0, 1, 2)]
        best_after = []
        for k in (1, 2, 3):
            best_after.append(max(attempts[:k], key=lambda row: float(row["pred_score"])))
        escalated = lookup[(pid, 3, 0)]
        cost_prefix = []
        total = 0.0
        for attempt in attempts:
            total += float(attempt["weighted_output_cost"])
            cost_prefix.append(total)

        def score_for(k: int, best: dict[str, Any]) -> float:
            if state_scores is None:
                return float(best["pred_score"])
            return float(state_scores[(pid, k)])

        examples.append(
            {
                "pid": pid,
                "s1": score_for(1, best_after[0]),
                "s2": score_for(2, best_after[1]),
                "s3": score_for(3, best_after[2]),
                "r1": float(best_after[0]["real_success"]),
                "r2": float(best_after[1]["real_success"]),
                "r3": float(best_after[2]["real_success"]),
                "r120": float(escalated["real_success"]),
                "c1": float(cost_prefix[0]),
                "c2": float(cost_prefix[1]),
                "c3": float(cost_prefix[2]),
                "c120": float(escalated["weighted_output_cost"]),
            }
        )
    return examples


def _threshold_candidates(examples: list[dict[str, Any]], max_thresholds: int) -> list[float]:
    values = np.asarray([float(ex[key]) for ex in examples for key in ("s1", "s2", "s3")], dtype=float)
    quantiles = np.linspace(0.0, 1.0, int(max_thresholds))
    candidates = sorted(set(float(value) for value in np.quantile(values, quantiles)))
    eps = max(1.0e-6, float(np.std(values)) * 1.0e-6)
    return [float(np.min(values) - eps)] + candidates + [float(np.max(values) + eps)]


def _eval_policy(examples: list[dict[str, Any]], policy: dict[str, float]) -> dict[str, Any]:
    rewards: list[float] = []
    costs: list[float] = []
    actions = defaultdict(int)
    rollouts: list[int] = []
    for ex in examples:
        if float(ex["s1"]) >= float(policy["submit_tau_1"]):
            rewards.append(float(ex["r1"]))
            costs.append(float(ex["c1"]))
            actions["submit_20b_after_1"] += 1
            rollouts.append(1)
        elif float(ex["s1"]) < float(policy["escalate_tau_1"]):
            rewards.append(float(ex["r120"]))
            costs.append(float(ex["c1"]) + float(ex["c120"]))
            actions["escalate_120b_after_1"] += 1
            rollouts.append(1)
        elif float(ex["s2"]) >= float(policy["submit_tau_2"]):
            rewards.append(float(ex["r2"]))
            costs.append(float(ex["c2"]))
            actions["submit_20b_after_2"] += 1
            rollouts.append(2)
        elif float(ex["s2"]) < float(policy["escalate_tau_2"]):
            rewards.append(float(ex["r120"]))
            costs.append(float(ex["c2"]) + float(ex["c120"]))
            actions["escalate_120b_after_2"] += 1
            rollouts.append(2)
        elif float(ex["s3"]) < float(policy["final_escalate_tau"]):
            rewards.append(float(ex["r120"]))
            costs.append(float(ex["c3"]) + float(ex["c120"]))
            actions["escalate_120b_after_3"] += 1
            rollouts.append(3)
        else:
            rewards.append(float(ex["r3"]))
            costs.append(float(ex["c3"]))
            actions["submit_20b_after_3"] += 1
            rollouts.append(3)
    row = {
        "mean_reward": float(np.mean(rewards)) if rewards else math.nan,
        "mean_cost": float(np.mean(costs)) if costs else math.nan,
        "mean_20b_rollouts": float(np.mean(rollouts)) if rollouts else math.nan,
    }
    row.update({key: int(value) for key, value in actions.items()})
    row["escalate_120b"] = int(sum(value for key, value in actions.items() if key.startswith("escalate_120b")))
    return row


def _grid_policies(examples: list[dict[str, Any]], max_thresholds: int) -> list[dict[str, float]]:
    candidates = _threshold_candidates(examples, max_thresholds)
    pairs = [(lo, hi) for lo in candidates for hi in candidates if lo <= hi]
    policies: list[dict[str, float]] = []
    for escalate_tau_1, submit_tau_1 in pairs:
        for escalate_tau_2, submit_tau_2 in pairs:
            for final_escalate_tau in candidates:
                policies.append(
                    {
                        "escalate_tau_1": float(escalate_tau_1),
                        "submit_tau_1": float(submit_tau_1),
                        "escalate_tau_2": float(escalate_tau_2),
                        "submit_tau_2": float(submit_tau_2),
                        "final_escalate_tau": float(final_escalate_tau),
                    }
                )
    return policies


def _oracle_dynamic(examples: list[dict[str, Any]], lambda_value: float) -> dict[str, Any]:
    rewards: list[float] = []
    costs: list[float] = []
    actions = defaultdict(int)
    rollouts: list[int] = []

    def best_action(ex: dict[str, Any], k: int) -> tuple[float, float, str, int]:
        submit_reward = float(ex[f"r{k}"])
        submit_cost = float(ex[f"c{k}"])
        choices = [(submit_reward - float(lambda_value) * submit_cost, submit_reward, submit_cost, f"submit_20b_after_{k}", k)]
        escalate_reward = float(ex["r120"])
        escalate_cost = float(ex[f"c{k}"]) + float(ex["c120"])
        choices.append(
            (
                escalate_reward - float(lambda_value) * escalate_cost,
                escalate_reward,
                escalate_cost,
                f"escalate_120b_after_{k}",
                k,
            )
        )
        if k < 3:
            future = best_action(ex, k + 1)
            choices.append((future[0] - float(lambda_value) * 0.0, future[1], future[2], future[3], future[4]))
        return max(choices, key=lambda item: item[0])

    for ex in examples:
        _, reward, cost, action, n_rollouts = best_action(ex, 1)
        rewards.append(float(reward))
        costs.append(float(cost))
        actions[action] += 1
        rollouts.append(int(n_rollouts))
    row = {
        "mean_reward": float(np.mean(rewards)) if rewards else math.nan,
        "mean_cost": float(np.mean(costs)) if costs else math.nan,
        "mean_utility": float(np.mean(rewards) - float(lambda_value) * np.mean(costs)) if rewards else math.nan,
        "mean_20b_rollouts": float(np.mean(rollouts)) if rollouts else math.nan,
    }
    row.update({key: int(value) for key, value in actions.items()})
    row["escalate_120b"] = int(sum(value for key, value in actions.items() if key.startswith("escalate_120b")))
    return row


def _baselines(examples: list[dict[str, Any]], lambda_value: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, reward_key, cost_expr in [
        ("20B_r0", "r1", lambda ex: ex["c1"]),
        ("20B_x3_pred_best", "r3", lambda ex: ex["c3"]),
        ("120B_r0", "r120", lambda ex: ex["c120"]),
    ]:
        rewards = [float(ex[reward_key]) for ex in examples]
        costs = [float(cost_expr(ex)) for ex in examples]
        rows.append(
            {
                "policy": name,
                "mean_reward": float(np.mean(rewards)),
                "mean_cost": float(np.mean(costs)),
                "mean_utility": float(np.mean(rewards) - float(lambda_value) * np.mean(costs)),
            }
        )
    return rows


def _train_eval_policy_rows(
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    *,
    score_model: str,
    split_idx: int,
    lambdas: list[float],
    max_thresholds: int,
) -> list[dict[str, Any]]:
    policies = _grid_policies(train_examples, max_thresholds)
    train_eval = [{**policy, **_eval_policy(train_examples, policy)} for policy in policies]
    rows: list[dict[str, Any]] = []
    for lambda_value in lambdas:
        selected = max(
            train_eval,
            key=lambda row: float(row["mean_reward"]) - float(lambda_value) * float(row["mean_cost"]),
        )
        eval_row = _eval_policy(eval_examples, selected)
        eval_row["mean_utility"] = float(eval_row["mean_reward"]) - float(lambda_value) * float(eval_row["mean_cost"])
        base_rows = _baselines(eval_examples, float(lambda_value))
        best_baseline = max(base_rows, key=lambda row: float(row["mean_utility"]))
        oracle_row = _oracle_dynamic(eval_examples, float(lambda_value))
        rows.append(
            {
                "split_idx": int(split_idx),
                "score_model": score_model,
                "lambda": float(lambda_value),
                "train_utility": float(selected["mean_reward"]) - float(lambda_value) * float(selected["mean_cost"]),
                "eval_utility_minus_best_baseline": float(eval_row["mean_utility"]) - float(best_baseline["mean_utility"]),
                "best_baseline": str(best_baseline["policy"]),
                "best_baseline_utility": float(best_baseline["mean_utility"]),
                "oracle_dynamic_utility": float(oracle_row["mean_utility"]),
                "oracle_dynamic_reward": float(oracle_row["mean_reward"]),
                "oracle_dynamic_cost": float(oracle_row["mean_cost"]),
                **{f"eval_{key}": value for key, value in eval_row.items()},
                **{f"chosen_{key}": value for key, value in selected.items() if "tau" in key},
            }
        )
    return rows


def _summarize(rows: list[dict[str, Any]], group_keys: list[str], metric_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, subset in sorted(groups.items()):
        result = dict(zip(group_keys, key))
        result["n"] = len(subset)
        for metric in metric_keys:
            values = np.asarray(
                [float(row[metric]) for row in subset if row.get(metric) is not None and math.isfinite(float(row[metric]))],
                dtype=float,
            )
            result[f"{metric}_mean"] = float(np.mean(values)) if len(values) else math.nan
            result[f"{metric}_std"] = float(np.std(values)) if len(values) else math.nan
        out.append(result)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-jsonl", required=True)
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--route-cost-weights", default=",".join(str(value) for value in DEFAULT_COST_WEIGHTS))
    parser.add_argument("--attempt-modes", default="global,global_len,route_intercept,route_intercept_len")
    parser.add_argument("--state-modes", default="state_score,state_score_n,state_full")
    parser.add_argument("--lambdas", default=",".join(str(value) for value in DEFAULT_LAMBDAS))
    parser.add_argument("--n-splits", type=int, default=50)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-thresholds", type=int, default=5)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    route_cost_weights = [float(part.strip()) for part in str(args.route_cost_weights).split(",") if part.strip()]
    lambdas = [float(part.strip()) for part in str(args.lambdas).split(",") if part.strip()]
    attempt_modes = [part.strip() for part in str(args.attempt_modes).split(",") if part.strip()]
    state_modes = [part.strip() for part in str(args.state_modes).split(",") if part.strip()]

    attempts = _build_attempt_table(_read_jsonl(Path(args.scores_jsonl)), Path(args.report_root), route_cost_weights)
    lookup = _attempt_lookup(attempts)
    ids = _complete_ids(lookup)
    if len(ids) < 4:
        raise ValueError(f"Too few complete ids for diagnostics: {len(ids)}")
    route_rollouts = [(1, 0), (1, 1), (1, 2), (3, 0)]
    route_values = [1, 3]

    calibration_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    rng = random.Random(int(args.seed))
    for split_idx in range(int(args.n_splits)):
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n_train = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * float(args.train_frac)))))
        train_ids = sorted(shuffled[:n_train])
        eval_ids = sorted(shuffled[n_train:])

        train_attempts = _attempt_rows_for_ids(lookup, train_ids, route_rollouts)
        eval_attempts = _attempt_rows_for_ids(lookup, eval_ids, route_rollouts)
        train_states = _state_rows_for_ids(lookup, train_ids)
        eval_states = _state_rows_for_ids(lookup, eval_ids)

        raw_attempt_probs = np.asarray([float(row["pred_score"]) for row in eval_attempts], dtype=float)
        attempt_labels = np.asarray([int(row["real_success"]) for row in eval_attempts], dtype=int)
        calibration_rows.append(
            _metric_row(raw_attempt_probs, attempt_labels, model="raw_score", split_idx=split_idx, level="attempt")
        )
        reliability_rows.extend(
            _reliability_rows(raw_attempt_probs, attempt_labels, model="raw_score", split_idx=split_idx, level="attempt")
        )
        for route_idx in route_values:
            subset = [row for row in eval_attempts if int(row["route_idx"]) == route_idx]
            calibration_rows.append(
                _metric_row(
                    np.asarray([float(row["pred_score"]) for row in subset], dtype=float),
                    np.asarray([int(row["real_success"]) for row in subset], dtype=int),
                    model="raw_score",
                    split_idx=split_idx,
                    level="attempt_by_route",
                    route_idx=route_idx,
                )
            )

        for mode in attempt_modes:
            y_train = np.asarray([int(row["real_success"]) for row in train_attempts], dtype=int)
            weights = _fit_logistic(
                _attempt_features(train_attempts, mode, route_values),
                y_train,
                l2=float(args.l2),
                steps=int(args.steps),
                lr=float(args.lr),
            )
            probs = _predict_logistic(weights, _attempt_features(eval_attempts, mode, route_values))
            calibration_rows.append(_metric_row(probs, attempt_labels, model=mode, split_idx=split_idx, level="attempt"))
            reliability_rows.extend(_reliability_rows(probs, attempt_labels, model=mode, split_idx=split_idx, level="attempt"))
            for route_idx in route_values:
                subset_idx = [idx for idx, row in enumerate(eval_attempts) if int(row["route_idx"]) == route_idx]
                calibration_rows.append(
                    _metric_row(
                        probs[subset_idx],
                        attempt_labels[subset_idx],
                        model=mode,
                        split_idx=split_idx,
                        level="attempt_by_route",
                        route_idx=route_idx,
                    )
                )

        raw_state_probs = np.asarray([float(row["best_score"]) for row in eval_states], dtype=float)
        state_labels = np.asarray([int(row["real_success"]) for row in eval_states], dtype=int)
        calibration_rows.append(
            _metric_row(raw_state_probs, state_labels, model="raw_state_score", split_idx=split_idx, level="best_seen_state")
        )
        reliability_rows.extend(
            _reliability_rows(raw_state_probs, state_labels, model="raw_state_score", split_idx=split_idx, level="best_seen_state")
        )
        raw_train_examples = _state_examples(lookup, train_ids, None)
        raw_eval_examples = _state_examples(lookup, eval_ids, None)
        policy_rows.extend(
            _train_eval_policy_rows(
                raw_train_examples,
                raw_eval_examples,
                score_model="raw_state_score",
                split_idx=split_idx,
                lambdas=lambdas,
                max_thresholds=int(args.max_thresholds),
            )
        )

        for mode in state_modes:
            y_train = np.asarray([int(row["real_success"]) for row in train_states], dtype=int)
            weights = _fit_logistic(
                _state_features(train_states, mode),
                y_train,
                l2=float(args.l2),
                steps=int(args.steps),
                lr=float(args.lr),
            )
            probs = _predict_logistic(weights, _state_features(eval_states, mode))
            calibration_rows.append(
                _metric_row(probs, state_labels, model=mode, split_idx=split_idx, level="best_seen_state")
            )
            reliability_rows.extend(
                _reliability_rows(probs, state_labels, model=mode, split_idx=split_idx, level="best_seen_state")
            )
            state_score_map = _score_state_map(
                train_states,
                eval_states,
                mode,
                l2=float(args.l2),
                steps=int(args.steps),
                lr=float(args.lr),
            )
            train_score_map = _score_state_map(
                train_states,
                train_states,
                mode,
                l2=float(args.l2),
                steps=int(args.steps),
                lr=float(args.lr),
            )
            policy_rows.extend(
                _train_eval_policy_rows(
                    _state_examples(lookup, train_ids, train_score_map),
                    _state_examples(lookup, eval_ids, state_score_map),
                    score_model=mode,
                    split_idx=split_idx,
                    lambdas=lambdas,
                    max_thresholds=int(args.max_thresholds),
                )
            )

    calibration_summary = _summarize(
        calibration_rows,
        ["level", "model"],
        ["auc", "pearson", "brier", "nll", "ece", "mean_pred", "positive_rate"],
    )
    route_calibration_summary = _summarize(
        [row for row in calibration_rows if row["level"] == "attempt_by_route"],
        ["route", "model"],
        ["auc", "pearson", "brier", "nll", "ece", "mean_pred", "positive_rate"],
    )
    policy_summary = _summarize(
        policy_rows,
        ["score_model", "lambda"],
        [
            "eval_mean_reward",
            "eval_mean_cost",
            "eval_mean_utility",
            "eval_utility_minus_best_baseline",
            "oracle_dynamic_utility",
            "oracle_dynamic_reward",
            "oracle_dynamic_cost",
            "eval_escalate_120b",
            "eval_mean_20b_rollouts",
        ],
    )

    _write_csv(output_dir / "calibration_rows.csv", calibration_rows)
    _write_csv(output_dir / "calibration_summary.csv", calibration_summary)
    _write_csv(output_dir / "route_calibration_summary.csv", route_calibration_summary)
    _write_csv(output_dir / "reliability_rows.csv", reliability_rows)
    _write_csv(output_dir / "policy_rows.csv", policy_rows)
    _write_csv(output_dir / "policy_summary.csv", policy_summary)

    manifest = {
        "scores_jsonl": str(args.scores_jsonl),
        "report_root": str(args.report_root),
        "output_dir": str(output_dir),
        "n_attempts_joined": len(attempts),
        "n_complete_ids": len(ids),
        "route_cost_weights": route_cost_weights,
        "route_rollouts": route_rollouts,
        "attempt_modes": attempt_modes,
        "state_modes": state_modes,
        "lambdas": lambdas,
        "n_splits": int(args.n_splits),
        "train_frac": float(args.train_frac),
        "seed": int(args.seed),
        "max_thresholds": int(args.max_thresholds),
        "calibration_summary": calibration_summary,
        "route_calibration_summary": route_calibration_summary,
        "policy_summary": policy_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

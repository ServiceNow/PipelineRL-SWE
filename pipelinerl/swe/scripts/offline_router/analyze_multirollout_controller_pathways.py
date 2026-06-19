#!/usr/bin/env python
"""CPU-only controller diagnostics for SWE-Smith multi-rollout verifier scores.

This intentionally avoids proxy-trained controllers. It consumes existing
verifier scores plus real pass/fail labels and asks four bounded questions:

1. What is the controller ceiling if ranking/control were oracle?
2. Can a tiny learned controller generalize on heldout problem IDs?
3. How do calibrated verifier scores behave? (kept in the separate launcher)
4. How far do threshold/controller policy families get? (kept in the separate launcher)
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
    _attempt_lookup,
    _build_attempt_table,
    _read_jsonl,
)


DEFAULT_REPORT_ROOT = (
    "router_analysis/uploaded_eval_full_20260617/"
    "swe_smith_multirollout_eval150_1781382734/logs/run_evaluation"
)
DEFAULT_LAMBDAS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
ACTION_NAMES = ["submit_20b", "resample_20b", "escalate_120b"]
SUBMIT, RESAMPLE, ESCALATE = 0, 1, 2


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


def _summarize(rows: list[dict[str, Any]], group_keys: list[str], metric_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, subset in sorted(groups.items()):
        result = dict(zip(group_keys, key))
        result["n"] = len(subset)
        for metric in metric_keys:
            vals = [
                float(row[metric])
                for row in subset
                if row.get(metric) is not None and math.isfinite(float(row[metric]))
            ]
            result[f"{metric}_mean"] = float(np.mean(vals)) if vals else math.nan
            result[f"{metric}_std"] = float(np.std(vals)) if vals else math.nan
        out.append(result)
    return out


def _complete_ids(lookup: dict[tuple[str, int, int], dict[str, Any]]) -> list[str]:
    ids = sorted({pid for pid, _, _ in lookup})
    return [
        pid
        for pid in ids
        if all((pid, 1, rollout_idx) in lookup for rollout_idx in (0, 1, 2))
        and (pid, 3, 0) in lookup
    ]


def _build_examples(
    lookup: dict[tuple[str, int, int], dict[str, Any]],
    ids: list[str],
    *,
    score_source: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for pid in ids:
        attempts = [lookup[(pid, 1, rollout_idx)] for rollout_idx in (0, 1, 2)]
        scores = [float(row["pred_score"]) for row in attempts]
        costs = [float(row["weighted_output_cost"]) for row in attempts]
        successes = [float(row["real_success"]) for row in attempts]
        c_prefix = np.cumsum(np.asarray(costs, dtype=float)).tolist()
        escalated = lookup[(pid, 3, 0)]
        ex: dict[str, Any] = {
            "pid": pid,
            "r120": float(escalated["real_success"]),
            "c120": float(escalated["weighted_output_cost"]),
        }
        for k in (1, 2, 3):
            if score_source == "learned":
                best_idx = int(np.argmax(scores[:k]))
                state_score = float(scores[best_idx])
                state_reward = float(successes[best_idx])
                margin_source = sorted(scores[:k], reverse=True)
            elif score_source == "oracle":
                best_idx = int(np.argmax(successes[:k]))
                state_score = float(successes[best_idx])
                state_reward = float(max(successes[:k]))
                margin_source = sorted(successes[:k], reverse=True)
            else:
                raise ValueError(f"Unknown score_source={score_source!r}")
            second = margin_source[1] if len(margin_source) > 1 else margin_source[0]
            ex[f"s{k}"] = state_score
            ex[f"r{k}"] = state_reward
            ex[f"c{k}"] = float(c_prefix[k - 1])
            ex[f"margin{k}"] = float(margin_source[0] - second)
            ex[f"mean_score{k}"] = float(np.mean(scores[:k]))
            ex[f"std_score{k}"] = float(np.std(scores[:k]))
            ex[f"best_output_tokens{k}"] = float(attempts[best_idx].get("output_tokens") or 0.0)
            for rollout_idx in (0, 1, 2):
                ex[f"score_r{rollout_idx}_at{k}"] = float(scores[rollout_idx]) if rollout_idx < k else -1.0
        examples.append(ex)
    return examples


def _threshold_candidates(examples: list[dict[str, Any]], max_thresholds: int) -> list[float]:
    vals = np.asarray([float(ex[f"s{k}"]) for ex in examples for k in (1, 2, 3)], dtype=float)
    quantiles = np.linspace(0.0, 1.0, int(max_thresholds))
    candidates = sorted(set(float(value) for value in np.quantile(vals, quantiles)))
    eps = max(1.0e-6, float(np.std(vals)) * 1.0e-6)
    return [float(np.min(vals) - eps)] + candidates + [float(np.max(vals) + eps)]


def _policy_grid(examples: list[dict[str, Any]], max_thresholds: int) -> list[dict[str, float]]:
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


def _eval_threshold_policy(examples: list[dict[str, Any]], policy: dict[str, float]) -> dict[str, Any]:
    rewards: list[float] = []
    costs: list[float] = []
    actions = defaultdict(int)
    n_rollouts: list[int] = []
    for ex in examples:
        if float(ex["s1"]) >= float(policy["submit_tau_1"]):
            reward = float(ex["r1"])
            cost = float(ex["c1"])
            action = "submit_20b_after_1"
            k = 1
        elif float(ex["s1"]) < float(policy["escalate_tau_1"]):
            reward = float(ex["r120"])
            cost = float(ex["c1"]) + float(ex["c120"])
            action = "escalate_120b_after_1"
            k = 1
        elif float(ex["s2"]) >= float(policy["submit_tau_2"]):
            reward = float(ex["r2"])
            cost = float(ex["c2"])
            action = "submit_20b_after_2"
            k = 2
        elif float(ex["s2"]) < float(policy["escalate_tau_2"]):
            reward = float(ex["r120"])
            cost = float(ex["c2"]) + float(ex["c120"])
            action = "escalate_120b_after_2"
            k = 2
        elif float(ex["s3"]) < float(policy["final_escalate_tau"]):
            reward = float(ex["r120"])
            cost = float(ex["c3"]) + float(ex["c120"])
            action = "escalate_120b_after_3"
            k = 3
        else:
            reward = float(ex["r3"])
            cost = float(ex["c3"])
            action = "submit_20b_after_3"
            k = 3
        rewards.append(reward)
        costs.append(cost)
        actions[action] += 1
        n_rollouts.append(k)
    row = {
        "mean_reward": float(np.mean(rewards)) if rewards else math.nan,
        "mean_cost": float(np.mean(costs)) if costs else math.nan,
        "mean_20b_rollouts": float(np.mean(n_rollouts)) if n_rollouts else math.nan,
    }
    row.update({key: int(value) for key, value in actions.items()})
    row["escalate_120b"] = int(sum(value for key, value in actions.items() if key.startswith("escalate_120b")))
    return row


def _oracle_dynamic_policy(examples: list[dict[str, Any]], lambda_value: float) -> dict[str, Any]:
    rewards: list[float] = []
    costs: list[float] = []
    actions = defaultdict(int)
    rollouts: list[int] = []
    for ex in examples:
        values: dict[int, tuple[float, float, float, str, int]] = {}
        for k in (3, 2, 1):
            submit = (
                float(ex[f"r{k}"]) - float(lambda_value) * float(ex[f"c{k}"]),
                float(ex[f"r{k}"]),
                float(ex[f"c{k}"]),
                f"submit_20b_after_{k}",
                k,
            )
            escalate = (
                float(ex["r120"]) - float(lambda_value) * (float(ex[f"c{k}"]) + float(ex["c120"])),
                float(ex["r120"]),
                float(ex[f"c{k}"]) + float(ex["c120"]),
                f"escalate_120b_after_{k}",
                k,
            )
            choices = [submit, escalate]
            if k < 3:
                choices.append(values[k + 1])
            values[k] = max(choices, key=lambda item: item[0])
        _, reward, cost, action, k_used = values[1]
        rewards.append(reward)
        costs.append(cost)
        actions[action] += 1
        rollouts.append(k_used)
    row = {
        "mean_reward": float(np.mean(rewards)) if rewards else math.nan,
        "mean_cost": float(np.mean(costs)) if costs else math.nan,
        "mean_utility": float(np.mean(rewards) - float(lambda_value) * np.mean(costs)) if rewards else math.nan,
        "mean_20b_rollouts": float(np.mean(rollouts)) if rollouts else math.nan,
    }
    row.update({key: int(value) for key, value in actions.items()})
    row["escalate_120b"] = int(sum(value for key, value in actions.items() if key.startswith("escalate_120b")))
    return row


def _baseline_rows(examples: list[dict[str, Any]], lambda_value: float) -> list[dict[str, Any]]:
    specs = [
        ("always_20b_r0", "r1", "c1"),
        ("always_20b_x3_learned_best", "r3", "c3"),
        ("always_120b_r0", "r120", "c120"),
    ]
    rows: list[dict[str, Any]] = []
    for name, reward_key, cost_key in specs:
        rewards = np.asarray([float(ex[reward_key]) for ex in examples], dtype=float)
        costs = np.asarray([float(ex[cost_key]) for ex in examples], dtype=float)
        rows.append(
            {
                "policy": name,
                "mean_reward": float(np.mean(rewards)),
                "mean_cost": float(np.mean(costs)),
                "mean_utility": float(np.mean(rewards) - float(lambda_value) * np.mean(costs)),
            }
        )
    oracle_rewards = np.asarray([max(float(ex["r3"]), float(ex["r120"])) for ex in examples], dtype=float)
    oracle_costs = np.asarray([float(ex["c3"]) + float(ex["c120"]) for ex in examples], dtype=float)
    rows.append(
        {
            "policy": "oracle_best_after_20b_x3_and_120b",
            "mean_reward": float(np.mean(oracle_rewards)),
            "mean_cost": float(np.mean(oracle_costs)),
            "mean_utility": float(np.mean(oracle_rewards) - float(lambda_value) * np.mean(oracle_costs)),
        }
    )
    return rows


def _ceiling_analysis(
    lookup: dict[tuple[str, int, int], dict[str, Any]],
    ids: list[str],
    *,
    output_dir: Path,
    lambdas: list[float],
    max_thresholds: int,
    n_splits: int,
    seed: int,
) -> dict[str, Any]:
    learned_examples = _build_examples(lookup, ids, score_source="learned")
    oracle_examples = _build_examples(lookup, ids, score_source="oracle")
    learned_policies = _policy_grid(learned_examples, max_thresholds)
    oracle_score_policies = _policy_grid(oracle_examples, max_thresholds)
    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    rng = random.Random(int(seed))
    for lambda_value in lambdas:
        for base in _baseline_rows(learned_examples, lambda_value):
            rows.append({"lambda": lambda_value, "score_source": "n/a", "policy_type": base["policy"], **base})
        oracle_dyn = _oracle_dynamic_policy(learned_examples, lambda_value)
        rows.append({"lambda": lambda_value, "score_source": "learned_rank_oracle_control", "policy_type": "oracle_dynamic", **oracle_dyn})
        oracle_verifier_dyn = _oracle_dynamic_policy(oracle_examples, lambda_value)
        rows.append({"lambda": lambda_value, "score_source": "oracle_rank_oracle_control", "policy_type": "oracle_dynamic", **oracle_verifier_dyn})
        for source_name, examples, policies in [
            ("learned_score", learned_examples, learned_policies),
            ("oracle_score", oracle_examples, oracle_score_policies),
        ]:
            evals = [{**policy, **_eval_threshold_policy(examples, policy)} for policy in policies]
            selected = max(evals, key=lambda row: float(row["mean_reward"]) - lambda_value * float(row["mean_cost"]))
            rows.append(
                {
                    "lambda": lambda_value,
                    "score_source": source_name,
                    "policy_type": "full_eval_tuned_threshold",
                    "mean_utility": float(selected["mean_reward"]) - lambda_value * float(selected["mean_cost"]),
                    **selected,
                }
            )

    for split_idx in range(int(n_splits)):
        shuffled = list(ids)
        rng.shuffle(shuffled)
        mid = len(shuffled) // 2
        train_ids = sorted(shuffled[:mid])
        eval_ids = sorted(shuffled[mid:])
        for source_name, all_examples in [("learned_score", learned_examples), ("oracle_score", oracle_examples)]:
            by_id = {str(ex["pid"]): ex for ex in all_examples}
            train_examples = [by_id[pid] for pid in train_ids]
            eval_examples = [by_id[pid] for pid in eval_ids]
            policies = _policy_grid(train_examples, max_thresholds)
            train_evals = [{**policy, **_eval_threshold_policy(train_examples, policy)} for policy in policies]
            for lambda_value in lambdas:
                selected = max(train_evals, key=lambda row: float(row["mean_reward"]) - lambda_value * float(row["mean_cost"]))
                eval_row = _eval_threshold_policy(eval_examples, selected)
                eval_row["mean_utility"] = float(eval_row["mean_reward"]) - lambda_value * float(eval_row["mean_cost"])
                best_baseline = max(
                    [row for row in _baseline_rows(eval_examples, lambda_value) if row["policy"] != "oracle_best_after_20b_x3_and_120b"],
                    key=lambda row: row["mean_utility"],
                )
                split_rows.append(
                    {
                        "split_idx": split_idx,
                        "lambda": lambda_value,
                        "score_source": source_name,
                        "policy_type": "train_tuned_threshold",
                        "best_baseline": best_baseline["policy"],
                        "best_baseline_utility": best_baseline["mean_utility"],
                        "eval_utility_minus_best_baseline": eval_row["mean_utility"] - best_baseline["mean_utility"],
                        **{f"eval_{key}": value for key, value in eval_row.items()},
                    }
                )

    summary_rows = _summarize(
        split_rows,
        ["score_source", "lambda"],
        ["eval_mean_reward", "eval_mean_cost", "eval_mean_utility", "eval_utility_minus_best_baseline", "eval_escalate_120b"],
    )
    _write_csv(output_dir / "ceiling_fullset_rows.csv", rows)
    _write_csv(output_dir / "ceiling_split_rows.csv", split_rows)
    _write_csv(output_dir / "ceiling_split_summary.csv", summary_rows)
    return {
        "n_ids": len(ids),
        "lambdas": lambdas,
        "max_thresholds": max_thresholds,
        "n_splits": n_splits,
        "split_summary": summary_rows,
    }


def _state_feature(ex: dict[str, Any], k: int, lambda_value: float) -> list[float]:
    return [
        float(ex[f"s{k}"]),
        float(ex[f"margin{k}"]),
        float(ex[f"mean_score{k}"]),
        float(ex[f"std_score{k}"]),
        float(k) / 3.0,
        math.log1p(float(ex[f"c{k}"])) / 10.0,
        math.log1p(float(ex[f"best_output_tokens{k}"])) / 10.0,
        float(ex["score_r0_at%s" % k]),
        float(ex["score_r1_at%s" % k]),
        float(ex["score_r2_at%s" % k]),
        math.log1p(float(lambda_value)) / 5.0,
    ]


def _optimal_action_values(ex: dict[str, Any], lambda_value: float) -> dict[int, tuple[int, float]]:
    values: dict[int, tuple[int, float]] = {}
    for k in (3, 2, 1):
        submit_value = float(ex[f"r{k}"]) - float(lambda_value) * float(ex[f"c{k}"])
        escalate_value = float(ex["r120"]) - float(lambda_value) * (float(ex[f"c{k}"]) + float(ex["c120"]))
        choices = [(SUBMIT, submit_value), (ESCALATE, escalate_value)]
        if k < 3:
            choices.append((RESAMPLE, values[k + 1][1]))
        values[k] = max(choices, key=lambda item: item[1])
    return values


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(logits, -40.0, 40.0))
    return exp / np.sum(exp, axis=1, keepdims=True)


def _fit_softmax(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    steps: int,
    lr: float,
    l2: float,
    class_weight: str,
) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0.0] = 1.0
    z = (x - mean) / std
    design = np.concatenate([np.ones((z.shape[0], 1), dtype=float), z], axis=1)
    weights = np.zeros((design.shape[1], int(n_classes)), dtype=float)
    sample_weight = np.ones(len(y), dtype=float)
    if class_weight == "balanced":
        counts = np.bincount(y, minlength=int(n_classes)).astype(float)
        counts[counts == 0.0] = 1.0
        inv = float(len(y)) / (float(n_classes) * counts)
        sample_weight = inv[y]
        sample_weight = sample_weight / float(np.mean(sample_weight))
    reg = np.ones_like(weights)
    reg[0, :] = 0.0
    y_onehot = np.zeros((len(y), int(n_classes)), dtype=float)
    y_onehot[np.arange(len(y)), y] = 1.0
    for _ in range(int(steps)):
        probs = _softmax(design @ weights)
        err = (probs - y_onehot) * sample_weight[:, None]
        grad = design.T @ err / float(len(y))
        grad += float(l2) * reg * weights / float(len(y))
        weights -= float(lr) * grad
    return {"weights": weights, "mean": mean, "std": std}


def _predict_action(model: dict[str, Any], features: list[float], k: int) -> int:
    x = np.asarray(features, dtype=float)[None, :]
    z = (x - model["mean"]) / model["std"]
    design = np.concatenate([np.ones((1, 1), dtype=float), z], axis=1)
    logits = design @ model["weights"]
    if k >= 3:
        logits[0, RESAMPLE] = -1.0e9
    return int(np.argmax(logits[0]))


def _evaluate_learned_controller(
    examples: list[dict[str, Any]],
    model: dict[str, Any],
    lambda_value: float,
) -> dict[str, Any]:
    rewards: list[float] = []
    costs: list[float] = []
    actions = defaultdict(int)
    n_rollouts: list[int] = []
    for ex in examples:
        k = 1
        while True:
            action = _predict_action(model, _state_feature(ex, k, lambda_value), k)
            if action == RESAMPLE and k < 3:
                k += 1
                continue
            if action == RESAMPLE:
                action = SUBMIT
            if action == SUBMIT:
                rewards.append(float(ex[f"r{k}"]))
                costs.append(float(ex[f"c{k}"]))
                actions[f"submit_20b_after_{k}"] += 1
                n_rollouts.append(k)
                break
            rewards.append(float(ex["r120"]))
            costs.append(float(ex[f"c{k}"]) + float(ex["c120"]))
            actions[f"escalate_120b_after_{k}"] += 1
            n_rollouts.append(k)
            break
    row = {
        "mean_reward": float(np.mean(rewards)) if rewards else math.nan,
        "mean_cost": float(np.mean(costs)) if costs else math.nan,
        "mean_utility": float(np.mean(rewards) - float(lambda_value) * np.mean(costs)) if rewards else math.nan,
        "mean_20b_rollouts": float(np.mean(n_rollouts)) if n_rollouts else math.nan,
    }
    row.update({key: int(value) for key, value in actions.items()})
    row["escalate_120b"] = int(sum(value for key, value in actions.items() if key.startswith("escalate_120b")))
    return row


def _learned_controller_cv(
    lookup: dict[tuple[str, int, int], dict[str, Any]],
    ids: list[str],
    *,
    output_dir: Path,
    lambdas: list[float],
    n_splits: int,
    seed: int,
    train_frac: float,
    steps: int,
    lr: float,
    l2: float,
    class_weight: str,
) -> dict[str, Any]:
    examples = _build_examples(lookup, ids, score_source="learned")
    by_id = {str(ex["pid"]): ex for ex in examples}
    rng = random.Random(int(seed))
    rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for split_idx in range(int(n_splits)):
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n_train = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * float(train_frac)))))
        train_ids = sorted(shuffled[:n_train])
        eval_ids = sorted(shuffled[n_train:])
        x_train: list[list[float]] = []
        y_train: list[int] = []
        for pid in train_ids:
            ex = by_id[pid]
            for lambda_value in lambdas:
                values = _optimal_action_values(ex, lambda_value)
                for k in (1, 2, 3):
                    action = int(values[k][0])
                    x_train.append(_state_feature(ex, k, lambda_value))
                    y_train.append(action)
                    label_rows.append(
                        {
                            "split_idx": split_idx,
                            "lambda": lambda_value,
                            "k": k,
                            "action": ACTION_NAMES[action],
                        }
                    )
        model = _fit_softmax(
            np.asarray(x_train, dtype=float),
            np.asarray(y_train, dtype=int),
            n_classes=3,
            steps=steps,
            lr=lr,
            l2=l2,
            class_weight=class_weight,
        )
        eval_examples = [by_id[pid] for pid in eval_ids]
        for lambda_value in lambdas:
            eval_row = _evaluate_learned_controller(eval_examples, model, lambda_value)
            best_baseline = max(
                [row for row in _baseline_rows(eval_examples, lambda_value) if row["policy"] != "oracle_best_after_20b_x3_and_120b"],
                key=lambda row: float(row["mean_utility"]),
            )
            oracle_dyn = _oracle_dynamic_policy(eval_examples, lambda_value)
            rows.append(
                {
                    "split_idx": split_idx,
                    "lambda": lambda_value,
                    "best_baseline": best_baseline["policy"],
                    "best_baseline_utility": best_baseline["mean_utility"],
                    "utility_minus_best_baseline": eval_row["mean_utility"] - best_baseline["mean_utility"],
                    "oracle_dynamic_utility": oracle_dyn["mean_utility"],
                    "oracle_gap_captured": (
                        (eval_row["mean_utility"] - best_baseline["mean_utility"])
                        / (oracle_dyn["mean_utility"] - best_baseline["mean_utility"])
                        if oracle_dyn["mean_utility"] != best_baseline["mean_utility"]
                        else math.nan
                    ),
                    **eval_row,
                }
            )
    label_summary = _summarize(label_rows, ["lambda", "k", "action"], [])
    summary = _summarize(
        rows,
        ["lambda"],
        [
            "mean_reward",
            "mean_cost",
            "mean_utility",
            "utility_minus_best_baseline",
            "oracle_dynamic_utility",
            "oracle_gap_captured",
            "escalate_120b",
            "mean_20b_rollouts",
        ],
    )
    _write_csv(output_dir / "learned_controller_cv_rows.csv", rows)
    _write_csv(output_dir / "learned_controller_cv_summary.csv", summary)
    _write_csv(output_dir / "learned_controller_label_summary.csv", label_summary)
    return {
        "n_ids": len(ids),
        "n_splits": n_splits,
        "train_frac": train_frac,
        "lambdas": lambdas,
        "class_weight": class_weight,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", choices=["ceiling", "learned_controller_cv", "all"], required=True)
    parser.add_argument("--scores-jsonl", required=True)
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--route-cost-weights", default=",".join(str(value) for value in DEFAULT_COST_WEIGHTS))
    parser.add_argument("--lambdas", default=",".join(str(value) for value in DEFAULT_LAMBDAS))
    parser.add_argument("--max-thresholds", type=int, default=7)
    parser.add_argument("--n-splits", type=int, default=50)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    route_cost_weights = [float(part.strip()) for part in str(args.route_cost_weights).split(",") if part.strip()]
    lambdas = [float(part.strip()) for part in str(args.lambdas).split(",") if part.strip()]
    attempts = _build_attempt_table(_read_jsonl(Path(args.scores_jsonl)), Path(args.report_root), route_cost_weights)
    lookup = _attempt_lookup(attempts)
    ids = _complete_ids(lookup)
    if len(ids) < 8:
        raise ValueError(f"Too few complete ids for controller diagnostics: {len(ids)}")

    manifest: dict[str, Any] = {
        "analysis": args.analysis,
        "scores_jsonl": str(args.scores_jsonl),
        "report_root": str(args.report_root),
        "output_dir": str(output_dir),
        "n_attempts_joined": len(attempts),
        "n_complete_ids": len(ids),
        "route_cost_weights": route_cost_weights,
        "lambdas": lambdas,
    }
    if args.analysis in {"ceiling", "all"}:
        manifest["ceiling"] = _ceiling_analysis(
            lookup,
            ids,
            output_dir=output_dir / "ceiling" if args.analysis == "all" else output_dir,
            lambdas=lambdas,
            max_thresholds=int(args.max_thresholds),
            n_splits=int(args.n_splits),
            seed=int(args.seed),
        )
    if args.analysis in {"learned_controller_cv", "all"}:
        manifest["learned_controller_cv"] = _learned_controller_cv(
            lookup,
            ids,
            output_dir=output_dir / "learned_controller_cv" if args.analysis == "all" else output_dir,
            lambdas=lambdas,
            n_splits=int(args.n_splits),
            seed=int(args.seed),
            train_frac=float(args.train_frac),
            steps=int(args.steps),
            lr=float(args.lr),
            l2=float(args.l2),
            class_weight=str(args.class_weight),
        )
    (output_dir / "summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

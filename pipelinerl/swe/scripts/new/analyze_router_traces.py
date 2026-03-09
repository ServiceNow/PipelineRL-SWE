#!/usr/bin/env python
import argparse
import csv
import importlib.util
import itertools
import json
import logging
import math
from pathlib import Path
from typing import Any

try:
    import numpy as np

    NP_AVAILABLE = True
except ModuleNotFoundError:
    NP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt  # type: ignore

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    from pipelinerl.swe.scripts.new.router_trace_utils import (
        extract_reward_vector,
        extract_route_labels,
        extract_score_vector,
        extract_token_vector,
        load_router_traces,
    )
except ModuleNotFoundError:
    # Allow running this script directly even when the full package extras are not installed.
    module_path = Path(__file__).with_name("router_trace_utils.py")
    spec = importlib.util.spec_from_file_location("router_trace_utils_local", module_path)
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extract_reward_vector = module.extract_reward_vector
    extract_route_labels = module.extract_route_labels
    extract_score_vector = module.extract_score_vector
    extract_token_vector = module.extract_token_vector
    load_router_traces = module.load_router_traces

logger = logging.getLogger(__name__)


def _frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    values: list[float] = []
    cur = start
    while cur <= stop + 1e-12:
        values.append(round(cur, 8))
        cur += step
    return values


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _route_share_key(label: str) -> str:
    cleaned = []
    for ch in label.lower():
        cleaned.append(ch if ch.isalnum() else "_")
    slug = "".join(cleaned).strip("_")
    return f"share_{slug or 'route'}"


def _route_slug(label: str) -> str:
    return _route_share_key(label).removeprefix("share_")


def _parse_expert_costs(cost_arg: str, n_experts: int, default: float) -> list[float]:
    if n_experts <= 0:
        return []
    if not cost_arg:
        return [default] * n_experts

    values = [_safe_float(chunk.strip(), default) for chunk in cost_arg.split(",") if chunk.strip()]
    if not values:
        return [default] * n_experts
    if len(values) < n_experts:
        values.extend([values[-1]] * (n_experts - len(values)))
    return values[:n_experts]


def _parse_lambda_values(args: argparse.Namespace) -> list[float]:
    if args.lambda_values:
        values = [_safe_float(v.strip(), 0.0) for v in args.lambda_values.split(",") if v.strip()]
        if not values:
            raise ValueError("--lambda-values was provided but no values were parsed")
        return sorted(set(values))
    return _frange(args.lambda_start, args.lambda_stop, args.lambda_step)


def _parse_tau_values(args: argparse.Namespace) -> list[float]:
    tau_start = args.tau_start if args.tau_start is not None else args.threshold_start
    tau_stop = args.tau_stop if args.tau_stop is not None else args.threshold_stop
    tau_step = args.tau_step if args.tau_step is not None else args.threshold_step
    return _frange(tau_start, tau_stop, tau_step)


def _evaluate_operating_point(
    traces: list[dict[str, Any]],
    score_key: str,
    lambda_cost: float,
    tau: float,
    allow_abstain: bool,
    abstain_quality: float,
    route_labels: list[str],
    policy_cost_per_1k: float,
    expert_costs_per_1k: list[float],
    fixed_scores: list[float] | None = None,
) -> dict[str, Any]:
    total = 0
    scored = 0
    missing_score = 0
    non_abstain = 0

    quality_sum = 0.0
    cost_sum = 0.0
    utility_sum = 0.0
    token_sum = 0.0

    quality_sum_routed = 0.0
    cost_sum_routed = 0.0

    route_counts: dict[str, int] = {label: 0 for label in route_labels}
    route_counts["abstain"] = 0

    for trace in traces:
        rewards = extract_reward_vector(trace)
        tokens = extract_token_vector(trace)
        dim = min(len(rewards), len(tokens), len(route_labels))
        if dim <= 0:
            continue

        rewards = rewards[:dim]
        tokens = tokens[:dim]

        rates = [policy_cost_per_1k] + expert_costs_per_1k
        rates = rates[:dim]
        costs = [(tokens[i] * rates[i]) / 1000.0 for i in range(dim)]

        total += 1
        if fixed_scores is None:
            scores = extract_score_vector(trace, score_key=score_key, expected_dim=dim)
        else:
            scores = [float(value) for value in fixed_scores[:dim]]

        chosen_idx = -1
        if scores is None:
            missing_score += 1
        else:
            scored += 1
            utility_scores = [scores[i] - lambda_cost * costs[i] for i in range(dim)]
            best_idx = _select_best_route(utility_scores, costs)
            best_utility = utility_scores[best_idx]
            if not allow_abstain or best_utility >= tau:
                chosen_idx = best_idx

        if chosen_idx < 0:
            route_counts["abstain"] += 1
            chosen_quality = abstain_quality
            chosen_cost = 0.0
            chosen_tokens = 0.0
        else:
            route_name = route_labels[chosen_idx]
            route_counts[route_name] += 1
            non_abstain += 1
            chosen_quality = rewards[chosen_idx]
            chosen_cost = costs[chosen_idx]
            chosen_tokens = float(tokens[chosen_idx])
            quality_sum_routed += chosen_quality
            cost_sum_routed += chosen_cost

        chosen_utility = chosen_quality - lambda_cost * chosen_cost
        quality_sum += chosen_quality
        cost_sum += chosen_cost
        utility_sum += chosen_utility
        token_sum += chosen_tokens

    if total == 0:
        return {
            "lambda": lambda_cost,
            "tau": tau,
            "n_total": 0,
            "n_scored": 0,
            "n_missing_score": 0,
            "n_non_abstain": 0,
            "coverage": 0.0,
            "score_coverage": 0.0,
            "abstain_rate": 0.0,
            "avg_quality": 0.0,
            "avg_cost": 0.0,
            "avg_utility": 0.0,
            "avg_tokens": 0.0,
            "avg_quality_when_routed": 0.0,
            "avg_cost_when_routed": 0.0,
            "_route_counts": route_counts,
        }

    row = {
        "lambda": lambda_cost,
        "tau": tau,
        "n_total": total,
        "n_scored": scored,
        "n_missing_score": missing_score,
        "n_non_abstain": non_abstain,
        "coverage": non_abstain / total,
        "score_coverage": scored / total,
        "abstain_rate": 1.0 - (non_abstain / total),
        "avg_quality": quality_sum / total,
        "avg_cost": cost_sum / total,
        "avg_utility": utility_sum / total,
        "avg_tokens": token_sum / total,
        "avg_quality_when_routed": (quality_sum_routed / non_abstain) if non_abstain else 0.0,
        "avg_cost_when_routed": (cost_sum_routed / non_abstain) if non_abstain else 0.0,
        "_route_counts": route_counts,
    }

    for name, count in route_counts.items():
        row[_route_share_key(name)] = count / total

    return row


def _evaluate_oracle_operating_point(
    traces: list[dict[str, Any]],
    lambda_cost: float,
    tau: float,
    allow_abstain: bool,
    abstain_quality: float,
    route_labels: list[str],
    policy_cost_per_1k: float,
    expert_costs_per_1k: list[float],
) -> dict[str, Any]:
    total = 0
    non_abstain = 0

    quality_sum = 0.0
    cost_sum = 0.0
    utility_sum = 0.0
    token_sum = 0.0

    quality_sum_routed = 0.0
    cost_sum_routed = 0.0

    route_counts: dict[str, int] = {label: 0 for label in route_labels}
    route_counts["abstain"] = 0

    for trace in traces:
        rewards = extract_reward_vector(trace)
        tokens = extract_token_vector(trace)
        dim = min(len(rewards), len(tokens), len(route_labels))
        if dim <= 0:
            continue

        rewards = rewards[:dim]
        tokens = tokens[:dim]
        rates = [policy_cost_per_1k] + expert_costs_per_1k
        rates = rates[:dim]
        costs = [(tokens[i] * rates[i]) / 1000.0 for i in range(dim)]

        total += 1
        utility_scores = [rewards[i] - lambda_cost * costs[i] for i in range(dim)]
        best_idx = _select_best_route(utility_scores, costs)
        best_utility = utility_scores[best_idx]
        chosen_idx = best_idx if (not allow_abstain or best_utility >= tau) else -1

        if chosen_idx < 0:
            route_counts["abstain"] += 1
            chosen_quality = abstain_quality
            chosen_cost = 0.0
            chosen_tokens = 0.0
        else:
            route_name = route_labels[chosen_idx]
            route_counts[route_name] += 1
            non_abstain += 1
            chosen_quality = rewards[chosen_idx]
            chosen_cost = costs[chosen_idx]
            chosen_tokens = float(tokens[chosen_idx])
            quality_sum_routed += chosen_quality
            cost_sum_routed += chosen_cost

        chosen_utility = chosen_quality - lambda_cost * chosen_cost
        quality_sum += chosen_quality
        cost_sum += chosen_cost
        utility_sum += chosen_utility
        token_sum += chosen_tokens

    if total == 0:
        return {
            "lambda": lambda_cost,
            "tau": tau,
            "n_total": 0,
            "n_scored": 0,
            "n_missing_score": 0,
            "n_non_abstain": 0,
            "coverage": 0.0,
            "score_coverage": 0.0,
            "abstain_rate": 0.0,
            "avg_quality": 0.0,
            "avg_cost": 0.0,
            "avg_utility": 0.0,
            "avg_tokens": 0.0,
            "avg_quality_when_routed": 0.0,
            "avg_cost_when_routed": 0.0,
            "_route_counts": route_counts,
        }

    row = {
        "lambda": lambda_cost,
        "tau": tau,
        "n_total": total,
        "n_scored": total,
        "n_missing_score": 0,
        "n_non_abstain": non_abstain,
        "coverage": non_abstain / total,
        "score_coverage": 1.0,
        "abstain_rate": 1.0 - (non_abstain / total),
        "avg_quality": quality_sum / total,
        "avg_cost": cost_sum / total,
        "avg_utility": utility_sum / total,
        "avg_tokens": token_sum / total,
        "avg_quality_when_routed": (quality_sum_routed / non_abstain) if non_abstain else 0.0,
        "avg_cost_when_routed": (cost_sum_routed / non_abstain) if non_abstain else 0.0,
        "_route_counts": route_counts,
    }
    for name, count in route_counts.items():
        row[_route_share_key(name)] = count / total
    return row


def _build_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    # Collapse duplicate cost points to their best quality.
    by_cost: dict[float, dict[str, Any]] = {}
    for row in rows:
        cost = float(row["avg_cost"])
        prev = by_cost.get(cost)
        if prev is None or row["avg_quality"] > prev["avg_quality"]:
            by_cost[cost] = row

    ordered = sorted(by_cost.values(), key=lambda row: (row["avg_cost"], -row["avg_quality"]))
    frontier: list[dict[str, Any]] = []
    best_quality = -1e30
    for row in ordered:
        quality = float(row["avg_quality"])
        if quality > best_quality + 1e-12:
            frontier.append(row)
            best_quality = quality
    return frontier


def _compute_auc_qc(frontier: list[dict[str, Any]]) -> tuple[float, float]:
    if len(frontier) < 2:
        return 0.0, 0.0
    x = [float(row["avg_cost"]) for row in frontier]
    y = [float(row["avg_quality"]) for row in frontier]
    auc = 0.0
    for i in range(len(x) - 1):
        auc += (x[i + 1] - x[i]) * (y[i + 1] + y[i]) * 0.5
    span = x[-1] - x[0]
    auc_normalized = auc / span if span > 1e-12 else 0.0
    return auc, auc_normalized


def _select_best_route(utilities: list[float], costs: list[float]) -> int:
    if not utilities:
        return -1
    best_idx = 0
    best_utility = utilities[0]
    best_cost = costs[0] if costs else 0.0
    for idx in range(1, len(utilities)):
        utility = utilities[idx]
        cost = costs[idx] if idx < len(costs) else 0.0
        if utility > best_utility + 1e-12:
            best_idx = idx
            best_utility = utility
            best_cost = cost
            continue
        if abs(utility - best_utility) > 1e-12:
            continue
        if cost < best_cost - 1e-12:
            best_idx = idx
            best_utility = utility
            best_cost = cost
            continue
        if abs(cost - best_cost) <= 1e-12 and idx < best_idx:
            best_idx = idx
            best_utility = utility
            best_cost = cost
    return best_idx


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if NP_AVAILABLE:
        return float(np.percentile(values, q))
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (q / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(ordered[lo])
    lo_v = ordered[lo]
    hi_v = ordered[hi]
    weight = rank - lo
    return float(lo_v * (1.0 - weight) + hi_v * weight)


def _r2_score(y_true: list[float], y_pred: list[float]) -> float | None:
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return None
    mean_true = sum(y_true) / len(y_true)
    ss_tot = sum((value - mean_true) ** 2 for value in y_true)
    if ss_tot <= 1e-12:
        return None
    ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    return 1.0 - (ss_res / ss_tot)


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    if NP_AVAILABLE:
        return float(np.var(values))
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _pearson_corr(x: list[float], y: list[float]) -> float | None:
    if not x or not y or len(x) != len(y):
        return None
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    vx = sum((value - mx) ** 2 for value in x)
    vy = sum((value - my) ** 2 for value in y)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    return cov / math.sqrt(vx * vy)


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda it: it[1])
    ranks = [0.0 for _ in values]
    pos = 0
    while pos < len(indexed):
        start = pos
        current = indexed[pos][1]
        while pos + 1 < len(indexed) and abs(indexed[pos + 1][1] - current) <= 1e-12:
            pos += 1
        end = pos
        avg_rank = (start + end) / 2.0 + 1.0
        for slot in range(start, end + 1):
            original_idx = indexed[slot][0]
            ranks[original_idx] = avg_rank
        pos += 1
    return ranks


def _spearman_corr(x: list[float], y: list[float]) -> float | None:
    if not x or not y or len(x) != len(y):
        return None
    return _pearson_corr(_average_ranks(x), _average_ranks(y))


def _roc_auc_binary(labels: list[int], scores: list[float]) -> float | None:
    if not labels or not scores or len(labels) != len(scores):
        return None
    positives = sum(1 for value in labels if value == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0 for _ in scores]
    pos = 0
    while pos < len(order):
        start = pos
        cur = scores[order[pos]]
        while pos + 1 < len(order) and abs(scores[order[pos + 1]] - cur) <= 1e-12:
            pos += 1
        end = pos
        avg_rank = (start + end) / 2.0 + 1.0
        for slot in range(start, end + 1):
            ranks[order[slot]] = avg_rank
        pos += 1

    rank_sum_pos = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    u_stat = rank_sum_pos - positives * (positives + 1) / 2.0
    return u_stat / (positives * negatives)


def _precision_recall_from_scores(labels: list[int], scores: list[float], threshold: float = 0.0) -> tuple[float, float]:
    if not labels or not scores or len(labels) != len(scores):
        return (0.0, 0.0)
    tp = fp = fn = 0
    for idx in range(len(labels)):
        pred_pos = scores[idx] > threshold
        true_pos = labels[idx] == 1
        if pred_pos and true_pos:
            tp += 1
        elif pred_pos and not true_pos:
            fp += 1
        elif (not pred_pos) and true_pos:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _compute_best_expert_identity_stats(
    traces: list[dict[str, Any]],
    route_labels: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_experts = max(0, len(route_labels) - 1)
    if n_experts <= 0:
        return {
            "n_tasks": 0,
            "n_experts": 0,
            "best_expert_variance": 0.0,
            "best_expert_change_prob_random_pair": 0.0,
            "dominant_expert_share": 0.0,
            "dominant_expert": None,
        }, []

    best_expert_ids: list[int] = []
    for trace in traces:
        rewards = extract_reward_vector(trace)
        available_experts = min(n_experts, max(0, len(rewards) - 1))
        if available_experts <= 0:
            continue
        expert_rewards = rewards[1 : 1 + available_experts]
        best_idx = max(range(available_experts), key=lambda idx: expert_rewards[idx])
        best_expert_ids.append(best_idx)

    if not best_expert_ids:
        stats = {
            "n_tasks": 0,
            "n_experts": n_experts,
            "best_expert_variance": 0.0,
            "best_expert_change_prob_random_pair": 0.0,
            "dominant_expert_share": 0.0,
            "dominant_expert": None,
        }
        rows = [
            {
                "expert_index": i,
                "expert_label": route_labels[i + 1],
                "count_best": 0,
                "share_best": 0.0,
            }
            for i in range(n_experts)
        ]
        return stats, rows

    total = len(best_expert_ids)
    counts = [0 for _ in range(n_experts)]
    for expert_idx in best_expert_ids:
        if 0 <= expert_idx < n_experts:
            counts[expert_idx] += 1

    shares = [count / total for count in counts]
    dominant_idx = max(range(n_experts), key=lambda i: counts[i])
    change_prob = 1.0 - sum(share * share for share in shares)
    stats = {
        "n_tasks": total,
        "n_experts": n_experts,
        "best_expert_variance": _variance([float(idx) for idx in best_expert_ids]),
        "best_expert_change_prob_random_pair": change_prob,
        "dominant_expert_share": shares[dominant_idx],
        "dominant_expert": route_labels[dominant_idx + 1],
    }
    rows = [
        {
            "expert_index": i,
            "expert_label": route_labels[i + 1],
            "count_best": counts[i],
            "share_best": shares[i],
        }
        for i in range(n_experts)
    ]
    return stats, rows


def _compute_route_mean_rewards(
    traces: list[dict[str, Any]],
    route_labels: list[str],
) -> list[float]:
    sums = [0.0 for _ in route_labels]
    counts = [0 for _ in route_labels]
    for trace in traces:
        rewards = extract_reward_vector(trace)
        dim = min(len(rewards), len(route_labels))
        if dim <= 0:
            continue
        for idx in range(dim):
            sums[idx] += float(rewards[idx])
            counts[idx] += 1
    means: list[float] = []
    for idx in range(len(route_labels)):
        if counts[idx] > 0:
            means.append(sums[idx] / counts[idx])
        else:
            means.append(0.0)
    return means


def _resolve_baseline_index(route_labels: list[str], baseline_route: str) -> int | None:
    text = (baseline_route or "").strip()
    if not text or text.lower() == "none":
        return None
    if text == "policy":
        return 0
    if text in route_labels:
        return route_labels.index(text)
    if text.startswith("expert_"):
        suffix = text[len("expert_") :]
        if suffix.isdigit():
            idx = int(suffix) + 1
            if idx < len(route_labels):
                return idx
    for idx, label in enumerate(route_labels):
        if label.startswith(text + ":"):
            return idx
    logger.warning("Baseline route '%s' not found; falling back to policy", baseline_route)
    return 0 if route_labels else None


def _evaluate_oracle_predictor_regret(
    traces: list[dict[str, Any]],
    score_key: str,
    lambda_cost: float,
    tau: float,
    allow_abstain: bool,
    route_labels: list[str],
    policy_cost_per_1k: float,
    expert_costs_per_1k: list[float],
    baseline_route_idx: int | None,
) -> dict[str, Any]:
    total = 0
    scored = 0
    missing_score = 0

    util_oracle_sum = 0.0
    util_pred_sum = 0.0
    util_base_sum = 0.0

    oracle_non_abstain = 0
    pred_non_abstain = 0
    baseline_count = 0
    regrets: list[float] = []

    for trace in traces:
        rewards = extract_reward_vector(trace)
        tokens = extract_token_vector(trace)
        dim = min(len(rewards), len(tokens), len(route_labels))
        if dim <= 0:
            continue
        total += 1

        scores = extract_score_vector(trace, score_key=score_key, expected_dim=dim)
        if scores is None:
            missing_score += 1
            continue
        scored += 1

        rewards = rewards[:dim]
        tokens = tokens[:dim]
        rates = [policy_cost_per_1k] + expert_costs_per_1k
        rates = rates[:dim]
        costs = [(tokens[i] * rates[i]) / 1000.0 for i in range(dim)]

        true_utilities = [rewards[i] - lambda_cost * costs[i] for i in range(dim)]
        pred_utilities = [scores[i] - lambda_cost * costs[i] for i in range(dim)]

        oracle_idx = _select_best_route(true_utilities, costs)
        pred_idx = _select_best_route(pred_utilities, costs)

        best_true_utility = true_utilities[oracle_idx]
        # Regret is intentionally ranking-only: tau/abstention is disabled.
        best_pred_utility = pred_utilities[pred_idx]
        oracle_abstain = False
        pred_abstain = False

        util_oracle = 0.0 if oracle_abstain else best_true_utility
        util_pred = 0.0 if pred_abstain else true_utilities[pred_idx]

        if not oracle_abstain:
            oracle_non_abstain += 1
        if not pred_abstain:
            pred_non_abstain += 1

        util_oracle_sum += util_oracle
        util_pred_sum += util_pred
        regrets.append(util_oracle - util_pred)

        if baseline_route_idx is not None and baseline_route_idx < dim:
            util_base_sum += true_utilities[baseline_route_idx]
            baseline_count += 1

    if scored == 0:
        return {
            "lambda": lambda_cost,
            "tau": tau,
            "n_total": total,
            "n_scored": 0,
            "n_missing_score": missing_score,
            "score_coverage": 0.0,
            "mean_util_oracle": 0.0,
            "mean_util_pred": 0.0,
            "mean_util_base": None,
            "mean_regret": 0.0,
            "p95_regret": 0.0,
            "oracle_coverage": 0.0,
            "predictor_coverage": 0.0,
            "oracle_abstain_rate": 0.0,
            "predictor_abstain_rate": 0.0,
            "capture": None,
            "n_baseline_available": baseline_count,
        }

    mean_util_oracle = util_oracle_sum / scored
    mean_util_pred = util_pred_sum / scored
    mean_util_base: float | None = None
    capture: float | None = None
    utility_gap: float | None = None
    if baseline_count > 0:
        mean_util_base = util_base_sum / baseline_count
        utility_gap = mean_util_oracle - mean_util_base
        if abs(utility_gap) > 1e-12:
            capture = (mean_util_pred - mean_util_base) / utility_gap

    return {
        "lambda": lambda_cost,
        "tau": tau,
        "n_total": total,
        "n_scored": scored,
        "n_missing_score": missing_score,
        "score_coverage": scored / total if total else 0.0,
        "mean_util_oracle": mean_util_oracle,
        "mean_util_pred": mean_util_pred,
        "mean_util_base": mean_util_base,
        "mean_regret": sum(regrets) / len(regrets),
        "p95_regret": _percentile(regrets, 95.0),
        "oracle_coverage": oracle_non_abstain / scored,
        "predictor_coverage": pred_non_abstain / scored,
        "oracle_abstain_rate": 1.0 - (oracle_non_abstain / scored),
        "predictor_abstain_rate": 1.0 - (pred_non_abstain / scored),
        "capture": capture,
        "utility_gap": utility_gap,
        "n_baseline_available": baseline_count,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _plot_cost_quality_scatter(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping cost-quality scatter")
        return
    if not rows:
        return

    x = [row["avg_cost"] for row in rows]
    y = [row["avg_quality"] for row in rows]
    c = [row["coverage"] for row in rows]

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(x, y, c=c, cmap="viridis", alpha=0.8)
    plt.colorbar(scatter, label="coverage")
    plt.xlabel("Average cost")
    plt.ylabel("Average quality")
    plt.title("Cost vs quality (all lambda/tau operating points)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_frontier(rows: list[dict[str, Any]], frontier: list[dict[str, Any]], auc_qc: float, out_path: Path) -> None:
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping frontier plot")
        return
    if not rows:
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(
        [row["avg_cost"] for row in rows],
        [row["avg_quality"] for row in rows],
        alpha=0.25,
        color="gray",
        label="Operating points",
    )

    if frontier:
        fx = [row["avg_cost"] for row in frontier]
        fy = [row["avg_quality"] for row in frontier]
        plt.plot(fx, fy, color="tab:blue", linewidth=2.0, marker="o", label="Pareto frontier")
        if len(frontier) >= 2:
            plt.fill_between(fx, fy, color="tab:blue", alpha=0.12)

    plt.xlabel("Average cost")
    plt.ylabel("Average quality")
    plt.title(f"Pareto frontier (AUC-QC={auc_qc:.4f})")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_combined_frontiers(
    actual_frontier: list[dict[str, Any]],
    baseline_frontier: list[dict[str, Any]],
    oracle_frontier: list[dict[str, Any]],
    actual_auc_qc: float,
    baseline_auc_qc: float,
    oracle_auc_qc: float,
    out_path: Path,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping combined frontier plot")
        return
    if not actual_frontier and not baseline_frontier and not oracle_frontier:
        return

    plt.figure(figsize=(8, 5.5))

    if actual_frontier:
        x = [row["avg_cost"] for row in actual_frontier]
        y = [row["avg_quality"] for row in actual_frontier]
        plt.plot(x, y, color="tab:blue", marker="o", linewidth=2.0, label=f"Actual (AUC={actual_auc_qc:.4f})")

    if baseline_frontier:
        x = [row["avg_cost"] for row in baseline_frontier]
        y = [row["avg_quality"] for row in baseline_frontier]
        plt.plot(
            x,
            y,
            color="tab:orange",
            marker="s",
            linewidth=2.0,
            linestyle="--",
            label=f"Baseline mu (AUC={baseline_auc_qc:.4f})",
        )

    if oracle_frontier:
        x = [row["avg_cost"] for row in oracle_frontier]
        y = [row["avg_quality"] for row in oracle_frontier]
        plt.plot(x, y, color="tab:green", marker="^", linewidth=2.0, label=f"Oracle (AUC={oracle_auc_qc:.4f})")

    plt.xlabel("Average cost")
    plt.ylabel("Average quality")
    plt.title("Pareto frontiers: actual vs baseline vs oracle")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_heatmap(
    rows: list[dict[str, Any]],
    lambdas: list[float],
    taus: list[float],
    metric_key: str,
    title: str,
    out_path: Path,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping heatmap for %s", metric_key)
        return

    grid = [[float("nan") for _ in taus] for _ in lambdas]
    index = {(row["lambda"], row["tau"]): row for row in rows}
    for i, lam in enumerate(lambdas):
        for j, tau in enumerate(taus):
            row = index.get((lam, tau))
            if row is not None:
                grid[i][j] = float(row.get(metric_key, float("nan")))

    plt.figure(figsize=(8, 5))
    image = plt.imshow(grid, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(image, label=metric_key)
    plt.xlabel("tau (abstention threshold)")
    plt.ylabel("lambda (cost sensitivity)")
    plt.title(title)

    # Keep tick count manageable.
    x_step = max(1, len(taus) // 10)
    y_step = max(1, len(lambdas) // 10)
    plt.xticks(range(0, len(taus), x_step), [f"{taus[j]:.2f}" for j in range(0, len(taus), x_step)], rotation=45)
    plt.yticks(range(0, len(lambdas), y_step), [f"{lambdas[i]:.3f}" for i in range(0, len(lambdas), y_step)])

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_regret_curves(
    rows: list[dict[str, Any]],
    lambdas: list[float],
    taus: list[float],
    metric_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping %s plot", metric_key)
        return
    if not rows:
        return

    index = {(row["lambda"], row["tau"]): row for row in rows}
    plt.figure(figsize=(8, 5))
    for tau in taus:
        x_values: list[float] = []
        y_values: list[float] = []
        for lam in lambdas:
            row = index.get((lam, tau))
            if row is None:
                continue
            value = row.get(metric_key)
            if value is None:
                continue
            x_values.append(lam)
            y_values.append(float(value))
        if x_values:
            plt.plot(x_values, y_values, linewidth=1.5, label=f"tau={tau:.2f}")

    plt.xlabel("lambda (cost sensitivity)")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(taus) <= 16:
        plt.legend(loc="best", fontsize=8)
    else:
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_predicted_vs_realized_by_route(
    traces: list[dict[str, Any]],
    score_key: str,
    route_labels: list[str],
    out_dir: Path,
) -> None:
    if not traces or not route_labels:
        return
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping predicted-vs-realized scatter plots")

    pred_by_route: list[list[float]] = [[] for _ in route_labels]
    real_by_route: list[list[float]] = [[] for _ in route_labels]

    for trace in traces:
        rewards = extract_reward_vector(trace)
        dim = min(len(rewards), len(route_labels))
        if dim <= 0:
            continue
        scores = extract_score_vector(trace, score_key=score_key, expected_dim=dim)
        if scores is None:
            continue
        for route_idx in range(dim):
            pred_by_route[route_idx].append(float(scores[route_idx]))
            real_by_route[route_idx].append(float(rewards[route_idx]))

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_rows: list[dict[str, Any]] = []
    for route_idx, route_label in enumerate(route_labels):
        pred = pred_by_route[route_idx]
        real = real_by_route[route_idx]
        if not pred or not real:
            continue
        n_points = len(pred)
        pred_mean = sum(pred) / n_points
        real_mean = sum(real) / n_points
        pred_var = sum((value - pred_mean) ** 2 for value in pred) / n_points
        real_var = sum((value - real_mean) ** 2 for value in real) / n_points
        pred_std = math.sqrt(max(pred_var, 0.0))
        real_std = math.sqrt(max(real_var, 0.0))

        lower = min(min(pred), min(real))
        upper = max(max(pred), max(real))
        if abs(upper - lower) < 1e-12:
            lower -= 0.05
            upper += 0.05
        r2 = _r2_score(real, pred)
        stats_rows.append(
            {
                "route_index": route_idx,
                "route_label": route_label,
                "n_points": n_points,
                "r2": r2,
                "pred_mean": pred_mean,
                "pred_std": pred_std,
                "realized_mean": real_mean,
                "realized_std": real_std,
            }
        )

        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=(9.0, 5.5))
            plt.scatter(pred, real, alpha=0.35, s=14, color="tab:blue")
            plt.plot([lower, upper], [lower, upper], linestyle="--", color="gray", linewidth=1.0)
            plt.xlabel("Predicted reward")
            plt.ylabel("Realized reward")
            title = route_label
            if r2 is not None:
                title += f" | R^2={r2:.3f}"
            plt.title(title, fontsize=10)
            plt.xlim(lower, upper)
            plt.ylim(lower, upper)
            plt.tight_layout()
            out_path = out_dir / f"pred_vs_realized_{route_idx}_{_route_slug(route_label)}.png"
            plt.savefig(out_path)
            plt.close()

    _write_csv(
        out_dir / "pred_vs_realized_stats.csv",
        stats_rows,
        [
            "route_index",
            "route_label",
            "n_points",
            "r2",
            "pred_mean",
            "pred_std",
            "realized_mean",
            "realized_std",
        ],
    )


def _plot_pairwise_delta_scatter_and_stats(
    traces: list[dict[str, Any]],
    score_key: str,
    route_labels: list[str],
    out_dir: Path,
) -> None:
    if len(route_labels) < 2:
        return
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping pairwise delta scatter plots")

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_rows: list[dict[str, Any]] = []
    for left_idx, right_idx in itertools.combinations(range(len(route_labels)), 2):
        left_label = route_labels[left_idx]
        right_label = route_labels[right_idx]
        delta_pred: list[float] = []
        delta_true: list[float] = []
        for trace in traces:
            rewards = extract_reward_vector(trace)
            dim = min(len(rewards), len(route_labels))
            if dim <= max(left_idx, right_idx):
                continue
            scores = extract_score_vector(trace, score_key=score_key, expected_dim=dim)
            if scores is None:
                continue
            pred_delta = float(scores[left_idx]) - float(scores[right_idx])
            true_delta = float(rewards[left_idx]) - float(rewards[right_idx])
            delta_pred.append(pred_delta)
            delta_true.append(true_delta)

        if not delta_pred:
            continue

        abs_err = [abs(delta_pred[i] - delta_true[i]) for i in range(len(delta_pred))]
        pearson = _pearson_corr(delta_pred, delta_true)
        spearman = _spearman_corr(delta_pred, delta_true)
        mae = sum(abs_err) / len(abs_err)
        r2 = _r2_score(delta_true, delta_pred)
        stats_rows.append(
            {
                "pair_left_idx": left_idx,
                "pair_right_idx": right_idx,
                "pair_left_label": left_label,
                "pair_right_label": right_label,
                "n_points": len(delta_pred),
                "pearson": pearson,
                "spearman": spearman,
                "delta_mae": mae,
                "r2": r2,
            }
        )

        if not MATPLOTLIB_AVAILABLE:
            continue

        lower = min(min(delta_pred), min(delta_true))
        upper = max(max(delta_pred), max(delta_true))
        if abs(upper - lower) < 1e-12:
            lower -= 0.05
            upper += 0.05
        plt.figure(figsize=(10.0, 5.8))
        plt.scatter(delta_pred, delta_true, alpha=0.35, s=14, color="tab:blue")
        plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        plt.axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
        if len(delta_pred) >= 2:
            slope, intercept = np.polyfit(delta_pred, delta_true, 1) if NP_AVAILABLE else (None, None)
            if slope is not None and intercept is not None:
                x0, x1 = lower, upper
                y0 = float(slope * x0 + intercept)
                y1 = float(slope * x1 + intercept)
                plt.plot([x0, x1], [y0, y1], color="tab:red", linewidth=1.5, label="Least-squares fit")
                plt.legend(loc="best", fontsize=8)
        title = f"Delta scatter: {left_label} - {right_label}"
        if pearson is not None and spearman is not None:
            title += f" | Pearson={pearson:.3f}, Spearman={spearman:.3f}, MAE={mae:.3f}"
        plt.title(title, fontsize=10)
        plt.xlabel("Predicted delta")
        plt.ylabel("Realized delta")
        plt.xlim(lower, upper)
        plt.ylim(lower, upper)
        plt.tight_layout()
        out_path = out_dir / f"delta_scatter_{left_idx}_vs_{right_idx}_{_route_slug(left_label)}__{_route_slug(right_label)}.png"
        plt.savefig(out_path)
        plt.close()

    _write_csv(
        out_dir / "pairwise_delta_stats.csv",
        stats_rows,
        [
            "pair_left_idx",
            "pair_right_idx",
            "pair_left_label",
            "pair_right_label",
            "n_points",
            "pearson",
            "spearman",
            "delta_mae",
            "r2",
        ],
    )


def _compute_pairwise_ranking_metrics(
    traces: list[dict[str, Any]],
    score_key: str,
    route_labels: list[str],
    out_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_idx, right_idx in itertools.combinations(range(len(route_labels)), 2):
        delta_pred: list[float] = []
        delta_true: list[float] = []
        for trace in traces:
            rewards = extract_reward_vector(trace)
            dim = min(len(rewards), len(route_labels))
            if dim <= max(left_idx, right_idx):
                continue
            scores = extract_score_vector(trace, score_key=score_key, expected_dim=dim)
            if scores is None:
                continue
            delta_pred.append(float(scores[left_idx]) - float(scores[right_idx]))
            delta_true.append(float(rewards[left_idx]) - float(rewards[right_idx]))
        if not delta_pred:
            continue

        labels = [1 if value > 0.0 else 0 for value in delta_true]
        sign_match = 0
        for idx in range(len(delta_pred)):
            pred_sign = 1 if delta_pred[idx] > 0 else (-1 if delta_pred[idx] < 0 else 0)
            true_sign = 1 if delta_true[idx] > 0 else (-1 if delta_true[idx] < 0 else 0)
            if pred_sign == true_sign:
                sign_match += 1
        ranking_acc = sign_match / len(delta_pred)
        roc_auc = _roc_auc_binary(labels, delta_pred)
        precision, recall = _precision_recall_from_scores(labels, delta_pred, threshold=0.0)

        rows.append(
            {
                "pair_left_idx": left_idx,
                "pair_right_idx": right_idx,
                "pair_left_label": route_labels[left_idx],
                "pair_right_label": route_labels[right_idx],
                "n_points": len(delta_pred),
                "positive_rate_true": sum(labels) / len(labels),
                "ranking_accuracy_sign": ranking_acc,
                "roc_auc": roc_auc,
                "precision_pos_pred_gt_0": precision,
                "recall_pos_pred_gt_0": recall,
            }
        )

    _write_csv(
        out_path,
        rows,
        [
            "pair_left_idx",
            "pair_right_idx",
            "pair_left_label",
            "pair_right_label",
            "n_points",
            "positive_rate_true",
            "ranking_accuracy_sign",
            "roc_auc",
            "precision_pos_pred_gt_0",
            "recall_pos_pred_gt_0",
        ],
    )
    return rows


def _float_token(value: float, decimals: int) -> str:
    token = f"{value:.{decimals}f}"
    token = token.replace("-", "m").replace(".", "p")
    return token


def _plot_routing_pies(
    rows: list[dict[str, Any]],
    route_labels: list[str],
    out_dir: Path,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping routing pies")
        return
    if not rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    route_order = route_labels + ["abstain"]
    for row in rows:
        route_counts = row.get("_route_counts")
        if not isinstance(route_counts, dict):
            continue

        labels: list[str] = []
        values: list[int] = []
        for route_name in route_order:
            count = int(route_counts.get(route_name, 0))
            if count <= 0:
                continue
            labels.append(route_name)
            values.append(count)
        if not values:
            continue

        lambda_cost = float(row.get("lambda", 0.0))
        tau = float(row.get("tau", 0.0))
        lambda_token = _float_token(lambda_cost, decimals=4)
        tau_token = _float_token(tau, decimals=3)
        lambda_dir = out_dir / f"lambda_{lambda_token}"
        lambda_dir.mkdir(parents=True, exist_ok=True)
        pie_path = lambda_dir / f"routing_lambda_{lambda_token}_tau_{tau_token}.png"

        plt.figure(figsize=(5, 5))
        plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title(f"Routing share (lambda={lambda_cost:.4f}, tau={tau:.3f})")
        plt.tight_layout()
        plt.savefig(pie_path)
        plt.close()


def _plot_capture_vs_lambda(regret_rows: list[dict[str, Any]], out_path: Path) -> None:
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping oracle-gap-capture plot")
        return
    if not regret_rows:
        return
    x_values: list[float] = []
    y_values: list[float] = []
    for row in sorted(regret_rows, key=lambda r: float(r["lambda"])):
        capture = row.get("capture")
        if capture is None:
            continue
        x_values.append(float(row["lambda"]))
        y_values.append(float(capture))
    if not x_values:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o", linewidth=1.8, color="tab:purple")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    plt.axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
    plt.xlabel("lambda (cost sensitivity)")
    plt.ylabel("Oracle gap capture")
    plt.title("Oracle gap capture vs lambda (tau=0)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _rows_for_tau(rows: list[dict[str, Any]], tau: float) -> list[dict[str, Any]]:
    return [row for row in rows if abs(float(row.get("tau", 0.0)) - tau) <= 1e-12]


def _frontier_cost_range(frontier: list[dict[str, Any]]) -> tuple[float, float, float]:
    if not frontier:
        return (0.0, 0.0, 0.0)
    costs = [float(row["avg_cost"]) for row in frontier]
    min_cost = min(costs)
    max_cost = max(costs)
    return (min_cost, max_cost, max_cost - min_cost)


def _compute_pred_std_over_model_version(
    traces: list[dict[str, Any]],
    score_key: str,
    route_labels: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    versions = sorted({int(trace.get("model_version")) for trace in traces if trace.get("model_version") is not None})
    if not versions:
        return [], []

    true_std_rows: list[dict[str, Any]] = []
    for route_idx, route_label in enumerate(route_labels):
        realized: list[float] = []
        for trace in traces:
            rewards = extract_reward_vector(trace)
            if route_idx < len(rewards):
                realized.append(float(rewards[route_idx]))
        if not realized:
            continue
        mean_val = sum(realized) / len(realized)
        var_val = sum((value - mean_val) ** 2 for value in realized) / len(realized)
        true_std_rows.append(
            {
                "route_index": route_idx,
                "route_label": route_label,
                "n_points": len(realized),
                "realized_mean": mean_val,
                "realized_std": math.sqrt(max(var_val, 0.0)),
            }
        )

    rows: list[dict[str, Any]] = []
    for version in versions:
        version_traces = [trace for trace in traces if int(trace.get("model_version")) == version]
        for route_idx, route_label in enumerate(route_labels):
            preds: list[float] = []
            for trace in version_traces:
                rewards = extract_reward_vector(trace)
                dim = min(len(rewards), len(route_labels))
                if route_idx >= dim:
                    continue
                scores = extract_score_vector(trace, score_key=score_key, expected_dim=dim)
                if scores is None:
                    continue
                preds.append(float(scores[route_idx]))
            if not preds:
                continue
            mean_val = sum(preds) / len(preds)
            var_val = sum((value - mean_val) ** 2 for value in preds) / len(preds)
            rows.append(
                {
                    "model_version": version,
                    "route_index": route_idx,
                    "route_label": route_label,
                    "n_points": len(preds),
                    "pred_mean": mean_val,
                    "pred_std": math.sqrt(max(var_val, 0.0)),
                }
            )
    return rows, true_std_rows


def _plot_pred_std_over_model_version(
    pred_rows: list[dict[str, Any]],
    true_std_rows: list[dict[str, Any]],
    route_labels: list[str],
    out_path: Path,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping predicted std over model version plot")
        return
    if not pred_rows:
        return
    plt.figure(figsize=(10.5, 6.2))
    true_std_by_idx = {
        int(row["route_index"]): float(row["realized_std"]) for row in true_std_rows if row.get("realized_std") is not None
    }
    for route_idx, route_label in enumerate(route_labels):
        route_rows = [row for row in pred_rows if int(row["route_index"]) == route_idx]
        if not route_rows:
            continue
        route_rows = sorted(route_rows, key=lambda row: int(row["model_version"]))
        x = [int(row["model_version"]) for row in route_rows]
        y = [float(row["pred_std"]) for row in route_rows]
        plt.plot(x, y, marker="o", linewidth=1.6, label=f"pred_std | {route_label}")
        true_std = true_std_by_idx.get(route_idx)
        if true_std is not None:
            plt.axhline(true_std, linestyle="--", linewidth=1.0, alpha=0.55)
    plt.xlabel("model_version")
    plt.ylabel("Std of predicted reward")
    plt.title("Predicted std over model versions (with realized std reference lines)")
    if len(route_labels) <= 8:
        plt.legend(loc="best", fontsize=8)
    else:
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, ncol=1)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _print_top_rows(rows: list[dict[str, Any]], limit: int = 8) -> None:
    ranked = sorted(rows, key=lambda row: row["avg_utility"], reverse=True)
    logger.info("Top %d operating points by avg_utility:", min(limit, len(ranked)))
    for row in ranked[:limit]:
        logger.info(
            "lambda=%.4f tau=%.3f utility=%.4f quality=%.4f cost=%.4f coverage=%.3f",
            row["lambda"],
            row["tau"],
            row["avg_utility"],
            row["avg_quality"],
            row["avg_cost"],
            row["coverage"],
        )


def _print_low_regret_rows(rows: list[dict[str, Any]], limit: int = 8) -> None:
    ranked = sorted(rows, key=lambda row: (row["mean_regret"], row["p95_regret"]))
    logger.info("Top %d operating points by lowest mean_regret:", min(limit, len(ranked)))
    for row in ranked[:limit]:
        logger.info(
            "lambda=%.4f tau=%.3f mean_regret=%.4f p95_regret=%.4f util_pred=%.4f util_oracle=%.4f capture=%s",
            row["lambda"],
            row["tau"],
            row["mean_regret"],
            row["p95_regret"],
            row["mean_util_pred"],
            row["mean_util_oracle"],
            f"{row['capture']:.4f}" if row.get("capture") is not None else "n/a",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline utility-based router analysis on runtime traces.")
    parser.add_argument(
        "--input-glob",
        action="append",
        required=True,
        help="Glob pattern for router trace JSONL files. Can be provided multiple times.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-key", default="policy_value_prompt_last_all")
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--all-model-versions", action="store_true", help="Do not filter to latest model version")
    parser.add_argument("--keep-duplicates", action="store_true", help="Do not dedupe by problem id")

    parser.add_argument("--lambda-values", default="", help="Comma-separated lambda values (overrides start/stop/step)")
    parser.add_argument("--lambda-start", type=float, default=0.0)
    parser.add_argument("--lambda-stop", type=float, default=0.0)
    parser.add_argument("--lambda-step", type=float, default=0.01)

    # Preferred abstention knob names.
    parser.add_argument("--tau-start", type=float, default=None)
    parser.add_argument("--tau-stop", type=float, default=None)
    parser.add_argument("--tau-step", type=float, default=None)

    # Backward-compatible aliases.
    parser.add_argument("--threshold-start", type=float, default=0.0)
    parser.add_argument("--threshold-stop", type=float, default=1.0)
    parser.add_argument("--threshold-step", type=float, default=0.05)

    parser.add_argument("--no-abstain", action="store_true", help="Disable abstention")
    parser.add_argument("--abstain-quality", type=float, default=0.0)

    parser.add_argument(
        "--policy-cost-per-1k",
        type=float,
        default=1.0,
        help="Cost units per 1K policy tokens (set to USD rate if desired)",
    )
    parser.add_argument(
        "--expert-costs-per-1k",
        default="",
        help="Comma-separated cost per 1K tokens for each expert, in expert order",
    )
    parser.add_argument(
        "--regret-baseline-route",
        default="policy",
        help="Route label for capture metric baseline (e.g. policy, expert_0). Use 'none' to disable.",
    )
    parser.add_argument("--max-log-points", type=int, default=12)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    split = None if args.split == "all" else args.split
    traces = load_router_traces(
        input_globs=args.input_glob,
        split=split,
        latest_model_only=not args.all_model_versions,
        dedupe_by_problem=not args.keep_duplicates,
    )
    if not traces:
        raise ValueError("No traces found after filtering")

    lambdas = _parse_lambda_values(args)
    taus = _parse_tau_values(args)

    n_experts = max((len(trace.get("experts") or []) for trace in traces), default=0)
    route_labels = extract_route_labels(traces, n_experts)
    expert_costs_per_1k = _parse_expert_costs(
        args.expert_costs_per_1k,
        n_experts=n_experts,
        default=args.policy_cost_per_1k,
    )
    best_expert_stats, best_expert_rows = _compute_best_expert_identity_stats(
        traces=traces,
        route_labels=route_labels,
    )
    route_mean_rewards = _compute_route_mean_rewards(
        traces=traces,
        route_labels=route_labels,
    )
    baseline_route_idx = _resolve_baseline_index(route_labels, args.regret_baseline_route)
    baseline_route_label = route_labels[baseline_route_idx] if baseline_route_idx is not None else None

    rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []
    for lambda_cost in lambdas:
        for tau in taus:
            row = _evaluate_operating_point(
                traces=traces,
                score_key=args.score_key,
                lambda_cost=lambda_cost,
                tau=tau,
                allow_abstain=not args.no_abstain,
                abstain_quality=args.abstain_quality,
                route_labels=route_labels,
                policy_cost_per_1k=args.policy_cost_per_1k,
                expert_costs_per_1k=expert_costs_per_1k,
            )
            rows.append(row)
            oracle_rows.append(
                _evaluate_oracle_operating_point(
                    traces=traces,
                    lambda_cost=lambda_cost,
                    tau=tau,
                    allow_abstain=not args.no_abstain,
                    abstain_quality=args.abstain_quality,
                    route_labels=route_labels,
                    policy_cost_per_1k=args.policy_cost_per_1k,
                    expert_costs_per_1k=expert_costs_per_1k,
                )
            )
            baseline_rows.append(
                _evaluate_operating_point(
                    traces=traces,
                    score_key=args.score_key,
                    lambda_cost=lambda_cost,
                    tau=tau,
                    allow_abstain=not args.no_abstain,
                    abstain_quality=args.abstain_quality,
                    route_labels=route_labels,
                    policy_cost_per_1k=args.policy_cost_per_1k,
                    expert_costs_per_1k=expert_costs_per_1k,
                    fixed_scores=route_mean_rewards,
                )
            )
        regret_rows.append(
            _evaluate_oracle_predictor_regret(
                traces=traces,
                score_key=args.score_key,
                lambda_cost=lambda_cost,
                tau=0.0,
                allow_abstain=False,
                route_labels=route_labels,
                policy_cost_per_1k=args.policy_cost_per_1k,
                expert_costs_per_1k=expert_costs_per_1k,
                baseline_route_idx=baseline_route_idx,
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    share_columns = [_route_share_key(name) for name in route_labels] + [_route_share_key("abstain")]
    headers = [
        "lambda",
        "tau",
        "n_total",
        "n_scored",
        "n_missing_score",
        "n_non_abstain",
        "coverage",
        "score_coverage",
        "abstain_rate",
        "avg_quality",
        "avg_cost",
        "avg_utility",
        "avg_tokens",
        "avg_quality_when_routed",
        "avg_cost_when_routed",
    ] + share_columns
    _write_csv(output_dir / "operating_points.csv", rows, headers)
    _write_csv(output_dir / "oracle_operating_points.csv", oracle_rows, headers)
    _write_csv(
        output_dir / "best_expert_identity.csv",
        best_expert_rows,
        ["expert_index", "expert_label", "count_best", "share_best"],
    )

    frontier = _build_frontier(rows)
    _write_csv(
        output_dir / "frontier_points.csv",
        frontier,
        ["lambda", "tau", "avg_cost", "avg_quality", "avg_utility", "coverage", "abstain_rate"],
    )
    oracle_frontier = _build_frontier(oracle_rows)
    _write_csv(
        output_dir / "oracle_frontier_points.csv",
        oracle_frontier,
        ["lambda", "tau", "avg_cost", "avg_quality", "avg_utility", "coverage", "abstain_rate"],
    )
    _write_csv(
        output_dir / "oracle_predictor_regret.csv",
        regret_rows,
        [
            "lambda",
            "tau",
            "n_total",
            "n_scored",
            "n_missing_score",
            "score_coverage",
            "mean_util_oracle",
            "mean_util_pred",
            "mean_util_base",
            "mean_regret",
            "p95_regret",
            "oracle_coverage",
            "predictor_coverage",
            "oracle_abstain_rate",
            "predictor_abstain_rate",
            "capture",
            "utility_gap",
            "n_baseline_available",
        ],
    )
    auc_qc, auc_qc_norm = _compute_auc_qc(frontier)
    oracle_auc_qc, oracle_auc_qc_norm = _compute_auc_qc(oracle_frontier)

    best_utility = max(rows, key=lambda row: row["avg_utility"])
    best_quality = max(rows, key=lambda row: row["avg_quality"])
    min_cost = min(rows, key=lambda row: row["avg_cost"])
    best_mean_regret = min(regret_rows, key=lambda row: row["mean_regret"])
    best_p95_regret = min(regret_rows, key=lambda row: row["p95_regret"])
    capture_rows = [row for row in regret_rows if row.get("capture") is not None]
    best_capture = max(capture_rows, key=lambda row: row["capture"]) if capture_rows else None
    summary = {
        "n_traces": len(traces),
        "score_key": args.score_key,
        "route_labels": route_labels,
        "n_operating_points": len(rows),
        "lambda_values": lambdas,
        "tau_values": taus,
        "auc_qc": auc_qc,
        "auc_qc_normalized": auc_qc_norm,
        "oracle_auc_qc": oracle_auc_qc,
        "oracle_auc_qc_normalized": oracle_auc_qc_norm,
        "best_expert_identity": best_expert_stats,
        "best_avg_utility": {
            "lambda": best_utility["lambda"],
            "tau": best_utility["tau"],
            "avg_utility": best_utility["avg_utility"],
            "avg_quality": best_utility["avg_quality"],
            "avg_cost": best_utility["avg_cost"],
            "coverage": best_utility["coverage"],
        },
        "best_avg_quality": {
            "lambda": best_quality["lambda"],
            "tau": best_quality["tau"],
            "avg_quality": best_quality["avg_quality"],
            "avg_cost": best_quality["avg_cost"],
            "coverage": best_quality["coverage"],
        },
        "min_avg_cost": {
            "lambda": min_cost["lambda"],
            "tau": min_cost["tau"],
            "avg_cost": min_cost["avg_cost"],
            "avg_quality": min_cost["avg_quality"],
            "coverage": min_cost["coverage"],
        },
        "oracle_predictor_regret": {
            "baseline_route": baseline_route_label,
            "best_mean_regret": {
                "lambda": best_mean_regret["lambda"],
                "tau": best_mean_regret["tau"],
                "mean_regret": best_mean_regret["mean_regret"],
                "p95_regret": best_mean_regret["p95_regret"],
                "mean_util_oracle": best_mean_regret["mean_util_oracle"],
                "mean_util_pred": best_mean_regret["mean_util_pred"],
                "capture": best_mean_regret.get("capture"),
            },
            "best_p95_regret": {
                "lambda": best_p95_regret["lambda"],
                "tau": best_p95_regret["tau"],
                "mean_regret": best_p95_regret["mean_regret"],
                "p95_regret": best_p95_regret["p95_regret"],
                "mean_util_oracle": best_p95_regret["mean_util_oracle"],
                "mean_util_pred": best_p95_regret["mean_util_pred"],
                "capture": best_p95_regret.get("capture"),
            },
            "best_capture": (
                {
                    "lambda": best_capture["lambda"],
                    "tau": best_capture["tau"],
                    "capture": best_capture["capture"],
                    "mean_regret": best_capture["mean_regret"],
                    "p95_regret": best_capture["p95_regret"],
                }
                if best_capture is not None
                else None
            ),
        },
        "pairwise_ranking_metrics_file": "pairwise_ranking_metrics.csv",
        "pairwise_delta_stats_file": "pairwise_delta/pairwise_delta_stats.csv",
        "frontier_metrics_file": "frontier_metrics.csv",
        "oracle_gap_capture_file": "oracle_gap_capture_vs_lambda.csv",
        "pred_std_over_model_version_file": "pred_std_over_model_version.csv",
        "true_reward_std_reference_file": "true_reward_std_reference.csv",
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    _plot_cost_quality_scatter(rows, output_dir / "cost_vs_quality_scatter.png")
    _plot_frontier(rows, frontier, auc_qc, output_dir / "pareto_frontier_auc.png")
    _plot_frontier(
        oracle_rows,
        oracle_frontier,
        oracle_auc_qc,
        output_dir / "oracle_pareto_frontier_auc.png",
    )
    _plot_heatmap(
        rows,
        lambdas,
        taus,
        metric_key="coverage",
        title="Coverage heatmap (non-abstain rate)",
        out_path=output_dir / "coverage_heatmap_lambda_tau.png",
    )
    _plot_heatmap(
        rows,
        lambdas,
        taus,
        metric_key="avg_utility",
        title="Average utility heatmap",
        out_path=output_dir / "utility_heatmap_lambda_tau.png",
    )
    _plot_regret_curves(
        regret_rows,
        lambdas,
        [0.0],
        metric_key="mean_regret",
        title="Mean oracle-vs-predictor utility regret vs lambda",
        ylabel="Mean regret",
        out_path=output_dir / "mean_regret_vs_lambda.png",
    )
    _plot_regret_curves(
        regret_rows,
        lambdas,
        [0.0],
        metric_key="p95_regret",
        title="P95 oracle-vs-predictor utility regret vs lambda",
        ylabel="P95 regret",
        out_path=output_dir / "p95_regret_vs_lambda.png",
    )
    _plot_capture_vs_lambda(
        regret_rows,
        out_path=output_dir / "oracle_gap_capture_vs_lambda.png",
    )
    _plot_predicted_vs_realized_by_route(
        traces=traces,
        score_key=args.score_key,
        route_labels=route_labels,
        out_dir=output_dir / "pred_vs_realized",
    )
    _plot_pairwise_delta_scatter_and_stats(
        traces=traces,
        score_key=args.score_key,
        route_labels=route_labels,
        out_dir=output_dir / "pairwise_delta",
    )
    pairwise_ranking_rows = _compute_pairwise_ranking_metrics(
        traces=traces,
        score_key=args.score_key,
        route_labels=route_labels,
        out_path=output_dir / "pairwise_ranking_metrics.csv",
    )
    _plot_routing_pies(
        rows,
        route_labels=route_labels,
        out_dir=output_dir / "routing_pies",
    )

    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(baseline_dir / "operating_points.csv", baseline_rows, headers)
    baseline_frontier = _build_frontier(baseline_rows)
    _write_csv(
        baseline_dir / "frontier_points.csv",
        baseline_frontier,
        ["lambda", "tau", "avg_cost", "avg_quality", "avg_utility", "coverage", "abstain_rate"],
    )
    _write_csv(
        baseline_dir / "route_mean_rewards.csv",
        [
            {"route_index": idx, "route_label": route_labels[idx], "mean_reward": route_mean_rewards[idx]}
            for idx in range(len(route_labels))
        ],
        ["route_index", "route_label", "mean_reward"],
    )
    baseline_auc_qc, baseline_auc_qc_norm = _compute_auc_qc(baseline_frontier)
    baseline_best_utility = max(baseline_rows, key=lambda row: row["avg_utility"])
    baseline_best_quality = max(baseline_rows, key=lambda row: row["avg_quality"])
    baseline_min_cost = min(baseline_rows, key=lambda row: row["avg_cost"])
    baseline_summary = {
        "description": "Constant-prediction baseline: each route prediction is its mean realized reward.",
        "n_traces": len(traces),
        "route_labels": route_labels,
        "route_mean_rewards": {
            route_labels[idx]: route_mean_rewards[idx] for idx in range(len(route_labels))
        },
        "n_operating_points": len(baseline_rows),
        "lambda_values": lambdas,
        "tau_values": taus,
        "auc_qc": baseline_auc_qc,
        "auc_qc_normalized": baseline_auc_qc_norm,
        "best_avg_utility": {
            "lambda": baseline_best_utility["lambda"],
            "tau": baseline_best_utility["tau"],
            "avg_utility": baseline_best_utility["avg_utility"],
            "avg_quality": baseline_best_utility["avg_quality"],
            "avg_cost": baseline_best_utility["avg_cost"],
            "coverage": baseline_best_utility["coverage"],
        },
        "best_avg_quality": {
            "lambda": baseline_best_quality["lambda"],
            "tau": baseline_best_quality["tau"],
            "avg_quality": baseline_best_quality["avg_quality"],
            "avg_cost": baseline_best_quality["avg_cost"],
            "coverage": baseline_best_quality["coverage"],
        },
        "min_avg_cost": {
            "lambda": baseline_min_cost["lambda"],
            "tau": baseline_min_cost["tau"],
            "avg_cost": baseline_min_cost["avg_cost"],
            "avg_quality": baseline_min_cost["avg_quality"],
            "coverage": baseline_min_cost["coverage"],
        },
    }
    with (baseline_dir / "summary.json").open("w") as handle:
        json.dump(baseline_summary, handle, indent=2)
    _plot_cost_quality_scatter(baseline_rows, baseline_dir / "cost_vs_quality_scatter.png")
    _plot_frontier(baseline_rows, baseline_frontier, baseline_auc_qc, baseline_dir / "pareto_frontier_auc.png")
    _plot_heatmap(
        baseline_rows,
        lambdas,
        taus,
        metric_key="coverage",
        title="Coverage heatmap (baseline, non-abstain rate)",
        out_path=baseline_dir / "coverage_heatmap_lambda_tau.png",
    )
    _plot_heatmap(
        baseline_rows,
        lambdas,
        taus,
        metric_key="avg_utility",
        title="Average utility heatmap (baseline)",
        out_path=baseline_dir / "utility_heatmap_lambda_tau.png",
    )
    _plot_routing_pies(
        baseline_rows,
        route_labels=route_labels,
        out_dir=baseline_dir / "routing_pies",
    )
    _plot_combined_frontiers(
        actual_frontier=frontier,
        baseline_frontier=baseline_frontier,
        oracle_frontier=oracle_frontier,
        actual_auc_qc=auc_qc,
        baseline_auc_qc=baseline_auc_qc,
        oracle_auc_qc=oracle_auc_qc,
        out_path=output_dir / "pareto_frontier_combined.png",
    )
    tau0_rows = _rows_for_tau(rows, 0.0)
    tau0_baseline_rows = _rows_for_tau(baseline_rows, 0.0)
    tau0_oracle_rows = _rows_for_tau(oracle_rows, 0.0)
    tau0_frontier = _build_frontier(tau0_rows)
    tau0_baseline_frontier = _build_frontier(tau0_baseline_rows)
    tau0_oracle_frontier = _build_frontier(tau0_oracle_rows)
    tau0_auc, tau0_auc_norm = _compute_auc_qc(tau0_frontier)
    tau0_baseline_auc, tau0_baseline_auc_norm = _compute_auc_qc(tau0_baseline_frontier)
    tau0_oracle_auc, tau0_oracle_auc_norm = _compute_auc_qc(tau0_oracle_frontier)
    _plot_combined_frontiers(
        actual_frontier=tau0_frontier,
        baseline_frontier=tau0_baseline_frontier,
        oracle_frontier=tau0_oracle_frontier,
        actual_auc_qc=tau0_auc,
        baseline_auc_qc=tau0_baseline_auc,
        oracle_auc_qc=tau0_oracle_auc,
        out_path=output_dir / "pareto_frontier_combined_tau0.png",
    )
    frontier_metric_rows = [
        {
            "method": "learned_all_tau",
            "tau_filter": "all",
            "auc_qc_raw": auc_qc,
            "auc_qc_normalized": auc_qc_norm,
            "cost_min": _frontier_cost_range(frontier)[0],
            "cost_max": _frontier_cost_range(frontier)[1],
            "cost_range": _frontier_cost_range(frontier)[2],
            "n_frontier_points": len(frontier),
        },
        {
            "method": "baseline_all_tau",
            "tau_filter": "all",
            "auc_qc_raw": baseline_auc_qc,
            "auc_qc_normalized": baseline_auc_qc_norm,
            "cost_min": _frontier_cost_range(baseline_frontier)[0],
            "cost_max": _frontier_cost_range(baseline_frontier)[1],
            "cost_range": _frontier_cost_range(baseline_frontier)[2],
            "n_frontier_points": len(baseline_frontier),
        },
        {
            "method": "oracle_all_tau",
            "tau_filter": "all",
            "auc_qc_raw": oracle_auc_qc,
            "auc_qc_normalized": oracle_auc_qc_norm,
            "cost_min": _frontier_cost_range(oracle_frontier)[0],
            "cost_max": _frontier_cost_range(oracle_frontier)[1],
            "cost_range": _frontier_cost_range(oracle_frontier)[2],
            "n_frontier_points": len(oracle_frontier),
        },
        {
            "method": "learned_tau0",
            "tau_filter": "0.0",
            "auc_qc_raw": tau0_auc,
            "auc_qc_normalized": tau0_auc_norm,
            "cost_min": _frontier_cost_range(tau0_frontier)[0],
            "cost_max": _frontier_cost_range(tau0_frontier)[1],
            "cost_range": _frontier_cost_range(tau0_frontier)[2],
            "n_frontier_points": len(tau0_frontier),
        },
        {
            "method": "baseline_tau0",
            "tau_filter": "0.0",
            "auc_qc_raw": tau0_baseline_auc,
            "auc_qc_normalized": tau0_baseline_auc_norm,
            "cost_min": _frontier_cost_range(tau0_baseline_frontier)[0],
            "cost_max": _frontier_cost_range(tau0_baseline_frontier)[1],
            "cost_range": _frontier_cost_range(tau0_baseline_frontier)[2],
            "n_frontier_points": len(tau0_baseline_frontier),
        },
        {
            "method": "oracle_tau0",
            "tau_filter": "0.0",
            "auc_qc_raw": tau0_oracle_auc,
            "auc_qc_normalized": tau0_oracle_auc_norm,
            "cost_min": _frontier_cost_range(tau0_oracle_frontier)[0],
            "cost_max": _frontier_cost_range(tau0_oracle_frontier)[1],
            "cost_range": _frontier_cost_range(tau0_oracle_frontier)[2],
            "n_frontier_points": len(tau0_oracle_frontier),
        },
    ]
    _write_csv(
        output_dir / "frontier_metrics.csv",
        frontier_metric_rows,
        [
            "method",
            "tau_filter",
            "auc_qc_raw",
            "auc_qc_normalized",
            "cost_min",
            "cost_max",
            "cost_range",
            "n_frontier_points",
        ],
    )
    _write_csv(
        output_dir / "oracle_gap_capture_vs_lambda.csv",
        regret_rows,
        [
            "lambda",
            "tau",
            "mean_util_oracle",
            "mean_util_pred",
            "mean_util_base",
            "utility_gap",
            "capture",
            "mean_regret",
            "p95_regret",
            "n_scored",
        ],
    )
    pred_std_rows, true_std_rows = _compute_pred_std_over_model_version(
        traces=traces,
        score_key=args.score_key,
        route_labels=route_labels,
    )
    _write_csv(
        output_dir / "pred_std_over_model_version.csv",
        pred_std_rows,
        ["model_version", "route_index", "route_label", "n_points", "pred_mean", "pred_std"],
    )
    _write_csv(
        output_dir / "true_reward_std_reference.csv",
        true_std_rows,
        ["route_index", "route_label", "n_points", "realized_mean", "realized_std"],
    )
    _plot_pred_std_over_model_version(
        pred_rows=pred_std_rows,
        true_std_rows=true_std_rows,
        route_labels=route_labels,
        out_path=output_dir / "pred_std_over_model_version.png",
    )

    _print_top_rows(rows, limit=args.max_log_points)
    _print_low_regret_rows(regret_rows, limit=args.max_log_points)
    logger.info(
        "Best-expert identity: n_tasks=%s variance=%.4f random_pair_change_prob=%.4f dominant=%s (share=%.3f)",
        best_expert_stats["n_tasks"],
        best_expert_stats["best_expert_variance"],
        best_expert_stats["best_expert_change_prob_random_pair"],
        best_expert_stats["dominant_expert"],
        best_expert_stats["dominant_expert_share"],
    )
    logger.info(
        "Baseline folder written to %s (constant route means). best_utility=(lambda=%.4f,tau=%.3f,utility=%.4f)",
        baseline_dir,
        baseline_best_utility["lambda"],
        baseline_best_utility["tau"],
        baseline_best_utility["avg_utility"],
    )
    logger.info(
        "Pairwise ranking metrics rows=%d written to %s",
        len(pairwise_ranking_rows),
        output_dir / "pairwise_ranking_metrics.csv",
    )
    logger.info(
        "Tau=0 frontier AUCs: learned=%.4f baseline=%.4f oracle=%.4f",
        tau0_auc,
        tau0_baseline_auc,
        tau0_oracle_auc,
    )
    logger.info(
        "Predicted std over model_version points=%d (versions=%d)",
        len(pred_std_rows),
        len({row['model_version'] for row in pred_std_rows}),
    )
    logger.info("Oracle frontier AUC-QC=%.4f (normalized %.4f)", oracle_auc_qc, oracle_auc_qc_norm)
    logger.info("Wrote outputs to %s", output_dir)


if __name__ == "__main__":
    main()

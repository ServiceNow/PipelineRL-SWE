#!/usr/bin/env python
import argparse
import csv
import importlib.util
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
    if baseline_count > 0:
        mean_util_base = util_base_sum / baseline_count
        denom = mean_util_oracle - mean_util_base
        if abs(denom) > 1e-12:
            capture = (mean_util_pred - mean_util_base) / denom

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
    _plot_predicted_vs_realized_by_route(
        traces=traces,
        score_key=args.score_key,
        route_labels=route_labels,
        out_dir=output_dir / "pred_vs_realized",
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
    logger.info("Oracle frontier AUC-QC=%.4f (normalized %.4f)", oracle_auc_qc, oracle_auc_qc_norm)
    logger.info("Wrote outputs to %s", output_dir)


if __name__ == "__main__":
    main()

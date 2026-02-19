#!/usr/bin/env python
import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt  # type: ignore

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

from pipelinerl.swe.scripts.new.router_trace_utils import (
    extract_reward_vector,
    extract_route_labels,
    extract_score_vector,
    extract_token_vector,
    load_router_traces,
)

logger = logging.getLogger(__name__)


def _frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("threshold_step must be > 0")
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 8))
        current += step
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


def _evaluate_threshold(
    traces: list[dict[str, Any]],
    score_key: str,
    threshold: float,
    allow_abstain: bool,
    abstain_reward: float,
    success_threshold: float,
    route_labels: list[str],
    policy_cost_per_1k: float,
    expert_costs_per_1k: list[float],
) -> dict[str, Any]:
    total = 0
    routed = 0
    missing_score = 0

    reward_sum = 0.0
    regret_sum = 0.0
    token_sum = 0.0
    cost_sum = 0.0
    success_sum = 0.0
    oracle_reward_sum = 0.0

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
        scores = extract_score_vector(trace, score_key=score_key, expected_dim=dim)
        total += 1

        if scores is None:
            missing_score += 1
            continue

        routed += 1
        oracle_reward = max(rewards)
        oracle_reward_sum += oracle_reward

        best_score = max(scores)
        best_route = max(range(dim), key=lambda idx: scores[idx])
        if allow_abstain and best_score < threshold:
            chosen_route = -1
        else:
            chosen_route = best_route

        if chosen_route < 0:
            chosen_reward = abstain_reward
            chosen_tokens = 0.0
            chosen_cost = 0.0
            route_counts["abstain"] += 1
        else:
            route_name = route_labels[chosen_route]
            route_counts[route_name] += 1
            chosen_reward = rewards[chosen_route]
            chosen_tokens = float(tokens[chosen_route])
            if chosen_route == 0:
                cost_rate = policy_cost_per_1k
            else:
                cost_rate = expert_costs_per_1k[chosen_route - 1]
            chosen_cost = chosen_tokens * cost_rate / 1000.0

        reward_sum += chosen_reward
        token_sum += chosen_tokens
        cost_sum += chosen_cost
        regret_sum += oracle_reward - chosen_reward
        success_sum += 1.0 if chosen_reward >= success_threshold else 0.0

    if routed == 0:
        row = {
            "threshold": threshold,
            "n_total": total,
            "n_routed": 0,
            "n_missing_score": missing_score,
            "coverage": 0.0,
            "avg_reward": 0.0,
            "avg_regret": 0.0,
            "avg_tokens": 0.0,
            "avg_cost": 0.0,
            "oracle_avg_reward": 0.0,
            "success_rate": 0.0,
            "_route_counts": route_counts,
        }
    else:
        row = {
            "threshold": threshold,
            "n_total": total,
            "n_routed": routed,
            "n_missing_score": missing_score,
            "coverage": routed / total if total else 0.0,
            "avg_reward": reward_sum / routed,
            "avg_regret": regret_sum / routed,
            "avg_tokens": token_sum / routed,
            "avg_cost": cost_sum / routed,
            "oracle_avg_reward": oracle_reward_sum / routed,
            "success_rate": success_sum / routed,
            "_route_counts": route_counts,
        }

    for name, count in route_counts.items():
        row[_route_share_key(name)] = count / routed if routed else 0.0

    return row


def _evaluate_baselines(
    traces: list[dict[str, Any]],
    success_threshold: float,
    policy_cost_per_1k: float,
    expert_costs_per_1k: list[float],
) -> list[dict[str, Any]]:
    rows = []
    modes = ["policy_only", "oracle_best", "best_single_expert"]

    for mode in modes:
        rewards_sum = 0.0
        regret_sum = 0.0
        tokens_sum = 0.0
        cost_sum = 0.0
        success_sum = 0.0
        oracle_sum = 0.0
        count = 0

        for trace in traces:
            rewards = extract_reward_vector(trace)
            tokens = extract_token_vector(trace)
            dim = min(len(rewards), len(tokens))
            if dim <= 0:
                continue

            rewards = rewards[:dim]
            tokens = tokens[:dim]
            oracle_reward = max(rewards)
            oracle_sum += oracle_reward

            if mode == "policy_only":
                idx = 0
            elif mode == "oracle_best":
                idx = max(range(dim), key=lambda i: rewards[i])
            else:
                if dim <= 1:
                    idx = 0
                else:
                    idx = 1 + max(range(dim - 1), key=lambda i: rewards[i + 1])

            chosen_reward = rewards[idx]
            chosen_tokens = float(tokens[idx])
            if idx == 0:
                cost_rate = policy_cost_per_1k
            else:
                cost_rate = expert_costs_per_1k[idx - 1] if idx - 1 < len(expert_costs_per_1k) else policy_cost_per_1k
            chosen_cost = chosen_tokens * cost_rate / 1000.0

            rewards_sum += chosen_reward
            regret_sum += oracle_reward - chosen_reward
            tokens_sum += chosen_tokens
            cost_sum += chosen_cost
            success_sum += 1.0 if chosen_reward >= success_threshold else 0.0
            count += 1

        rows.append(
            {
                "mode": mode,
                "n": count,
                "avg_reward": rewards_sum / count if count else 0.0,
                "avg_regret": regret_sum / count if count else 0.0,
                "avg_tokens": tokens_sum / count if count else 0.0,
                "avg_cost": cost_sum / count if count else 0.0,
                "oracle_avg_reward": oracle_sum / count if count else 0.0,
                "success_rate": success_sum / count if count else 0.0,
            }
        )

    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _plot_metric_curves(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping metric curves")
        return

    thresholds = [row["threshold"] for row in rows]
    rewards = [row["avg_reward"] for row in rows]
    regrets = [row["avg_regret"] for row in rows]
    costs = [row["avg_cost"] for row in rows]

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, rewards, label="avg_reward", color="tab:blue")
    plt.plot(thresholds, regrets, label="avg_regret", color="tab:red")
    plt.plot(thresholds, costs, label="avg_cost", color="tab:green")
    plt.xlabel("threshold")
    plt.ylabel("metric")
    plt.title("Router threshold sweep")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _pick_pie_rows(rows: list[dict[str, Any]], max_pies: int) -> list[dict[str, Any]]:
    rows = [row for row in rows if row.get("n_routed", 0) > 0]
    if len(rows) <= max_pies:
        return rows

    best_reward = max(rows, key=lambda row: row["avg_reward"])
    best_regret = min(rows, key=lambda row: row["avg_regret"])
    best_cost = min(rows, key=lambda row: row["avg_cost"])

    selected = {rows[0]["threshold"], rows[-1]["threshold"], best_reward["threshold"], best_regret["threshold"], best_cost["threshold"]}

    stride = max(1, len(rows) // max_pies)
    for idx in range(0, len(rows), stride):
        selected.add(rows[idx]["threshold"])

    selected_rows = [row for row in rows if row["threshold"] in selected]
    return selected_rows[:max_pies]


def _plot_pies(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping routing pies")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        counts = row.get("_route_counts") or {}
        labels = []
        values = []
        for name, value in counts.items():
            if value <= 0:
                continue
            labels.append(name)
            values.append(value)
        if not values:
            continue

        plt.figure(figsize=(5, 5))
        plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title(f"Routing share @ threshold={row['threshold']:.2f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"routing_pie_{row['threshold']:.2f}.png")
        plt.close()


def _print_top_rows(rows: list[dict[str, Any]], limit: int = 8) -> None:
    ranked = sorted(rows, key=lambda row: row["avg_reward"], reverse=True)
    logger.info("Top %d thresholds by avg_reward:", min(limit, len(ranked)))
    for row in ranked[:limit]:
        logger.info(
            "thr=%.2f reward=%.4f regret=%.4f cost=%.4f coverage=%.3f",
            row["threshold"],
            row["avg_reward"],
            row["avg_regret"],
            row["avg_cost"],
            row["coverage"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline router analysis on runtime router traces.")
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
    parser.add_argument("--keep-duplicates", action="store_true", help="Do not dedupe by problem_id")
    parser.add_argument("--threshold-start", type=float, default=0.0)
    parser.add_argument("--threshold-stop", type=float, default=1.0)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--no-abstain", action="store_true", help="Disable abstention thresholding")
    parser.add_argument("--abstain-reward", type=float, default=0.0)
    parser.add_argument("--success-threshold", type=float, default=0.8)
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
    parser.add_argument("--max-pies", type=int, default=8, help="Maximum number of pie charts to render")
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

    n_experts = max((len(trace.get("experts") or []) for trace in traces), default=0)
    route_labels = extract_route_labels(traces, n_experts)
    expert_costs_per_1k = _parse_expert_costs(
        args.expert_costs_per_1k,
        n_experts=n_experts,
        default=args.policy_cost_per_1k,
    )

    thresholds = _frange(args.threshold_start, args.threshold_stop, args.threshold_step)
    threshold_rows = [
        _evaluate_threshold(
            traces=traces,
            score_key=args.score_key,
            threshold=threshold,
            allow_abstain=not args.no_abstain,
            abstain_reward=args.abstain_reward,
            success_threshold=args.success_threshold,
            route_labels=route_labels,
            policy_cost_per_1k=args.policy_cost_per_1k,
            expert_costs_per_1k=expert_costs_per_1k,
        )
        for threshold in thresholds
    ]

    baseline_rows = _evaluate_baselines(
        traces=traces,
        success_threshold=args.success_threshold,
        policy_cost_per_1k=args.policy_cost_per_1k,
        expert_costs_per_1k=expert_costs_per_1k,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    share_columns = [_route_share_key(name) for name in route_labels] + [_route_share_key("abstain")]
    threshold_headers = [
        "threshold",
        "n_total",
        "n_routed",
        "n_missing_score",
        "coverage",
        "avg_reward",
        "avg_regret",
        "avg_tokens",
        "avg_cost",
        "oracle_avg_reward",
        "success_rate",
    ] + share_columns

    _write_csv(output_dir / "threshold_metrics.csv", threshold_rows, threshold_headers)
    _write_csv(
        output_dir / "baseline_metrics.csv",
        baseline_rows,
        ["mode", "n", "avg_reward", "avg_regret", "avg_tokens", "avg_cost", "oracle_avg_reward", "success_rate"],
    )

    best_reward = max(threshold_rows, key=lambda row: row["avg_reward"])
    best_regret = min(threshold_rows, key=lambda row: row["avg_regret"])
    best_cost = min(threshold_rows, key=lambda row: row["avg_cost"])
    summary = {
        "n_traces": len(traces),
        "score_key": args.score_key,
        "route_labels": route_labels,
        "best_reward_threshold": best_reward["threshold"],
        "best_reward": best_reward["avg_reward"],
        "best_regret_threshold": best_regret["threshold"],
        "best_regret": best_regret["avg_regret"],
        "min_cost_threshold": best_cost["threshold"],
        "min_cost": best_cost["avg_cost"],
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    _plot_metric_curves(threshold_rows, output_dir / "threshold_metrics.png")
    pie_rows = _pick_pie_rows(threshold_rows, max_pies=args.max_pies)
    _plot_pies(pie_rows, output_dir / "pies")

    _print_top_rows(threshold_rows)
    logger.info("Wrote outputs to %s", output_dir)


if __name__ == "__main__":
    main()

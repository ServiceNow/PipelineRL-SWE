#!/usr/bin/env python
"""Score-bin marginal value diagnostics for multi-rollout SWE-Smith control.

This is CPU-only. It joins verifier scores with real pass/fail labels, then asks:
for a controller state after k observed 20B rollouts, how much value is left in
one more 20B sample, all remaining 20B samples, or a 120B escalation?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

DEFAULT_SCORE_JSONL = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "score_swe_smith_multirollout_eval150_proxy_init_random128_verifier_1781857392/"
    "scores/eval_verifier_scores.jsonl"
)
DEFAULT_REPORT_ROOT = (
    "router_analysis/uploaded_eval_full_20260617/"
    "swe_smith_multirollout_eval150_1781382734/logs/run_evaluation"
)
DEFAULT_OUTPUT_DIR = "router_analysis/swe_smith_multirollout_marginal_value_bins"
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _complete_ids(lookup: dict[tuple[str, int, int], dict[str, Any]]) -> list[str]:
    ids = sorted({pid for pid, _, _ in lookup})
    return [
        pid
        for pid in ids
        if all((pid, 1, rollout_idx) in lookup for rollout_idx in (0, 1, 2))
        and (pid, 3, 0) in lookup
    ]


def _select_by_score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: float(row["pred_score"]))


def _select_by_real(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: float(row["real_success"]))


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _quantile_bins(values: np.ndarray, n_bins: int) -> list[tuple[float, float]]:
    if values.size == 0:
        return []
    edges = np.quantile(values.astype(float), np.linspace(0.0, 1.0, int(n_bins) + 1))
    # Keep duplicate edges from producing empty labels, but still assign with searchsorted later.
    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]


def _assign_bin(value: float, edges: list[tuple[float, float]]) -> int:
    if not edges:
        return -1
    for idx, (lo, hi) in enumerate(edges):
        if idx == len(edges) - 1:
            if float(lo) <= float(value) <= float(hi):
                return idx
        elif float(lo) <= float(value) < float(hi):
            return idx
    # Duplicate quantile edges can put values outside strict intervals; use nearest fallback.
    mids = [0.5 * (lo + hi) for lo, hi in edges]
    return int(np.argmin(np.abs(np.asarray(mids, dtype=float) - float(value))))


def _state_rows(
    lookup: dict[tuple[str, int, int], dict[str, Any]],
    ids: list[str],
    lambdas: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid in ids:
        attempts_20b = [lookup[(pid, 1, rollout_idx)] for rollout_idx in (0, 1, 2)]
        attempt_120b = lookup[(pid, 3, 0)]
        for k in (1, 2):
            seen = attempts_20b[:k]
            remaining = attempts_20b[k:]
            current = _select_by_score(seen)
            after_one_seen = seen + remaining[:1]
            after_all_seen = seen + remaining
            best_after_one_by_score = _select_by_score(after_one_seen)
            best_after_all_by_score = _select_by_score(after_all_seen)
            best_after_one_oracle = _select_by_real(after_one_seen)
            best_after_all_oracle = _select_by_real(after_all_seen)
            best_current_or_120b_by_score = _select_by_score([current, attempt_120b])
            best_current_or_120b_oracle = _select_by_real([current, attempt_120b])
            current_reward = float(current["real_success"])
            current_cost = float(sum(float(row["weighted_output_cost"]) for row in seen))
            next20_cost = float(remaining[0]["weighted_output_cost"])
            all_remaining_cost = float(sum(float(row["weighted_output_cost"]) for row in remaining))
            cost_120b = float(attempt_120b["weighted_output_cost"])
            row: dict[str, Any] = {
                "original_problem_id": pid,
                "state_k": k,
                "current_score": float(current["pred_score"]),
                "current_success": current_reward,
                "current_cost": current_cost,
                "current_output_tokens": float(current.get("output_tokens") or 0.0),
                "next20b_success": float(remaining[0]["real_success"]),
                "next20b_score": float(remaining[0]["pred_score"]),
                "next20b_cost": next20_cost,
                "all_remaining_20b_cost": all_remaining_cost,
                "success_120b": float(attempt_120b["real_success"]),
                "score_120b": float(attempt_120b["pred_score"]),
                "cost_120b": cost_120b,
                "pass_after_one_20b_by_score": float(best_after_one_by_score["real_success"]),
                "pass_after_all_20b_by_score": float(best_after_all_by_score["real_success"]),
                "pass_after_one_20b_oracle": float(best_after_one_oracle["real_success"]),
                "pass_after_all_20b_oracle": float(best_after_all_oracle["real_success"]),
                "pass_current_or_120b_by_score": float(best_current_or_120b_by_score["real_success"]),
                "pass_current_or_120b_oracle": float(best_current_or_120b_oracle["real_success"]),
            }
            gain_specs = {
                "next20b_by_score": (
                    row["pass_after_one_20b_by_score"] - current_reward,
                    next20_cost,
                ),
                "next20b_oracle": (
                    row["pass_after_one_20b_oracle"] - current_reward,
                    next20_cost,
                ),
                "all20b_by_score": (
                    row["pass_after_all_20b_by_score"] - current_reward,
                    all_remaining_cost,
                ),
                "all20b_oracle": (
                    row["pass_after_all_20b_oracle"] - current_reward,
                    all_remaining_cost,
                ),
                "submit_120b": (
                    row["success_120b"] - current_reward,
                    cost_120b,
                ),
                "current_or_120b_by_score": (
                    row["pass_current_or_120b_by_score"] - current_reward,
                    cost_120b,
                ),
                "current_or_120b_oracle": (
                    row["pass_current_or_120b_oracle"] - current_reward,
                    cost_120b,
                ),
            }
            for action_name, (reward_gain, extra_cost) in gain_specs.items():
                row[f"reward_gain_{action_name}"] = float(reward_gain)
                row[f"extra_cost_{action_name}"] = float(extra_cost)
                for lambda_value in lambdas:
                    row[f"utility_gain_{action_name}_lambda_{lambda_value:g}"] = float(
                        reward_gain - float(lambda_value) * extra_cost
                    )
            rows.append(row)
    return rows


def _bin_summary(rows: list[dict[str, Any]], n_bins: int, lambdas: list[float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    actions = [
        "next20b_by_score",
        "next20b_oracle",
        "all20b_by_score",
        "all20b_oracle",
        "submit_120b",
        "current_or_120b_by_score",
        "current_or_120b_oracle",
    ]
    for state_k in sorted({int(row["state_k"]) for row in rows}):
        state_rows = [row for row in rows if int(row["state_k"]) == state_k]
        edges = _quantile_bins(np.asarray([row["current_score"] for row in state_rows], dtype=float), n_bins)
        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in state_rows:
            groups[_assign_bin(float(row["current_score"]), edges)].append(row)
        for bin_idx in sorted(groups):
            subset = groups[bin_idx]
            lo, hi = edges[bin_idx] if 0 <= bin_idx < len(edges) else (math.nan, math.nan)
            summary: dict[str, Any] = {
                "state_k": state_k,
                "score_bin": bin_idx,
                "score_lo": lo,
                "score_hi": hi,
                "n": len(subset),
                "current_score_mean": _mean([float(row["current_score"]) for row in subset]),
                "current_pass_rate": _mean([float(row["current_success"]) for row in subset]),
                "next20b_pass_rate": _mean([float(row["next20b_success"]) for row in subset]),
                "pass_after_one_20b_by_score": _mean([float(row["pass_after_one_20b_by_score"]) for row in subset]),
                "pass_after_all_20b_by_score": _mean([float(row["pass_after_all_20b_by_score"]) for row in subset]),
                "pass_after_one_20b_oracle": _mean([float(row["pass_after_one_20b_oracle"]) for row in subset]),
                "pass_after_all_20b_oracle": _mean([float(row["pass_after_all_20b_oracle"]) for row in subset]),
                "pass_120b": _mean([float(row["success_120b"]) for row in subset]),
                "pass_current_or_120b_by_score": _mean([float(row["pass_current_or_120b_by_score"]) for row in subset]),
                "pass_current_or_120b_oracle": _mean([float(row["pass_current_or_120b_oracle"]) for row in subset]),
            }
            for action in actions:
                gains = [float(row[f"reward_gain_{action}"]) for row in subset]
                costs = [float(row[f"extra_cost_{action}"]) for row in subset]
                summary[f"reward_gain_{action}"] = _mean(gains)
                summary[f"extra_cost_{action}"] = _mean(costs)
                summary[f"positive_reward_gain_rate_{action}"] = _mean([1.0 if gain > 0 else 0.0 for gain in gains])
                for lambda_value in lambdas:
                    key = f"utility_gain_{action}_lambda_{lambda_value:g}"
                    utility_gains = [float(row[key]) for row in subset]
                    summary[key] = _mean(utility_gains)
                    summary[f"positive_{key}_rate"] = _mean([1.0 if gain > 0 else 0.0 for gain in utility_gains])
            out.append(summary)
    return out


def _overall_summary(rows: list[dict[str, Any]], lambdas: list[float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    actions = [
        "next20b_by_score",
        "next20b_oracle",
        "all20b_by_score",
        "all20b_oracle",
        "submit_120b",
        "current_or_120b_by_score",
        "current_or_120b_oracle",
    ]
    for state_k in sorted({int(row["state_k"]) for row in rows}):
        subset = [row for row in rows if int(row["state_k"]) == state_k]
        for action in actions:
            result: dict[str, Any] = {
                "state_k": state_k,
                "action": action,
                "n": len(subset),
                "current_pass_rate": _mean([float(row["current_success"]) for row in subset]),
                "reward_gain": _mean([float(row[f"reward_gain_{action}"]) for row in subset]),
                "extra_cost": _mean([float(row[f"extra_cost_{action}"]) for row in subset]),
                "positive_reward_gain_rate": _mean([
                    1.0 if float(row[f"reward_gain_{action}"]) > 0 else 0.0 for row in subset
                ]),
            }
            for lambda_value in lambdas:
                key = f"utility_gain_{action}_lambda_{lambda_value:g}"
                vals = [float(row[key]) for row in subset]
                result[key] = _mean(vals)
                result[f"positive_{key}_rate"] = _mean([1.0 if value > 0 else 0.0 for value in vals])
            out.append(result)
    return out


def _maybe_plot(output_dir: Path, bin_rows: list[dict[str, Any]], lambdas: list[float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    for state_k in sorted({int(row["state_k"]) for row in bin_rows}):
        rows = [row for row in bin_rows if int(row["state_k"]) == state_k]
        x = [float(row["current_score_mean"]) for row in rows]
        plt.figure(figsize=(8, 5))
        plt.plot(x, [float(row["current_pass_rate"]) for row in rows], marker="o", label="current")
        plt.plot(x, [float(row["pass_after_one_20b_by_score"]) for row in rows], marker="o", label="+1 20B, verifier best")
        plt.plot(x, [float(row["pass_after_all_20b_by_score"]) for row in rows], marker="o", label="all 20B, verifier best")
        plt.plot(x, [float(row["pass_current_or_120b_by_score"]) for row in rows], marker="o", label="current or 120B, verifier best")
        plt.plot(x, [float(row["pass_after_all_20b_oracle"]) for row in rows], marker="o", linestyle="--", label="all 20B oracle")
        plt.xlabel("Current best verifier score")
        plt.ylabel("Real pass rate")
        plt.title(f"Pass rate by verifier-score bin after {state_k} observed 20B rollout(s)")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"pass_rate_by_score_bin_state{state_k}.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 5))
        for action, label in [
            ("next20b_by_score", "+1 20B, verifier best"),
            ("all20b_by_score", "all 20B, verifier best"),
            ("current_or_120b_by_score", "current or 120B, verifier best"),
            ("next20b_oracle", "+1 20B oracle"),
            ("current_or_120b_oracle", "current or 120B oracle"),
        ]:
            plt.plot(x, [float(row[f"reward_gain_{action}"]) for row in rows], marker="o", label=label)
        plt.axhline(0.0, color="black", linewidth=1)
        plt.xlabel("Current best verifier score")
        plt.ylabel("Mean reward gain over current")
        plt.title(f"Marginal reward by verifier-score bin after {state_k} observed 20B rollout(s)")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"marginal_reward_by_score_bin_state{state_k}.png", dpi=160)
        plt.close()

        for lambda_value in lambdas:
            plt.figure(figsize=(8, 5))
            for action, label in [
                ("next20b_by_score", "+1 20B, verifier best"),
                ("all20b_by_score", "all 20B, verifier best"),
                ("current_or_120b_by_score", "current or 120B, verifier best"),
            ]:
                col = f"utility_gain_{action}_lambda_{lambda_value:g}"
                plt.plot(x, [float(row[col]) for row in rows], marker="o", label=label)
            plt.axhline(0.0, color="black", linewidth=1)
            plt.xlabel("Current best verifier score")
            plt.ylabel("Mean utility gain over current")
            plt.title(f"Marginal utility by score bin, state {state_k}, lambda={lambda_value:g}")
            plt.grid(alpha=0.25)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(output_dir / f"marginal_utility_by_score_bin_state{state_k}_lambda{lambda_value:g}.png", dpi=160)
            plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze score-bin marginal value for 20B resample / 120B escalation.")
    parser.add_argument("--score-jsonl", default=DEFAULT_SCORE_JSONL)
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=6)
    parser.add_argument("--lambdas", default=",".join(str(x) for x in DEFAULT_LAMBDAS))
    parser.add_argument("--route-cost-weights", default=",".join(str(x) for x in DEFAULT_COST_WEIGHTS))
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lambdas = [float(piece) for piece in str(args.lambdas).split(",") if piece.strip()]
    route_cost_weights = [float(piece) for piece in str(args.route_cost_weights).split(",") if piece.strip()]
    output_dir = Path(args.output_dir)
    score_rows = _read_jsonl(Path(args.score_jsonl))
    attempts = _build_attempt_table(score_rows, Path(args.report_root), route_cost_weights)
    lookup = _attempt_lookup(attempts)
    ids = _complete_ids(lookup)
    state_rows = _state_rows(lookup, ids, lambdas)
    bin_rows = _bin_summary(state_rows, int(args.bins), lambdas)
    overall_rows = _overall_summary(state_rows, lambdas)
    _write_csv(output_dir / "state_action_rows.csv", state_rows)
    _write_csv(output_dir / "score_bin_marginal_value_summary.csv", bin_rows)
    _write_csv(output_dir / "overall_marginal_value_summary.csv", overall_rows)
    summary = {
        "score_jsonl": str(args.score_jsonl),
        "report_root": str(args.report_root),
        "output_dir": str(output_dir),
        "n_attempts": len(attempts),
        "n_complete_ids": len(ids),
        "bins": int(args.bins),
        "lambdas": lambdas,
        "route_cost_weights": route_cost_weights,
    }
    _write_json(output_dir / "summary.json", summary)
    if not args.no_plots:
        _maybe_plot(output_dir, bin_rows, lambdas)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Simulate active real-label acquisition for SWE patch verifiers.

This is intentionally lightweight: it uses existing verifier scores and
existing real pass/fail reports, hides real labels on a train split, selects a
small budget of labels to reveal, fits a simple calibration model, and evaluates
on heldout instances.
"""

from __future__ import annotations

import argparse
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
    _write_csv,
)


DEFAULT_PROXY_SCORES = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "score_swe_smith_multirollout_eval150_proxy_verifier_soft_bce_1781735614/"
    "scores/eval_verifier_scores.jsonl"
)
DEFAULT_REPORT_ROOT = (
    "router_analysis/uploaded_eval_full_20260617/"
    "swe_smith_multirollout_eval150_1781382734/logs/run_evaluation"
)

PORTFOLIOS: dict[str, list[tuple[int, int]]] = {
    "20b_x3": [(1, 0), (1, 1), (1, 2)],
    "20b_120b_x3": [(1, 0), (1, 1), (1, 2), (3, 0), (3, 1), (3, 2)],
    "all_4route_x3": [(route_idx, rollout_idx) for route_idx in (0, 1, 2, 3) for rollout_idx in (0, 1, 2)],
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _brier(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(probs) == 0:
        return math.nan
    return float(np.mean((probs.astype(float) - labels.astype(float)) ** 2))


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(probs) == 0:
        return math.nan
    probs = np.clip(probs.astype(float), 1.0e-6, 1.0 - 1.0e-6)
    labels = labels.astype(float)
    return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))


def _complete_ids(
    lookup: dict[tuple[str, int, int], dict[str, Any]],
    portfolio: list[tuple[int, int]],
) -> list[str]:
    all_ids = sorted({pid for pid, _, _ in lookup})
    return [
        pid
        for pid in all_ids
        if all((pid, int(route_idx), int(rollout_idx)) in lookup for route_idx, rollout_idx in portfolio)
    ]


def _rows_for_ids(
    lookup: dict[tuple[str, int, int], dict[str, Any]],
    ids: list[str],
    portfolio: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid in ids:
        for route_idx, rollout_idx in portfolio:
            rows.append(lookup[(pid, int(route_idx), int(rollout_idx))])
    return rows


def _portfolio_metrics(rows: list[dict[str, Any]], score_key: str = "pred_score") -> dict[str, float]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id[str(row["original_problem_id"])].append(row)
    selected_success: list[float] = []
    oracle_success: list[float] = []
    selected_cost: list[float] = []
    portfolio_cost: list[float] = []
    for pid, candidates in by_id.items():
        _ = pid
        selected = max(candidates, key=lambda row: float(row[score_key]))
        selected_success.append(float(selected["real_success"]))
        oracle_success.append(float(max(float(row["real_success"]) for row in candidates)))
        selected_cost.append(float(selected["weighted_output_cost"]))
        portfolio_cost.append(float(sum(float(row["weighted_output_cost"]) for row in candidates)))
    return {
        "best_of_n_pass_rate": float(np.mean(selected_success)) if selected_success else math.nan,
        "oracle_best_of_n_pass_rate": float(np.mean(oracle_success)) if oracle_success else math.nan,
        "mean_selected_attempt_cost": float(np.mean(selected_cost)) if selected_cost else math.nan,
        "mean_portfolio_cost": float(np.mean(portfolio_cost)) if portfolio_cost else math.nan,
    }


def _feature_matrix(rows: list[dict[str, Any]], mode: str, route_values: list[int]) -> np.ndarray:
    scores = np.asarray([float(row["pred_score"]) for row in rows], dtype=float)
    log_output_tokens = np.asarray([math.log1p(float(row.get("output_tokens") or 0.0)) / 10.0 for row in rows], dtype=float)
    if mode == "global":
        return np.column_stack([np.ones(len(rows), dtype=float), scores])
    if mode == "global_len":
        return np.column_stack([np.ones(len(rows), dtype=float), scores, log_output_tokens])
    if mode == "route_intercept":
        columns = [np.ones(len(rows), dtype=float), scores]
        routes = [int(row["route_idx"]) for row in rows]
        for route_idx in route_values[1:]:
            columns.append(np.asarray([1.0 if route == route_idx else 0.0 for route in routes], dtype=float))
        return np.column_stack(columns)
    if mode == "route_intercept_len":
        columns = [np.ones(len(rows), dtype=float), scores, log_output_tokens]
        routes = [int(row["route_idx"]) for row in rows]
        for route_idx in route_values[1:]:
            columns.append(np.asarray([1.0 if route == route_idx else 0.0 for route in routes], dtype=float))
        return np.column_stack(columns)
    raise ValueError(f"Unknown calibration mode: {mode}")


class _RawScoreCalibrator:
    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([float(row["pred_score"]) for row in rows], dtype=float)


class _ConstantCalibrator:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return np.full(len(rows), self.value, dtype=float)


class _LogisticCalibrator:
    def __init__(self, weights: np.ndarray, mode: str, route_values: list[int]):
        self.weights = weights.astype(float)
        self.mode = mode
        self.route_values = route_values

    def predict(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return _sigmoid(_feature_matrix(rows, self.mode, self.route_values) @ self.weights)


def _fit_calibrator(
    rows: list[dict[str, Any]],
    mode: str,
    route_values: list[int],
    l2: float,
    lr: float,
    steps: int,
) -> Any:
    if not rows:
        return _RawScoreCalibrator()
    labels = np.asarray([int(row["real_success"]) for row in rows], dtype=float)
    smoothed_rate = float((np.sum(labels) + 0.5) / (len(labels) + 1.0))
    if len(set(labels.astype(int).tolist())) < 2:
        return _ConstantCalibrator(smoothed_rate)
    x = _feature_matrix(rows, mode, route_values)
    weights = np.zeros(x.shape[1], dtype=float)
    weights[0] = math.log(smoothed_rate / max(1.0e-6, 1.0 - smoothed_rate))
    reg_mask = np.ones_like(weights)
    reg_mask[0] = 0.0
    for _ in range(int(steps)):
        probs = _sigmoid(x @ weights)
        grad = (x.T @ (probs - labels)) / float(len(labels))
        grad += float(l2) * reg_mask * weights / float(len(labels))
        weights -= float(lr) * grad
    return _LogisticCalibrator(weights, mode, route_values)


def _score_rows(rows: list[dict[str, Any]], calibrator: Any, output_key: str = "calibrated_score") -> list[dict[str, Any]]:
    probs = calibrator.predict(rows)
    return [{**row, output_key: float(prob)} for row, prob in zip(rows, probs)]


def _rank_top2_info(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], tuple[int, float]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id[str(row["original_problem_id"])].append(row)
    info: dict[tuple[str, int, int], tuple[int, float]] = {}
    for pid, candidates in by_id.items():
        ranked = sorted(candidates, key=lambda row: float(row["pred_score"]), reverse=True)
        if not ranked:
            continue
        top_score = float(ranked[0]["pred_score"])
        second_score = float(ranked[1]["pred_score"]) if len(ranked) > 1 else top_score
        margin = top_score - second_score
        for rank, row in enumerate(ranked):
            key = (pid, int(row["route_idx"]), int(row["rollout_idx"]))
            info[key] = (rank, margin)
    return info


def _select_rows(
    rows: list[dict[str, Any]],
    budget: int,
    strategy: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    budget = min(int(budget), len(rows))
    if budget <= 0:
        return []
    if strategy == "random":
        return rng.sample(rows, budget)
    if strategy == "uncertainty":
        ranked = sorted(rows, key=lambda row: abs(float(row["pred_score"]) - 0.5))
        return ranked[:budget]
    if strategy == "high_score":
        ranked = sorted(rows, key=lambda row: float(row["pred_score"]), reverse=True)
        return ranked[:budget]
    if strategy == "low_score":
        ranked = sorted(rows, key=lambda row: float(row["pred_score"]))
        return ranked[:budget]
    if strategy == "top2_margin":
        info = _rank_top2_info(rows)

        def key(row: dict[str, Any]) -> tuple[float, int, float]:
            rank, margin = info[(str(row["original_problem_id"]), int(row["route_idx"]), int(row["rollout_idx"]))]
            top2_penalty = 0 if rank < 2 else 1
            return (float(margin), top2_penalty, abs(float(row["pred_score"]) - 0.5))

        return sorted(rows, key=key)[:budget]
    if strategy == "route_balanced_random":
        by_route: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_route[int(row["route_idx"])].append(row)
        selected: list[dict[str, Any]] = []
        route_ids = sorted(by_route)
        per_route = max(1, budget // max(1, len(route_ids)))
        for route_idx in route_ids:
            candidates = by_route[route_idx]
            take = min(per_route, len(candidates), budget - len(selected))
            selected.extend(rng.sample(candidates, take))
        remaining = [row for row in rows if row not in selected]
        if len(selected) < budget and remaining:
            selected.extend(rng.sample(remaining, min(budget - len(selected), len(remaining))))
        return selected[:budget]
    raise ValueError(f"Unknown acquisition strategy: {strategy}")


def _evaluate_calibrator(
    eval_rows: list[dict[str, Any]],
    calibrator: Any,
) -> dict[str, float]:
    scored_rows = _score_rows(eval_rows, calibrator, "calibrated_score")
    labels = np.asarray([int(row["real_success"]) for row in scored_rows], dtype=int)
    raw_scores = np.asarray([float(row["pred_score"]) for row in scored_rows], dtype=float)
    calibrated_scores = np.asarray([float(row["calibrated_score"]) for row in scored_rows], dtype=float)
    with_raw_alias = [{**row, "raw_score": float(row["pred_score"])} for row in scored_rows]
    raw_portfolio = _portfolio_metrics(with_raw_alias, "raw_score")
    calibrated_portfolio = _portfolio_metrics(scored_rows, "calibrated_score")
    return {
        "patch_auc_raw": _roc_auc(raw_scores, labels),
        "patch_auc_calibrated": _roc_auc(calibrated_scores, labels),
        "patch_pearson_raw": _safe_corr(raw_scores, labels.astype(float)),
        "patch_pearson_calibrated": _safe_corr(calibrated_scores, labels.astype(float)),
        "brier_raw": _brier(raw_scores, labels),
        "brier_calibrated": _brier(calibrated_scores, labels),
        "nll_raw": _nll(raw_scores, labels),
        "nll_calibrated": _nll(calibrated_scores, labels),
        "best_of_n_pass_rate_raw": raw_portfolio["best_of_n_pass_rate"],
        "best_of_n_pass_rate_calibrated": calibrated_portfolio["best_of_n_pass_rate"],
        "oracle_best_of_n_pass_rate": calibrated_portfolio["oracle_best_of_n_pass_rate"],
        "mean_portfolio_cost": calibrated_portfolio["mean_portfolio_cost"],
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    group_keys = ["portfolio", "calibration", "budget", "strategy"]
    metric_keys = [
        "patch_auc_calibrated",
        "brier_calibrated",
        "nll_calibrated",
        "best_of_n_pass_rate_calibrated",
        "best_of_n_pass_rate_raw",
        "oracle_best_of_n_pass_rate",
        "best_of_n_lift_vs_raw",
        "best_of_n_oracle_gap_recovered",
    ]
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    summary: list[dict[str, Any]] = []
    for key, subset in sorted(groups.items()):
        out = dict(zip(group_keys, key))
        out["n_splits"] = len(subset)
        for metric in metric_keys:
            values = np.asarray([float(row[metric]) for row in subset if not math.isnan(float(row[metric]))], dtype=float)
            out[f"{metric}_mean"] = float(np.mean(values)) if len(values) else math.nan
            out[f"{metric}_std"] = float(np.std(values)) if len(values) else math.nan
        summary.append(out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-jsonl", default=DEFAULT_PROXY_SCORES)
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--route-cost-weights", default=",".join(str(value) for value in DEFAULT_COST_WEIGHTS))
    parser.add_argument("--portfolios", default="20b_x3,20b_120b_x3,all_4route_x3")
    parser.add_argument("--budgets", default="0,16,32,64,128,256")
    parser.add_argument("--strategies", default="random,uncertainty,top2_margin,route_balanced_random,high_score")
    parser.add_argument("--calibrations", default="global,global_len,route_intercept,route_intercept_len")
    parser.add_argument("--n-splits", type=int, default=100)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()

    route_cost_weights = [float(part.strip()) for part in str(args.route_cost_weights).split(",") if part.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attempts = _build_attempt_table(_read_jsonl(Path(args.scores_jsonl)), Path(args.report_root), route_cost_weights)
    lookup = _attempt_lookup(attempts)
    budgets = [int(part.strip()) for part in str(args.budgets).split(",") if part.strip()]
    strategies = [part.strip() for part in str(args.strategies).split(",") if part.strip()]
    calibrations = [part.strip() for part in str(args.calibrations).split(",") if part.strip()]
    portfolio_names = [part.strip() for part in str(args.portfolios).split(",") if part.strip()]

    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "scores_jsonl": str(args.scores_jsonl),
        "report_root": str(args.report_root),
        "route_cost_weights": route_cost_weights,
        "n_attempts": len(attempts),
        "portfolios": {},
        "budgets": budgets,
        "strategies": strategies,
        "calibrations": calibrations,
        "n_splits": int(args.n_splits),
        "train_frac": float(args.train_frac),
        "seed": int(args.seed),
    }

    for portfolio_name in portfolio_names:
        if portfolio_name not in PORTFOLIOS:
            raise ValueError(f"Unknown portfolio {portfolio_name}. Known: {sorted(PORTFOLIOS)}")
        portfolio = PORTFOLIOS[portfolio_name]
        ids = _complete_ids(lookup, portfolio)
        if len(ids) < 4:
            raise ValueError(f"Portfolio {portfolio_name} has too few complete ids: {len(ids)}")
        route_values = sorted({route_idx for route_idx, _ in portfolio})
        manifest["portfolios"][portfolio_name] = {
            "n_complete_ids": len(ids),
            "candidates_per_id": len(portfolio),
            "routes": [ROUTE_LABELS.get(route_idx, str(route_idx)) for route_idx in route_values],
        }
        for split_idx in range(int(args.n_splits)):
            rng = random.Random(int(args.seed) + split_idx * 1009)
            split_ids = list(ids)
            rng.shuffle(split_ids)
            n_train = max(1, min(len(split_ids) - 1, int(round(len(split_ids) * float(args.train_frac)))))
            train_ids = sorted(split_ids[:n_train])
            eval_ids = sorted(split_ids[n_train:])
            train_rows = _rows_for_ids(lookup, train_ids, portfolio)
            eval_rows = _rows_for_ids(lookup, eval_ids, portfolio)
            raw_eval = _evaluate_calibrator(eval_rows, _RawScoreCalibrator())
            raw_pass = float(raw_eval["best_of_n_pass_rate_raw"])
            oracle_pass = float(raw_eval["oracle_best_of_n_pass_rate"])
            oracle_gap = max(1.0e-12, oracle_pass - raw_pass)

            for calibration in calibrations:
                for budget in budgets:
                    if int(budget) == 0:
                        eval_metrics = raw_eval
                        rows.append(
                            {
                                "portfolio": portfolio_name,
                                "split_idx": split_idx,
                                "n_train_ids": len(train_ids),
                                "n_eval_ids": len(eval_ids),
                                "calibration": "raw",
                                "budget": 0,
                                "strategy": "none",
                                "n_selected_labels": 0,
                                "selected_positive_rate": math.nan,
                                "best_of_n_lift_vs_raw": 0.0,
                                "best_of_n_oracle_gap_recovered": 0.0,
                                **eval_metrics,
                            }
                        )
                        continue
                    for strategy in strategies:
                        selected = _select_rows(train_rows, int(budget), strategy, rng)
                        calibrator = _fit_calibrator(
                            selected,
                            calibration,
                            route_values,
                            float(args.l2),
                            float(args.lr),
                            int(args.steps),
                        )
                        eval_metrics = _evaluate_calibrator(eval_rows, calibrator)
                        calibrated_pass = float(eval_metrics["best_of_n_pass_rate_calibrated"])
                        rows.append(
                            {
                                "portfolio": portfolio_name,
                                "split_idx": split_idx,
                                "n_train_ids": len(train_ids),
                                "n_eval_ids": len(eval_ids),
                                "calibration": calibration,
                                "budget": int(budget),
                                "strategy": strategy,
                                "n_selected_labels": len(selected),
                                "selected_positive_rate": (
                                    float(np.mean([int(row["real_success"]) for row in selected])) if selected else math.nan
                                ),
                                "best_of_n_lift_vs_raw": calibrated_pass - raw_pass,
                                "best_of_n_oracle_gap_recovered": (calibrated_pass - raw_pass) / oracle_gap,
                                **eval_metrics,
                            }
                        )

    summary = _summarize(rows)
    _write_csv(output_dir / "active_label_sim_rows.csv", rows)
    _write_csv(output_dir / "active_label_sim_summary.csv", summary)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    best_rows: list[dict[str, Any]] = []
    for portfolio_name in portfolio_names:
        subset = [
            row
            for row in summary
            if str(row["portfolio"]) == portfolio_name and int(row["budget"]) > 0 and str(row["strategy"]) != "none"
        ]
        if subset:
            best_rows.append(
                max(
                    subset,
                    key=lambda row: float(row["best_of_n_pass_rate_calibrated_mean"]),
                )
            )
    result = {
        "manifest": manifest,
        "best_by_portfolio": best_rows,
        "summary_csv": str(output_dir / "active_label_sim_summary.csv"),
        "rows_csv": str(output_dir / "active_label_sim_rows.csv"),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

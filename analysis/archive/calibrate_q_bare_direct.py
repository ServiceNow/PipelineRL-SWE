#!/usr/bin/env python3
"""Calibrate bare-state Q success probabilities and rescore direct routing.

This is intentionally a no-retrain analysis:

* fit per-route Platt scalers on train bare-state predictions only
* apply them to eval bare-state predictions
* choose routes with p_success - lambda * decision_cost
* report realized utility using actual costs for the chosen route
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_Q_RUN = Path(
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_state_policy_allsubsets_5epoch_1780909738/"
    "train_qwen3_embedding_8b_lora_state_policy_5epoch"
)
DEFAULT_COST_RUN = Path(
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_trace_4route_qwen3_embedding_8b_lora_expert_cost_only_normalized_10epoch_1780967656/"
    "train_qwen3_embedding_8b_lora_expert_cost_only_normalized_10epoch"
)
DEFAULT_DATASET_DIR = Path(
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect"
)
DEFAULT_CASCADE_CSV = Path(
    "router_analysis/swe_smith_train1500_real_4route_cascade_aws_rescore_178099/"
    "cascade_weighted_cost_utility.csv"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def _bce(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    p = np.clip(p_pred, 1e-8, 1.0 - 1e-8)
    return float(np.mean(-(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))))


def _fit_platt(raw_probs: np.ndarray, targets: np.ndarray, l2: float = 1e-3) -> tuple[float, float]:
    """Fit sigmoid(intercept + slope * logit(raw_prob)) with Newton steps."""
    x = _logit(raw_probs).astype(np.float64)
    y = targets.astype(np.float64)
    if y.size == 0:
        return 0.0, 1.0
    if float(np.min(y)) == float(np.max(y)):
        mean = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
        return float(math.log(mean / (1.0 - mean))), 0.0

    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    if x_std < 1e-8:
        x_std = 1.0
    xs = (x - x_mean) / x_std
    X = np.stack([np.ones_like(xs), xs], axis=1)

    target_mean = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
    beta = np.asarray([math.log(target_mean / (1.0 - target_mean)), 1.0], dtype=np.float64)
    reg = np.diag([0.0, l2])
    for _ in range(50):
        z = X @ beta
        p = _sigmoid(z)
        w = np.clip(p * (1.0 - p), 1e-6, None)
        grad = X.T @ (p - y) + reg @ beta
        hess = (X.T * w) @ X + reg
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        step_norm = float(np.linalg.norm(step))
        if step_norm > 10.0:
            step *= 10.0 / step_norm
        beta -= step
        if float(np.linalg.norm(step)) < 1e-7:
            break

    # Convert standardized-x coefficients back to raw-logit coefficients.
    slope = float(beta[1] / x_std)
    intercept = float(beta[0] - beta[1] * x_mean / x_std)
    return intercept, slope


def _apply_platt(raw_probs: np.ndarray, params: list[tuple[float, float]]) -> np.ndarray:
    logits = _logit(raw_probs)
    out = np.empty_like(raw_probs, dtype=np.float64)
    for route_idx, (intercept, slope) in enumerate(params):
        out[:, route_idx] = _sigmoid(intercept + slope * logits[:, route_idx])
    return out


def _load_bare(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bare = [row for row in rows if row.get("state_kind") == "bare"]
    return sorted(bare, key=lambda row: int(row["source_idx"]))


def _read_split_parquet(dataset_dir: Path, split: str) -> pd.DataFrame:
    paths = sorted((dataset_dir / split).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {dataset_dir / split}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _matrix_from_column(df: pd.DataFrame, column: str) -> np.ndarray:
    return np.asarray([np.asarray(value, dtype=np.float64) for value in df[column]], dtype=np.float64)


def _align_eval_examples(
    bare_eval: list[dict[str, Any]],
    dataset_dir: Path,
    route_cost_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = _read_split_parquet(dataset_dir, "eval")
    by_problem_id = {str(row.problem_id): row for row in df.itertuples(index=False)}
    rewards: list[np.ndarray] = []
    costs: list[np.ndarray] = []
    problem_ids: list[str] = []
    for row in bare_eval:
        problem_id = str(row["problem_id"])
        if problem_id not in by_problem_id:
            raise KeyError(f"Missing eval parquet row for {problem_id}")
        rec = by_problem_id[problem_id]
        route_successes = np.asarray(rec.route_successes, dtype=np.float64)
        route_output_tokens = np.asarray(rec.route_output_tokens, dtype=np.float64)
        rewards.append(route_successes)
        costs.append(route_output_tokens * route_cost_weights)
        problem_ids.append(problem_id)
    return np.asarray(rewards), np.asarray(costs), problem_ids


def _load_predicted_eval_costs(
    cost_run: Path,
    problem_ids: list[str],
    route_count: int,
    route_cost_weights: np.ndarray,
    scout_cost: float,
) -> np.ndarray:
    path = cost_run / "eval_predictions.jsonl"
    rows = _read_jsonl(path)
    by_problem_id = {str(row["problem_id"]): row for row in rows}
    costs = np.full((len(problem_ids), route_count), float(scout_cost), dtype=np.float64)
    for idx, problem_id in enumerate(problem_ids):
        if problem_id not in by_problem_id:
            raise KeyError(f"Missing predicted cost row for {problem_id}")
        pred_tokens = np.asarray(by_problem_id[problem_id]["pred_output_tokens"], dtype=np.float64)
        if pred_tokens.shape[0] != route_count - 1:
            raise ValueError(f"Expected {route_count - 1} expert costs for {problem_id}, got {pred_tokens.shape[0]}")
        costs[idx, 1:] = pred_tokens * route_cost_weights[1:]
    return costs


def _choice_counts(choices: np.ndarray, route_labels: list[str]) -> dict[str, int]:
    return {route_labels[idx]: int(np.sum(choices == idx)) for idx in range(len(route_labels))}


def _simulate_direct(
    *,
    probs: np.ndarray,
    rewards: np.ndarray,
    realized_costs: np.ndarray,
    decision_costs: np.ndarray,
    route_labels: list[str],
    lambda_value: float,
    policy: str,
    cost_mode: str,
) -> dict[str, Any]:
    scores = probs - float(lambda_value) * decision_costs
    choices = np.argmax(scores, axis=1)
    chosen_reward = rewards[np.arange(rewards.shape[0]), choices]
    chosen_cost = realized_costs[np.arange(realized_costs.shape[0]), choices]
    mean_reward = float(np.mean(chosen_reward))
    mean_cost = float(np.mean(chosen_cost))
    return {
        "policy": policy,
        "cost_mode": cost_mode,
        "lambda": float(lambda_value),
        "mean_reward": mean_reward,
        "mean_cost": mean_cost,
        "mean_utility": mean_reward - float(lambda_value) * mean_cost,
        "choice_counts_by_route": json.dumps(_choice_counts(choices, route_labels), sort_keys=True),
    }


def _best_always(
    rewards: np.ndarray,
    realized_costs: np.ndarray,
    route_labels: list[str],
    lambda_value: float,
) -> dict[str, Any]:
    utilities = np.mean(rewards - float(lambda_value) * realized_costs, axis=0)
    choice = int(np.argmax(utilities))
    return {
        "policy": f"always::{route_labels[choice]}",
        "mean_utility": float(utilities[choice]),
        "mean_reward": float(np.mean(rewards[:, choice])),
        "mean_cost": float(np.mean(realized_costs[:, choice])),
    }


def _oracle_direct(
    rewards: np.ndarray,
    realized_costs: np.ndarray,
    lambda_value: float,
) -> dict[str, float]:
    scores = rewards - float(lambda_value) * realized_costs
    choices = np.argmax(scores, axis=1)
    chosen_reward = rewards[np.arange(rewards.shape[0]), choices]
    chosen_cost = realized_costs[np.arange(realized_costs.shape[0]), choices]
    return {
        "mean_utility": float(np.mean(chosen_reward - float(lambda_value) * chosen_cost)),
        "mean_reward": float(np.mean(chosen_reward)),
        "mean_cost": float(np.mean(chosen_cost)),
    }


def _read_cascade_utilities(path: Path) -> dict[float, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[float, dict[str, Any]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            lam = float(row["lambda"])
            rows[lam] = row
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(out_dir: Path, rows: list[dict[str, Any]], title_suffix: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is best effort.
        print(f"Skipping plots: {exc}")
        return

    for cost_mode in sorted({str(row["cost_mode"]) for row in rows}):
        subset = [row for row in rows if str(row["cost_mode"]) == cost_mode]
        if not subset:
            continue
        lambdas = sorted({float(row["lambda"]) for row in subset})
        series_names = [
            "raw_input_only",
            "platt_input_only",
            "best_always",
            "cascade",
            "oracle_direct",
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in series_names:
            ys = []
            xs = []
            for lam in lambdas:
                matches = [row for row in subset if float(row["lambda"]) == lam and row["series"] == name]
                if not matches:
                    continue
                xs.append(lam)
                ys.append(float(matches[0]["mean_utility"]))
            if xs:
                ax.plot(xs, ys, marker="o", label=name)
        ax.set_xlabel("lambda")
        ax.set_ylabel("mean true utility")
        ax.set_title(f"Bare direct calibration ({cost_mode}){title_suffix}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        safe_mode = cost_mode.replace("/", "_")
        fig.savefig(out_dir / f"bare_direct_calibration_{safe_mode}.png", dpi=180)
        fig.savefig(out_dir / f"bare_direct_calibration_{safe_mode}.pdf")
        ax.set_xlim(20, 60)
        fig.savefig(out_dir / f"bare_direct_calibration_{safe_mode}_zoom_20_60.png", dpi=180)
        fig.savefig(out_dir / f"bare_direct_calibration_{safe_mode}_zoom_20_60.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-run", type=Path, default=DEFAULT_Q_RUN)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--cost-run", type=Path, default=DEFAULT_COST_RUN)
    parser.add_argument("--cascade-csv", type=Path, default=DEFAULT_CASCADE_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0,5,10,15,20,25,30,35,40,45,50,60,75,100",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.q_run / "summary.json").read_text())
    config = summary["config"]
    route_labels = list(config["route_labels"])
    route_count = len(route_labels)
    fixed_train_costs = np.asarray(config["fixed_train_route_costs"], dtype=np.float64)
    route_cost_weights = np.asarray(config["route_output_cost_weights"], dtype=np.float64)
    lambdas = [float(value) for value in args.lambdas.split(",") if value.strip()]

    train_bare = _load_bare(_read_jsonl(args.q_run / "train_state_predictions.jsonl"))
    eval_bare = _load_bare(_read_jsonl(args.q_run / "eval_state_predictions.jsonl"))
    train_probs = np.asarray([row["pred_success_probs"] for row in train_bare], dtype=np.float64)
    train_targets = np.asarray([row["true_rewards"] for row in train_bare], dtype=np.float64)
    eval_probs = np.asarray([row["pred_success_probs"] for row in eval_bare], dtype=np.float64)
    eval_targets_from_q = np.asarray([row["true_rewards"] for row in eval_bare], dtype=np.float64)

    eval_rewards, realized_eval_costs, eval_problem_ids = _align_eval_examples(
        eval_bare, args.dataset_dir, route_cost_weights
    )
    if not np.allclose(eval_rewards, eval_targets_from_q):
        max_abs = float(np.max(np.abs(eval_rewards - eval_targets_from_q)))
        raise ValueError(f"Eval parquet rewards do not match Q dump true_rewards; max_abs_diff={max_abs}")

    params = [_fit_platt(train_probs[:, idx], train_targets[:, idx]) for idx in range(route_count)]
    train_cal = _apply_platt(train_probs, params)
    eval_cal = _apply_platt(eval_probs, params)

    calibration_rows: list[dict[str, Any]] = []
    for idx, label in enumerate(route_labels):
        calibration_rows.append(
            {
                "route_idx": idx,
                "route_label": label,
                "intercept": params[idx][0],
                "slope": params[idx][1],
                "train_mean_true": float(np.mean(train_targets[:, idx])),
                "train_mean_raw": float(np.mean(train_probs[:, idx])),
                "train_mean_calibrated": float(np.mean(train_cal[:, idx])),
                "train_bce_raw": _bce(train_targets[:, idx], train_probs[:, idx]),
                "train_bce_calibrated": _bce(train_targets[:, idx], train_cal[:, idx]),
                "eval_mean_true": float(np.mean(eval_rewards[:, idx])),
                "eval_mean_raw": float(np.mean(eval_probs[:, idx])),
                "eval_mean_calibrated": float(np.mean(eval_cal[:, idx])),
                "eval_bce_raw": _bce(eval_rewards[:, idx], eval_probs[:, idx]),
                "eval_bce_calibrated": _bce(eval_rewards[:, idx], eval_cal[:, idx]),
            }
        )

    cascade = _read_cascade_utilities(args.cascade_csv)
    predicted_eval_costs = _load_predicted_eval_costs(
        args.cost_run,
        eval_problem_ids,
        route_count,
        route_cost_weights,
        scout_cost=float(fixed_train_costs[0]),
    )
    fixed_eval_decision_costs = np.broadcast_to(fixed_train_costs[None, :], realized_eval_costs.shape)
    decision_cost_modes = {
        "fixed_train_mean": fixed_eval_decision_costs,
        "predicted_expert_fixed_scout": predicted_eval_costs,
    }

    policy_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for cost_mode, decision_costs in decision_cost_modes.items():
        for lambda_value in lambdas:
            raw = _simulate_direct(
                probs=eval_probs,
                rewards=eval_rewards,
                realized_costs=realized_eval_costs,
                decision_costs=decision_costs,
                route_labels=route_labels,
                lambda_value=lambda_value,
                policy="raw_input_only",
                cost_mode=cost_mode,
            )
            cal = _simulate_direct(
                probs=eval_cal,
                rewards=eval_rewards,
                realized_costs=realized_eval_costs,
                decision_costs=decision_costs,
                route_labels=route_labels,
                lambda_value=lambda_value,
                policy="platt_input_only",
                cost_mode=cost_mode,
            )
            always = _best_always(eval_rewards, realized_eval_costs, route_labels, lambda_value)
            oracle = _oracle_direct(eval_rewards, realized_eval_costs, lambda_value)
            cascade_row = cascade.get(float(lambda_value), {})
            cascade_utility = (
                float(cascade_row["mean_true_utility"]) if cascade_row and "mean_true_utility" in cascade_row else math.nan
            )
            series = [
                ("raw_input_only", raw),
                ("platt_input_only", cal),
                ("best_always", always),
                ("oracle_direct", oracle),
            ]
            if not math.isnan(cascade_utility):
                series.append(
                    (
                        "cascade",
                        {
                            "policy": "cascade_train_tau_weighted",
                            "mean_utility": cascade_utility,
                            "mean_reward": float(cascade_row["mean_true_reward"]),
                            "mean_cost": float(cascade_row["mean_true_cost"]),
                        },
                    )
                )
            for name, row in series:
                out = {
                    "series": name,
                    "cost_mode": cost_mode,
                    "lambda": float(lambda_value),
                    **row,
                }
                policy_rows.append(out)
            summary_rows.append(
                {
                    "cost_mode": cost_mode,
                    "lambda": float(lambda_value),
                    "raw_utility": raw["mean_utility"],
                    "calibrated_utility": cal["mean_utility"],
                    "best_always_policy": always["policy"],
                    "best_always_utility": always["mean_utility"],
                    "cascade_utility": cascade_utility,
                    "oracle_direct_utility": oracle["mean_utility"],
                    "calibrated_minus_raw": cal["mean_utility"] - raw["mean_utility"],
                    "calibrated_minus_always": cal["mean_utility"] - always["mean_utility"],
                    "calibrated_minus_cascade": cal["mean_utility"] - cascade_utility
                    if not math.isnan(cascade_utility)
                    else math.nan,
                    "calibrated_reward": cal["mean_reward"],
                    "calibrated_cost": cal["mean_cost"],
                    "calibrated_choice_counts_by_route": cal["choice_counts_by_route"],
                    "raw_reward": raw["mean_reward"],
                    "raw_cost": raw["mean_cost"],
                    "raw_choice_counts_by_route": raw["choice_counts_by_route"],
                }
            )

    _write_csv(args.output_dir / "calibration_params.csv", calibration_rows)
    _write_csv(args.output_dir / "bare_direct_calibration_policy_rows.csv", policy_rows)
    _write_csv(args.output_dir / "bare_direct_calibration_summary.csv", summary_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "q_run": str(args.q_run),
                "dataset_dir": str(args.dataset_dir),
                "cost_run": str(args.cost_run),
                "cascade_csv": str(args.cascade_csv),
                "n_train_bare": len(train_bare),
                "n_eval_bare": len(eval_bare),
                "route_labels": route_labels,
                "fixed_train_costs": fixed_train_costs.tolist(),
                "route_output_cost_weights": route_cost_weights.tolist(),
                "lambdas": lambdas,
            },
            indent=2,
            sort_keys=True,
        )
    )
    _plot_summary(args.output_dir, policy_rows, title_suffix="")

    print(f"Wrote {args.output_dir}")
    print("Best calibrated improvements over raw:")
    for cost_mode in sorted(decision_cost_modes):
        subset = [row for row in summary_rows if row["cost_mode"] == cost_mode]
        best = max(subset, key=lambda row: float(row["calibrated_minus_raw"]))
        print(
            f"  {cost_mode}: lambda={best['lambda']:.1f} "
            f"cal-raw={best['calibrated_minus_raw']:+.4f} "
            f"cal-always={best['calibrated_minus_always']:+.4f} "
            f"cal-cascade={best['calibrated_minus_cascade']:+.4f}"
        )


if __name__ == "__main__":
    main()

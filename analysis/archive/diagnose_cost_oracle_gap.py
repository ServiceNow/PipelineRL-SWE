#!/usr/bin/env python3
"""Diagnose how much Q routing loses to imperfect per-instance cost estimates.

This is an analysis-only script. Oracle cost modes use eval/train realized route
costs for action selection, so they are upper bounds, not deployable policies.
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_bare_state_rows(path: Path) -> list[dict[str, Any]]:
    return sorted(
        [row for row in read_jsonl(path) if row.get("state_kind") == "bare"],
        key=lambda row: int(row["source_idx"]),
    )


def read_split(dataset_dir: Path, split: str) -> pd.DataFrame:
    paths = sorted((dataset_dir / split).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {dataset_dir / split}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def align_split(
    *,
    pred_rows: list[dict[str, Any]],
    dataset_dir: Path,
    split: str,
    route_cost_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = read_split(dataset_dir, split)
    by_problem = {str(row.problem_id): row for row in df.itertuples(index=False)}
    rewards = []
    costs = []
    problem_ids = []
    for row in pred_rows:
        problem_id = str(row["problem_id"])
        rec = by_problem[problem_id]
        rewards.append(np.asarray(rec.route_successes, dtype=np.float64))
        output_tokens = np.asarray(rec.route_output_tokens, dtype=np.float64)
        costs.append(output_tokens * route_cost_weights)
        problem_ids.append(problem_id)
    return np.asarray(rewards), np.asarray(costs), problem_ids


def load_predicted_costs(
    *,
    cost_run: Path,
    split: str,
    problem_ids: list[str],
    route_count: int,
    route_cost_weights: np.ndarray,
    scout_cost: float,
) -> np.ndarray | None:
    path = cost_run / f"{split}_predictions.jsonl"
    if not path.exists():
        return None
    by_problem = {str(row["problem_id"]): row for row in read_jsonl(path)}
    if not all(problem_id in by_problem for problem_id in problem_ids):
        missing = [problem_id for problem_id in problem_ids if problem_id not in by_problem]
        raise ValueError(f"{path} is missing {len(missing)} problem ids; first={missing[:3]}")
    costs = np.full((len(problem_ids), route_count), float(scout_cost), dtype=np.float64)
    for idx, problem_id in enumerate(problem_ids):
        pred_tokens = np.asarray(by_problem[problem_id]["pred_output_tokens"], dtype=np.float64)
        if len(pred_tokens) == route_count:
            costs[idx, :] = pred_tokens * route_cost_weights
        elif len(pred_tokens) == route_count - 1:
            costs[idx, 1:] = pred_tokens * route_cost_weights[1:]
        else:
            raise ValueError(
                f"Unexpected pred_output_tokens length for {problem_id}: "
                f"{len(pred_tokens)} vs route_count={route_count}"
            )
    return costs


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return math.nan
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def choice_counts(choices: np.ndarray, route_labels: list[str]) -> str:
    counts = {label: int(np.sum(choices == idx)) for idx, label in enumerate(route_labels)}
    return json.dumps(counts, sort_keys=True)


def summarize_policy(
    *,
    split: str,
    policy: str,
    cost_mode: str,
    lambda_value: float,
    choices: np.ndarray,
    rewards: np.ndarray,
    realized_costs: np.ndarray,
    route_labels: list[str],
    oracle_cost_choices: np.ndarray | None,
) -> dict[str, Any]:
    idx = np.arange(len(choices))
    selected_rewards = rewards[idx, choices]
    selected_costs = realized_costs[idx, choices]
    out = {
        "split": split,
        "policy": policy,
        "cost_mode": cost_mode,
        "lambda": float(lambda_value),
        "mean_reward": float(np.mean(selected_rewards)),
        "mean_cost": float(np.mean(selected_costs)),
        "mean_utility": float(np.mean(selected_rewards) - float(lambda_value) * np.mean(selected_costs)),
        "choice_counts_by_route": choice_counts(choices, route_labels),
    }
    if oracle_cost_choices is not None:
        out["choice_agreement_with_oracle_cost_q"] = float(np.mean(choices == oracle_cost_choices))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def cost_diagnostics(
    *,
    split: str,
    predicted_costs: np.ndarray,
    realized_costs: np.ndarray,
    route_labels: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for route_idx, route_label in enumerate(route_labels):
        y_true = realized_costs[:, route_idx]
        y_pred = predicted_costs[:, route_idx]
        rows.append(
            {
                "split": split,
                "route_idx": route_idx,
                "route_label": route_label,
                "mean_true_cost": float(np.mean(y_true)),
                "mean_pred_cost": float(np.mean(y_pred)),
                "std_true_cost": float(np.std(y_true)),
                "std_pred_cost": float(np.std(y_pred)),
                "std_ratio_pred_over_true": float(np.std(y_pred) / np.std(y_true))
                if float(np.std(y_true)) > 0
                else math.nan,
                "mae_cost": float(np.mean(np.abs(y_pred - y_true))),
                "rmse_cost": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
                "pearson_cost": safe_corr(y_pred, y_true),
                "pearson_log1p_cost": safe_corr(np.log1p(y_pred), np.log1p(y_true)),
            }
        )
    return rows


def analyze_split(
    *,
    split: str,
    q_run: Path,
    cost_run: Path,
    dataset_dir: Path,
    route_labels: list[str],
    fixed_costs: np.ndarray,
    route_cost_weights: np.ndarray,
    lambdas: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pred_path = q_run / f"{split}_state_predictions.jsonl"
    if not pred_path.exists():
        return [], [], []
    pred_rows = read_bare_state_rows(pred_path)
    q_probs = np.asarray([row["pred_success_probs"] for row in pred_rows], dtype=np.float64)
    rewards, realized_costs, problem_ids = align_split(
        pred_rows=pred_rows,
        dataset_dir=dataset_dir,
        split=split,
        route_cost_weights=route_cost_weights,
    )
    if not np.allclose(rewards, np.asarray([row["true_rewards"] for row in pred_rows], dtype=np.float64)):
        raise ValueError(f"{split}: Q dump true_rewards do not match dataset rewards")

    route_count = len(route_labels)
    fixed_decision_costs = np.broadcast_to(fixed_costs[None, :], realized_costs.shape)
    predicted_costs = load_predicted_costs(
        cost_run=cost_run,
        split=split,
        problem_ids=problem_ids,
        route_count=route_count,
        route_cost_weights=route_cost_weights,
        scout_cost=float(fixed_costs[0]),
    )

    decision_cost_modes: dict[str, np.ndarray] = {
        "fixed_train_mean": fixed_decision_costs,
        "oracle_actual_cost": realized_costs,
    }
    if predicted_costs is not None:
        decision_cost_modes["predicted_expert_fixed_scout"] = predicted_costs

    policy_rows = []
    summary_rows = []
    for lambda_value in lambdas:
        oracle_cost_choices = np.argmax(q_probs - float(lambda_value) * realized_costs, axis=1)
        true_oracle_choices = np.argmax(rewards - float(lambda_value) * realized_costs, axis=1)
        always_utils = np.mean(rewards - float(lambda_value) * realized_costs, axis=0)
        best_always_idx = int(np.argmax(always_utils))
        best_always_choices = np.full(len(pred_rows), best_always_idx, dtype=np.int64)

        per_mode = {}
        for cost_mode, decision_costs in decision_cost_modes.items():
            choices = np.argmax(q_probs - float(lambda_value) * decision_costs, axis=1)
            row = summarize_policy(
                split=split,
                policy="q_bare_direct",
                cost_mode=cost_mode,
                lambda_value=lambda_value,
                choices=choices,
                rewards=rewards,
                realized_costs=realized_costs,
                route_labels=route_labels,
                oracle_cost_choices=oracle_cost_choices,
            )
            policy_rows.append(row)
            per_mode[cost_mode] = row

        best_always = summarize_policy(
            split=split,
            policy="best_always",
            cost_mode="actual_realized",
            lambda_value=lambda_value,
            choices=best_always_choices,
            rewards=rewards,
            realized_costs=realized_costs,
            route_labels=route_labels,
            oracle_cost_choices=None,
        )
        true_oracle = summarize_policy(
            split=split,
            policy="oracle_true_success_and_cost",
            cost_mode="actual_realized",
            lambda_value=lambda_value,
            choices=true_oracle_choices,
            rewards=rewards,
            realized_costs=realized_costs,
            route_labels=route_labels,
            oracle_cost_choices=None,
        )
        policy_rows.extend([best_always, true_oracle])

        oracle_q = per_mode["oracle_actual_cost"]
        summary = {
            "split": split,
            "lambda": float(lambda_value),
            "best_always_policy": route_labels[best_always_idx],
            "best_always_utility": best_always["mean_utility"],
            "fixed_q_utility": per_mode["fixed_train_mean"]["mean_utility"],
            "oracle_cost_q_utility": oracle_q["mean_utility"],
            "oracle_true_success_and_cost_utility": true_oracle["mean_utility"],
            "oracle_cost_minus_fixed": oracle_q["mean_utility"] - per_mode["fixed_train_mean"]["mean_utility"],
            "oracle_cost_q_minus_always": oracle_q["mean_utility"] - best_always["mean_utility"],
            "fixed_q_agree_oracle_cost": per_mode["fixed_train_mean"]["choice_agreement_with_oracle_cost_q"],
        }
        if "predicted_expert_fixed_scout" in per_mode:
            pred = per_mode["predicted_expert_fixed_scout"]
            summary.update(
                {
                    "predicted_q_utility": pred["mean_utility"],
                    "oracle_cost_minus_predicted": oracle_q["mean_utility"] - pred["mean_utility"],
                    "predicted_q_minus_fixed": pred["mean_utility"] - per_mode["fixed_train_mean"]["mean_utility"],
                    "predicted_q_agree_oracle_cost": pred["choice_agreement_with_oracle_cost_q"],
                }
            )
        else:
            summary.update(
                {
                    "predicted_q_utility": math.nan,
                    "oracle_cost_minus_predicted": math.nan,
                    "predicted_q_minus_fixed": math.nan,
                    "predicted_q_agree_oracle_cost": math.nan,
                }
            )
        summary_rows.append(summary)

    diag_rows = []
    if predicted_costs is not None:
        diag_rows.extend(
            cost_diagnostics(
                split=split,
                predicted_costs=predicted_costs,
                realized_costs=realized_costs,
                route_labels=route_labels,
            )
        )
    return policy_rows, summary_rows, diag_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-run", type=Path, default=DEFAULT_Q_RUN)
    parser.add_argument("--cost-run", type=Path, default=DEFAULT_COST_RUN)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lambdas",
        default="0,5,10,15,20,25,30,35,40,45,50,55,60,75,100,150,200",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    q_summary = json.loads((args.q_run / "summary.json").read_text())
    config = q_summary["config"]
    route_labels = list(config["route_labels"])
    fixed_costs = np.asarray(config["fixed_train_route_costs"], dtype=np.float64)
    route_cost_weights = np.asarray(config["route_output_cost_weights"], dtype=np.float64)
    lambdas = [float(value) for value in args.lambdas.split(",") if value.strip()]

    all_policy_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    all_cost_diag_rows: list[dict[str, Any]] = []
    for split in ["train", "eval"]:
        policy_rows, summary_rows, diag_rows = analyze_split(
            split=split,
            q_run=args.q_run,
            cost_run=args.cost_run,
            dataset_dir=args.dataset_dir,
            route_labels=route_labels,
            fixed_costs=fixed_costs,
            route_cost_weights=route_cost_weights,
            lambdas=lambdas,
        )
        all_policy_rows.extend(policy_rows)
        all_summary_rows.extend(summary_rows)
        all_cost_diag_rows.extend(diag_rows)

    write_csv(args.output_dir / "cost_oracle_gap_policy_rows.csv", all_policy_rows)
    write_csv(args.output_dir / "cost_oracle_gap_summary.csv", all_summary_rows)
    if all_cost_diag_rows:
        write_csv(args.output_dir / "cost_prediction_diagnostics.csv", all_cost_diag_rows)

    missing_train_cost_dump = not (args.cost_run / "train_predictions.jsonl").exists()
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "q_run": str(args.q_run),
                "cost_run": str(args.cost_run),
                "dataset_dir": str(args.dataset_dir),
                "route_labels": route_labels,
                "fixed_train_costs": fixed_costs.tolist(),
                "route_output_cost_weights": route_cost_weights.tolist(),
                "lambdas": lambdas,
                "missing_train_cost_prediction_dump": missing_train_cost_dump,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Wrote {args.output_dir}")
    if missing_train_cost_dump:
        print("Note: cost run has no train_predictions.jsonl, so predicted-cost train diagnostics are unavailable.")
    for row in all_summary_rows:
        if row["split"] == "eval" and row["lambda"] in {25.0, 35.0, 50.0, 75.0}:
            print(row)


if __name__ == "__main__":
    main()

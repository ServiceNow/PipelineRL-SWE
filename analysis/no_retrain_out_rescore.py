#!/usr/bin/env python3
"""No-retrain OUT/abstention rescoring for bare-state Q predictions."""

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bare_rows(path: Path) -> list[dict[str, Any]]:
    return sorted(
        [row for row in read_jsonl(path) if row.get("state_kind") == "bare"],
        key=lambda row: int(row["source_idx"]),
    )


def read_split(dataset_dir: Path, split: str) -> pd.DataFrame:
    paths = sorted((dataset_dir / split).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {dataset_dir / split}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def align_eval(
    rows: list[dict[str, Any]],
    dataset_dir: Path,
    route_cost_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = read_split(dataset_dir, "eval")
    by_problem = {str(row.problem_id): row for row in df.itertuples(index=False)}
    rewards = []
    costs = []
    problem_ids = []
    for row in rows:
        problem_id = str(row["problem_id"])
        rec = by_problem[problem_id]
        route_successes = np.asarray(rec.route_successes, dtype=np.float64)
        route_output_tokens = np.asarray(rec.route_output_tokens, dtype=np.float64)
        rewards.append(route_successes)
        costs.append(route_output_tokens * route_cost_weights)
        problem_ids.append(problem_id)
    return np.asarray(rewards), np.asarray(costs), problem_ids


def load_predicted_eval_costs(
    cost_run: Path,
    problem_ids: list[str],
    route_count: int,
    route_cost_weights: np.ndarray,
    scout_cost: float,
) -> np.ndarray:
    rows = read_jsonl(cost_run / "eval_predictions.jsonl")
    by_problem = {str(row["problem_id"]): row for row in rows}
    out = np.full((len(problem_ids), route_count), float(scout_cost), dtype=np.float64)
    for idx, problem_id in enumerate(problem_ids):
        pred_tokens = np.asarray(by_problem[problem_id]["pred_output_tokens"], dtype=np.float64)
        out[idx, 1:] = pred_tokens * route_cost_weights[1:]
    return out


def choice_counts(choices: np.ndarray, route_labels: list[str]) -> dict[str, int]:
    counts = {label: int(np.sum(choices == idx)) for idx, label in enumerate(route_labels)}
    counts["OUT"] = int(np.sum(choices < 0))
    return counts


def summarize(
    *,
    policy: str,
    cost_mode: str,
    lambda_value: float,
    choices: np.ndarray,
    rewards: np.ndarray,
    realized_costs: np.ndarray,
    route_labels: list[str],
) -> dict[str, Any]:
    selected_reward = np.where(choices >= 0, rewards[np.arange(len(choices)), np.maximum(choices, 0)], 0.0)
    selected_cost = np.where(choices >= 0, realized_costs[np.arange(len(choices)), np.maximum(choices, 0)], 0.0)
    mean_reward = float(np.mean(selected_reward))
    mean_cost = float(np.mean(selected_cost))
    all_fail = np.sum(rewards, axis=1) == 0
    abstained = choices < 0
    return {
        "policy": policy,
        "cost_mode": cost_mode,
        "lambda": float(lambda_value),
        "mean_reward": mean_reward,
        "mean_cost": mean_cost,
        "mean_utility": mean_reward - float(lambda_value) * mean_cost,
        "out_rate": float(np.mean(abstained)),
        "all_fail_out_recall": float(np.mean(abstained[all_fail])) if bool(np.any(all_fail)) else math.nan,
        "out_precision_all_fail": float(np.mean(all_fail[abstained])) if bool(np.any(abstained)) else math.nan,
        "choice_counts_by_route": json.dumps(choice_counts(choices, route_labels), sort_keys=True),
    }


def tune_all_fail_threshold(train_probs: np.ndarray, train_rewards: np.ndarray) -> tuple[float, dict[str, float]]:
    max_prob = np.max(train_probs, axis=1)
    all_fail = np.sum(train_rewards, axis=1) == 0
    candidates = sorted(set([0.0, 1.0] + [float(value) for value in max_prob]))
    best_tau = 0.0
    best = {"f1": -1.0, "precision": 0.0, "recall": 0.0, "out_rate": 0.0}
    for tau in candidates:
        pred_out = max_prob <= float(tau)
        tp = float(np.sum(pred_out & all_fail))
        fp = float(np.sum(pred_out & ~all_fail))
        fn = float(np.sum(~pred_out & all_fail))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        if f1 > best["f1"]:
            best_tau = float(tau)
            best = {
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "out_rate": float(np.mean(pred_out)),
            }
    return best_tau, best


def read_cascade(path: Path) -> dict[float, float]:
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[float(row["lambda"])] = float(row["mean_true_utility"])
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots: {exc}")
        return
    for cost_mode in sorted({row["cost_mode"] for row in rows}):
        subset = [row for row in rows if row["cost_mode"] == cost_mode]
        lambdas = sorted({float(row["lambda"]) for row in subset})
        fig, ax = plt.subplots(figsize=(8, 5))
        for policy in ["raw_no_out", "utility_out", "all_fail_tau_out", "oracle_all_fail_out", "oracle_utility_out", "best_always", "cascade"]:
            xs = []
            ys = []
            for lam in lambdas:
                match = [row for row in subset if row["policy"] == policy and float(row["lambda"]) == lam]
                if match:
                    xs.append(lam)
                    ys.append(float(match[0]["mean_utility"]))
            if xs:
                ax.plot(xs, ys, marker="o", label=policy)
        ax.set_xlabel("lambda")
        ax.set_ylabel("mean true utility")
        ax.set_title(f"No-retrain OUT rescore ({cost_mode})")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        safe_mode = cost_mode.replace("/", "_")
        fig.savefig(out_dir / f"no_retrain_out_{safe_mode}.png", dpi=180)
        fig.savefig(out_dir / f"no_retrain_out_{safe_mode}.pdf")
        ax.set_xlim(20, 80)
        fig.savefig(out_dir / f"no_retrain_out_{safe_mode}_zoom_20_80.png", dpi=180)
        fig.savefig(out_dir / f"no_retrain_out_{safe_mode}_zoom_20_80.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-run", type=Path, default=DEFAULT_Q_RUN)
    parser.add_argument("--cost-run", type=Path, default=DEFAULT_COST_RUN)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--cascade-csv", type=Path, default=DEFAULT_CASCADE_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambdas", default="0,5,10,15,20,25,30,35,40,45,50,55,60,75,100")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.q_run / "summary.json").read_text())
    config = summary["config"]
    route_labels = list(config["route_labels"])
    route_count = len(route_labels)
    fixed_costs = np.asarray(config["fixed_train_route_costs"], dtype=np.float64)
    route_cost_weights = np.asarray(config["route_output_cost_weights"], dtype=np.float64)
    lambdas = [float(value) for value in args.lambdas.split(",") if value.strip()]

    train = bare_rows(args.q_run / "train_state_predictions.jsonl")
    eval_rows = bare_rows(args.q_run / "eval_state_predictions.jsonl")
    train_probs = np.asarray([row["pred_success_probs"] for row in train], dtype=np.float64)
    train_rewards = np.asarray([row["true_rewards"] for row in train], dtype=np.float64)
    eval_probs = np.asarray([row["pred_success_probs"] for row in eval_rows], dtype=np.float64)
    eval_rewards, realized_costs, eval_problem_ids = align_eval(eval_rows, args.dataset_dir, route_cost_weights)

    tau, tau_stats = tune_all_fail_threshold(train_probs, train_rewards)
    eval_all_fail = np.sum(eval_rewards, axis=1) == 0
    eval_pred_all_fail = np.max(eval_probs, axis=1) <= tau
    eval_tau_stats = {
        "precision": float(np.mean(eval_all_fail[eval_pred_all_fail])) if bool(np.any(eval_pred_all_fail)) else math.nan,
        "recall": float(np.mean(eval_pred_all_fail[eval_all_fail])) if bool(np.any(eval_all_fail)) else math.nan,
        "out_rate": float(np.mean(eval_pred_all_fail)),
        "all_fail_rate": float(np.mean(eval_all_fail)),
    }

    predicted_costs = load_predicted_eval_costs(
        args.cost_run,
        eval_problem_ids,
        route_count,
        route_cost_weights,
        scout_cost=float(fixed_costs[0]),
    )
    fixed_decision_costs = np.broadcast_to(fixed_costs[None, :], realized_costs.shape)
    decision_cost_modes = {
        "fixed_train_mean": fixed_decision_costs,
        "predicted_expert_fixed_scout": predicted_costs,
    }
    cascade = read_cascade(args.cascade_csv)

    rows = []
    for cost_mode, decision_costs in decision_cost_modes.items():
        for lam in lambdas:
            scores = eval_probs - float(lam) * decision_costs
            raw_choices = np.argmax(scores, axis=1)
            utility_out_choices = np.where(np.max(scores, axis=1) <= 0.0, -1, raw_choices)
            all_fail_tau_choices = np.where(eval_pred_all_fail, -1, raw_choices)
            oracle_all_fail_choices = np.where(eval_all_fail, -1, raw_choices)
            true_scores = eval_rewards - float(lam) * realized_costs
            oracle_utility_choices = np.argmax(true_scores, axis=1)
            oracle_utility_choices = np.where(np.max(true_scores, axis=1) <= 0.0, -1, oracle_utility_choices)
            always_utils = np.mean(eval_rewards - float(lam) * realized_costs, axis=0)
            best_always_idx = int(np.argmax(always_utils))
            best_always_choices = np.full(eval_rewards.shape[0], best_always_idx, dtype=np.int64)

            for policy, choices in [
                ("raw_no_out", raw_choices),
                ("utility_out", utility_out_choices),
                ("all_fail_tau_out", all_fail_tau_choices),
                ("oracle_all_fail_out", oracle_all_fail_choices),
                ("oracle_utility_out", oracle_utility_choices),
                ("best_always", best_always_choices),
            ]:
                rows.append(
                    summarize(
                        policy=policy,
                        cost_mode=cost_mode,
                        lambda_value=lam,
                        choices=choices,
                        rewards=eval_rewards,
                        realized_costs=realized_costs,
                        route_labels=route_labels,
                    )
                )
            if lam in cascade:
                rows.append(
                    {
                        "policy": "cascade",
                        "cost_mode": cost_mode,
                        "lambda": float(lam),
                        "mean_reward": math.nan,
                        "mean_cost": math.nan,
                        "mean_utility": float(cascade[lam]),
                        "out_rate": math.nan,
                        "all_fail_out_recall": math.nan,
                        "out_precision_all_fail": math.nan,
                        "choice_counts_by_route": "{}",
                    }
                )

    write_csv(args.output_dir / "no_retrain_out_rows.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "q_run": str(args.q_run),
                "cost_run": str(args.cost_run),
                "dataset_dir": str(args.dataset_dir),
                "route_labels": route_labels,
                "fixed_train_costs": fixed_costs.tolist(),
                "route_output_cost_weights": route_cost_weights.tolist(),
                "train_all_fail_threshold": tau,
                "train_all_fail_threshold_stats": tau_stats,
                "eval_all_fail_threshold_stats": eval_tau_stats,
                "n_train": len(train),
                "n_eval": len(eval_rows),
                "lambdas": lambdas,
            },
            indent=2,
            sort_keys=True,
        )
    )
    plot(args.output_dir, rows)

    print(f"Wrote {args.output_dir}")
    print(json.dumps({
        "train_tau": tau,
        "train_tau_stats": tau_stats,
        "eval_tau_stats": eval_tau_stats,
    }, indent=2))


if __name__ == "__main__":
    main()

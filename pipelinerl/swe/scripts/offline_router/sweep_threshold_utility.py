#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from pipelinerl.swe.scripts.offline_router.common import problem_id_from_item, write_json

DEFAULT_LAMBDAS = [0.0, 1.0e-5, 2.0e-5, 5.0e-5, 1.0e-4, 2.0e-4]


def _load_split(dataset_dir: Path, split_name: str):
    files = sorted((dataset_dir / split_name).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards found for split={split_name} in {dataset_dir / split_name}")
    return load_dataset("parquet", data_files={split_name: [str(path) for path in files]})[split_name]


def _prediction_problem_key(dataset: Any, problem_id: Any) -> str:
    return f"{dataset}::{problem_id}"


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _argmax_index(values: list[float]) -> int:
    if not values:
        raise ValueError("Cannot choose argmax for an empty list")
    best_idx = 0
    best_value = float(values[0])
    for idx, value in enumerate(values[1:], start=1):
        numeric = float(value)
        if numeric > best_value:
            best_idx = idx
            best_value = numeric
    return int(best_idx)


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _build_thresholds(deltas: list[float]) -> list[float]:
    values = sorted(set([-1.0, 1.0, *[float(delta) for delta in deltas]]))
    thresholds = list(values)
    for left, right in zip(values, values[1:]):
        thresholds.append((left + right) / 2.0)
    return sorted(set(thresholds))


def _summarize_policy(examples: list[dict[str, Any]], choice_fn, cost_key: str) -> dict[str, Any]:
    if cost_key not in {"output_tokens", "total_tokens"}:
        raise ValueError(f"Unsupported cost_key={cost_key}")
    route_labels = list(examples[0]["route_labels"]) if examples else []
    route_choice_counts = [0] * len(route_labels)
    reward_sum = 0.0
    cost_sum = 0.0
    for example in examples:
        choice = int(choice_fn(example))
        route_choice_counts[choice] += 1
        reward_sum += float(example["rewards"][choice])
        cost_sum += float(example[cost_key][choice])
    count = len(examples)
    return {
        "n_examples": int(count),
        "mean_reward": float(reward_sum / count) if count else float("nan"),
        "mean_cost": float(cost_sum / count) if count else float("nan"),
        "choice_counts_by_route": {
            str(route_label): int(route_choice_counts[idx]) for idx, route_label in enumerate(route_labels)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep 2-route decision thresholds against realized utility")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Collected offline-router dataset dir")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to eval_predictions.jsonl")
    parser.add_argument("--output-json", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--output-csv", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--lambdas", type=float, nargs="*", default=DEFAULT_LAMBDAS, help="Utility lambdas")
    args = parser.parse_args()

    predictions = _load_predictions(args.predictions)
    metadata = json.loads((args.dataset_dir / "metadata.json").read_text())
    route_labels = list(metadata.get("route_labels") or [])
    if len(route_labels) != 2:
        raise ValueError(
            f"Threshold sweep currently requires exactly 2 routes; found {len(route_labels)} in {args.dataset_dir / 'metadata.json'}"
        )
    eval_dataset = _load_split(args.dataset_dir, "eval")
    eval_lookup: dict[str, dict[str, Any]] = {}
    for row in eval_dataset:
        key = _prediction_problem_key(row.get("dataset"), problem_id_from_item(row))
        if key not in eval_lookup:
            eval_lookup[key] = row

    examples: list[dict[str, Any]] = []
    skipped_missing_eval = 0
    skipped_invalid_rows = 0
    for row in predictions:
        pred_rewards = row.get("pred_rewards")
        if not isinstance(pred_rewards, list) or len(pred_rewards) != 2:
            skipped_invalid_rows += 1
            continue
        key = _prediction_problem_key(row.get("dataset"), row.get("problem_id"))
        source_row = eval_lookup.get(key)
        if source_row is None:
            skipped_missing_eval += 1
            continue
        rewards = source_row.get("performance_targets")
        prompt_tokens = source_row.get("route_prompt_tokens")
        output_tokens = source_row.get("route_output_tokens")
        if (
            not isinstance(rewards, list)
            or not isinstance(prompt_tokens, list)
            or not isinstance(output_tokens, list)
            or len(rewards) != 2
            or len(prompt_tokens) != 2
            or len(output_tokens) != 2
        ):
            skipped_invalid_rows += 1
            continue
        try:
            pred_rewards = [float(value) for value in pred_rewards]
            rewards = [float(value) for value in rewards]
            prompt_tokens = [float(value) for value in prompt_tokens]
            output_tokens = [float(value) for value in output_tokens]
        except (TypeError, ValueError):
            skipped_invalid_rows += 1
            continue
        examples.append(
            {
                "dataset": row.get("dataset"),
                "problem_id": row.get("problem_id"),
                "route_labels": list(route_labels),
                "pred_rewards": pred_rewards,
                "pred_delta": float(pred_rewards[0] - pred_rewards[1]),
                "rewards": rewards,
                "output_tokens": output_tokens,
                "total_tokens": [prompt_tokens[idx] + output_tokens[idx] for idx in range(2)],
            }
        )

    if not examples:
        raise ValueError("No usable eval examples found for threshold sweep")

    thresholds = _build_thresholds([example["pred_delta"] for example in examples])
    csv_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for cost_metric in ("output_tokens", "total_tokens"):
        always_primary = _summarize_policy(examples, lambda example: 0, cost_metric)
        always_expert = _summarize_policy(examples, lambda example: 1, cost_metric)
        oracle_reward = _summarize_policy(
            examples,
            lambda example: 0 if example["rewards"][0] >= example["rewards"][1] else 1,
            cost_metric,
        )
        default_tau0 = _summarize_policy(
            examples,
            lambda example: 0 if example["pred_delta"] > 0.0 else 1,
            cost_metric,
        )

        for lambda_value in args.lambdas:
            best_summary = None
            best_tau = None
            for tau in thresholds:
                summary = _summarize_policy(
                    examples,
                    lambda example, tau=tau: 0 if example["pred_delta"] > tau else 1,
                    cost_metric,
                )
                mean_utility = float(summary["mean_reward"] - (float(lambda_value) * summary["mean_cost"]))
                if best_summary is None or mean_utility > best_summary["mean_utility"]:
                    best_summary = {
                        **summary,
                        "mean_utility": mean_utility,
                    }
                    best_tau = float(tau)
            assert best_summary is not None
            row = {
                "cost_metric": cost_metric,
                "lambda": float(lambda_value),
                "best_tau": best_tau,
                "best_mean_reward": float(best_summary["mean_reward"]),
                "best_mean_cost": float(best_summary["mean_cost"]),
                "best_mean_utility": float(best_summary["mean_utility"]),
                "best_choice_counts_by_route": dict(best_summary["choice_counts_by_route"]),
                "always_primary_mean_utility": float(
                    always_primary["mean_reward"] - (float(lambda_value) * always_primary["mean_cost"])
                ),
                "always_expert_mean_utility": float(
                    always_expert["mean_reward"] - (float(lambda_value) * always_expert["mean_cost"])
                ),
                "default_tau0_mean_utility": float(
                    default_tau0["mean_reward"] - (float(lambda_value) * default_tau0["mean_cost"])
                ),
                "oracle_reward_mean_utility": float(
                    oracle_reward["mean_reward"] - (float(lambda_value) * oracle_reward["mean_cost"])
                ),
                "always_primary_choice_counts_by_route": dict(always_primary["choice_counts_by_route"]),
                "always_expert_choice_counts_by_route": dict(always_expert["choice_counts_by_route"]),
                "default_tau0_choice_counts_by_route": dict(default_tau0["choice_counts_by_route"]),
                "oracle_reward_choice_counts_by_route": dict(oracle_reward["choice_counts_by_route"]),
            }
            results.append(row)
            csv_rows.append(
                {
                    "cost_metric": cost_metric,
                    "lambda": float(lambda_value),
                    "best_tau": best_tau,
                    "best_mean_reward": float(best_summary["mean_reward"]),
                    "best_mean_cost": float(best_summary["mean_cost"]),
                    "best_mean_utility": float(best_summary["mean_utility"]),
                    "best_chosen_primary": int(best_summary["choice_counts_by_route"][route_labels[0]]),
                    "best_chosen_expert": int(best_summary["choice_counts_by_route"][route_labels[1]]),
                    "always_primary_mean_utility": row["always_primary_mean_utility"],
                    "always_expert_mean_utility": row["always_expert_mean_utility"],
                    "default_tau0_mean_utility": row["default_tau0_mean_utility"],
                    "oracle_reward_mean_utility": row["oracle_reward_mean_utility"],
                }
            )

    output_json = args.output_json or args.predictions.with_name("threshold_sweep_utility.json")
    output_csv = args.output_csv or args.predictions.with_name("threshold_sweep_utility.csv")
    report = {
        "dataset_dir": str(args.dataset_dir),
        "predictions": str(args.predictions),
        "route_labels": route_labels,
        "n_predictions": int(len(predictions)),
        "n_examples_with_utility": int(len(examples)),
        "skipped_missing_eval": int(skipped_missing_eval),
        "skipped_invalid_rows": int(skipped_invalid_rows),
        "lambda_values": [float(value) for value in args.lambdas],
        "delta_summary": {
            "min": float(min(example["pred_delta"] for example in examples)),
            "max": float(max(example["pred_delta"] for example in examples)),
            "mean": float(sum(example["pred_delta"] for example in examples) / len(examples)),
            "num_thresholds": int(len(thresholds)),
        },
        "results": results,
    }
    write_json(output_json, report)
    _write_csv(
        output_csv,
        csv_rows,
        [
            "cost_metric",
            "lambda",
            "best_tau",
            "best_mean_reward",
            "best_mean_cost",
            "best_mean_utility",
            "best_chosen_primary",
            "best_chosen_expert",
            "always_primary_mean_utility",
            "always_expert_mean_utility",
            "default_tau0_mean_utility",
            "oracle_reward_mean_utility",
        ],
    )
    print(json.dumps({"output_json": str(output_json), "output_csv": str(output_csv), "n_examples": len(examples)}))


if __name__ == "__main__":
    main()

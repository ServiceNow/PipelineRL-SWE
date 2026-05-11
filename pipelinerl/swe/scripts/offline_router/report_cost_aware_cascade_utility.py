#!/usr/bin/env python
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_LAMBDAS = [
    0.0,
    1.0e-5,
    2.0e-5,
    5.0e-5,
    1.0e-4,
    2.0e-4,
    5.0e-4,
    1.0e-3,
]


def _read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header) for header in headers})


def _key(row: dict[str, Any]) -> str:
    return f"{row.get('dataset')}::{row.get('problem_id')}"


def _parse_lambdas(value: str) -> list[float]:
    values: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values or list(DEFAULT_LAMBDAS)


def _mean(values: list[float]) -> float:
    return math.nan if not values else float(sum(values) / len(values))


def _argmax(values: list[float]) -> int:
    best_idx = 0
    best_value = float(values[0])
    for idx, value in enumerate(values[1:], start=1):
        value = float(value)
        if value > best_value:
            best_idx = idx
            best_value = value
    return int(best_idx)


def _lookup_policy(summary: dict[str, Any], route_label: str) -> dict[str, Any] | None:
    policies = ((summary.get("utility") or {}).get("policies") or {})
    policy = policies.get(f"always::{route_label}")
    return policy if isinstance(policy, dict) else None


def _infer_primary_mean_output_cost(
    reward_summary: dict[str, Any],
    primary_label: str,
    fallback: float | None,
) -> float:
    if fallback is not None:
        return float(fallback)
    policy = _lookup_policy(reward_summary, primary_label)
    if policy is None:
        raise ValueError(
            "Could not infer primary mean output tokens from reward summary; pass --primary-mean-output-tokens"
        )
    value = policy.get("mean_output_tokens")
    if value is None:
        raise ValueError(
            "Reward summary policy has no mean_output_tokens; pass --primary-mean-output-tokens"
        )
    return float(value)


def _load_joined_examples(
    reward_predictions: Path,
    cost_predictions: Path,
    expert_route_idx: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    reward_rows = _read_jsonl(reward_predictions)
    cost_rows = {_key(row): row for row in _read_jsonl(cost_predictions)}
    examples: list[dict[str, Any]] = []
    route_labels: list[str] | None = None
    skipped_missing_cost = 0
    skipped_invalid = 0

    for reward_row in reward_rows:
        key = _key(reward_row)
        cost_row = cost_rows.get(key)
        if cost_row is None:
            skipped_missing_cost += 1
            continue
        true_rewards = reward_row.get("true_rewards")
        pred_rewards = reward_row.get("pred_rewards")
        row_route_labels = reward_row.get("route_labels")
        pred_expert_tokens = cost_row.get("pred_output_tokens")
        true_expert_tokens = cost_row.get("true_output_tokens")
        if (
            not isinstance(true_rewards, list)
            or not isinstance(pred_rewards, list)
            or not isinstance(row_route_labels, list)
            or not isinstance(pred_expert_tokens, list)
            or not isinstance(true_expert_tokens, list)
            or len(true_rewards) != len(pred_rewards)
            or len(row_route_labels) != len(pred_rewards)
            or len(pred_expert_tokens) < 1
            or len(true_expert_tokens) < 1
            or not 0 <= int(expert_route_idx) < len(pred_rewards)
        ):
            skipped_invalid += 1
            continue
        if route_labels is None:
            route_labels = [str(label) for label in row_route_labels]
        examples.append(
            {
                "key": key,
                "dataset": reward_row.get("dataset"),
                "problem_id": reward_row.get("problem_id"),
                "true_primary_reward": float(true_rewards[0]),
                "true_expert_reward": float(true_rewards[int(expert_route_idx)]),
                "pred_primary_reward": float(pred_rewards[0]),
                "pred_expert_reward": float(pred_rewards[int(expert_route_idx)]),
                "pred_expert_output_tokens": max(0.0, float(pred_expert_tokens[0])),
                "true_expert_output_tokens": max(0.0, float(true_expert_tokens[0])),
            }
        )

    if route_labels is None:
        raise ValueError("Could not infer route labels from reward predictions")
    if not examples:
        raise ValueError(
            f"No joined examples. skipped_missing_cost={skipped_missing_cost} skipped_invalid={skipped_invalid}"
        )
    return examples, route_labels


def _summarize_policy(
    name: str,
    choices: list[int],
    examples: list[dict[str, Any]],
    lambda_value: float,
    primary_mean_output_tokens: float,
    primary_cost_weight: float,
    expert_cost_weight: float,
    oracle_choices: list[int],
) -> dict[str, Any]:
    rewards: list[float] = []
    expert_output_costs: list[float] = []
    weighted_costs: list[float] = []
    direct_weighted_costs: list[float] = []
    for choice, example in zip(choices, examples, strict=True):
        if int(choice) == 1:
            reward = float(example["true_expert_reward"])
            expert_cost = float(example["true_expert_output_tokens"])
        else:
            reward = float(example["true_primary_reward"])
            expert_cost = 0.0
        cascade_cost = (primary_cost_weight * primary_mean_output_tokens) + (
            expert_cost_weight * expert_cost
        )
        direct_cost = expert_cost_weight * expert_cost if int(choice) == 1 else (
            primary_cost_weight * primary_mean_output_tokens
        )
        rewards.append(reward)
        expert_output_costs.append(expert_cost)
        weighted_costs.append(cascade_cost)
        direct_weighted_costs.append(direct_cost)

    expert_calls = sum(1 for choice in choices if int(choice) == 1)
    oracle_matches = sum(
        1 for choice, oracle_choice in zip(choices, oracle_choices, strict=True) if int(choice) == int(oracle_choice)
    )
    mean_reward = _mean(rewards)
    mean_weighted_cost = _mean(weighted_costs)
    mean_direct_weighted_cost = _mean(direct_weighted_costs)
    return {
        "policy": name,
        "lambda": float(lambda_value),
        "n_eval": int(len(examples)),
        "expert_call_count": int(expert_calls),
        "expert_call_rate": float(expert_calls / len(examples)),
        "mean_reward": mean_reward,
        "mean_expert_output_tokens": _mean(expert_output_costs),
        "mean_cascade_weighted_output_cost": mean_weighted_cost,
        "mean_direct_weighted_output_cost": mean_direct_weighted_cost,
        "mean_cascade_utility": mean_reward - (float(lambda_value) * mean_weighted_cost),
        "mean_direct_utility": mean_reward - (float(lambda_value) * mean_direct_weighted_cost),
        "oracle_match_rate": float(oracle_matches / len(examples)),
    }


def _decision_rows(
    examples: list[dict[str, Any]],
    lambdas: list[float],
    primary_mean_output_tokens: float,
    primary_cost_weight: float,
    expert_cost_weight: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lambda_value in lambdas:
        router_choices: list[int] = []
        oracle_cascade_choices: list[int] = []
        for example in examples:
            pred_primary_score = float(example["pred_primary_reward"])
            pred_expert_score = float(example["pred_expert_reward"]) - (
                float(lambda_value) * expert_cost_weight * float(example["pred_expert_output_tokens"])
            )
            router_choices.append(1 if pred_expert_score > pred_primary_score else 0)

            true_primary_score = float(example["true_primary_reward"])
            true_expert_score = float(example["true_expert_reward"]) - (
                float(lambda_value) * expert_cost_weight * float(example["true_expert_output_tokens"])
            )
            oracle_cascade_choices.append(1 if true_expert_score > true_primary_score else 0)

        always_primary = [0] * len(examples)
        always_expert = [1] * len(examples)
        policy_summaries = [
            _summarize_policy(
                "router_cascade",
                router_choices,
                examples,
                lambda_value,
                primary_mean_output_tokens,
                primary_cost_weight,
                expert_cost_weight,
                oracle_cascade_choices,
            ),
            _summarize_policy(
                "oracle_cascade",
                oracle_cascade_choices,
                examples,
                lambda_value,
                primary_mean_output_tokens,
                primary_cost_weight,
                expert_cost_weight,
                oracle_cascade_choices,
            ),
            _summarize_policy(
                "always_primary_direct",
                always_primary,
                examples,
                lambda_value,
                primary_mean_output_tokens,
                primary_cost_weight,
                expert_cost_weight,
                oracle_cascade_choices,
            ),
            _summarize_policy(
                "always_expert_direct",
                always_expert,
                examples,
                lambda_value,
                primary_mean_output_tokens,
                primary_cost_weight,
                expert_cost_weight,
                oracle_cascade_choices,
            ),
            _summarize_policy(
                "always_expert_after_primary",
                always_expert,
                examples,
                lambda_value,
                primary_mean_output_tokens,
                primary_cost_weight,
                expert_cost_weight,
                oracle_cascade_choices,
            ),
        ]
        # always_expert_direct should not pay the primary cascade overhead.
        for row in policy_summaries:
            if row["policy"] == "always_expert_direct":
                row["mean_cascade_weighted_output_cost"] = row["mean_direct_weighted_output_cost"]
                row["mean_cascade_utility"] = row["mean_direct_utility"]
        rows.extend(policy_summaries)
    baseline_by_lambda = {
        (row["lambda"], row["policy"]): row for row in rows
    }
    for row in rows:
        expert = baseline_by_lambda[(row["lambda"], "always_expert_direct")]
        primary = baseline_by_lambda[(row["lambda"], "always_primary_direct")]
        oracle = baseline_by_lambda[(row["lambda"], "oracle_cascade")]
        row["delta_utility_vs_always_expert_direct"] = (
            float(row["mean_cascade_utility"]) - float(expert["mean_cascade_utility"])
        )
        row["delta_utility_vs_always_primary_direct"] = (
            float(row["mean_cascade_utility"]) - float(primary["mean_cascade_utility"])
        )
        denominator = float(oracle["mean_cascade_utility"]) - max(
            float(expert["mean_cascade_utility"]),
            float(primary["mean_cascade_utility"]),
        )
        numerator = float(row["mean_cascade_utility"]) - max(
            float(expert["mean_cascade_utility"]),
            float(primary["mean_cascade_utility"]),
        )
        row["oracle_gap_capture_vs_best_direct"] = numerator / denominator if abs(denominator) > 1.0e-12 else None
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-predictions", required=True)
    parser.add_argument("--reward-summary", required=True)
    parser.add_argument("--cost-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expert-route-idx", type=int, default=1)
    parser.add_argument("--primary-cost-weight", type=float, default=0.1)
    parser.add_argument("--expert-cost-weight", type=float, default=1.0)
    parser.add_argument("--primary-mean-output-tokens", type=float, default=None)
    parser.add_argument(
        "--lambdas",
        default=",".join(str(value) for value in DEFAULT_LAMBDAS),
        help="Comma-separated lambda values in reward per weighted output token.",
    )
    args = parser.parse_args()

    reward_predictions = Path(args.reward_predictions)
    reward_summary_path = Path(args.reward_summary)
    cost_predictions = Path(args.cost_predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples, route_labels = _load_joined_examples(
        reward_predictions=reward_predictions,
        cost_predictions=cost_predictions,
        expert_route_idx=int(args.expert_route_idx),
    )
    reward_summary = _read_json(reward_summary_path)
    primary_label = route_labels[0]
    expert_label = route_labels[int(args.expert_route_idx)]
    primary_mean_output_tokens = _infer_primary_mean_output_cost(
        reward_summary=reward_summary,
        primary_label=primary_label,
        fallback=args.primary_mean_output_tokens,
    )
    lambdas = _parse_lambdas(str(args.lambdas))
    rows = _decision_rows(
        examples=examples,
        lambdas=lambdas,
        primary_mean_output_tokens=primary_mean_output_tokens,
        primary_cost_weight=float(args.primary_cost_weight),
        expert_cost_weight=float(args.expert_cost_weight),
    )

    headers = [
        "policy",
        "lambda",
        "n_eval",
        "expert_call_count",
        "expert_call_rate",
        "mean_reward",
        "mean_expert_output_tokens",
        "mean_cascade_weighted_output_cost",
        "mean_direct_weighted_output_cost",
        "mean_cascade_utility",
        "mean_direct_utility",
        "oracle_match_rate",
        "delta_utility_vs_always_expert_direct",
        "delta_utility_vs_always_primary_direct",
        "oracle_gap_capture_vs_best_direct",
    ]
    _write_csv(output_dir / "cost_aware_cascade_utility.csv", rows, headers)

    summary = {
        "reward_predictions": str(reward_predictions),
        "reward_summary": str(reward_summary_path),
        "cost_predictions": str(cost_predictions),
        "n_joined_examples": int(len(examples)),
        "route_labels": route_labels,
        "primary_label": primary_label,
        "expert_label": expert_label,
        "cost_metric": "output_tokens",
        "primary_cost_weight": float(args.primary_cost_weight),
        "expert_cost_weight": float(args.expert_cost_weight),
        "primary_mean_output_tokens": float(primary_mean_output_tokens),
        "lambdas": lambdas,
        "rows": rows,
    }
    _write_json(output_dir / "cost_aware_cascade_utility.json", summary)
    print(f"Wrote {output_dir / 'cost_aware_cascade_utility.csv'}")
    print(f"Wrote {output_dir / 'cost_aware_cascade_utility.json'}")


if __name__ == "__main__":
    main()

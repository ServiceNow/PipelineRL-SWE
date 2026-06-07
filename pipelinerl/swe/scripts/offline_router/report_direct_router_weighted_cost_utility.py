#!/usr/bin/env python
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

DEFAULT_LAMBDAS = [0.0, 1.0e-5, 2.0e-5, 5.0e-5, 1.0e-4, 2.0e-4]
DEFAULT_ROUTE_COST_WEIGHTS = {
    "primary_model": 1.0,
    "expert_0:openai/gpt-oss-120b": 3.0,
    "expert_0:google/gemini-3-flash-preview": 20.0,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header) for header in headers})


def _key(row: dict[str, Any]) -> str:
    return f"{row.get('dataset')}::{row.get('problem_id')}"


def _argmax(values: list[float]) -> int:
    best_idx = 0
    best_value = float(values[0])
    for idx, value in enumerate(values[1:], start=1):
        value = float(value)
        if value > best_value:
            best_idx = idx
            best_value = value
    return int(best_idx)


def _mean(values: list[float]) -> float:
    return math.nan if not values else float(sum(values) / len(values))


def _parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    return values or list(DEFAULT_LAMBDAS)


def _parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def _parse_cost_metrics(value: str) -> list[str]:
    metrics = [part.strip() for part in value.split(",") if part.strip()]
    for metric in metrics:
        if metric not in {"output_tokens", "total_tokens"}:
            raise ValueError(f"Unsupported cost metric: {metric}")
    return metrics or ["output_tokens"]


def _parse_route_cost_weights(value: str | None, route_labels: list[str]) -> list[float]:
    if value is None or not value.strip():
        return [float(DEFAULT_ROUTE_COST_WEIGHTS.get(label, 1.0)) for label in route_labels]

    raw_parts = [part.strip() for part in value.split(",") if part.strip()]
    if all("=" not in part for part in raw_parts):
        weights = [float(part) for part in raw_parts]
        if len(weights) != len(route_labels):
            raise ValueError(
                f"Positional --route-cost-weights needs {len(route_labels)} values, got {len(weights)}"
            )
        return weights

    mapping: dict[str, float] = {}
    for part in raw_parts:
        if "=" not in part:
            raise ValueError("Use either all positional weights or all label=value weights")
        label, weight = part.split("=", 1)
        mapping[label.strip()] = float(weight.strip())
    return [float(mapping.get(label, DEFAULT_ROUTE_COST_WEIGHTS.get(label, 1.0))) for label in route_labels]


def _collect_eval_dir(path: Path) -> Path:
    if (path / "eval").is_dir():
        return path / "eval"
    return path


def _read_eval_rows(collect_dir: Path) -> dict[str, dict[str, Any]]:
    eval_dir = _collect_eval_dir(collect_dir)
    parquet_paths = sorted(eval_dir.glob("*.parquet"))
    if not parquet_paths:
        raise ValueError(f"No parquet shards found under {eval_dir}")
    lookup: dict[str, dict[str, Any]] = {}
    columns = [
        "dataset",
        "problem_id",
        "performance_targets",
        "route_prompt_tokens",
        "route_output_tokens",
    ]
    for path in parquet_paths:
        table = pq.read_table(path, columns=columns)
        for row in table.to_pylist():
            lookup[_key(row)] = row
    return lookup


def _cost_for_route(
    *,
    prompt_tokens: list[float],
    output_tokens: list[float],
    route_idx: int,
    cost_metric: str,
    route_cost_weights: list[float],
) -> float:
    token_count = float(output_tokens[route_idx])
    if cost_metric == "total_tokens":
        token_count += float(prompt_tokens[route_idx])
    return float(route_cost_weights[route_idx]) * token_count


def _after_primary_cost(
    *,
    prompt_tokens: list[float],
    output_tokens: list[float],
    route_idx: int,
    cost_metric: str,
    route_cost_weights: list[float],
) -> float:
    cost = _cost_for_route(
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        route_idx=0,
        cost_metric=cost_metric,
        route_cost_weights=route_cost_weights,
    )
    if int(route_idx) != 0:
        cost += _cost_for_route(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            route_idx=route_idx,
            cost_metric=cost_metric,
            route_cost_weights=route_cost_weights,
        )
    return cost


def _build_examples(
    *,
    reward_predictions: Path,
    cost_predictions: Path,
    collect_dir: Path,
    cost_route_idxs: list[int],
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    reward_rows = _read_jsonl(reward_predictions)
    cost_lookup = {_key(row): row for row in _read_jsonl(cost_predictions)}
    eval_lookup = _read_eval_rows(collect_dir)
    route_labels: list[str] | None = None
    skipped = {"missing_cost": 0, "missing_eval": 0, "invalid": 0}
    examples: list[dict[str, Any]] = []

    for reward_row in reward_rows:
        key = _key(reward_row)
        cost_row = cost_lookup.get(key)
        eval_row = eval_lookup.get(key)
        if cost_row is None:
            skipped["missing_cost"] += 1
            continue
        if eval_row is None:
            skipped["missing_eval"] += 1
            continue

        row_route_labels = reward_row.get("route_labels")
        pred_rewards = reward_row.get("pred_rewards")
        true_rewards = eval_row.get("performance_targets")
        prompt_tokens = eval_row.get("route_prompt_tokens")
        output_tokens = eval_row.get("route_output_tokens")
        pred_cost_outputs = cost_row.get("pred_output_tokens")
        if (
            not isinstance(row_route_labels, list)
            or not isinstance(pred_rewards, list)
            or not isinstance(true_rewards, list)
            or not isinstance(prompt_tokens, list)
            or not isinstance(output_tokens, list)
            or not isinstance(pred_cost_outputs, list)
            or len(row_route_labels) != len(pred_rewards)
            or len(true_rewards) != len(pred_rewards)
            or len(prompt_tokens) != len(pred_rewards)
            or len(output_tokens) != len(pred_rewards)
            or len(pred_cost_outputs) != len(cost_route_idxs)
        ):
            skipped["invalid"] += 1
            continue

        if route_labels is None:
            route_labels = [str(label) for label in row_route_labels]
        elif route_labels != [str(label) for label in row_route_labels]:
            raise ValueError(f"Route labels changed within predictions at {key}")

        pred_output_tokens = [float(value) for value in output_tokens]
        for pos, route_idx in enumerate(cost_route_idxs):
            pred_output_tokens[int(route_idx)] = max(0.0, float(pred_cost_outputs[pos]))

        examples.append(
            {
                "key": key,
                "dataset": reward_row.get("dataset"),
                "problem_id": reward_row.get("problem_id"),
                "true_rewards": [float(value) for value in true_rewards],
                "pred_rewards": [float(value) for value in pred_rewards],
                "prompt_tokens": [float(value) for value in prompt_tokens],
                "true_output_tokens": [float(value) for value in output_tokens],
                "pred_output_tokens": pred_output_tokens,
            }
        )

    if route_labels is None:
        raise ValueError("Could not infer route labels from reward predictions")
    if not examples:
        raise ValueError(f"No joined examples: {skipped}")
    return examples, route_labels, skipped


def _choice_counts(choices: list[int], route_labels: list[str]) -> dict[str, int]:
    counts = {label: 0 for label in route_labels}
    for choice in choices:
        counts[route_labels[int(choice)]] += 1
    return counts


def _summarize_choices(
    *,
    policy: str,
    policy_type: str,
    choices: list[int],
    examples: list[dict[str, Any]],
    route_labels: list[str],
    route_cost_weights: list[float],
    lambda_value: float,
    cost_metric: str,
    accounting: str,
    utility_oracle_choices: list[int],
    reward_oracle_choices: list[int],
) -> dict[str, Any]:
    true_rewards: list[float] = []
    pred_rewards: list[float] = []
    true_costs: list[float] = []
    pred_costs: list[float] = []
    utility_oracle_matches = 0
    reward_oracle_matches = 0

    for choice, example, utility_oracle_choice, reward_oracle_choice in zip(
        choices, examples, utility_oracle_choices, reward_oracle_choices, strict=True
    ):
        route_idx = int(choice)
        if accounting == "direct":
            true_cost = _cost_for_route(
                prompt_tokens=example["prompt_tokens"],
                output_tokens=example["true_output_tokens"],
                route_idx=route_idx,
                cost_metric=cost_metric,
                route_cost_weights=route_cost_weights,
            )
            pred_cost = _cost_for_route(
                prompt_tokens=example["prompt_tokens"],
                output_tokens=example["pred_output_tokens"],
                route_idx=route_idx,
                cost_metric=cost_metric,
                route_cost_weights=route_cost_weights,
            )
        elif accounting == "after_primary":
            true_cost = _after_primary_cost(
                prompt_tokens=example["prompt_tokens"],
                output_tokens=example["true_output_tokens"],
                route_idx=route_idx,
                cost_metric=cost_metric,
                route_cost_weights=route_cost_weights,
            )
            pred_cost = _after_primary_cost(
                prompt_tokens=example["prompt_tokens"],
                output_tokens=example["pred_output_tokens"],
                route_idx=route_idx,
                cost_metric=cost_metric,
                route_cost_weights=route_cost_weights,
            )
        else:
            raise ValueError(f"Unsupported accounting mode: {accounting}")

        true_rewards.append(float(example["true_rewards"][route_idx]))
        pred_rewards.append(float(example["pred_rewards"][route_idx]))
        true_costs.append(float(true_cost))
        pred_costs.append(float(pred_cost))
        if route_idx == int(utility_oracle_choice):
            utility_oracle_matches += 1
        if route_idx == int(reward_oracle_choice):
            reward_oracle_matches += 1

    mean_true_reward = _mean(true_rewards)
    mean_pred_reward = _mean(pred_rewards)
    mean_true_cost = _mean(true_costs)
    mean_pred_cost = _mean(pred_costs)
    n = len(examples)
    return {
        "policy": policy,
        "policy_type": policy_type,
        "lambda": float(lambda_value),
        "cost_metric": cost_metric,
        "cost_accounting": accounting,
        "mean_true_reward": mean_true_reward,
        "mean_pred_reward": mean_pred_reward,
        "mean_true_cost": mean_true_cost,
        "mean_pred_cost": mean_pred_cost,
        "mean_true_utility": mean_true_reward - (float(lambda_value) * mean_true_cost),
        "mean_pred_utility": mean_pred_reward - (float(lambda_value) * mean_pred_cost),
        "utility_oracle_match_rate": float(utility_oracle_matches / n),
        "reward_oracle_match_rate": float(reward_oracle_matches / n),
        "choice_counts_by_route": json.dumps(_choice_counts(choices, route_labels), sort_keys=True),
    }


def _oracle_choices(
    *,
    examples: list[dict[str, Any]],
    route_cost_weights: list[float],
    lambda_value: float,
    cost_metric: str,
    accounting: str,
) -> list[int]:
    choices: list[int] = []
    for example in examples:
        scores: list[float] = []
        for route_idx, reward in enumerate(example["true_rewards"]):
            if accounting == "direct":
                cost = _cost_for_route(
                    prompt_tokens=example["prompt_tokens"],
                    output_tokens=example["true_output_tokens"],
                    route_idx=route_idx,
                    cost_metric=cost_metric,
                    route_cost_weights=route_cost_weights,
                )
            elif accounting == "after_primary":
                cost = _after_primary_cost(
                    prompt_tokens=example["prompt_tokens"],
                    output_tokens=example["true_output_tokens"],
                    route_idx=route_idx,
                    cost_metric=cost_metric,
                    route_cost_weights=route_cost_weights,
                )
            else:
                raise ValueError(f"Unsupported accounting mode: {accounting}")
            scores.append(float(reward) - (float(lambda_value) * float(cost)))
        choices.append(_argmax(scores))
    return choices


def _router_choices(
    *,
    examples: list[dict[str, Any]],
    route_cost_weights: list[float],
    lambda_value: float,
    cost_metric: str,
    include_predicted_cost: bool,
    accounting: str,
) -> list[int]:
    choices: list[int] = []
    for example in examples:
        scores: list[float] = []
        for route_idx, pred_reward in enumerate(example["pred_rewards"]):
            if include_predicted_cost:
                if accounting == "direct":
                    pred_cost = _cost_for_route(
                        prompt_tokens=example["prompt_tokens"],
                        output_tokens=example["pred_output_tokens"],
                        route_idx=route_idx,
                        cost_metric=cost_metric,
                        route_cost_weights=route_cost_weights,
                    )
                elif accounting == "after_primary":
                    pred_cost = _after_primary_cost(
                        prompt_tokens=example["prompt_tokens"],
                        output_tokens=example["pred_output_tokens"],
                        route_idx=route_idx,
                        cost_metric=cost_metric,
                        route_cost_weights=route_cost_weights,
                    )
                else:
                    raise ValueError(f"Unsupported router accounting mode: {accounting}")
            else:
                pred_cost = 0.0
            scores.append(float(pred_reward) - (float(lambda_value) * float(pred_cost)))
        choices.append(_argmax(scores))
    return choices


def _compute_rows(
    *,
    examples: list[dict[str, Any]],
    route_labels: list[str],
    route_cost_weights: list[float],
    lambdas: list[float],
    cost_metrics: list[str],
    router_accounting: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_dim = len(route_labels)
    reward_oracle_choices = [_argmax(example["true_rewards"]) for example in examples]
    reward_router_choices = [_argmax(example["pred_rewards"]) for example in examples]

    for lambda_value in lambdas:
        for cost_metric in cost_metrics:
            after_primary_utility_oracle = _oracle_choices(
                examples=examples,
                route_cost_weights=route_cost_weights,
                lambda_value=float(lambda_value),
                cost_metric=cost_metric,
                accounting="after_primary",
            )
            direct_utility_oracle = _oracle_choices(
                examples=examples,
                route_cost_weights=route_cost_weights,
                lambda_value=float(lambda_value),
                cost_metric=cost_metric,
                accounting="direct",
            )
            router_cost_choices = _router_choices(
                examples=examples,
                route_cost_weights=route_cost_weights,
                lambda_value=float(lambda_value),
                cost_metric=cost_metric,
                include_predicted_cost=True,
                accounting=router_accounting,
            )

            policy_specs: list[tuple[str, str, list[int], str]] = [
                (
                    f"direct_router_{router_accounting}_pred_reward_pred_cost_weighted",
                    f"router_{router_accounting}",
                    router_cost_choices,
                    router_accounting,
                ),
                (
                    f"reward_only_router_{router_accounting}_weighted",
                    f"router_{router_accounting}",
                    reward_router_choices,
                    router_accounting,
                ),
                (
                    "oracle_after_primary_utility_weighted",
                    "oracle_after_primary",
                    after_primary_utility_oracle,
                    "after_primary",
                ),
                (
                    "oracle_direct_utility_weighted",
                    "oracle_direct",
                    direct_utility_oracle,
                    "direct",
                ),
                (
                    "oracle_reward_after_primary_weighted",
                    "oracle_after_primary",
                    reward_oracle_choices,
                    "after_primary",
                ),
            ]
            for route_idx, route_label in enumerate(route_labels):
                choices = [route_idx for _ in examples]
                policy_specs.append(
                    (
                        f"always_direct::{route_label}",
                        "always_direct",
                        choices,
                        "direct",
                    )
                )
                if route_idx != 0:
                    policy_specs.append(
                        (
                            f"always_after_primary::{route_label}",
                            "always_after_primary",
                            choices,
                            "after_primary",
                        )
                    )

            for policy, policy_type, choices, accounting in policy_specs:
                utility_oracle = (
                    direct_utility_oracle if accounting == "direct" else after_primary_utility_oracle
                )
                rows.append(
                    _summarize_choices(
                        policy=policy,
                        policy_type=policy_type,
                        choices=choices,
                        examples=examples,
                        route_labels=route_labels,
                        route_cost_weights=route_cost_weights,
                        lambda_value=float(lambda_value),
                        cost_metric=cost_metric,
                        accounting=accounting,
                        utility_oracle_choices=utility_oracle,
                        reward_oracle_choices=reward_oracle_choices,
                    )
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report realized utility for a 3+ route direct router with fixed per-route cost multipliers."
    )
    parser.add_argument("--reward-predictions", type=Path, required=True)
    parser.add_argument("--cost-predictions", type=Path, required=True)
    parser.add_argument("--collect-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cost-route-idxs", default="1,2")
    parser.add_argument("--route-cost-weights", default=None)
    parser.add_argument("--lambdas", default=",".join(str(value) for value in DEFAULT_LAMBDAS))
    parser.add_argument("--cost-metrics", default="output_tokens,total_tokens")
    parser.add_argument(
        "--router-accounting",
        choices=["direct", "after_primary"],
        default="after_primary",
        help="Cost accounting for the learned router policy. Use after_primary for scout-conditioned routers and direct for input-only routers.",
    )
    args = parser.parse_args()

    examples, route_labels, skipped = _build_examples(
        reward_predictions=args.reward_predictions,
        cost_predictions=args.cost_predictions,
        collect_dir=args.collect_dir,
        cost_route_idxs=_parse_int_list(str(args.cost_route_idxs)),
    )
    route_cost_weights = _parse_route_cost_weights(args.route_cost_weights, route_labels)
    lambdas = _parse_float_list(str(args.lambdas))
    cost_metrics = _parse_cost_metrics(str(args.cost_metrics))
    rows = _compute_rows(
        examples=examples,
        route_labels=route_labels,
        route_cost_weights=route_cost_weights,
        lambdas=lambdas,
        cost_metrics=cost_metrics,
        router_accounting=str(args.router_accounting),
    )

    headers = [
        "policy",
        "policy_type",
        "lambda",
        "cost_metric",
        "cost_accounting",
        "mean_true_reward",
        "mean_pred_reward",
        "mean_true_cost",
        "mean_pred_cost",
        "mean_true_utility",
        "mean_pred_utility",
        "utility_oracle_match_rate",
        "reward_oracle_match_rate",
        "choice_counts_by_route",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "direct_router_weighted_cost_utility.csv", rows, headers)
    _write_json(
        args.output_dir / "direct_router_weighted_cost_utility.json",
        {
            "reward_predictions": str(args.reward_predictions),
            "cost_predictions": str(args.cost_predictions),
            "collect_dir": str(args.collect_dir),
            "n_examples": len(examples),
            "skipped": skipped,
            "route_labels": route_labels,
            "route_cost_weights": {
                route_label: float(weight) for route_label, weight in zip(route_labels, route_cost_weights, strict=True)
            },
            "cost_route_idxs": _parse_int_list(str(args.cost_route_idxs)),
            "lambdas": lambdas,
            "cost_metrics": cost_metrics,
            "router_accounting": str(args.router_accounting),
            "rows": rows,
        },
    )
    print(f"Wrote {args.output_dir / 'direct_router_weighted_cost_utility.csv'}")
    print(f"Wrote {args.output_dir / 'direct_router_weighted_cost_utility.json'}")


if __name__ == "__main__":
    main()

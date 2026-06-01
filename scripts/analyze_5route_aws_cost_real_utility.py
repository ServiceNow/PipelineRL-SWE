#!/usr/bin/env python3
"""Real-reward 5-route utility simulation with AWS/OpenRouter cost accounting.

This intentionally uses only the Python stdlib so it can run in the thin
workspace environment. It consumes existing router prediction JSONL files and
the SWE-bench real pass matrix, then emits CSVs plus simple SVG plots.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ROUTES = [
    "scout:Qwen/Qwen3-4B-Instruct-2507",
    "solver:openai/gpt-oss-20b",
    "solver:Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "solver:openai/gpt-oss-120b",
    "solver:google/gemini-3-flash-preview",
]

SHORT = {
    "scout:Qwen/Qwen3-4B-Instruct-2507": "4B scout",
    "solver:openai/gpt-oss-20b": "OSS-20B",
    "solver:Qwen/Qwen3-Coder-30B-A3B-Instruct": "Qwen-30B",
    "solver:openai/gpt-oss-120b": "OSS-120B",
    "solver:google/gemini-3-flash-preview": "Gemini",
}

REAL_KEY_BY_ROUTE = {
    "scout:Qwen/Qwen3-4B-Instruct-2507": "qwen3_4b_instruct_2507",
    "solver:openai/gpt-oss-20b": "gpt_oss_20b",
    "solver:Qwen/Qwen3-Coder-30B-A3B-Instruct": "qwen3_coder_30b_a3b",
    "solver:openai/gpt-oss-120b": "gpt_oss_120b",
    "solver:google/gemini-3-flash-preview": "gemini",
}

LOCAL_COST_PER_M_OUTPUT = {
    "scout:Qwen/Qwen3-4B-Instruct-2507": 0.278,
    "solver:openai/gpt-oss-20b": 1.299,
    "solver:Qwen/Qwen3-Coder-30B-A3B-Instruct": 4.640,
    "solver:openai/gpt-oss-120b": 11.130,
}

GEMINI_LABEL = "solver:google/gemini-3-flash-preview"
GEMINI_INPUT_PER_M = 0.50
GEMINI_OUTPUT_PER_M = 3.00

DEFAULT_LAMBDAS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100, 150, 200]


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def read_jsonl_by_problem_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[str(row["problem_id"])] = row
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda idx: float(values[idx]))


def normalize_token_vector(values: Any, route_labels: list[str], scout_mean_output: float) -> list[float]:
    if not isinstance(values, list):
        raise ValueError("token vector is not a list")
    floats = [max(0.0, float(value)) for value in values]
    if len(floats) == len(route_labels):
        return floats
    if len(floats) == len(route_labels) - 1:
        return [float(scout_mean_output)] + floats
    raise ValueError(f"unexpected token vector length {len(floats)} for {len(route_labels)} routes")


def build_examples(
    *,
    reward_path: Path,
    cost_path: Path,
    pass_matrix: dict[str, Any],
    scout_mean_output: float,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    reward_rows = read_jsonl_by_problem_id(reward_path)
    cost_rows = read_jsonl_by_problem_id(cost_path)
    pass_ids = [str(value) for value in pass_matrix["instance_ids"]]
    pass_by = pass_matrix["pass_by"]

    examples: list[dict[str, Any]] = []
    route_labels: list[str] | None = None
    skipped = {"missing_reward": 0, "missing_cost": 0, "missing_pass": 0, "invalid": 0}

    for problem_id in pass_ids:
        reward_row = reward_rows.get(problem_id)
        cost_row = cost_rows.get(problem_id)
        if reward_row is None:
            skipped["missing_reward"] += 1
            continue
        if cost_row is None:
            skipped["missing_cost"] += 1
            continue

        labels = [str(label) for label in reward_row.get("route_labels") or []]
        if route_labels is None:
            route_labels = labels
        if labels != route_labels or labels != [str(label) for label in cost_row.get("route_labels") or []]:
            skipped["invalid"] += 1
            continue

        pred_rewards = reward_row.get("pred_rewards")
        if not isinstance(pred_rewards, list) or len(pred_rewards) != len(labels):
            skipped["invalid"] += 1
            continue

        try:
            pred_output_tokens = normalize_token_vector(
                cost_row.get("pred_output_tokens"),
                labels,
                scout_mean_output,
            )
            true_output_tokens = normalize_token_vector(
                cost_row.get("true_output_tokens"),
                labels,
                scout_mean_output,
            )
        except ValueError:
            skipped["invalid"] += 1
            continue

        true_rewards: list[float] = []
        missing_pass = False
        for label in labels:
            model_key = REAL_KEY_BY_ROUTE[label]
            model_passes = pass_by[model_key]
            if problem_id not in model_passes:
                missing_pass = True
                break
            true_rewards.append(float(model_passes[problem_id]))
        if missing_pass:
            skipped["missing_pass"] += 1
            continue

        examples.append(
            {
                "problem_id": problem_id,
                "route_labels": labels,
                "pred_rewards": [float(value) for value in pred_rewards],
                "true_rewards": true_rewards,
                "pred_output_tokens": pred_output_tokens,
                "true_output_tokens": true_output_tokens,
            }
        )

    if route_labels is None:
        raise ValueError("Could not infer route labels")
    if not examples:
        raise ValueError(f"No joined examples. skipped={skipped}")
    return examples, route_labels, skipped


def model_cost_dollars(label: str, output_tokens: float, gemini_prompt_tokens: float) -> float:
    if label == GEMINI_LABEL:
        return (
            float(gemini_prompt_tokens) * GEMINI_INPUT_PER_M
            + float(output_tokens) * GEMINI_OUTPUT_PER_M
        ) / 1_000_000.0
    return float(output_tokens) * LOCAL_COST_PER_M_OUTPUT[label] / 1_000_000.0


def route_cost(
    example: dict[str, Any],
    route_idx: int,
    *,
    pred: bool,
    accounting: str,
    gemini_prompt_tokens: float,
) -> float:
    token_key = "pred_output_tokens" if pred else "true_output_tokens"
    labels = example["route_labels"]
    output_tokens = example[token_key]
    cost = model_cost_dollars(labels[route_idx], output_tokens[route_idx], gemini_prompt_tokens)

    if accounting == "direct":
        return cost
    if accounting == "after_scout":
        if route_idx == 0:
            return cost
        scout_cost = model_cost_dollars(labels[0], output_tokens[0], gemini_prompt_tokens)
        return scout_cost + cost
    raise ValueError(f"Unsupported accounting mode: {accounting}")


def choose_router(
    examples: list[dict[str, Any]],
    lambda_value: float,
    *,
    accounting: str,
    gemini_prompt_tokens: float,
) -> list[int]:
    choices: list[int] = []
    for example in examples:
        scores = []
        for idx, pred_reward in enumerate(example["pred_rewards"]):
            pred_cost = route_cost(
                example,
                idx,
                pred=True,
                accounting=accounting,
                gemini_prompt_tokens=gemini_prompt_tokens,
            )
            scores.append(float(pred_reward) - float(lambda_value) * pred_cost)
        choices.append(argmax(scores))
    return choices


def choose_oracle(
    examples: list[dict[str, Any]],
    lambda_value: float,
    *,
    accounting: str,
    gemini_prompt_tokens: float,
) -> list[int]:
    choices: list[int] = []
    for example in examples:
        scores = []
        for idx, true_reward in enumerate(example["true_rewards"]):
            true_cost = route_cost(
                example,
                idx,
                pred=False,
                accounting=accounting,
                gemini_prompt_tokens=gemini_prompt_tokens,
            )
            scores.append(float(true_reward) - float(lambda_value) * true_cost)
        choices.append(argmax(scores))
    return choices


def summarize_choices(
    *,
    policy: str,
    family: str,
    examples: list[dict[str, Any]],
    route_labels: list[str],
    choices: list[int],
    lambda_value: float,
    accounting: str,
    gemini_prompt_tokens: float,
) -> dict[str, Any]:
    rewards: list[float] = []
    costs: list[float] = []
    counts = {SHORT[label]: 0 for label in route_labels}
    for example, choice in zip(examples, choices, strict=True):
        rewards.append(float(example["true_rewards"][choice]))
        costs.append(
            route_cost(
                example,
                choice,
                pred=False,
                accounting=accounting,
                gemini_prompt_tokens=gemini_prompt_tokens,
            )
        )
        counts[SHORT[route_labels[choice]]] += 1

    mean_reward = mean(rewards)
    mean_cost = mean(costs)
    return {
        "policy": policy,
        "family": family,
        "n": len(examples),
        "lambda_dollar": float(lambda_value),
        "real_pass_rate": mean_reward,
        "mean_cost_dollars": mean_cost,
        "mean_cost_per_1000": mean_cost * 1000.0,
        "real_utility": mean_reward - float(lambda_value) * mean_cost,
        "choice_counts_by_route": json.dumps(counts, sort_keys=True),
    }


def compute_family_rows(
    *,
    family: str,
    examples: list[dict[str, Any]],
    route_labels: list[str],
    accounting: str,
    lambdas: list[float],
    gemini_prompt_tokens: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lambda_value in lambdas:
        router_choices = choose_router(
            examples,
            lambda_value,
            accounting=accounting,
            gemini_prompt_tokens=gemini_prompt_tokens,
        )
        oracle_choices = choose_oracle(
            examples,
            lambda_value,
            accounting=accounting,
            gemini_prompt_tokens=gemini_prompt_tokens,
        )
        rows.append(
            summarize_choices(
                policy=f"{family} router",
                family=family,
                examples=examples,
                route_labels=route_labels,
                choices=router_choices,
                lambda_value=lambda_value,
                accounting=accounting,
                gemini_prompt_tokens=gemini_prompt_tokens,
            )
        )
        rows.append(
            summarize_choices(
                policy=f"{family} oracle",
                family=family,
                examples=examples,
                route_labels=route_labels,
                choices=oracle_choices,
                lambda_value=lambda_value,
                accounting=accounting,
                gemini_prompt_tokens=gemini_prompt_tokens,
            )
        )
        for route_idx, route_label in enumerate(route_labels):
            rows.append(
                summarize_choices(
                    policy=f"always::{SHORT[route_label]}",
                    family="always",
                    examples=examples,
                    route_labels=route_labels,
                    choices=[route_idx] * len(examples),
                    lambda_value=lambda_value,
                    accounting="direct",
                    gemini_prompt_tokens=gemini_prompt_tokens,
                )
            )
    return rows


def svg_line_plot(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    keep_always_only_first: bool,
) -> None:
    colors = {
        "4B post-scout router": "#1f77b4",
        "4B input-only router": "#ff7f0e",
        "always::4B scout": "#2ca02c",
        "always::OSS-20B": "#8c564b",
        "always::Qwen-30B": "#9467bd",
        "always::OSS-120B": "#7f7f7f",
        "always::Gemini": "#d62728",
    }
    present_policies = {str(row["policy"]) for row in rows}
    policies = [policy for policy in colors if policy in present_policies]
    plot_rows = [
        row
        for row in rows
        if row["policy"] in policies
        and (not keep_always_only_first or not str(row["policy"]).startswith("always::") or row["lambda_dollar"] == 0)
    ]

    width, height = 960, 620
    ml, mr, mt, mb = 90, 35, 35, 80
    xs = [float(row[x_key]) for row in plot_rows]
    ys = [float(row[y_key]) for row in plot_rows]
    xmin, xmax = (0.0, max(xs) * 1.08) if x_key == "mean_cost_per_1000" else (min(xs), max(xs))
    ymin, ymax = max(0.0, min(ys) - 0.03), min(0.62, max(ys) + 0.04)
    if y_key == "real_utility":
        ymin, ymax = min(ys) - 0.02, max(ys) + 0.02

    def sx(value: float) -> float:
        return ml + (value - xmin) / (xmax - xmin) * (width - ml - mr)

    def sy(value: float) -> float:
        return height - mb - (value - ymin) / (ymax - ymin) * (height - mt - mb)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in plot_rows:
        grouped.setdefault(str(row["policy"]), []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: float(row[x_key]))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
    ]

    if x_key == "mean_cost_per_1000":
        step = 1 if xmax < 12 else 2
        xticks = list(range(0, int(math.ceil(xmax / step)) * step + 1, step))
    else:
        xticks = [0, 25, 50, 75, 100, 125, 150, 175, 200]
    for tick in xticks:
        if xmin <= tick <= xmax:
            x = sx(float(tick))
            parts.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{height - mb}" stroke="#ddd"/>')
            parts.append(
                f'<text x="{x:.1f}" y="{height - mb + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{tick:g}</text>'
            )

    for idx in range(9):
        value = ymin + idx * (ymax - ymin) / 8
        y = sy(value)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width - mr}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(
            f'<text x="{ml - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.2f}</text>'
        )

    parts.extend(
        [
            f'<line x1="{ml}" y1="{height - mb}" x2="{width - mr}" y2="{height - mb}" stroke="#333"/>',
            f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height - mb}" stroke="#333"/>',
            f'<text x="{width / 2}" y="{height - 28}" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(x_label)}</text>',
            f'<text transform="translate(25 {height / 2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(y_label)}</text>',
        ]
    )

    for policy, values in grouped.items():
        color = colors[policy]
        if not policy.startswith("always::") or not keep_always_only_first:
            points = " ".join(f'{sx(float(row[x_key])):.1f},{sy(float(row[y_key])):.1f}' for row in values)
            width_line = 3 if policy in {"4B post-scout router", "4B input-only router"} else 1.8
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width_line}" opacity="0.9"/>'
            )
        for row in values:
            x = sx(float(row[x_key]))
            y = sy(float(row[y_key]))
            radius = 5 if not policy.startswith("always::") else 7
            title_text = (
                f'{policy} lambda={row["lambda_dollar"]:g} '
                f'reward={row["real_pass_rate"]:.3f} '
                f'cost=${row["mean_cost_per_1000"]:.3f}/1k'
            )
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" stroke="white" stroke-width="1.5">'
                f"<title>{html.escape(title_text)}</title></circle>"
            )
            if not policy.startswith("always::") and float(row["lambda_dollar"]) in {0, 15, 25, 40, 75, 100}:
                parts.append(
                    f'<text x="{x + 7:.1f}" y="{y - 7:.1f}" font-family="sans-serif" font-size="10" fill="{color}">'
                    f'lambda={row["lambda_dollar"]:g}</text>'
                )

    lx, ly = width - 270, 60
    for idx, policy in enumerate(policies):
        y = ly + idx * 24
        color = colors[policy]
        parts.append(f'<circle cx="{lx}" cy="{y}" r="6" fill="{color}"/>')
        parts.append(f'<text x="{lx + 14}" y="{y + 4}" font-family="sans-serif" font-size="12">{html.escape(policy)}</text>')

    parts.append("</svg>")
    output_path.write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-post", type=Path, required=True)
    parser.add_argument("--cost-post", type=Path, required=True)
    parser.add_argument("--reward-input", type=Path, required=True)
    parser.add_argument("--cost-input", type=Path, required=True)
    parser.add_argument("--pass-matrix", type=Path, required=True)
    parser.add_argument("--collection-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambdas", default=",".join(str(value) for value in DEFAULT_LAMBDAS))
    parser.add_argument(
        "--exclude-route-labels",
        default="",
        help="Comma-separated route labels to remove from selectable/evaluated routes.",
    )
    args = parser.parse_args()

    lambdas = [float(part.strip()) for part in str(args.lambdas).split(",") if part.strip()]
    excluded_route_labels = {
        part.strip() for part in str(args.exclude_route_labels).split(",") if part.strip()
    }
    pass_matrix = read_json(args.pass_matrix)
    collection_summary = read_json(args.collection_summary)
    scout_mean_output = float(
        collection_summary["splits"]["eval"]["by_route"][ROUTES[0]]["output_tokens_mean"]
    )
    gemini_mean_prompt = float(
        collection_summary["splits"]["eval"]["by_route"][GEMINI_LABEL]["prompt_tokens_mean"]
    )

    post_examples, route_labels, post_skipped = build_examples(
        reward_path=args.reward_post,
        cost_path=args.cost_post,
        pass_matrix=pass_matrix,
        scout_mean_output=scout_mean_output,
    )
    input_examples, input_route_labels, input_skipped = build_examples(
        reward_path=args.reward_input,
        cost_path=args.cost_input,
        pass_matrix=pass_matrix,
        scout_mean_output=scout_mean_output,
    )
    if route_labels != input_route_labels:
        raise ValueError("Post-scout and input-only route labels differ")

    if excluded_route_labels:
        keep_idxs = [idx for idx, label in enumerate(route_labels) if label not in excluded_route_labels]
        if not keep_idxs:
            raise ValueError("Excluded every route")

        def filter_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
            filtered = []
            for example in examples:
                filtered.append(
                    {
                        **example,
                        "route_labels": [example["route_labels"][idx] for idx in keep_idxs],
                        "pred_rewards": [example["pred_rewards"][idx] for idx in keep_idxs],
                        "true_rewards": [example["true_rewards"][idx] for idx in keep_idxs],
                        "pred_output_tokens": [example["pred_output_tokens"][idx] for idx in keep_idxs],
                        "true_output_tokens": [example["true_output_tokens"][idx] for idx in keep_idxs],
                    }
                )
            return filtered

        post_examples = filter_examples(post_examples)
        input_examples = filter_examples(input_examples)
        route_labels = [route_labels[idx] for idx in keep_idxs]

    rows = []
    rows.extend(
        compute_family_rows(
            family="4B post-scout",
            examples=post_examples,
            route_labels=route_labels,
            accounting="after_scout",
            lambdas=lambdas,
            gemini_prompt_tokens=gemini_mean_prompt,
        )
    )
    rows.extend(
        compute_family_rows(
            family="4B input-only",
            examples=input_examples,
            route_labels=route_labels,
            accounting="direct",
            lambdas=lambdas,
            gemini_prompt_tokens=gemini_mean_prompt,
        )
    )

    headers = [
        "policy",
        "family",
        "n",
        "lambda_dollar",
        "real_pass_rate",
        "mean_cost_dollars",
        "mean_cost_per_1000",
        "real_utility",
        "choice_counts_by_route",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "direct_policy_real_utility_aws_cost_exact_gemini.csv", rows, headers)
    write_csv(
        args.output_dir / "router_choice_breakdown_aws_cost_exact_gemini.csv",
        [row for row in rows if row["policy"] in {"4B post-scout router", "4B input-only router"}],
        headers,
    )
    write_csv(
        args.output_dir / "selected_policy_real_utility_aws_cost_exact_gemini.csv",
        [
            row
            for row in rows
            if row["lambda_dollar"] in {0, 10, 15, 20, 25, 30, 40, 50, 75, 100}
            and row["policy"]
            in {
                "4B post-scout router",
                "4B input-only router",
                "always::4B scout",
                "always::OSS-20B",
                "always::Qwen-30B",
                "always::OSS-120B",
                "always::Gemini",
                "4B post-scout oracle",
                "4B input-only oracle",
            }
        ],
        headers,
    )

    svg_line_plot(
        rows=rows,
        output_path=args.output_dir / "real_reward_cost_front_aws_cost_exact_gemini.svg",
        title="Real SWE-Bench Verified tradeoff with AWS/OpenRouter cost scale",
        x_key="mean_cost_per_1000",
        y_key="real_pass_rate",
        x_label="Mean cost ($ / 1000 instances)",
        y_label="Real pass rate",
        keep_always_only_first=True,
    )
    svg_line_plot(
        rows=rows,
        output_path=args.output_dir / "real_utility_vs_lambda_aws_cost_exact_gemini.svg",
        title="Real utility vs lambda using AWS/OpenRouter cost scale",
        x_key="lambda_dollar",
        y_key="real_utility",
        x_label="lambda in utility = pass rate - lambda * dollars/instance",
        y_label="Mean true utility",
        keep_always_only_first=False,
    )

    summary = {
        "local_cost_per_m_output_tokens_usd": LOCAL_COST_PER_M_OUTPUT,
        "gemini_cost_per_m_input_tokens_usd": GEMINI_INPUT_PER_M,
        "gemini_cost_per_m_output_tokens_usd": GEMINI_OUTPUT_PER_M,
        "gemini_mean_eval_prompt_tokens_used_for_cost": gemini_mean_prompt,
        "scout_mean_eval_output_tokens_used_for_post_scout_cost": scout_mean_output,
        "lambdas_dollar": lambdas,
        "excluded_route_labels": sorted(excluded_route_labels),
        "route_labels_evaluated": route_labels,
        "n_post_examples": len(post_examples),
        "n_input_examples": len(input_examples),
        "post_skipped": post_skipped,
        "input_skipped": input_skipped,
        "paths": {
            "reward_post": str(args.reward_post),
            "cost_post": str(args.cost_post),
            "reward_input": str(args.reward_input),
            "cost_input": str(args.cost_input),
            "pass_matrix": str(args.pass_matrix),
            "collection_summary": str(args.collection_summary),
        },
    }
    (args.output_dir / "summary_aws_cost_exact_gemini.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    print(args.output_dir)
    for lambda_value in [0, 10, 15, 20, 25, 30, 40, 50, 75, 100]:
        print(f"\nlambda {lambda_value:g}")
        for policy in [
            "4B post-scout router",
            "4B input-only router",
            "always::4B scout",
            "always::Gemini",
            "always::Qwen-30B",
            "always::OSS-20B",
            "always::OSS-120B",
        ]:
            matches = [
                row
                for row in rows
                if row["policy"] == policy and float(row["lambda_dollar"]) == float(lambda_value)
            ]
            if not matches:
                continue
            row = matches[0]
            print(
                f"{policy:24s} "
                f"util={row['real_utility']:.4f} "
                f"pass={row['real_pass_rate']:.4f} "
                f"cost/1k=${row['mean_cost_per_1000']:.3f} "
                f"choices={row['choice_counts_by_route']}"
            )


if __name__ == "__main__":
    main()

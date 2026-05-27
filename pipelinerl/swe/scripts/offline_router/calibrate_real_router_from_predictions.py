#!/usr/bin/env python
"""Calibrate proxy-router predictions against real SWE-Bench pass/fail labels.

This script is intentionally lightweight: it fits tiny route-specific logistic
calibrators on top of existing deployable router predictions, using grouped
cross-validation by SWE instance. It does not fine-tune the embedding model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize

DEFAULT_COLLECT_DIR = Path(
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_bench_train_all_16k_verified_eval_collect_3route_oss_gemini/collect"
)
DEFAULT_EXPORT_DIR = Path("eval_exports/swebench_verified_3route_predictions_369_1779753111")
DEFAULT_RESULTS_DIR = Path("res")
DEFAULT_POST_REWARD = Path(
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_bench_train_all_16k_verified_eval_3route_qwen3_embedding_8b_lora_reward_mse_delta_5epoch_1778485045/"
    "train_qwen3_embedding_8b_lora_reward_mse_delta_5epoch/eval_predictions.jsonl"
)
DEFAULT_POST_COST = Path(
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_bench_train_all_16k_verified_eval_3route_qwen3_embedding_8b_lora_expert_cost_only_5epoch_1778523269/"
    "train_qwen3_embedding_8b_lora_expert_cost_only_5epoch/eval_predictions.jsonl"
)
DEFAULT_INPUT_REWARD = Path(
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_bench_train_all_16k_verified_eval_3route_qwen3_embedding_8b_lora_reward_mse_delta_input_only_5epoch_1778791805/"
    "train_qwen3_embedding_8b_lora_reward_mse_delta_5epoch/eval_predictions.jsonl"
)
DEFAULT_INPUT_COST = Path(
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_bench_train_all_16k_verified_eval_3route_qwen3_embedding_8b_lora_expert_cost_only_input_only_5epoch_1778790758/"
    "train_qwen3_embedding_8b_lora_expert_cost_only_5epoch/eval_predictions.jsonl"
)

ROUTE_NAMES = ["scout", "oss", "gemini"]
ROUTE_LABELS = ["primary_model", "expert_0:openai/gpt-oss-120b", "expert_0:google/gemini-3-flash-preview"]
ROUTE_WEIGHTS = np.asarray([1.0, 3.0, 20.0], dtype=np.float64)
DEFAULT_LAMBDAS = [0.0, 1.0e-5, 2.0e-5, 5.0e-5, 1.0e-4, 2.0e-4]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _key(row: dict[str, Any]) -> str:
    return str(row.get("problem_id") or row.get("instance_id") or row.get("id"))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _logit(p: float) -> float:
    p = min(max(float(p), 1.0e-6), 1.0 - 1.0e-6)
    return float(math.log(p / (1.0 - p)))


def _ranks(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = ((i + j - 1) / 2.0) + 1.0
        i = j
    return ranks


def _pearson(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if len(x_arr) < 2 or float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return math.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _spearman(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    return _pearson(_ranks(x), _ranks(y))


def _roc_auc(scores: list[float] | np.ndarray, labels: list[float] | np.ndarray) -> float:
    score_arr = np.asarray(scores, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int64)
    n_pos = int(np.sum(label_arr == 1))
    n_neg = int(np.sum(label_arr == 0))
    if n_pos == 0 or n_neg == 0:
        return math.nan
    rank_scores = _ranks(score_arr)
    sum_pos = float(np.sum(rank_scores[label_arr == 1]))
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _brier(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(scores, dtype=np.float64) - np.asarray(labels, dtype=np.float64)) ** 2))


def _logloss(scores: np.ndarray, labels: np.ndarray) -> float:
    eps = 1.0e-6
    p = np.clip(np.asarray(scores, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _parse_lambdas(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    return values or list(DEFAULT_LAMBDAS)


def _direct_cost(output_tokens: np.ndarray, route_idx: int) -> float:
    return float(output_tokens[int(route_idx)]) * float(ROUTE_WEIGHTS[int(route_idx)])


def _after_scout_cost(output_tokens: np.ndarray, route_idx: int) -> float:
    cost = _direct_cost(output_tokens, 0)
    if int(route_idx) != 0:
        cost += _direct_cost(output_tokens, int(route_idx))
    return float(cost)


def _load_real_reports(results_dir: Path) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    paths = {
        "scout": results_dir / "always_scout_fixed_results.json",
        "oss": results_dir / "always_oss_fixed_results.json",
        "gemini": results_dir / "always_gemini_fixed_results.json",
    }
    real_by_route: dict[str, dict[str, int]] = {}
    summary: dict[str, Any] = {}
    for route, path in paths.items():
        data = json.loads(path.read_text())
        resolved = set(data.get("resolved") or [])
        unresolved = set(data.get("unresolved") or [])
        all_ids = resolved | unresolved
        real_by_route[route] = {instance_id: int(instance_id in resolved) for instance_id in all_ids}
        summary[route] = {
            "resolved": len(resolved),
            "unresolved": len(unresolved),
            "reported_total": len(all_ids),
            "score": float(data.get("score", math.nan)),
        }
    return real_by_route, summary


def _load_collect_rows(collect_dir: Path) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in sorted((collect_dir / "eval").glob("*.parquet")):
        rows.extend(pq.read_table(shard).to_pylist())
    return {str(row["problem_id"]): row for row in rows}


def _load_prediction_lookup(path: Path) -> dict[str, dict[str, Any]]:
    return {_key(row): row for row in _read_jsonl(path)}


def _build_examples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    real_by_route, report_summary = _load_real_reports(Path(args.results_dir))
    collect_by_id = _load_collect_rows(Path(args.collect_dir))
    export_ids_path = Path(args.export_dir) / "instance_ids.txt"
    export_ids = [line.strip() for line in export_ids_path.read_text().splitlines() if line.strip()]
    reported_sets = [set(real_by_route[route]) for route in ROUTE_NAMES]
    if args.instance_mode == "intersection":
        instance_ids = sorted(set(export_ids).intersection(*reported_sets))
    elif args.instance_mode == "missing_as_fail":
        instance_ids = list(export_ids)
    else:
        raise ValueError(f"Unsupported instance_mode={args.instance_mode}")

    post_reward = _load_prediction_lookup(Path(args.post_reward_predictions))
    post_cost = _load_prediction_lookup(Path(args.post_cost_predictions))
    input_reward = _load_prediction_lookup(Path(args.input_reward_predictions))
    input_cost = _load_prediction_lookup(Path(args.input_cost_predictions))

    examples: list[dict[str, Any]] = []
    skipped = Counter()
    for instance_id in instance_ids:
        collect_row = collect_by_id.get(instance_id)
        if collect_row is None:
            skipped["missing_collect"] += 1
            continue
        if instance_id not in post_reward or instance_id not in post_cost or instance_id not in input_reward or instance_id not in input_cost:
            skipped["missing_predictions"] += 1
            continue
        route_real = []
        for route in ROUTE_NAMES:
            if instance_id in real_by_route[route]:
                route_real.append(float(real_by_route[route][instance_id]))
            elif args.instance_mode == "missing_as_fail":
                route_real.append(0.0)
            else:
                raise ValueError(f"Missing real result for {route}::{instance_id}")
        output_tokens = np.asarray([float(value) for value in collect_row["route_output_tokens"]], dtype=np.float64)
        examples.append(
            {
                "instance_id": instance_id,
                "repo": collect_row.get("repo"),
                "real_pass": np.asarray(route_real, dtype=np.float64),
                "proxy_rewards": np.asarray([float(value) for value in collect_row["performance_targets"]], dtype=np.float64),
                "output_tokens": output_tokens,
                "post_pred_rewards": np.asarray([float(value) for value in post_reward[instance_id]["pred_rewards"]], dtype=np.float64),
                "input_pred_rewards": np.asarray([float(value) for value in input_reward[instance_id]["pred_rewards"]], dtype=np.float64),
                "post_pred_output_tokens": np.asarray(
                    [
                        output_tokens[0],
                        float(post_cost[instance_id]["pred_output_tokens"][0]),
                        float(post_cost[instance_id]["pred_output_tokens"][1]),
                    ],
                    dtype=np.float64,
                ),
                "input_pred_output_tokens": np.asarray(
                    [
                        output_tokens[0],
                        float(input_cost[instance_id]["pred_output_tokens"][0]),
                        float(input_cost[instance_id]["pred_output_tokens"][1]),
                    ],
                    dtype=np.float64,
                ),
            }
        )
    metadata = {
        "instance_mode": args.instance_mode,
        "n_export_instances": len(export_ids),
        "n_examples": len(examples),
        "n_reported_intersection": len(set.intersection(*reported_sets)),
        "n_reported_union": len(set.union(*reported_sets)),
        "skipped": dict(skipped),
        "report_summary": report_summary,
    }
    return examples, metadata


def _make_splits(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, int(n_folds))
    return [np.asarray(fold, dtype=np.int64) for fold in folds if len(fold) > 0]


def _fit_logistic(x_train: np.ndarray, y_train: np.ndarray, l2: float) -> np.ndarray:
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    prior = (float(np.sum(y)) + 0.5) / (len(y) + 1.0)
    init = np.zeros(design.shape[1], dtype=np.float64)
    init[-1] = _logit(prior)

    def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        z = design @ params
        # mean BCE with logits, plus L2 on non-intercept terms.
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
        reg_mask = np.ones_like(params)
        reg_mask[-1] = 0.0
        loss += 0.5 * float(l2) * float(np.sum((params * reg_mask) ** 2))
        pred = _sigmoid(z)
        grad = design.T @ (pred - y) / len(y)
        grad += float(l2) * params * reg_mask
        return loss, grad

    result = minimize(
        lambda p: loss_and_grad(p)[0],
        init,
        jac=lambda p: loss_and_grad(p)[1],
        method="L-BFGS-B",
        options={"maxiter": 500},
    )
    if not result.success:
        return init
    return np.asarray(result.x, dtype=np.float64)


def _feature_matrix(examples: list[dict[str, Any]], pred_key: str, route_idx: int, model: str) -> np.ndarray:
    values = np.asarray([example[pred_key] for example in examples], dtype=np.float64)
    if model == "route_prior":
        return np.zeros((len(examples), 0), dtype=np.float64)
    if model == "route_scalar":
        return values[:, [int(route_idx)]]
    if model == "route_vector":
        return values
    raise ValueError(f"Unsupported calibrator model: {model}")


def _fit_route_calibrators(
    train_examples: list[dict[str, Any]],
    pred_key: str,
    model: str,
    l2: float,
) -> list[np.ndarray]:
    calibrators: list[np.ndarray] = []
    y_all = np.asarray([example["real_pass"] for example in train_examples], dtype=np.float64)
    for route_idx in range(len(ROUTE_NAMES)):
        y = y_all[:, route_idx]
        if model == "route_prior":
            prior = (float(np.sum(y)) + 0.5) / (len(y) + 1.0)
            calibrators.append(np.asarray([_logit(prior)], dtype=np.float64))
            continue
        x = _feature_matrix(train_examples, pred_key, route_idx, model)
        calibrators.append(_fit_logistic(x, y, l2=float(l2)))
    return calibrators


def _predict_route_probs(
    examples: list[dict[str, Any]],
    pred_key: str,
    model: str,
    calibrators: list[np.ndarray],
) -> np.ndarray:
    probs = np.zeros((len(examples), len(ROUTE_NAMES)), dtype=np.float64)
    for route_idx, params in enumerate(calibrators):
        if model == "route_prior":
            logits = np.full(len(examples), float(params[-1]), dtype=np.float64)
        else:
            x = _feature_matrix(examples, pred_key, route_idx, model)
            design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
            logits = design @ params
        probs[:, route_idx] = _sigmoid(logits)
    return probs


def _cross_validated_predictions(
    examples: list[dict[str, Any]],
    pred_key: str,
    model: str,
    n_folds: int,
    seed: int,
    l2: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    folds = _make_splits(len(examples), n_folds=n_folds, seed=seed)
    oof = np.full((len(examples), len(ROUTE_NAMES)), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    all_indices = np.arange(len(examples))
    for fold_idx, valid_idx in enumerate(folds):
        train_idx = np.asarray([idx for idx in all_indices if idx not in set(valid_idx.tolist())], dtype=np.int64)
        train_examples = [examples[int(idx)] for idx in train_idx]
        valid_examples = [examples[int(idx)] for idx in valid_idx]
        calibrators = _fit_route_calibrators(train_examples, pred_key=pred_key, model=model, l2=l2)
        valid_probs = _predict_route_probs(valid_examples, pred_key=pred_key, model=model, calibrators=calibrators)
        oof[valid_idx] = valid_probs
        for route_idx, params in enumerate(calibrators):
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "pred_key": pred_key,
                    "calibrator_model": model,
                    "route": ROUTE_NAMES[route_idx],
                    "params_json": json.dumps([float(value) for value in params]),
                    "train_n": int(len(train_examples)),
                    "valid_n": int(len(valid_examples)),
                    "train_positive_rate": float(np.mean([example["real_pass"][route_idx] for example in train_examples])),
                }
            )
    if np.any(np.isnan(oof)):
        raise ValueError("Some examples did not receive out-of-fold predictions")
    return oof, fold_rows


def _cost_vector(example: dict[str, Any], token_key: str, accounting: str) -> np.ndarray:
    tokens = np.asarray(example[token_key], dtype=np.float64)
    if accounting == "direct":
        return np.asarray([_direct_cost(tokens, idx) for idx in range(len(ROUTE_NAMES))], dtype=np.float64)
    if accounting == "after_scout":
        return np.asarray([_after_scout_cost(tokens, idx) for idx in range(len(ROUTE_NAMES))], dtype=np.float64)
    raise ValueError(f"Unsupported accounting={accounting}")


def _true_cost_vector(example: dict[str, Any], accounting: str) -> np.ndarray:
    return _cost_vector(example, "output_tokens", accounting=accounting)


def _summarize_policy(
    *,
    examples: list[dict[str, Any]],
    policy: str,
    policy_type: str,
    lambda_value: float,
    choices: list[int],
    cost_accounting: str,
) -> dict[str, Any]:
    rewards: list[float] = []
    proxy_rewards: list[float] = []
    costs: list[float] = []
    choice_counts: Counter[str] = Counter()
    called_counts: Counter[str] = Counter()
    for example, choice in zip(examples, choices, strict=True):
        idx = int(choice)
        true_costs = _true_cost_vector(example, accounting=cost_accounting)
        rewards.append(float(example["real_pass"][idx]))
        proxy_rewards.append(float(example["proxy_rewards"][idx]))
        costs.append(float(true_costs[idx]))
        choice_counts[ROUTE_NAMES[idx]] += 1
        if cost_accounting == "direct":
            called = [idx]
        else:
            called = [0] if idx == 0 else [0, idx]
        for called_idx in called:
            called_counts[ROUTE_NAMES[called_idx]] += 1
    mean_reward = float(np.mean(rewards))
    mean_cost = float(np.mean(costs))
    return {
        "policy": policy,
        "policy_type": policy_type,
        "lambda": float(lambda_value),
        "n": len(examples),
        "real_pass_rate": mean_reward,
        "mean_proxy_reward_of_chosen": float(np.mean(proxy_rewards)),
        "mean_true_weighted_cost": mean_cost,
        "real_utility": mean_reward - float(lambda_value) * mean_cost,
        "cost_accounting": cost_accounting,
        "choice_counts": json.dumps(dict(choice_counts), sort_keys=True),
        "called_counts": json.dumps(dict(called_counts), sort_keys=True),
    }


def _choices_from_scores(
    examples: list[dict[str, Any]],
    scores: np.ndarray,
    token_key: str,
    cost_accounting_for_decision: str,
    lambda_value: float,
) -> list[int]:
    choices: list[int] = []
    for row_idx, example in enumerate(examples):
        costs = _cost_vector(example, token_key=token_key, accounting=cost_accounting_for_decision)
        values = scores[row_idx] - float(lambda_value) * costs
        choices.append(int(np.argmax(values)))
    return choices


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _score_metrics(name: str, probs: np.ndarray, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    y = np.asarray([example["real_pass"] for example in examples], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for route_idx, route in enumerate(ROUTE_NAMES):
        rows.append(
            {
                "score_name": name,
                "route": route,
                "n": len(examples),
                "positive_rate": float(np.mean(y[:, route_idx])),
                "mean_score": float(np.mean(probs[:, route_idx])),
                "pearson": _pearson(probs[:, route_idx], y[:, route_idx]),
                "spearman": _spearman(probs[:, route_idx], y[:, route_idx]),
                "roc_auc": _roc_auc(probs[:, route_idx], y[:, route_idx]),
                "brier": _brier(probs[:, route_idx], y[:, route_idx]),
                "logloss": _logloss(probs[:, route_idx], y[:, route_idx]),
            }
        )
    rows.append(
        {
            "score_name": name,
            "route": "pooled",
            "n": int(y.size),
            "positive_rate": float(np.mean(y.reshape(-1))),
            "mean_score": float(np.mean(probs.reshape(-1))),
            "pearson": _pearson(probs.reshape(-1), y.reshape(-1)),
            "spearman": _spearman(probs.reshape(-1), y.reshape(-1)),
            "roc_auc": _roc_auc(probs.reshape(-1), y.reshape(-1)),
            "brier": _brier(probs.reshape(-1), y.reshape(-1)),
            "logloss": _logloss(probs.reshape(-1), y.reshape(-1)),
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-dir", type=Path, default=DEFAULT_COLLECT_DIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--post-reward-predictions", type=Path, default=DEFAULT_POST_REWARD)
    parser.add_argument("--post-cost-predictions", type=Path, default=DEFAULT_POST_COST)
    parser.add_argument("--input-reward-predictions", type=Path, default=DEFAULT_INPUT_REWARD)
    parser.add_argument("--input-cost-predictions", type=Path, default=DEFAULT_INPUT_COST)
    parser.add_argument("--output-dir", type=Path, default=Path("router_analysis/real_swebench_verified_3route_1779753111/calibration_cv"))
    parser.add_argument("--instance-mode", choices=["intersection", "missing_as_fail"], default="intersection")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--lambdas", default=",".join(str(value) for value in DEFAULT_LAMBDAS))
    parser.add_argument("--calibrator-models", default="route_prior,route_scalar,route_vector")
    args = parser.parse_args()

    examples, metadata = _build_examples(args)
    if len(examples) < int(args.n_folds):
        raise ValueError(f"Not enough examples for {args.n_folds} folds: {len(examples)}")
    calibrator_models = [part.strip() for part in str(args.calibrator_models).split(",") if part.strip()]
    lambdas = _parse_lambdas(str(args.lambdas))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_sets: dict[str, np.ndarray] = {
        "post_uncalibrated_pred_reward": np.asarray([example["post_pred_rewards"] for example in examples], dtype=np.float64),
        "input_uncalibrated_pred_reward": np.asarray([example["input_pred_rewards"] for example in examples], dtype=np.float64),
    }
    fold_rows: list[dict[str, Any]] = []
    for model in calibrator_models:
        post_probs, post_folds = _cross_validated_predictions(
            examples,
            pred_key="post_pred_rewards",
            model=model,
            n_folds=int(args.n_folds),
            seed=int(args.seed),
            l2=float(args.l2),
        )
        input_probs, input_folds = _cross_validated_predictions(
            examples,
            pred_key="input_pred_rewards",
            model=model,
            n_folds=int(args.n_folds),
            seed=int(args.seed),
            l2=float(args.l2),
        )
        prediction_sets[f"post_calibrated_{model}"] = post_probs
        prediction_sets[f"input_calibrated_{model}"] = input_probs
        fold_rows.extend(post_folds)
        fold_rows.extend(input_folds)

    metric_rows: list[dict[str, Any]] = []
    for name, scores in prediction_sets.items():
        metric_rows.extend(_score_metrics(name, scores, examples))

    policy_rows: list[dict[str, Any]] = []
    for lambda_value in lambdas:
        # Direct always policies.
        for route_idx, route in enumerate(ROUTE_NAMES):
            policy_rows.append(
                _summarize_policy(
                    examples=examples,
                    policy=f"always_{route}",
                    policy_type="always_direct",
                    lambda_value=float(lambda_value),
                    choices=[route_idx for _ in examples],
                    cost_accounting="direct",
                )
            )

        # Uncalibrated and calibrated routers. Post-scout uses after-scout accounting.
        for name, scores in prediction_sets.items():
            if name.startswith("post_"):
                choices = _choices_from_scores(
                    examples,
                    scores=scores,
                    token_key="post_pred_output_tokens",
                    cost_accounting_for_decision="after_scout",
                    lambda_value=float(lambda_value),
                )
                policy_rows.append(
                    _summarize_policy(
                        examples=examples,
                        policy=name,
                        policy_type="post_scout_router",
                        lambda_value=float(lambda_value),
                        choices=choices,
                        cost_accounting="after_scout",
                    )
                )
            elif name.startswith("input_"):
                # Deployment-realistic input-only: no scout cost unless scout is chosen.
                choices_direct = _choices_from_scores(
                    examples,
                    scores=scores,
                    token_key="input_pred_output_tokens",
                    cost_accounting_for_decision="direct",
                    lambda_value=float(lambda_value),
                )
                policy_rows.append(
                    _summarize_policy(
                        examples=examples,
                        policy=f"{name}_direct_accounting",
                        policy_type="input_only_router_direct",
                        lambda_value=float(lambda_value),
                        choices=choices_direct,
                        cost_accounting="direct",
                    )
                )
                # Information-state ablation: same post-scout cost, but hidden attempt.
                choices_after = _choices_from_scores(
                    examples,
                    scores=scores,
                    token_key="input_pred_output_tokens",
                    cost_accounting_for_decision="after_scout",
                    lambda_value=float(lambda_value),
                )
                policy_rows.append(
                    _summarize_policy(
                        examples=examples,
                        policy=f"{name}_hidden_scout_accounting",
                        policy_type="input_only_router_hidden_scout",
                        lambda_value=float(lambda_value),
                        choices=choices_after,
                        cost_accounting="after_scout",
                    )
                )

        # Real-label utility oracles for context.
        choices = []
        for example in examples:
            costs = _true_cost_vector(example, accounting="after_scout")
            choices.append(int(np.argmax(example["real_pass"] - float(lambda_value) * costs)))
        policy_rows.append(
            _summarize_policy(
                examples=examples,
                policy="oracle_after_scout_real_utility",
                policy_type="oracle_after_scout",
                lambda_value=float(lambda_value),
                choices=choices,
                cost_accounting="after_scout",
            )
        )
        choices = []
        for example in examples:
            costs = _true_cost_vector(example, accounting="direct")
            choices.append(int(np.argmax(example["real_pass"] - float(lambda_value) * costs)))
        policy_rows.append(
            _summarize_policy(
                examples=examples,
                policy="oracle_direct_real_utility",
                policy_type="oracle_direct",
                lambda_value=float(lambda_value),
                choices=choices,
                cost_accounting="direct",
            )
        )

    oof_rows: list[dict[str, Any]] = []
    y = np.asarray([example["real_pass"] for example in examples], dtype=np.float64)
    for row_idx, example in enumerate(examples):
        row: dict[str, Any] = {"instance_id": example["instance_id"], "repo": example.get("repo")}
        for route_idx, route in enumerate(ROUTE_NAMES):
            row[f"real_{route}"] = float(y[row_idx, route_idx])
            row[f"proxy_{route}"] = float(example["proxy_rewards"][route_idx])
        for name, scores in prediction_sets.items():
            for route_idx, route in enumerate(ROUTE_NAMES):
                row[f"{name}_{route}"] = float(scores[row_idx, route_idx])
        oof_rows.append(row)

    _write_csv(args.output_dir / "calibration_prediction_metrics.csv", metric_rows)
    _write_csv(args.output_dir / "calibrated_real_policy_utility.csv", policy_rows)
    _write_csv(args.output_dir / "calibrator_fold_params.csv", fold_rows)
    _write_csv(args.output_dir / "calibration_oof_predictions.csv", oof_rows)
    summary = {
        **metadata,
        "output_dir": str(args.output_dir),
        "n_folds": int(args.n_folds),
        "seed": int(args.seed),
        "l2": float(args.l2),
        "lambda_values": lambdas,
        "calibrator_models": calibrator_models,
        "route_names": ROUTE_NAMES,
        "route_labels": ROUTE_LABELS,
        "route_cost_weights": {route: float(weight) for route, weight in zip(ROUTE_NAMES, ROUTE_WEIGHTS, strict=True)},
        "notes": [
            "All calibrated scores are out-of-fold by instance.",
            "post_* policies use post-scout accounting and include scout cost.",
            "input_*_direct_accounting policies route from the original input and do not pay scout cost unless scout is selected.",
            "input_*_hidden_scout_accounting policies are information-state ablations: they pay scout cost but hide the scout attempt from the router.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(args.output_dir)
    print(args.output_dir / "calibrated_real_policy_utility.csv")
    print(args.output_dir / "calibration_prediction_metrics.csv")
    selected_lambda = 2.0e-5
    print(f"\nSelected lambda={selected_lambda:g}")
    interesting = {
        "always_scout",
        "always_oss",
        "always_gemini",
        "post_uncalibrated_pred_reward",
        "input_uncalibrated_pred_reward_direct_accounting",
        "post_calibrated_route_scalar",
        "input_calibrated_route_scalar_direct_accounting",
        "input_calibrated_route_scalar_hidden_scout_accounting",
        "post_calibrated_route_vector",
        "input_calibrated_route_vector_direct_accounting",
        "oracle_after_scout_real_utility",
    }
    for row in policy_rows:
        if abs(float(row["lambda"]) - selected_lambda) < 1.0e-12 and row["policy"] in interesting:
            print(
                row["policy"],
                "pass=", f"{float(row['real_pass_rate']):.3f}",
                "cost=", f"{float(row['mean_true_weighted_cost']):.0f}",
                "util=", f"{float(row['real_utility']):.3f}",
                "choices=", row["choice_counts"],
            )


if __name__ == "__main__":
    main()

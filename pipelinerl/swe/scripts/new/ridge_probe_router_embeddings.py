#!/usr/bin/env python
import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from pipelinerl.finetune.value_model import AutoModelForCausalLMWithValueHead
from pipelinerl.swe.scripts.new.router_trace_utils import (
    extract_reward_vector,
    extract_route_labels,
    load_router_traces,
)

logger = logging.getLogger(__name__)


def _pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size == 0 or y.size == 0 or x.shape != y.shape:
        return None
    x_center = x - x.mean()
    y_center = y - y.mean()
    denom = float(np.sqrt(np.sum(x_center**2) * np.sum(y_center**2)))
    if denom <= 1e-12:
        return None
    return float(np.sum(x_center * y_center) / denom)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    pos = 0
    while pos < len(order):
        start = pos
        current = values[order[pos]]
        while pos + 1 < len(order) and abs(values[order[pos + 1]] - current) <= 1e-12:
            pos += 1
        end = pos
        avg_rank = (start + end) / 2.0 + 1.0
        ranks[order[start : end + 1]] = avg_rank
        pos += 1
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size == 0 or y.size == 0 or x.shape != y.shape:
        return None
    return _pearson_corr(_average_ranks(x), _average_ranks(y))


def _roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if labels.size == 0 or scores.size == 0 or labels.shape != scores.shape:
        return None
    positives = int(np.sum(labels == 1))
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(scores, dtype=np.float64)
    pos = 0
    while pos < len(order):
        start = pos
        current = scores[order[pos]]
        while pos + 1 < len(order) and abs(scores[order[pos + 1]] - current) <= 1e-12:
            pos += 1
        end = pos
        avg_rank = (start + end) / 2.0 + 1.0
        ranks[order[start : end + 1]] = avg_rank
        pos += 1
    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    u_stat = rank_sum_pos - positives * (positives + 1) / 2.0
    return float(u_stat / (positives * negatives))


def _select_best_split(traces: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return traces
    return [trace for trace in traces if trace.get("split") == split]


def _split_within_repo(
    traces: list[dict[str, Any]],
    train_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_fraction <= 0.0 or train_fraction >= 1.0:
        raise ValueError(f"train_fraction must be in (0,1), got {train_fraction}")

    by_repo: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        repo = str(trace.get("repo") or "unknown")
        by_repo.setdefault(repo, []).append(trace)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    eval_: list[dict[str, Any]] = []

    for repo, repo_traces in by_repo.items():
        local = list(repo_traces)
        rng.shuffle(local)
        n = len(local)
        if n == 1:
            train.extend(local)
            continue
        n_train = int(round(n * train_fraction))
        n_train = max(1, min(n - 1, n_train))
        train.extend(local[:n_train])
        eval_.extend(local[n_train:])

    rng.shuffle(train)
    rng.shuffle(eval_)
    logger.info(
        "Within-repo split built: repos=%d train=%d eval=%d train_fraction=%.3f seed=%d",
        len(by_repo),
        len(train),
        len(eval_),
        train_fraction,
        seed,
    )
    return train, eval_


def _encode_prompt_last_embedding(
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: Any,
    device: torch.device,
    prompt_text: str,
    output_text: str,
) -> np.ndarray | None:
    if not prompt_text:
        return None

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt_text + (output_text or ""), add_special_tokens=False, return_tensors="pt").input_ids

    prompt_len = int(prompt_ids.shape[1])
    full_len = int(full_ids.shape[1])
    if prompt_len <= 0 or full_len <= prompt_len:
        return None

    input_ids = full_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        return None
    last_hidden = hidden_states[-1]
    prompt_last_idx = prompt_len - 1
    if prompt_last_idx >= last_hidden.shape[1]:
        return None
    embedding = last_hidden[0, prompt_last_idx].detach().float().cpu().numpy()
    return embedding


def _fit_ridge_multi_target(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    x_norm = (x_train - x_mean) / x_std
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_center = y_train - y_mean

    n, d = x_norm.shape
    if n <= d:
        k = x_norm @ x_norm.T
        k_reg = k + alpha * np.eye(n, dtype=x_norm.dtype)
        dual = np.linalg.solve(k_reg, y_center)
        weights = x_norm.T @ dual
    else:
        gram = x_norm.T @ x_norm
        gram_reg = gram + alpha * np.eye(d, dtype=x_norm.dtype)
        weights = np.linalg.solve(gram_reg, x_norm.T @ y_center)

    bias = y_mean - (x_mean / x_std) @ weights
    return weights, bias.reshape(-1), x_mean.reshape(-1), x_std.reshape(-1)


def _predict_ridge(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> np.ndarray:
    x_norm = (x - x_mean[None, :]) / x_std[None, :]
    return x_norm @ weights + bias[None, :]


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/eval ridge regression probe on frozen trunk prompt-last embeddings.")
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train-split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--eval-split", default="test", choices=["train", "test", "all"])
    parser.add_argument(
        "--split-mode",
        default="existing",
        choices=["existing", "within-repo"],
        help="existing: use trace split field; within-repo: create train/eval split inside each repo.",
    )
    parser.add_argument(
        "--within-repo-source-split",
        default="train",
        choices=["train", "test", "all"],
        help="Pool of traces used before within-repo split.",
    )
    parser.add_argument("--within-repo-train-fraction", type=float, default=0.8)
    parser.add_argument("--within-repo-seed", type=int, default=0)
    parser.add_argument("--all-model-versions", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-eval-records", type=int, default=None)
    parser.add_argument("--save-predictions-jsonl", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    traces = load_router_traces(
        input_globs=args.input_glob,
        split=None,
        latest_model_only=not args.all_model_versions,
        dedupe_by_problem=not args.keep_duplicates,
    )
    if not traces:
        raise ValueError("No traces found")

    if args.split_mode == "within-repo":
        source = _select_best_split(traces, args.within_repo_source_split)
        if not source:
            raise ValueError(f"No traces available for within-repo source split={args.within_repo_source_split}")
        train_traces, eval_traces = _split_within_repo(
            source,
            train_fraction=float(args.within_repo_train_fraction),
            seed=int(args.within_repo_seed),
        )
    else:
        train_traces = _select_best_split(traces, args.train_split)
        eval_traces = _select_best_split(traces, args.eval_split)
    if args.max_train_records is not None:
        train_traces = train_traces[: args.max_train_records]
    if args.max_eval_records is not None:
        eval_traces = eval_traces[: args.max_eval_records]
    if not train_traces or not eval_traces:
        raise ValueError(f"Empty split after filtering: train={len(train_traces)} eval={len(eval_traces)}")

    n_experts = max((len(trace.get("experts") or []) for trace in traces), default=0)
    route_labels = extract_route_labels(traces, n_experts=n_experts)
    target_dim = len(route_labels)
    logger.info("Loaded traces total=%d train=%d eval=%d target_dim=%d", len(traces), len(train_traces), len(eval_traces), target_dim)
    logger.info("Routes: %s", route_labels)

    device = _pick_device(args.device)
    logger.info("Loading frozen trunk from %s on %s", args.model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_path)
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    def build_xy(traces_in: list[dict[str, Any]], tag: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        kept: list[dict[str, Any]] = []
        skipped = 0
        for trace in tqdm(traces_in, desc=f"Embedding {tag}", unit="trace"):
            policy = trace.get("policy") or {}
            prompt_text = policy.get("prompt_text")
            output_text = policy.get("output_text") or ""
            if not isinstance(prompt_text, str):
                skipped += 1
                continue
            embedding = _encode_prompt_last_embedding(model, tokenizer, device, prompt_text, output_text)
            if embedding is None:
                skipped += 1
                continue
            rewards = extract_reward_vector(trace)
            if len(rewards) < target_dim:
                skipped += 1
                continue
            xs.append(embedding.astype(np.float32))
            ys.append(np.asarray(rewards[:target_dim], dtype=np.float32))
            kept.append(trace)
        if not xs:
            raise ValueError(f"No valid examples for {tag}")
        logger.info("%s examples kept=%d skipped=%d", tag, len(xs), skipped)
        return np.stack(xs), np.stack(ys), kept

    x_train, y_train, kept_train = build_xy(train_traces, "train")
    x_eval, y_eval, kept_eval = build_xy(eval_traces, "eval")
    logger.info("Embedding dims: train=%s eval=%s", x_train.shape, x_eval.shape)

    weights, bias, x_mean, x_std = _fit_ridge_multi_target(x_train, y_train, alpha=float(args.alpha))
    y_hat_eval = _predict_ridge(x_eval, weights, bias, x_mean, x_std)

    pair_rows: list[dict[str, Any]] = []
    for i in range(target_dim):
        for j in range(i + 1, target_dim):
            delta_true = y_eval[:, i] - y_eval[:, j]
            delta_pred = y_hat_eval[:, i] - y_hat_eval[:, j]
            labels = (delta_true > 0).astype(np.int64)

            sign_true = np.sign(delta_true)
            sign_pred = np.sign(delta_pred)
            ranking_acc = float(np.mean(sign_true == sign_pred))
            pearson = _pearson_corr(delta_pred, delta_true)
            spearman = _spearman_corr(delta_pred, delta_true)
            mae_delta = float(np.mean(np.abs(delta_pred - delta_true)))
            roc_auc = _roc_auc_binary(labels, delta_pred)

            pair_rows.append(
                {
                    "left_idx": i,
                    "right_idx": j,
                    "left_label": route_labels[i],
                    "right_label": route_labels[j],
                    "n_eval": int(delta_true.shape[0]),
                    "pearson_delta": pearson,
                    "spearman_delta": spearman,
                    "delta_mae": mae_delta,
                    "ranking_accuracy_sign": ranking_acc,
                    "roc_auc": roc_auc,
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        output_dir / "pairwise_metrics.csv",
        pair_rows,
        [
            "left_idx",
            "right_idx",
            "left_label",
            "right_label",
            "n_eval",
            "pearson_delta",
            "spearman_delta",
            "delta_mae",
            "ranking_accuracy_sign",
            "roc_auc",
        ],
    )

    policy_vs_gpt = None
    for row in pair_rows:
        left = str(row["left_label"]).lower()
        right = str(row["right_label"]).lower()
        labels = [left, right]
        if "policy" in labels[0] or "policy" in labels[1]:
            if any("gpt" in label for label in labels):
                policy_vs_gpt = row
                break

    summary = {
        "model_path": args.model_path,
        "alpha": float(args.alpha),
        "split_mode": args.split_mode,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "within_repo_source_split": args.within_repo_source_split,
        "within_repo_train_fraction": float(args.within_repo_train_fraction),
        "within_repo_seed": int(args.within_repo_seed),
        "n_train": int(x_train.shape[0]),
        "n_eval": int(x_eval.shape[0]),
        "embedding_dim": int(x_train.shape[1]),
        "target_dim": target_dim,
        "routes": route_labels,
        "policy_vs_gpt_pair_metrics": policy_vs_gpt,
        "notes": "Ridge probe trained on frozen prompt-last trunk embeddings.",
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    np.savez(
        output_dir / "ridge_probe_weights.npz",
        weights=weights,
        bias=bias,
        x_mean=x_mean,
        x_std=x_std,
        route_labels=np.asarray(route_labels),
    )

    if args.save_predictions_jsonl:
        pred_path = output_dir / "eval_predictions.jsonl"
        with pred_path.open("w") as sink:
            for idx, trace in enumerate(kept_eval):
                row = {
                    "problem_id": trace.get("problem_id"),
                    "split": trace.get("split"),
                    "model_version": trace.get("model_version"),
                    "routes": route_labels,
                    "true_rewards": y_eval[idx].tolist(),
                    "pred_rewards": y_hat_eval[idx].tolist(),
                }
                sink.write(json.dumps(row) + "\n")
        logger.info("Wrote eval predictions to %s", pred_path)

    logger.info("Done. pairwise metrics: %s", output_dir / "pairwise_metrics.csv")
    if policy_vs_gpt is None:
        logger.warning("Could not auto-detect a policy-vs-gpt pair in route labels.")
    else:
        logger.info(
            "Policy-vs-GPT: pearson=%.4f spearman=%.4f roc_auc=%s",
            _safe_float(policy_vs_gpt.get("pearson_delta")),
            _safe_float(policy_vs_gpt.get("spearman_delta")),
            f"{policy_vs_gpt.get('roc_auc'):.4f}" if policy_vs_gpt.get("roc_auc") is not None else "n/a",
        )


if __name__ == "__main__":
    main()

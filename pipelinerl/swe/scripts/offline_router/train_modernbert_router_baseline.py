#!/usr/bin/env python
import argparse
import csv
import glob
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from pipelinerl.swe.scripts.offline_router.common import (
    compute_pairwise_metrics,
    compute_per_route_metrics,
    csv_headers_for_pairwise_metrics,
    csv_headers_for_route_metrics,
    problem_id_from_item,
    write_json,
)


DEFAULT_UTILITY_LAMBDAS = [0.0, 1.0e-5, 2.0e-5, 5.0e-5, 1.0e-4, 2.0e-4]
DEFAULT_ORACLE_MARGIN_EPSILONS = [0.0, 0.01, 0.02, 0.05, 0.1]


def _load_split(dataset_dir: Path, split_name: str):
    files = sorted(glob.glob(str(dataset_dir / split_name / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet shards found for split={split_name} in {dataset_dir / split_name}")
    return load_dataset("parquet", data_files={split_name: files})[split_name]


def _get_primary_output_text(row: dict[str, Any]) -> str | None:
    primary_output_text = row.get("primary_output_text")
    if not isinstance(primary_output_text, str):
        primary_output_text = row.get("policy_output_text")
    return primary_output_text if isinstance(primary_output_text, str) else None


def _build_input_text(row: dict[str, Any], route_labels: list[str]) -> str | None:
    prompt_text = row.get("prompt_text")
    primary_output_text = _get_primary_output_text(row)
    if not isinstance(prompt_text, str) or not isinstance(primary_output_text, str):
        return None
    route_legend = "\n".join(f"{idx}: {label}" for idx, label in enumerate(route_labels))
    return (
        "Predict the realized proxy rewards for each model route.\n"
        "The proxy reward is computed after the repair run by comparing the route's patch against the gold patch.\n"
        "Only the primary model attempt is shown. Use it as context for all route predictions.\n\n"
        "[Route Order]\n"
        f"{route_legend}\n\n"
        "[Original Repair Prompt]\n"
        f"{prompt_text}\n\n"
        "[Primary Model Attempt]\n"
        f"{primary_output_text}"
    )


class RouterPairDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        route_labels: list[str],
        max_seq_length: int,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        target_dim = len(route_labels)
        for row in rows:
            targets = row.get("performance_targets")
            if not isinstance(targets, list) or len(targets) != target_dim:
                continue
            try:
                target_rewards = [float(value) for value in targets]
                problem_id = problem_id_from_item(row)
            except (TypeError, ValueError):
                continue
            input_text = _build_input_text(row, route_labels)
            if not input_text:
                continue
            encoded = tokenizer(
                input_text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_seq_length,
            )
            input_ids = encoded.get("input_ids")
            attention_mask = encoded.get("attention_mask")
            if not input_ids or not attention_mask:
                continue
            self.rows.append(
                {
                    "row_idx": len(self.rows),
                    "problem_id": problem_id,
                    "dataset": row.get("dataset"),
                    "repo": row.get("repo"),
                    "language": row.get("language"),
                    "input_ids": [int(value) for value in input_ids],
                    "attention_mask": [int(value) for value in attention_mask],
                    "targets": target_rewards,
                    "class_target": _argmax_index(target_rewards),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def _collate(batch: list[dict[str, Any]], pad_token_id: int, target_dim: int) -> dict[str, Any]:
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    targets = torch.zeros((len(batch), target_dim), dtype=torch.float32)
    class_targets = torch.zeros((len(batch),), dtype=torch.long)
    row_indices = torch.zeros((len(batch),), dtype=torch.long)
    for idx, row in enumerate(batch):
        seq_len = len(row["input_ids"])
        input_ids[idx, :seq_len] = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask[idx, :seq_len] = torch.tensor(row["attention_mask"], dtype=torch.long)
        targets[idx] = torch.tensor(row["targets"], dtype=torch.float32)
        class_targets[idx] = int(row["class_target"])
        row_indices[idx] = int(row["row_idx"])
    return {
        "problem_ids": [row["problem_id"] for row in batch],
        "datasets": [row["dataset"] for row in batch],
        "repos": [row["repo"] for row in batch],
        "languages": [row["language"] for row in batch],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "targets": targets,
        "class_targets": class_targets,
        "row_indices": row_indices,
    }


class ModernBertRouter(torch.nn.Module):
    def __init__(self, model_name: str, target_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = torch.nn.Dropout(float(dropout))
        self.head = torch.nn.Linear(hidden_size, int(target_dim))

    def gradient_checkpointing_enable(self) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = outputs.last_hidden_state[:, 0]
        return self.head(self.dropout(pooled))


def _prediction_problem_key(row: dict[str, Any]) -> str:
    return f"{row.get('dataset')}::{row.get('problem_id')}"


def _argmax_index(values: list[float]) -> int:
    best_idx = 0
    best_value = float(values[0])
    for idx, value in enumerate(values[1:], start=1):
        numeric = float(value)
        if numeric > best_value:
            best_idx = idx
            best_value = numeric
    return best_idx


def _mean(total: float, count: int) -> float:
    return math.nan if count <= 0 else float(total / count)


def _true_reward_margin(rewards: list[float]) -> float:
    if len(rewards) < 2:
        return 0.0
    ordered = sorted((float(value) for value in rewards), reverse=True)
    return float(ordered[0] - ordered[1])


def _compute_oracle_match_stats(
    valid_examples: list[dict[str, Any]],
    route_idx_key: str,
    margin_epsilons: list[float],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_epsilon: dict[str, dict[str, Any]] = {}
    for epsilon in margin_epsilons:
        epsilon = float(epsilon)
        kept = [
            example
            for example in valid_examples
            if float(example.get("oracle_reward_margin", 0.0)) > epsilon
        ]
        matches = sum(
            1
            for example in kept
            if int(example[route_idx_key]) == int(example["oracle_choice_idx"])
        )
        row = {
            "epsilon": epsilon,
            "n_examples": int(len(kept)),
            "oracle_match_rate": _mean(float(matches), len(kept)),
        }
        rows.append(row)
        by_epsilon[str(epsilon)] = row
    return {"rows": rows, "by_epsilon": by_epsilon}


def _oracle_margin_key(epsilon: float) -> str:
    return f"{float(epsilon):g}"


def _compute_utility_report(
    prediction_rows: list[dict[str, Any]],
    eval_source_rows: list[dict[str, Any]],
    route_labels: list[str],
    lambdas: list[float],
) -> dict[str, Any]:
    target_dim = len(route_labels)
    eval_lookup: dict[str, dict[str, Any]] = {}
    duplicate_eval_lookup_rows = 0
    invalid_eval_lookup_rows = 0
    for row in eval_source_rows:
        try:
            problem_id = problem_id_from_item(row)
        except ValueError:
            invalid_eval_lookup_rows += 1
            continue
        key = _prediction_problem_key({"dataset": row.get("dataset"), "problem_id": problem_id})
        if key in eval_lookup:
            duplicate_eval_lookup_rows += 1
            continue
        eval_lookup[key] = row

    valid_examples: list[dict[str, Any]] = []
    skipped_missing_eval_row = 0
    skipped_invalid_route_stats = 0
    for row in prediction_rows:
        pred_rewards = row.get("pred_rewards")
        if not isinstance(pred_rewards, list) or len(pred_rewards) != target_dim:
            skipped_invalid_route_stats += 1
            continue
        source_row = eval_lookup.get(_prediction_problem_key(row))
        if source_row is None:
            skipped_missing_eval_row += 1
            continue
        rewards = source_row.get("performance_targets")
        prompt_tokens = source_row.get("route_prompt_tokens")
        output_tokens = source_row.get("route_output_tokens")
        if (
            not isinstance(rewards, list)
            or not isinstance(prompt_tokens, list)
            or not isinstance(output_tokens, list)
            or len(rewards) != target_dim
            or len(prompt_tokens) != target_dim
            or len(output_tokens) != target_dim
        ):
            skipped_invalid_route_stats += 1
            continue
        rewards = [float(value) for value in rewards]
        prompt_tokens = [float(value) for value in prompt_tokens]
        output_tokens = [float(value) for value in output_tokens]
        pred_rewards = [float(value) for value in pred_rewards]
        valid_examples.append(
            {
                "rewards": rewards,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "router_choice_idx": _argmax_index(pred_rewards),
                "oracle_choice_idx": _argmax_index(rewards),
                "oracle_reward_margin": _true_reward_margin(rewards),
            }
        )

    policy_defs = [{"policy": "router", "policy_type": "router", "route_idx": None, "route_label": None}]
    for route_idx, route_label in enumerate(route_labels):
        policy_defs.append(
            {
                "policy": f"always::{route_label}",
                "policy_type": "always_route",
                "route_idx": route_idx,
                "route_label": route_label,
            }
        )
    policy_defs.append({"policy": "oracle", "policy_type": "oracle", "route_idx": None, "route_label": None})

    policy_summaries: dict[str, dict[str, Any]] = {}
    utility_rows: list[dict[str, Any]] = []
    valid_count = len(valid_examples)
    for policy_def in policy_defs:
        policy_name = str(policy_def["policy"])
        policy_type = str(policy_def["policy_type"])
        fixed_route_idx = policy_def["route_idx"]
        route_choice_counts = [0] * target_dim
        reward_sum = 0.0
        prompt_token_sum = 0.0
        output_token_sum = 0.0
        total_token_sum = 0.0
        oracle_match_sum = 0
        oracle_margin_matches = {float(epsilon): 0 for epsilon in DEFAULT_ORACLE_MARGIN_EPSILONS}
        oracle_margin_counts = {float(epsilon): 0 for epsilon in DEFAULT_ORACLE_MARGIN_EPSILONS}
        for example in valid_examples:
            if policy_type == "router":
                route_idx = int(example["router_choice_idx"])
            elif policy_type == "oracle":
                route_idx = int(example["oracle_choice_idx"])
            else:
                route_idx = int(fixed_route_idx)
            if route_idx == int(example["oracle_choice_idx"]):
                oracle_match_sum += 1
            for epsilon in DEFAULT_ORACLE_MARGIN_EPSILONS:
                epsilon = float(epsilon)
                if float(example["oracle_reward_margin"]) > epsilon:
                    oracle_margin_counts[epsilon] += 1
                    if route_idx == int(example["oracle_choice_idx"]):
                        oracle_margin_matches[epsilon] += 1
            route_choice_counts[route_idx] += 1
            reward_sum += float(example["rewards"][route_idx])
            prompt_token_sum += float(example["prompt_tokens"][route_idx])
            output_token_sum += float(example["output_tokens"][route_idx])
            total_token_sum += float(example["prompt_tokens"][route_idx] + example["output_tokens"][route_idx])

        policy_summary = {
            "policy": policy_name,
            "policy_type": policy_type,
            "route_idx": fixed_route_idx,
            "route_label": policy_def["route_label"],
            "n_examples": valid_count,
            "choice_counts_by_route": {
                str(route_label): int(route_choice_counts[idx]) for idx, route_label in enumerate(route_labels)
            },
            "mean_reward": _mean(reward_sum, valid_count),
            "mean_prompt_tokens": _mean(prompt_token_sum, valid_count),
            "mean_output_tokens": _mean(output_token_sum, valid_count),
            "mean_total_tokens": _mean(total_token_sum, valid_count),
            "oracle_match_rate": _mean(float(oracle_match_sum), valid_count),
            "oracle_match_by_margin": {
                _oracle_margin_key(epsilon): {
                    "epsilon": float(epsilon),
                    "n_examples": int(oracle_margin_counts[float(epsilon)]),
                    "oracle_match_rate": _mean(
                        float(oracle_margin_matches[float(epsilon)]),
                        int(oracle_margin_counts[float(epsilon)]),
                    ),
                }
                for epsilon in DEFAULT_ORACLE_MARGIN_EPSILONS
            },
        }
        policy_summaries[policy_name] = policy_summary
        for lambda_value in lambdas:
            lambda_value = float(lambda_value)
            for cost_metric, mean_cost in (
                ("output_tokens", policy_summary["mean_output_tokens"]),
                ("total_tokens", policy_summary["mean_total_tokens"]),
            ):
                utility_rows.append(
                    {
                        "policy": policy_name,
                        "policy_type": policy_type,
                        "route_idx": fixed_route_idx,
                        "route_label": policy_def["route_label"],
                        "lambda": lambda_value,
                        "cost_metric": cost_metric,
                        "mean_reward": policy_summary["mean_reward"],
                        "mean_cost": mean_cost,
                        "mean_utility": policy_summary["mean_reward"] - (lambda_value * mean_cost),
                        "oracle_match_rate": policy_summary["oracle_match_rate"],
                        "oracle_match_rate_margin_gt_0_05": policy_summary["oracle_match_by_margin"]["0.05"][
                            "oracle_match_rate"
                        ],
                        "oracle_match_n_margin_gt_0_05": policy_summary["oracle_match_by_margin"]["0.05"][
                            "n_examples"
                        ],
                    }
                )

    return {
        "n_eval_examples": len(prediction_rows),
        "n_examples_with_utility": valid_count,
        "skipped_missing_eval_row": skipped_missing_eval_row,
        "skipped_invalid_route_stats": skipped_invalid_route_stats,
        "eval_lookup_rows": len(eval_lookup),
        "duplicate_eval_lookup_rows": duplicate_eval_lookup_rows,
        "invalid_eval_lookup_rows": invalid_eval_lookup_rows,
        "lambdas": [float(value) for value in lambdas],
        "route_labels": list(route_labels),
        "policies": policy_summaries,
        "utility_rows": utility_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _compute_classifier_metrics(
    y_true_rewards: np.ndarray,
    y_pred_scores: np.ndarray,
    route_labels: list[str],
) -> dict[str, Any]:
    if y_true_rewards.size == 0 or y_pred_scores.size == 0:
        return {
            "n_eval": 0,
            "accuracy": math.nan,
            "target_counts_by_route": {label: 0 for label in route_labels},
            "pred_counts_by_route": {label: 0 for label in route_labels},
        }

    target_classes = np.argmax(y_true_rewards, axis=1)
    pred_classes = np.argmax(y_pred_scores, axis=1)
    target_counts = np.bincount(target_classes, minlength=len(route_labels))
    pred_counts = np.bincount(pred_classes, minlength=len(route_labels))
    metrics: dict[str, Any] = {
        "n_eval": int(target_classes.shape[0]),
        "accuracy": float(np.mean(target_classes == pred_classes)),
        "target_counts_by_route": {
            route_labels[idx]: int(target_counts[idx]) for idx in range(len(route_labels))
        },
        "pred_counts_by_route": {
            route_labels[idx]: int(pred_counts[idx]) for idx in range(len(route_labels))
        },
    }

    if len(route_labels) == 2:
        # Positive class is route 1, which is expert in the current two-route setup.
        positive_labels = (target_classes == 1).astype(np.int64)
        positive_scores = y_pred_scores[:, 1]
        try:
            from pipelinerl.swe.scripts.offline_router.common import _roc_auc_binary

            metrics["route_1_auc"] = _roc_auc_binary(positive_labels, positive_scores)
        except Exception:
            metrics["route_1_auc"] = None
    return metrics


def _load_route_labels(dataset_dir: Path) -> list[str]:
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open() as handle:
            metadata = json.load(handle)
        route_labels = metadata.get("route_labels")
        if isinstance(route_labels, list) and route_labels:
            return [str(label) for label in route_labels]
    raise ValueError(f"Could not read route_labels from {metadata_path}")


def _shuffle_rows(rows: list[dict[str, Any]], max_rows: int | None, seed: int) -> list[dict[str, Any]]:
    if max_rows is None or max_rows <= 0 or max_rows >= len(rows):
        return rows
    rng = random.Random(int(seed))
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    return [rows[idx] for idx in indices[: int(max_rows)]]


@torch.no_grad()
def _evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loader: DataLoader,
    eval_dataset: RouterPairDataset,
    route_labels: list[str],
    objective: str,
) -> tuple[float, list[dict[str, Any]], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    rows: list[dict[str, Any]] = []
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    for batch in tqdm(loader, desc="Eval ModernBERT router", disable=not accelerator.is_main_process):
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).float()
        targets = batch["targets"].float()
        if objective == "route_classifier":
            loss = F.cross_entropy(logits, batch["class_targets"].long(), reduction="sum")
            preds = torch.softmax(logits, dim=-1)
        else:
            loss = F.mse_loss(logits, targets, reduction="sum")
            preds = logits
        gathered_loss = accelerator.gather_for_metrics(loss.detach().reshape(1)).detach().cpu()
        gathered_preds = accelerator.gather_for_metrics(preds).detach().cpu()
        gathered_targets = accelerator.gather_for_metrics(targets).detach().cpu()
        gathered_indices = accelerator.gather_for_metrics(batch["row_indices"]).detach().cpu().tolist()
        if accelerator.is_main_process:
            total_loss += float(torch.sum(gathered_loss).item())
            total_examples += int(gathered_targets.shape[0])

        for idx in range(gathered_preds.shape[0]):
            source_meta = eval_dataset.rows[int(gathered_indices[idx])]
            true_rewards = [float(value) for value in gathered_targets[idx].tolist()]
            pred_rewards = [float(value) for value in gathered_preds[idx].tolist()]
            rows.append(
                {
                    "problem_id": source_meta["problem_id"],
                    "dataset": source_meta["dataset"],
                    "repo": source_meta["repo"],
                    "language": source_meta["language"],
                    "true_rewards": true_rewards,
                    "pred_rewards": pred_rewards,
                    "route_labels": list(route_labels),
                }
            )
        y_true_chunks.append(gathered_targets.numpy())
        y_pred_chunks.append(gathered_preds.numpy())

    if not accelerator.is_main_process:
        return math.nan, [], np.empty((0, len(route_labels))), np.empty((0, len(route_labels)))
    y_true = np.concatenate(y_true_chunks, axis=0) if y_true_chunks else np.empty((0, len(route_labels)))
    y_pred = np.concatenate(y_pred_chunks, axis=0) if y_pred_chunks else np.empty((0, len(route_labels)))
    if objective == "route_classifier":
        eval_loss = float(total_loss / total_examples) if total_examples > 0 else math.nan
    else:
        total_values = total_examples * len(route_labels)
        eval_loss = float(total_loss / total_values) if total_values > 0 else math.nan
    return eval_loss, rows, y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-large")
    parser.add_argument("--objective", choices=["reward_mse", "route_classifier"], default="reward_mse")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-train-rows", type=int, default=4096)
    parser.add_argument("--max-eval-rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)

    accelerator = Accelerator(gradient_accumulation_steps=int(args.gradient_accumulation_steps))
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    route_labels = _load_route_labels(dataset_dir)
    target_dim = len(route_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    train_rows = _shuffle_rows(list(_load_split(dataset_dir, "train")), args.max_train_rows, args.seed)
    eval_rows_source = _shuffle_rows(list(_load_split(dataset_dir, "eval")), args.max_eval_rows, args.seed + 1)
    train_dataset = RouterPairDataset(train_rows, tokenizer, route_labels, int(args.max_seq_length))
    eval_dataset = RouterPairDataset(eval_rows_source, tokenizer, route_labels, int(args.max_seq_length))
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError(f"Prepared empty dataset train={len(train_dataset)} eval={len(eval_dataset)}")

    collate_fn = lambda batch: _collate(batch, pad_token_id=int(pad_token_id), target_dim=target_dim)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = ModernBertRouter(args.model_name, target_dim=target_dim, dropout=float(args.dropout))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    update_steps_per_epoch = math.ceil(len(train_loader) / int(args.gradient_accumulation_steps))
    total_update_steps = max(1, int(args.num_epochs) * update_steps_per_epoch)
    warmup_steps = int(total_update_steps * float(args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    config = {
        "model_name": args.model_name,
        "objective": args.objective,
        "dataset_dir": str(dataset_dir),
        "route_labels": route_labels,
        "max_seq_length": int(args.max_seq_length),
        "num_epochs": int(args.num_epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "max_train_rows": int(args.max_train_rows),
        "max_eval_rows": int(args.max_eval_rows),
        "prepared_train_rows": len(train_dataset),
        "prepared_eval_rows": len(eval_dataset),
        "seed": int(args.seed),
    }
    if accelerator.is_main_process:
        write_json(output_dir / "train_config.json", config)

    history: list[dict[str, Any]] = []
    best_eval_loss = float("inf")
    best_payload: dict[str, Any] | None = None

    for epoch in range(int(args.num_epochs)):
        model.train()
        running_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"Train ModernBERT router epoch {epoch}", disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).float()
                if args.objective == "route_classifier":
                    loss = F.cross_entropy(logits, batch["class_targets"].long())
                else:
                    loss = F.mse_loss(logits, batch["targets"].float())
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                running_losses.append(float(loss.detach().item()))
        accelerator.wait_for_everyone()
        eval_loss, pred_rows, y_true, y_pred = _evaluate(
            accelerator,
            model,
            eval_loader,
            eval_dataset,
            route_labels,
            objective=str(args.objective),
        )
        train_loss = float(np.mean(running_losses)) if running_losses else math.nan
        if accelerator.is_main_process:
            epoch_summary = {"epoch": epoch, "train_loss": train_loss, "eval_loss": eval_loss}
            history.append(epoch_summary)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_payload = {
                    "epoch": epoch,
                    "prediction_rows": pred_rows,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
        accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return
    if best_payload is None:
        raise ValueError("No evaluation payload was captured")

    pred_rows = best_payload["prediction_rows"]
    y_true = best_payload["y_true"]
    y_pred = best_payload["y_pred"]
    route_metrics = compute_per_route_metrics(y_true, y_pred, route_labels)
    pairwise_metrics = compute_pairwise_metrics(y_true, y_pred, route_labels)
    classifier_metrics = _compute_classifier_metrics(y_true, y_pred, route_labels)
    utility_report = _compute_utility_report(pred_rows, eval_rows_source, route_labels, DEFAULT_UTILITY_LAMBDAS)

    _write_jsonl(output_dir / "eval_predictions.jsonl", pred_rows)
    _write_csv(output_dir / "route_metrics.csv", route_metrics, csv_headers_for_route_metrics())
    _write_csv(output_dir / "pairwise_metrics.csv", pairwise_metrics, csv_headers_for_pairwise_metrics())
    write_json(output_dir / "classifier_metrics.json", classifier_metrics)
    utility_headers = [
        "policy",
        "policy_type",
        "route_idx",
        "route_label",
        "lambda",
        "cost_metric",
        "mean_reward",
        "mean_cost",
        "mean_utility",
        "oracle_match_rate",
        "oracle_match_rate_margin_gt_0_05",
        "oracle_match_n_margin_gt_0_05",
    ]
    _write_csv(output_dir / "utility_vs_baselines.csv", utility_report["utility_rows"], utility_headers)
    write_json(output_dir / "utility_vs_baselines.json", utility_report)

    summary = {
        "best_epoch": int(best_payload["epoch"]),
        "best_eval_loss": float(best_eval_loss),
        "history": history,
        "route_metrics": route_metrics,
        "pairwise_metrics": pairwise_metrics,
        "classifier_metrics": classifier_metrics,
        "utility": utility_report,
        "config": config,
    }
    write_json(output_dir / "summary.json", summary)
    if args.save_model:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), output_dir / "model.pt")
        tokenizer.save_pretrained(output_dir / "tokenizer")


if __name__ == "__main__":
    main()

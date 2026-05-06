#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from pipelinerl.swe.scripts.offline_router.common import (
    compute_pairwise_metrics,
    compute_per_route_metrics,
    csv_headers_for_pairwise_metrics,
    csv_headers_for_route_metrics,
    _roc_auc_binary,
    write_json,
)
from pipelinerl.swe.scripts.offline_router.train_modernbert_router_baseline import (
    DEFAULT_UTILITY_LAMBDAS,
    _argmax_index,
    _build_input_text,
    _compute_classifier_metrics,
    _compute_utility_report,
    _load_route_labels,
    _load_split,
    _shuffle_rows,
)


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = bool(attention_mask[:, -1].sum().item() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def _collate_left_pad(
    batch: list[dict[str, Any]],
    pad_token_id: int,
    target_dim: int,
    cost_target_dim: int,
) -> dict[str, Any]:
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    targets = torch.zeros((len(batch), target_dim), dtype=torch.float32)
    output_token_targets_log = torch.zeros((len(batch), cost_target_dim), dtype=torch.float32)
    zero_reward_targets = torch.zeros((len(batch), target_dim), dtype=torch.float32)
    class_targets = torch.zeros((len(batch),), dtype=torch.long)
    row_indices = torch.zeros((len(batch),), dtype=torch.long)
    for idx, row in enumerate(batch):
        seq_len = len(row["input_ids"])
        start = max_len - seq_len
        input_ids[idx, start:] = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask[idx, start:] = torch.tensor(row["attention_mask"], dtype=torch.long)
        targets[idx] = torch.tensor(row["targets"], dtype=torch.float32)
        output_token_targets_log[idx] = torch.tensor(row["output_token_targets_log"], dtype=torch.float32)
        zero_reward_targets[idx] = torch.tensor(row["zero_reward_targets"], dtype=torch.float32)
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
        "output_token_targets_log": output_token_targets_log,
        "zero_reward_targets": zero_reward_targets,
        "class_targets": class_targets,
        "row_indices": row_indices,
    }


class RouterCostDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        route_labels: list[str],
        max_seq_length: int,
        require_cost_targets: bool,
        cost_route_idx: int,
        zero_reward_epsilon: float,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        target_dim = len(route_labels)
        for row in rows:
            targets = row.get("performance_targets")
            output_tokens = row.get("route_output_tokens")
            if not isinstance(targets, list) or len(targets) != target_dim:
                continue
            if require_cost_targets and (not isinstance(output_tokens, list) or len(output_tokens) != target_dim):
                continue
            if require_cost_targets and not 0 <= int(cost_route_idx) < target_dim:
                raise ValueError(f"cost_route_idx={cost_route_idx} is out of range for {target_dim} routes")
            try:
                target_rewards = [float(value) for value in targets]
                zero_reward_targets = [
                    1.0 if float(value) <= float(zero_reward_epsilon) else 0.0 for value in target_rewards
                ]
                output_token_targets_log = (
                    [float(math.log1p(float(output_tokens[int(cost_route_idx)])))]
                    if isinstance(output_tokens, list) and len(output_tokens) == target_dim
                    else [0.0]
                )
                problem_id = str(row.get("problem_id") or row.get("instance_id") or row.get("id"))
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
                    "output_token_targets_log": output_token_targets_log,
                    "zero_reward_targets": zero_reward_targets,
                    "class_target": _argmax_index(target_rewards),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class QwenEmbeddingRouter(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        target_dim: int,
        dropout: float,
        mlp_hidden_size: int,
        torch_dtype: torch.dtype,
        attn_implementation: str | None,
        encoder_frozen: bool,
        use_lora: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: list[str],
        gradient_checkpointing: bool,
        predict_costs: bool,
        cost_target_dim: int,
        cost_gradient_mode: str,
        predict_zero_reward_failure: bool,
    ) -> None:
        super().__init__()
        if bool(use_lora) and bool(encoder_frozen):
            raise ValueError("use_lora=true requires encoder_frozen=false")
        if cost_gradient_mode not in {"joint", "detached", "separate_adapter"}:
            raise ValueError(f"Unsupported cost_gradient_mode={cost_gradient_mode}")
        self.encoder_frozen = bool(encoder_frozen)
        self.cost_gradient_mode = str(cost_gradient_mode)
        self.target_dim = int(target_dim)
        self.reward_adapter_name = "reward_adapter"
        self.cost_adapter_name = "cost_adapter"
        model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.encoder = AutoModel.from_pretrained(model_name, **model_kwargs)
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
            if hasattr(self.encoder, "enable_input_require_grads"):
                self.encoder.enable_input_require_grads()
            if hasattr(self.encoder.config, "use_cache"):
                self.encoder.config.use_cache = False
        if use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as exc:
                raise ImportError("PEFT is required for --use-lora. Run in the pipeline-rl env.") from exc
            lora_config = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                target_modules=list(lora_target_modules),
                lora_dropout=float(lora_dropout),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.encoder = get_peft_model(self.encoder, lora_config, adapter_name=self.reward_adapter_name)
            if self.cost_gradient_mode == "separate_adapter" and predict_costs:
                self.encoder.add_adapter(self.cost_adapter_name, lora_config)
        else:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        hidden_size = int(self.encoder.config.hidden_size)
        self.predict_costs = bool(predict_costs)
        self.predict_zero_reward_failure = bool(predict_zero_reward_failure)
        self.reward_head = self._make_head(hidden_size, target_dim, dropout, mlp_hidden_size)
        self.head = self.reward_head
        self.cost_head = (
            self._make_head(hidden_size, cost_target_dim, dropout, mlp_hidden_size) if self.predict_costs else None
        )
        self.zero_reward_head = (
            self._make_head(hidden_size, target_dim, dropout, mlp_hidden_size)
            if self.predict_zero_reward_failure
            else None
        )

    @staticmethod
    def _make_head(
        hidden_size: int,
        target_dim: int,
        dropout: float,
        mlp_hidden_size: int,
    ) -> torch.nn.Module:
        if int(mlp_hidden_size) > 0:
            return torch.nn.Sequential(
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(hidden_size, int(mlp_hidden_size)),
                torch.nn.GELU(),
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(int(mlp_hidden_size), int(target_dim)),
            )
        return torch.nn.Sequential(
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(hidden_size, int(target_dim)),
        )

    def train(self, mode: bool = True) -> "QwenEmbeddingRouter":
        super().train(mode)
        if hasattr(self, "encoder") and self.encoder_frozen:
            self.encoder.eval()
        return self

    def _set_active_adapter(self, adapter_name: str) -> None:
        if hasattr(self.encoder, "set_adapter"):
            self.encoder.set_adapter(adapter_name)

    def encode_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adapter_name: str | None = None,
    ) -> torch.Tensor:
        if adapter_name is not None:
            self._set_active_adapter(adapter_name)
        if self.encoder_frozen:
            self.encoder.eval()
            with torch.no_grad():
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                pooled = _last_token_pool(outputs.last_hidden_state, attention_mask)
                return F.normalize(pooled.float(), p=2, dim=1)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = _last_token_pool(outputs.last_hidden_state, attention_mask)
            return F.normalize(pooled.float(), p=2, dim=1)

    def predict_from_embeddings(
        self,
        embeddings: torch.Tensor,
        detach_cost_embeddings: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        cost_embeddings = embeddings.detach() if detach_cost_embeddings else embeddings
        cost_logits = self.cost_head(cost_embeddings) if self.cost_head is not None else None
        zero_reward_logits = self.zero_reward_head(embeddings) if self.zero_reward_head is not None else None
        return self.reward_head(embeddings), cost_logits, zero_reward_logits

    def predict_cost_only_from_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.cost_head is None:
            raise ValueError("cost-only prediction requires a cost head")
        reward_logits = torch.zeros(
            (embeddings.shape[0], self.target_dim),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        return reward_logits, self.cost_head(embeddings), None

    def forward_cost_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.cost_head is None:
            raise ValueError("cost-only forward requires a cost head")
        adapter_name = (
            self.cost_adapter_name
            if self.cost_gradient_mode == "separate_adapter"
            else self.reward_adapter_name if hasattr(self.encoder, "set_adapter") else None
        )
        embeddings = self.encode_inputs(input_ids, attention_mask, adapter_name=adapter_name)
        return self.predict_cost_only_from_embeddings(embeddings)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        reward_embeddings = self.encode_inputs(
            input_ids,
            attention_mask,
            adapter_name=self.reward_adapter_name if hasattr(self.encoder, "set_adapter") else None,
        )
        if self.cost_head is None:
            return self.predict_from_embeddings(reward_embeddings)
        if self.cost_gradient_mode == "separate_adapter":
            cost_embeddings = self.encode_inputs(
                input_ids,
                attention_mask,
                adapter_name=self.cost_adapter_name,
            )
            zero_reward_logits = (
                self.zero_reward_head(reward_embeddings) if self.zero_reward_head is not None else None
            )
            return self.reward_head(reward_embeddings), self.cost_head(cost_embeddings), zero_reward_logits
        return self.predict_from_embeddings(
            reward_embeddings,
            detach_cost_embeddings=self.cost_gradient_mode == "detached",
        )


class PrecomputedEmbeddingDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def _collate_embeddings(batch: list[dict[str, Any]], target_dim: int, cost_target_dim: int) -> dict[str, Any]:
    embeddings = torch.stack([row["embedding"] for row in batch], dim=0).float()
    targets = torch.zeros((len(batch), target_dim), dtype=torch.float32)
    output_token_targets_log = torch.zeros((len(batch), cost_target_dim), dtype=torch.float32)
    zero_reward_targets = torch.zeros((len(batch), target_dim), dtype=torch.float32)
    class_targets = torch.zeros((len(batch),), dtype=torch.long)
    row_indices = torch.zeros((len(batch),), dtype=torch.long)
    for idx, row in enumerate(batch):
        targets[idx] = torch.tensor(row["targets"], dtype=torch.float32)
        output_token_targets_log[idx] = torch.tensor(row["output_token_targets_log"], dtype=torch.float32)
        zero_reward_targets[idx] = torch.tensor(row["zero_reward_targets"], dtype=torch.float32)
        class_targets[idx] = int(row["class_target"])
        row_indices[idx] = int(row["row_idx"])
    return {
        "problem_ids": [row["problem_id"] for row in batch],
        "datasets": [row["dataset"] for row in batch],
        "repos": [row["repo"] for row in batch],
        "languages": [row["language"] for row in batch],
        "embeddings": embeddings,
        "targets": targets,
        "output_token_targets_log": output_token_targets_log,
        "zero_reward_targets": zero_reward_targets,
        "class_targets": class_targets,
        "row_indices": row_indices,
    }


@torch.no_grad()
def _precompute_embeddings(
    accelerator: Accelerator,
    model: QwenEmbeddingRouter,
    loader: DataLoader,
    source_dataset: RouterCostDataset,
    desc: str,
) -> PrecomputedEmbeddingDataset:
    if accelerator.num_processes != 1:
        raise ValueError("Embedding precompute currently expects TRAIN_NPROC=1")
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc=desc, disable=not accelerator.is_main_process):
        input_ids = batch["input_ids"].to(accelerator.device)
        attention_mask = batch["attention_mask"].to(accelerator.device)
        embeddings = model.encode_inputs(input_ids=input_ids, attention_mask=attention_mask).detach().cpu()
        for idx in range(embeddings.shape[0]):
            source_meta = source_dataset.rows[int(batch["row_indices"][idx].item())]
            rows.append(
                {
                    "row_idx": len(rows),
                    "problem_id": source_meta["problem_id"],
                    "dataset": source_meta["dataset"],
                    "repo": source_meta["repo"],
                    "language": source_meta["language"],
                    "embedding": embeddings[idx].float(),
                    "targets": [float(value) for value in batch["targets"][idx].tolist()],
                    "output_token_targets_log": [
                        float(value) for value in batch["output_token_targets_log"][idx].tolist()
                    ],
                    "zero_reward_targets": [
                        float(value) for value in batch["zero_reward_targets"][idx].tolist()
                    ],
                    "class_target": int(batch["class_targets"][idx].item()),
                }
            )
    return PrecomputedEmbeddingDataset(rows)


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


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _compute_output_token_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    route_labels: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if y_true_log.size == 0 or y_pred_log.size == 0:
        return rows
    y_true_raw = np.expm1(y_true_log)
    y_pred_raw = np.maximum(0.0, np.expm1(y_pred_log))
    for idx, route_label in enumerate(route_labels):
        true_log = y_true_log[:, idx]
        pred_log = y_pred_log[:, idx]
        true_raw = y_true_raw[:, idx]
        pred_raw = y_pred_raw[:, idx]
        raw_err = pred_raw - true_raw
        log_err = pred_log - true_log
        rows.append(
            {
                "route_idx": idx,
                "route_label": route_label,
                "n_eval": int(true_log.shape[0]),
                "mean_true_output_tokens": float(np.mean(true_raw)),
                "mean_pred_output_tokens": float(np.mean(pred_raw)),
                "std_true_output_tokens": float(np.std(true_raw)),
                "std_pred_output_tokens": float(np.std(pred_raw)),
                "mae_output_tokens": float(np.mean(np.abs(raw_err))),
                "rmse_output_tokens": float(np.sqrt(np.mean(raw_err * raw_err))),
                "pearson_output_tokens": _safe_corr(true_raw, pred_raw),
                "mean_true_log1p_output_tokens": float(np.mean(true_log)),
                "mean_pred_log1p_output_tokens": float(np.mean(pred_log)),
                "std_true_log1p_output_tokens": float(np.std(true_log)),
                "std_pred_log1p_output_tokens": float(np.std(pred_log)),
                "mae_log1p_output_tokens": float(np.mean(np.abs(log_err))),
                "rmse_log1p_output_tokens": float(np.sqrt(np.mean(log_err * log_err))),
                "pearson_log1p_output_tokens": _safe_corr(true_log, pred_log),
            }
        )
    return rows


def _compute_zero_reward_failure_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    route_labels: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if y_true.size == 0 or y_pred_prob.size == 0:
        return rows
    eps = 1.0e-7
    clipped = np.clip(y_pred_prob, eps, 1.0 - eps)
    for idx, route_label in enumerate(route_labels):
        labels = y_true[:, idx].astype(np.int64)
        probs = clipped[:, idx]
        pred_labels = (probs >= 0.5).astype(np.int64)
        positives = int(np.sum(labels == 1))
        predicted_positives = int(np.sum(pred_labels == 1))
        true_positives = int(np.sum((labels == 1) & (pred_labels == 1)))
        rows.append(
            {
                "route_idx": idx,
                "route_label": route_label,
                "n_eval": int(labels.shape[0]),
                "positive_rate": float(np.mean(labels)),
                "mean_pred_prob": float(np.mean(probs)),
                "std_pred_prob": float(np.std(probs)),
                "bce": float(-np.mean((labels * np.log(probs)) + ((1 - labels) * np.log(1.0 - probs)))),
                "accuracy_at_0_5": float(np.mean(labels == pred_labels)),
                "precision_at_0_5": (
                    float(true_positives / predicted_positives) if predicted_positives > 0 else None
                ),
                "recall_at_0_5": float(true_positives / positives) if positives > 0 else None,
                "roc_auc": _roc_auc_binary(labels, probs),
            }
        )
    return rows


def _compute_predicted_cost_utility_report(
    prediction_rows: list[dict[str, Any]],
    eval_source_rows: list[dict[str, Any]],
    route_labels: list[str],
    lambdas: list[float],
    cost_route_idx: int,
) -> dict[str, Any]:
    target_dim = len(route_labels)
    eval_lookup: dict[str, dict[str, Any]] = {}
    for row in eval_source_rows:
        problem_id = str(row.get("problem_id") or row.get("instance_id") or row.get("id"))
        eval_lookup[f"{row.get('dataset')}::{problem_id}"] = row

    valid_examples: list[dict[str, Any]] = []
    skipped_missing_eval_row = 0
    skipped_invalid_stats = 0
    for row in prediction_rows:
        pred_rewards = row.get("pred_rewards")
        pred_output_tokens = row.get("pred_output_tokens")
        if (
            not isinstance(pred_rewards, list)
            or not isinstance(pred_output_tokens, list)
            or len(pred_rewards) != target_dim
            or len(pred_output_tokens) != 1
        ):
            skipped_invalid_stats += 1
            continue
        source_row = eval_lookup.get(f"{row.get('dataset')}::{row.get('problem_id')}")
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
            skipped_invalid_stats += 1
            continue
        valid_examples.append(
            {
                "rewards": [float(value) for value in rewards],
                "prompt_tokens": [float(value) for value in prompt_tokens],
                "output_tokens": [float(value) for value in output_tokens],
                "pred_rewards": [float(value) for value in pred_rewards],
                "pred_expert_output_tokens": max(0.0, float(pred_output_tokens[0])),
            }
        )

    utility_rows: list[dict[str, Any]] = []
    valid_count = len(valid_examples)
    for lambda_value in [float(value) for value in lambdas]:
        for cost_metric in ("output_tokens", "total_tokens"):
            route_choice_counts = [0] * target_dim
            reward_sum = 0.0
            prompt_token_sum = 0.0
            output_token_sum = 0.0
            total_token_sum = 0.0
            for example in valid_examples:
                scores = []
                for route_idx in range(target_dim):
                    pred_cost = (
                        example["pred_expert_output_tokens"]
                        if route_idx == int(cost_route_idx)
                        else example["output_tokens"][route_idx]
                    )
                    if cost_metric == "total_tokens":
                        pred_cost += example["prompt_tokens"][route_idx]
                    scores.append(example["pred_rewards"][route_idx] - (lambda_value * pred_cost))
                route_idx = _argmax_index(scores)
                route_choice_counts[route_idx] += 1
                reward_sum += example["rewards"][route_idx]
                prompt_token_sum += example["prompt_tokens"][route_idx]
                output_token_sum += example["output_tokens"][route_idx]
                total_token_sum += example["prompt_tokens"][route_idx] + example["output_tokens"][route_idx]
            mean_reward = math.nan if valid_count == 0 else reward_sum / valid_count
            mean_output_tokens = math.nan if valid_count == 0 else output_token_sum / valid_count
            mean_total_tokens = math.nan if valid_count == 0 else total_token_sum / valid_count
            mean_cost = mean_output_tokens if cost_metric == "output_tokens" else mean_total_tokens
            utility_rows.append(
                {
                    "policy": "router_predicted_cost",
                    "policy_type": "router_predicted_cost",
                    "route_idx": None,
                    "route_label": None,
                    "lambda": lambda_value,
                    "cost_metric": cost_metric,
                    "mean_reward": mean_reward,
                    "mean_cost": mean_cost,
                    "mean_utility": mean_reward - (lambda_value * mean_cost),
                    "choice_counts_by_route": {
                        str(route_labels[idx]): int(route_choice_counts[idx]) for idx in range(target_dim)
                    },
                }
            )
    return {
        "n_eval_examples": len(prediction_rows),
        "n_examples_with_utility": valid_count,
        "skipped_missing_eval_row": skipped_missing_eval_row,
        "skipped_invalid_stats": skipped_invalid_stats,
        "lambdas": [float(value) for value in lambdas],
        "route_labels": list(route_labels),
        "utility_rows": utility_rows,
    }


def _delta_loss(preds: torch.Tensor, targets: torch.Tensor, huber_delta: float) -> torch.Tensor:
    if preds.shape[1] != 2:
        raise ValueError("Delta auxiliary loss currently expects exactly two routes")
    pred_delta = preds[:, 1] - preds[:, 0]
    true_delta = targets[:, 1] - targets[:, 0]
    if float(huber_delta) > 0.0:
        return F.huber_loss(pred_delta, true_delta, delta=float(huber_delta))
    return F.mse_loss(pred_delta, true_delta)


def _compute_train_loss(
    reward_logits: torch.Tensor,
    cost_logits: torch.Tensor | None,
    zero_reward_logits: torch.Tensor | None,
    targets: torch.Tensor,
    output_token_targets_log: torch.Tensor,
    zero_reward_targets: torch.Tensor,
    class_targets: torch.Tensor,
    objective: str,
    reward_mse_weight: float,
    delta_aux_weight: float,
    delta_aux_huber_delta: float,
    predict_costs: bool,
    cost_mse_weight: float,
    cost_delta_aux_weight: float,
    predict_zero_reward_failure: bool,
    zero_reward_bce_weight: float,
) -> torch.Tensor:
    if objective == "cost_mse":
        if not predict_costs or cost_logits is None:
            raise ValueError("objective=cost_mse requires predict_costs=true and cost logits")
        if predict_zero_reward_failure:
            raise ValueError("objective=cost_mse is incompatible with zero-reward failure prediction")
        if float(cost_delta_aux_weight) != 0.0:
            raise ValueError("cost_delta_aux_weight is not supported for expert-only cost prediction")
        return float(cost_mse_weight) * F.mse_loss(cost_logits, output_token_targets_log.float())
    if objective == "route_classifier":
        reward_loss = F.cross_entropy(reward_logits, class_targets.long())
    else:
        reward_loss = F.mse_loss(reward_logits, targets.float()) * float(reward_mse_weight)
    if objective == "reward_mse_delta_aux":
        reward_loss = reward_loss + (
            float(delta_aux_weight) * _delta_loss(reward_logits, targets.float(), float(delta_aux_huber_delta))
        )
    if predict_costs:
        if cost_logits is None:
            raise ValueError("predict_costs=true but model did not return cost logits")
        reward_loss = reward_loss + (
            float(cost_mse_weight) * F.mse_loss(cost_logits, output_token_targets_log.float())
        )
        if float(cost_delta_aux_weight) != 0.0:
            raise ValueError("cost_delta_aux_weight is not supported for expert-only cost prediction")
    if predict_zero_reward_failure:
        if zero_reward_logits is None:
            raise ValueError("predict_zero_reward_failure=true but model did not return zero-reward logits")
        reward_loss = reward_loss + (
            float(zero_reward_bce_weight)
            * F.binary_cross_entropy_with_logits(zero_reward_logits, zero_reward_targets.float())
        )
    return reward_loss


def _predict_from_batch(
    model: QwenEmbeddingRouter,
    batch: dict[str, Any],
    cost_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if "embeddings" in batch:
        if cost_only:
            return model.predict_cost_only_from_embeddings(batch["embeddings"].float())
        return model.predict_from_embeddings(
            batch["embeddings"].float(),
            detach_cost_embeddings=model.cost_gradient_mode == "detached",
        )
    if cost_only:
        return model.forward_cost_only(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])


@torch.no_grad()
def _evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loader: DataLoader,
    eval_dataset: RouterCostDataset,
    route_labels: list[str],
    objective: str,
    predict_costs: bool,
    predict_zero_reward_failure: bool,
) -> tuple[float, list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    rows: list[dict[str, Any]] = []
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    cost_true_chunks: list[np.ndarray] = []
    cost_pred_chunks: list[np.ndarray] = []
    zero_reward_true_chunks: list[np.ndarray] = []
    zero_reward_pred_chunks: list[np.ndarray] = []
    cost_only = objective == "cost_mse"
    for batch in tqdm(loader, desc="Eval Qwen embedding router", disable=not accelerator.is_main_process):
        logits, cost_logits, zero_reward_logits = _predict_from_batch(model, batch, cost_only=cost_only)
        logits = logits.float()
        targets = batch["targets"].float()
        if objective == "cost_mse":
            if cost_logits is None:
                raise ValueError("objective=cost_mse requires cost logits")
            loss = F.mse_loss(cost_logits.float(), batch["output_token_targets_log"].float(), reduction="sum")
            preds = logits
        elif objective == "route_classifier":
            loss = F.cross_entropy(logits, batch["class_targets"].long(), reduction="sum")
            preds = torch.softmax(logits, dim=-1)
        else:
            loss = F.mse_loss(logits, targets, reduction="sum")
            preds = logits
            if predict_costs and cost_logits is not None:
                loss = loss + F.mse_loss(cost_logits.float(), batch["output_token_targets_log"].float(), reduction="sum")
        gathered_loss = accelerator.gather_for_metrics(loss.detach().reshape(1)).detach().cpu()
        gathered_preds = accelerator.gather_for_metrics(preds).detach().cpu()
        gathered_targets = accelerator.gather_for_metrics(targets).detach().cpu()
        if predict_costs:
            if cost_logits is None:
                raise ValueError("predict_costs=true but model did not return cost logits")
            gathered_cost_preds = accelerator.gather_for_metrics(cost_logits.float()).detach().cpu()
            gathered_cost_targets = accelerator.gather_for_metrics(batch["output_token_targets_log"].float()).detach().cpu()
        else:
            gathered_cost_preds = torch.empty((gathered_preds.shape[0], len(route_labels)))
            gathered_cost_targets = torch.empty((gathered_preds.shape[0], len(route_labels)))
        if predict_zero_reward_failure:
            if zero_reward_logits is None:
                raise ValueError("predict_zero_reward_failure=true but model did not return zero-reward logits")
            gathered_zero_reward_preds = accelerator.gather_for_metrics(
                torch.sigmoid(zero_reward_logits.float())
            ).detach().cpu()
            gathered_zero_reward_targets = accelerator.gather_for_metrics(
                batch["zero_reward_targets"].float()
            ).detach().cpu()
        else:
            gathered_zero_reward_preds = torch.empty((gathered_preds.shape[0], len(route_labels)))
            gathered_zero_reward_targets = torch.empty((gathered_preds.shape[0], len(route_labels)))
        gathered_indices = accelerator.gather_for_metrics(batch["row_indices"]).detach().cpu().tolist()
        if accelerator.is_main_process:
            total_loss += float(torch.sum(gathered_loss).item())
            total_examples += int(gathered_targets.shape[0])
        for idx in range(gathered_preds.shape[0]):
            source_meta = eval_dataset.rows[int(gathered_indices[idx])]
            rows.append(
                {
                    "problem_id": source_meta["problem_id"],
                    "dataset": source_meta["dataset"],
                    "repo": source_meta["repo"],
                    "language": source_meta["language"],
                    "true_rewards": [float(value) for value in gathered_targets[idx].tolist()],
                    "pred_rewards": [float(value) for value in gathered_preds[idx].tolist()],
                    "true_output_tokens": [
                        float(math.expm1(value)) for value in gathered_cost_targets[idx].tolist()
                    ] if predict_costs else None,
                    "pred_output_tokens": [
                        max(0.0, float(math.expm1(value))) for value in gathered_cost_preds[idx].tolist()
                    ] if predict_costs else None,
                    "true_output_tokens_log": [
                        float(value) for value in gathered_cost_targets[idx].tolist()
                    ] if predict_costs else None,
                    "pred_output_tokens_log": [
                        float(value) for value in gathered_cost_preds[idx].tolist()
                    ] if predict_costs else None,
                    "true_zero_reward_failure": [
                        float(value) for value in gathered_zero_reward_targets[idx].tolist()
                    ] if predict_zero_reward_failure else None,
                    "pred_zero_reward_failure_probs": [
                        float(value) for value in gathered_zero_reward_preds[idx].tolist()
                    ] if predict_zero_reward_failure else None,
                    "route_labels": list(route_labels),
                }
            )
        y_true_chunks.append(gathered_targets.numpy())
        y_pred_chunks.append(gathered_preds.numpy())
        if predict_costs:
            cost_true_chunks.append(gathered_cost_targets.numpy())
            cost_pred_chunks.append(gathered_cost_preds.numpy())
        if predict_zero_reward_failure:
            zero_reward_true_chunks.append(gathered_zero_reward_targets.numpy())
            zero_reward_pred_chunks.append(gathered_zero_reward_preds.numpy())

    if not accelerator.is_main_process:
        empty = np.empty((0, len(route_labels)))
        return math.nan, [], empty, empty, empty, empty, empty, empty
    y_true = np.concatenate(y_true_chunks, axis=0) if y_true_chunks else np.empty((0, len(route_labels)))
    y_pred = np.concatenate(y_pred_chunks, axis=0) if y_pred_chunks else np.empty((0, len(route_labels)))
    y_cost_true = (
        np.concatenate(cost_true_chunks, axis=0) if cost_true_chunks else np.empty((0, len(route_labels)))
    )
    y_cost_pred = (
        np.concatenate(cost_pred_chunks, axis=0) if cost_pred_chunks else np.empty((0, len(route_labels)))
    )
    y_zero_reward_true = (
        np.concatenate(zero_reward_true_chunks, axis=0)
        if zero_reward_true_chunks
        else np.empty((0, len(route_labels)))
    )
    y_zero_reward_pred = (
        np.concatenate(zero_reward_pred_chunks, axis=0)
        if zero_reward_pred_chunks
        else np.empty((0, len(route_labels)))
    )
    if objective == "route_classifier":
        eval_loss = float(total_loss / total_examples) if total_examples > 0 else math.nan
    elif objective == "cost_mse":
        total_values = total_examples * (int(y_cost_true.shape[1]) if y_cost_true.ndim == 2 else 1)
        eval_loss = float(total_loss / total_values) if total_values > 0 else math.nan
    else:
        total_values = total_examples * len(route_labels)
        eval_loss = float(total_loss / total_values) if total_values > 0 else math.nan
    return eval_loss, rows, y_true, y_pred, y_cost_true, y_cost_pred, y_zero_reward_true, y_zero_reward_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument(
        "--objective",
        choices=["reward_mse", "reward_mse_delta_aux", "route_classifier", "cost_mse"],
        default="reward_mse",
    )
    parser.add_argument("--max-seq-length", type=int, default=24000)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp-hidden-size", type=int, default=1024)
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32", "float32"], default="bf16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--precompute-embeddings", action="store_true")
    parser.add_argument("--encoder-frozen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--reward-mse-weight", type=float, default=1.0)
    parser.add_argument("--delta-aux-weight", type=float, default=1.0)
    parser.add_argument("--delta-aux-huber-delta", type=float, default=0.0)
    parser.add_argument("--predict-costs", action="store_true")
    parser.add_argument("--cost-route-idx", type=int, default=1)
    parser.add_argument(
        "--cost-gradient-mode",
        choices=["joint", "detached", "separate_adapter"],
        default="joint",
    )
    parser.add_argument("--cost-mse-weight", type=float, default=1.0)
    parser.add_argument("--cost-delta-aux-weight", type=float, default=0.0)
    parser.add_argument("--predict-zero-reward-failure", action="store_true")
    parser.add_argument("--zero-reward-epsilon", type=float, default=0.0)
    parser.add_argument("--zero-reward-bce-weight", type=float, default=1.0)
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)

    kwargs_handlers = []
    if args.ddp_find_unused_parameters:
        kwargs_handlers.append(DistributedDataParallelKwargs(find_unused_parameters=True))
    accelerator = Accelerator(
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        kwargs_handlers=kwargs_handlers,
    )
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    route_labels = _load_route_labels(dataset_dir)
    target_dim = len(route_labels)
    if args.objective == "reward_mse_delta_aux" and target_dim != 2:
        raise ValueError("reward_mse_delta_aux currently expects exactly two routes")
    if args.objective == "cost_mse" and not args.predict_costs:
        raise ValueError("objective=cost_mse requires --predict-costs")
    if args.objective == "cost_mse" and args.predict_zero_reward_failure:
        raise ValueError("objective=cost_mse is incompatible with --predict-zero-reward-failure")
    if args.objective == "cost_mse" and args.cost_gradient_mode != "joint":
        raise ValueError("objective=cost_mse expects --cost-gradient-mode=joint")
    if args.precompute_embeddings and not args.encoder_frozen:
        raise ValueError("--precompute-embeddings requires --encoder-frozen")
    if args.use_lora and args.encoder_frozen:
        raise ValueError("--use-lora requires --no-encoder-frozen")
    cost_route_idx = int(args.cost_route_idx)
    if bool(args.predict_costs) and not 0 <= cost_route_idx < target_dim:
        raise ValueError(f"--cost-route-idx={cost_route_idx} is out of range for {target_dim} routes")
    if args.cost_gradient_mode == "separate_adapter" and not args.use_lora:
        raise ValueError("--cost-gradient-mode=separate_adapter requires --use-lora")
    if args.precompute_embeddings and args.cost_gradient_mode == "separate_adapter":
        raise ValueError("--precompute-embeddings is incompatible with separate cost adapters")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    train_rows = _shuffle_rows(list(_load_split(dataset_dir, "train")), args.max_train_rows, args.seed)
    eval_rows_source = _shuffle_rows(list(_load_split(dataset_dir, "eval")), args.max_eval_rows, args.seed + 1)
    train_dataset = RouterCostDataset(
        train_rows,
        tokenizer,
        route_labels,
        int(args.max_seq_length),
        require_cost_targets=bool(args.predict_costs),
        cost_route_idx=cost_route_idx,
        zero_reward_epsilon=float(args.zero_reward_epsilon),
    )
    eval_dataset = RouterCostDataset(
        eval_rows_source,
        tokenizer,
        route_labels,
        int(args.max_seq_length),
        require_cost_targets=bool(args.predict_costs),
        cost_route_idx=cost_route_idx,
        zero_reward_epsilon=float(args.zero_reward_epsilon),
    )
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError(f"Prepared empty dataset train={len(train_dataset)} eval={len(eval_dataset)}")

    cost_target_dim = 1
    token_collate_fn = lambda batch: _collate_left_pad(
        batch,
        pad_token_id=int(pad_token_id),
        target_dim=target_dim,
        cost_target_dim=cost_target_dim,
    )
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=token_collate_fn,
        num_workers=0,
    )
    eval_loader: DataLoader = DataLoader(
        eval_dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        collate_fn=token_collate_fn,
        num_workers=0,
    )

    model = QwenEmbeddingRouter(
        args.model_name,
        target_dim=target_dim,
        dropout=float(args.dropout),
        mlp_hidden_size=int(args.mlp_hidden_size),
        torch_dtype=_dtype_from_name(str(args.torch_dtype)),
        attn_implementation=str(args.attn_implementation) if args.attn_implementation else None,
        encoder_frozen=bool(args.encoder_frozen),
        use_lora=bool(args.use_lora),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        lora_target_modules=[module.strip() for module in str(args.lora_target_modules).split(",") if module.strip()],
        gradient_checkpointing=bool(args.gradient_checkpointing),
        predict_costs=bool(args.predict_costs),
        cost_target_dim=cost_target_dim,
        cost_gradient_mode=str(args.cost_gradient_mode),
        predict_zero_reward_failure=bool(args.predict_zero_reward_failure),
    )
    if args.objective == "cost_mse":
        for parameter in model.reward_head.parameters():
            parameter.requires_grad_(False)
        if model.zero_reward_head is not None:
            for parameter in model.zero_reward_head.parameters():
                parameter.requires_grad_(False)
    if args.precompute_embeddings:
        model.to(accelerator.device)
        train_embed_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            collate_fn=token_collate_fn,
            num_workers=0,
        )
        train_dataset = _precompute_embeddings(
            accelerator, model, train_embed_loader, train_dataset, "Precompute train Qwen embeddings"
        )
        eval_dataset = _precompute_embeddings(
            accelerator, model, eval_loader, eval_dataset, "Precompute eval Qwen embeddings"
        )
        del model.encoder
        torch.cuda.empty_cache()
        embedding_collate_fn = lambda batch: _collate_embeddings(
            batch,
            target_dim=target_dim,
            cost_target_dim=cost_target_dim,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            collate_fn=embedding_collate_fn,
            num_workers=0,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            collate_fn=embedding_collate_fn,
            num_workers=0,
        )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(args.lr), weight_decay=float(args.weight_decay))
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
        "dropout": float(args.dropout),
        "mlp_hidden_size": int(args.mlp_hidden_size),
        "torch_dtype": str(args.torch_dtype),
        "attn_implementation": str(args.attn_implementation),
        "precompute_embeddings": bool(args.precompute_embeddings),
        "encoder_frozen": bool(args.encoder_frozen),
        "use_lora": bool(args.use_lora),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_target_modules": [module.strip() for module in str(args.lora_target_modules).split(",") if module.strip()],
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "reward_mse_weight": float(args.reward_mse_weight),
        "delta_aux_weight": float(args.delta_aux_weight),
        "delta_aux_huber_delta": float(args.delta_aux_huber_delta),
        "predict_costs": bool(args.predict_costs),
        "cost_target": "log1p_route_output_tokens",
        "cost_route_idx": cost_route_idx,
        "cost_route_label": route_labels[cost_route_idx] if bool(args.predict_costs) else None,
        "cost_gradient_mode": str(args.cost_gradient_mode),
        "cost_mse_weight": float(args.cost_mse_weight),
        "cost_delta_aux_weight": float(args.cost_delta_aux_weight),
        "predict_zero_reward_failure": bool(args.predict_zero_reward_failure),
        "zero_reward_failure_target": "reward <= zero_reward_epsilon",
        "zero_reward_epsilon": float(args.zero_reward_epsilon),
        "zero_reward_bce_weight": float(args.zero_reward_bce_weight),
        "ddp_find_unused_parameters": bool(args.ddp_find_unused_parameters),
    }
    if accelerator.is_main_process:
        write_json(output_dir / "train_config.json", config)

    history: list[dict[str, Any]] = []
    best_eval_loss = float("inf")
    best_payload: dict[str, Any] | None = None

    for epoch in range(int(args.num_epochs)):
        model.train()
        running_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"Train Qwen embedding router epoch {epoch}", disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                reward_logits, cost_logits, zero_reward_logits = _predict_from_batch(
                    model,
                    batch,
                    cost_only=args.objective == "cost_mse",
                )
                loss = _compute_train_loss(
                    reward_logits=reward_logits.float(),
                    cost_logits=cost_logits.float() if cost_logits is not None else None,
                    zero_reward_logits=zero_reward_logits.float() if zero_reward_logits is not None else None,
                    targets=batch["targets"].float(),
                    output_token_targets_log=batch["output_token_targets_log"].float(),
                    zero_reward_targets=batch["zero_reward_targets"].float(),
                    class_targets=batch["class_targets"],
                    objective=str(args.objective),
                    reward_mse_weight=float(args.reward_mse_weight),
                    delta_aux_weight=float(args.delta_aux_weight),
                    delta_aux_huber_delta=float(args.delta_aux_huber_delta),
                    predict_costs=bool(args.predict_costs),
                    cost_mse_weight=float(args.cost_mse_weight),
                    cost_delta_aux_weight=float(args.cost_delta_aux_weight),
                    predict_zero_reward_failure=bool(args.predict_zero_reward_failure),
                    zero_reward_bce_weight=float(args.zero_reward_bce_weight),
                )
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                running_losses.append(float(loss.detach().item()))
        accelerator.wait_for_everyone()
        (
            eval_loss,
            pred_rows,
            y_true,
            y_pred,
            y_cost_true,
            y_cost_pred,
            y_zero_reward_true,
            y_zero_reward_pred,
        ) = _evaluate(
            accelerator,
            model,
            eval_loader,
            eval_dataset,
            route_labels,
            objective=str(args.objective),
            predict_costs=bool(args.predict_costs),
            predict_zero_reward_failure=bool(args.predict_zero_reward_failure),
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
                    "y_cost_true": y_cost_true,
                    "y_cost_pred": y_cost_pred,
                    "y_zero_reward_true": y_zero_reward_true,
                    "y_zero_reward_pred": y_zero_reward_pred,
                }
        accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return
    if best_payload is None:
        raise ValueError("No evaluation payload was captured")

    pred_rows = best_payload["prediction_rows"]
    y_true = best_payload["y_true"]
    y_pred = best_payload["y_pred"]
    y_cost_true = best_payload["y_cost_true"]
    y_cost_pred = best_payload["y_cost_pred"]
    y_zero_reward_true = best_payload["y_zero_reward_true"]
    y_zero_reward_pred = best_payload["y_zero_reward_pred"]
    route_metrics = compute_per_route_metrics(y_true, y_pred, route_labels)
    pairwise_metrics = compute_pairwise_metrics(y_true, y_pred, route_labels)
    classifier_metrics = _compute_classifier_metrics(y_true, y_pred, route_labels)
    utility_report = _compute_utility_report(pred_rows, eval_rows_source, route_labels, DEFAULT_UTILITY_LAMBDAS)
    cost_route_labels = [route_labels[cost_route_idx]] if args.predict_costs else []
    cost_metrics = _compute_output_token_metrics(y_cost_true, y_cost_pred, cost_route_labels) if args.predict_costs else []
    zero_reward_failure_metrics = (
        _compute_zero_reward_failure_metrics(y_zero_reward_true, y_zero_reward_pred, route_labels)
        if args.predict_zero_reward_failure
        else []
    )
    predicted_cost_utility_report = (
        _compute_predicted_cost_utility_report(
            pred_rows,
            eval_rows_source,
            route_labels,
            DEFAULT_UTILITY_LAMBDAS,
            cost_route_idx=cost_route_idx,
        )
        if args.predict_costs
        else None
    )

    _write_jsonl(output_dir / "eval_predictions.jsonl", pred_rows)
    _write_csv(output_dir / "route_metrics.csv", route_metrics, csv_headers_for_route_metrics())
    _write_csv(output_dir / "pairwise_metrics.csv", pairwise_metrics, csv_headers_for_pairwise_metrics())
    write_json(output_dir / "classifier_metrics.json", classifier_metrics)
    if args.predict_costs:
        cost_headers = [
            "route_idx",
            "route_label",
            "n_eval",
            "mean_true_output_tokens",
            "mean_pred_output_tokens",
            "std_true_output_tokens",
            "std_pred_output_tokens",
            "mae_output_tokens",
            "rmse_output_tokens",
            "pearson_output_tokens",
            "mean_true_log1p_output_tokens",
            "mean_pred_log1p_output_tokens",
            "std_true_log1p_output_tokens",
            "std_pred_log1p_output_tokens",
            "mae_log1p_output_tokens",
            "rmse_log1p_output_tokens",
            "pearson_log1p_output_tokens",
        ]
        _write_csv(output_dir / "cost_metrics.csv", cost_metrics, cost_headers)
    if args.predict_zero_reward_failure:
        zero_reward_headers = [
            "route_idx",
            "route_label",
            "n_eval",
            "positive_rate",
            "mean_pred_prob",
            "std_pred_prob",
            "bce",
            "accuracy_at_0_5",
            "precision_at_0_5",
            "recall_at_0_5",
            "roc_auc",
        ]
        _write_csv(output_dir / "zero_reward_failure_metrics.csv", zero_reward_failure_metrics, zero_reward_headers)
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
    ]
    _write_csv(output_dir / "utility_vs_baselines.csv", utility_report["utility_rows"], utility_headers)
    write_json(output_dir / "utility_vs_baselines.json", utility_report)
    if predicted_cost_utility_report is not None:
        predicted_cost_headers = utility_headers + ["choice_counts_by_route"]
        _write_csv(
            output_dir / "utility_with_predicted_costs.csv",
            predicted_cost_utility_report["utility_rows"],
            predicted_cost_headers,
        )
        write_json(output_dir / "utility_with_predicted_costs.json", predicted_cost_utility_report)

    summary = {
        "best_epoch": int(best_payload["epoch"]),
        "best_eval_loss": float(best_eval_loss),
        "history": history,
        "route_metrics": route_metrics,
        "pairwise_metrics": pairwise_metrics,
        "classifier_metrics": classifier_metrics,
        "cost_metrics": cost_metrics,
        "zero_reward_failure_metrics": zero_reward_failure_metrics,
        "utility": utility_report,
        "utility_with_predicted_costs": predicted_cost_utility_report,
        "config": config,
    }
    write_json(output_dir / "summary.json", summary)
    if args.save_model:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.reward_head.state_dict(), output_dir / "reward_head.pt")
        if unwrapped.cost_head is not None:
            torch.save(unwrapped.cost_head.state_dict(), output_dir / "cost_head.pt")
        if unwrapped.zero_reward_head is not None:
            torch.save(unwrapped.zero_reward_head.state_dict(), output_dir / "zero_reward_head.pt")
        if hasattr(unwrapped, "encoder") and hasattr(unwrapped.encoder, "save_pretrained"):
            unwrapped.encoder.save_pretrained(output_dir / "encoder")
        tokenizer.save_pretrained(output_dir / "tokenizer")


if __name__ == "__main__":
    main()

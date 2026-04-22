#!/usr/bin/env python
import csv
import glob
import json
import logging
import math
import os
import random
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import DataLoaderConfiguration
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup
import wandb
import time

from pipelinerl.finetune.checkpoints import save_model_and_tokenizer, save_model_only
from pipelinerl.finetune.context import accelerator_is_initialized, configure_accelerator, get_accelerator
from pipelinerl.finetune.lora import has_lora_checkpoint, lora_load, prepare_lora_model
from pipelinerl.finetune.value_model import AutoModelForCausalLMWithValueHead
from pipelinerl.utils import init_wandb
from pipelinerl.swe.scripts.offline_router.common import (
    compute_pairwise_metrics,
    compute_per_route_metrics,
    configure_router_training_mode,
    count_parameters,
    csv_headers_for_pairwise_metrics,
    csv_headers_for_route_metrics,
    problem_id_from_item,
    sanitize_for_json,
    tokenize_prompt_completion,
    write_json,
)

logger = logging.getLogger(__name__)

TEXT_REWARD_SUPERVISION_MODES = {"text_reward_vector", "text_reward_scalar", "text_reward_bin"}
TEXT_PAIRWISE_SUPERVISION_MODES = {"text_pairwise_sign"}
TEXT_LM_SUPERVISION_MODES = TEXT_REWARD_SUPERVISION_MODES | TEXT_PAIRWISE_SUPERVISION_MODES
DEFAULT_UTILITY_LAMBDAS = [0.0, 1.0e-5, 2.0e-5, 5.0e-5, 1.0e-4, 2.0e-4]


def get_world_size() -> int:
    return int(get_accelerator().state.num_processes)


def get_rank() -> int:
    return int(get_accelerator().process_index)


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return get_world_size() > 1


def is_main_process() -> bool:
    return bool(get_accelerator().is_main_process)


def barrier_if_distributed() -> None:
    if is_distributed():
        get_accelerator().wait_for_everyone()


def _set_training_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return get_accelerator().device
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cast_auxiliary_heads_to_fp32(
    model: AutoModelForCausalLMWithValueHead,
    device: torch.device,
) -> None:
    model.value_head = model.value_head.to(device=device, dtype=torch.float32)
    model.performance_value_head = model.performance_value_head.to(device=device, dtype=torch.float32)


def _log_auxiliary_head_dtypes(prefix: str, model: Any) -> None:
    unwrapped_model = get_accelerator().unwrap_model(model)
    if not hasattr(unwrapped_model, "value_head") or not hasattr(unwrapped_model, "performance_value_head"):
        return
    logger.info(
        "Offline router %s head dtypes: value_head=%s performance_value_head=%s",
        prefix,
        next(unwrapped_model.value_head.parameters()).dtype,
        next(unwrapped_model.performance_value_head.parameters()).dtype,
    )


def _maybe_cuda_synchronize() -> None:
    device = get_accelerator().device
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)


def _maybe_reset_cuda_peak_memory() -> None:
    device = get_accelerator().device
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats(device)


def _cuda_memory_snapshot() -> tuple[float | None, float | None, float | None]:
    device = get_accelerator().device
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None, None
    mib = float(1024**2)
    return (
        float(torch.cuda.memory_allocated(device) / mib),
        float(torch.cuda.memory_reserved(device) / mib),
        float(torch.cuda.max_memory_allocated(device) / mib),
    )


def _log_text_train_step_debug(
    enabled: bool,
    epoch: int,
    step_idx: int,
    total_steps: int,
    phase: str,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor | None = None,
) -> None:
    if not enabled:
        return
    _maybe_cuda_synchronize()
    if phase == "before_forward":
        _maybe_reset_cuda_peak_memory()
    seq_lens = batch["attention_mask"].sum(dim=1).detach().cpu().tolist()
    target_tokens = (batch["labels"] != -100).sum(dim=1).detach().cpu().tolist()
    problem_ids = list(batch["problem_ids"])
    lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else math.nan
    loss_value = float(loss.detach().float().item()) if loss is not None else math.nan
    alloc_mb, reserved_mb, max_alloc_mb = _cuda_memory_snapshot()
    logger.info(
        "Offline router text debug epoch=%d rank=%d step=%d/%d phase=%s problems=%s seq_lens=%s target_tokens=%s loss=%.6f lr=%.2e alloc_mb=%s reserved_mb=%s max_alloc_mb=%s",
        epoch,
        get_rank(),
        step_idx,
        total_steps,
        phase,
        problem_ids,
        seq_lens,
        target_tokens,
        loss_value,
        lr,
        "n/a" if alloc_mb is None else f"{alloc_mb:.1f}",
        "n/a" if reserved_mb is None else f"{reserved_mb:.1f}",
        "n/a" if max_alloc_mb is None else f"{max_alloc_mb:.1f}",
    )


def _build_optimizer(
    model: torch.nn.Module,
    train_cfg: Any,
    supervision_mode: str,
    text_reward_cfg: Any,
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    assigned_param_ids: set[int] = set()
    param_groups: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    group_weight_decay = 0.0

    def _add_group(name: str, params: Any, lr: float) -> None:
        weight_decay = group_weight_decay
        selected = [param for param in params if param.requires_grad and id(param) not in assigned_param_ids]
        if not selected:
            return
        assigned_param_ids.update(id(param) for param in selected)
        param_groups.append(
            {
                "name": name,
                "params": selected,
                "lr": float(lr),
                "weight_decay": weight_decay,
            }
        )
        group_summaries.append(
            {
                "name": name,
                "lr": float(lr),
                "parameter_count": int(sum(param.numel() for param in selected)),
                "tensor_count": int(len(selected)),
            }
        )

    if supervision_mode in TEXT_LM_SUPERVISION_MODES:
        default_lr = float(text_reward_cfg.get("lr", 5.0e-6))
        group_weight_decay = float(text_reward_cfg.get("weight_decay", 0.01))
        _add_group("lora_adapters", model.parameters(), default_lr)
        _add_group(
            "other_trainable",
            (param for param in model.parameters() if param.requires_grad and id(param) not in assigned_param_ids),
            default_lr,
        )
    else:
        default_lr = float(train_cfg.get("lr", 1.0e-4))
        head_lr = float(train_cfg.get("head_lr", default_lr))
        trunk_lr = float(train_cfg.get("trunk_lr", default_lr))
        group_weight_decay = float(train_cfg.get("weight_decay", 0.0))
        _add_group("performance_value_head", model.performance_value_head.parameters(), head_lr)
        _add_group("pretrained_model", model.pretrained_model.parameters(), trunk_lr)
        _add_group(
            "other_trainable",
            (param for param in model.parameters() if param.requires_grad and id(param) not in assigned_param_ids),
            default_lr,
        )

    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer construction")

    optimizer = torch.optim.AdamW(param_groups)
    return optimizer, group_summaries


def _prediction_row_key(row: dict[str, Any]) -> str:
    dataset = row.get("dataset")
    problem_id = row.get("problem_id")
    if dataset is None and problem_id is None:
        return json.dumps(row, sort_keys=True)
    route_idx = row.get("route_idx")
    if route_idx is not None:
        return f"{dataset}::{problem_id}::route_idx={route_idx}"
    route_label = row.get("route_label")
    if route_label is not None:
        return f"{dataset}::{problem_id}::route_label={route_label}"
    return f"{dataset}::{problem_id}"


def _prediction_problem_key(row: dict[str, Any]) -> str:
    dataset = row.get("dataset")
    problem_id = row.get("problem_id")
    if dataset is None and problem_id is None:
        return json.dumps(row, sort_keys=True)
    return f"{dataset}::{problem_id}"


def _load_split(dataset_dir: Path, split_name: str):
    files = sorted(glob.glob(str(dataset_dir / split_name / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet shards found for split={split_name} in {dataset_dir / split_name}")
    return load_dataset("parquet", data_files={split_name: files})[split_name]


def _shuffle_and_truncate_dataset(dataset: Any, max_rows: int | None, seed: int, split_name: str) -> Any:
    if not max_rows:
        return dataset
    requested_rows = min(int(max_rows), len(dataset))
    if requested_rows <= 0:
        return dataset.select(range(0))
    logger.info(
        "Offline router randomly sampling split=%s rows=%d/%d with seed=%d before truncation",
        split_name,
        requested_rows,
        len(dataset),
        seed,
    )
    shuffled = dataset.shuffle(seed=int(seed))
    return shuffled.select(range(requested_rows))


def _pairwise_delta_sign(row: dict[str, Any]) -> int | None:
    targets = row.get("performance_targets")
    if not isinstance(targets, list) or len(targets) < 2:
        return None
    try:
        primary = float(targets[0])
        expert = float(targets[1])
    except (TypeError, ValueError):
        return None
    delta = primary - expert
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def _sample_indices_with_optional_replacement(
    indices: list[int],
    target_count: int,
    rng: random.Random,
) -> tuple[list[int], bool]:
    if target_count <= 0 or not indices:
        return [], False
    if target_count <= len(indices):
        return list(indices[:target_count]), False
    result = list(indices)
    while len(result) < target_count:
        result.append(rng.choice(indices))
    return result, True


def _balanced_pairwise_sign_dataset(
    dataset: Any,
    max_rows: int | None,
    seed: int,
    split_name: str,
    oversample: bool,
) -> tuple[Any, dict[str, Any]]:
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    tied_indices: list[int] = []
    invalid_rows = 0
    for idx, row in enumerate(dataset):
        sign = _pairwise_delta_sign(row)
        if sign is None:
            invalid_rows += 1
        elif sign > 0:
            positive_indices.append(idx)
        elif sign < 0:
            negative_indices.append(idx)
        else:
            tied_indices.append(idx)

    rng = random.Random(int(seed))
    rng.shuffle(positive_indices)
    rng.shuffle(negative_indices)

    requested_rows = int(max_rows) if max_rows else None
    if requested_rows is None:
        if oversample:
            requested_rows = len(positive_indices) + len(negative_indices)
        else:
            requested_rows = 2 * min(len(positive_indices), len(negative_indices))
    requested_rows = max(0, int(requested_rows))
    if requested_rows % 2 == 1:
        requested_rows -= 1

    available_balanced_rows = 2 * min(len(positive_indices), len(negative_indices))
    per_class_target = requested_rows // 2
    positive_selected, positive_used_replacement = _sample_indices_with_optional_replacement(
        positive_indices,
        per_class_target,
        rng,
    ) if oversample else (list(positive_indices[: min(per_class_target, len(positive_indices))]), False)
    negative_selected, negative_used_replacement = _sample_indices_with_optional_replacement(
        negative_indices,
        per_class_target,
        rng,
    ) if oversample else (list(negative_indices[: min(per_class_target, len(negative_indices))]), False)

    final_per_class = min(len(positive_selected), len(negative_selected))
    selected_indices = positive_selected[:final_per_class] + negative_selected[:final_per_class]
    rng.shuffle(selected_indices)

    if not oversample and 2 * final_per_class < requested_rows:
        logger.warning(
            "Offline router could not satisfy balanced %s sampling for split=%s rows=%d because only %d positive and %d negative rows are available; using rows=%d instead",
            "oversample" if oversample else "undersample",
            split_name,
            requested_rows,
            len(positive_indices),
            len(negative_indices),
            2 * final_per_class,
        )

    logger.info(
        "Offline router pairwise-sign-balanced sampling split=%s oversample=%s rows=%d/%d seed=%d positives=%d negatives=%d ties=%d invalid=%d available_balanced_rows=%d",
        split_name,
        oversample,
        2 * final_per_class,
        len(dataset),
        seed,
        len(positive_indices),
        len(negative_indices),
        len(tied_indices),
        invalid_rows,
        available_balanced_rows,
    )
    if positive_used_replacement or negative_used_replacement:
        logger.info(
            "Offline router pairwise-sign-balanced sampling reused minority-class rows split=%s positive_replacement=%s negative_replacement=%s",
            split_name,
            positive_used_replacement,
            negative_used_replacement,
        )

    summary = {
        "strategy": "pairwise_sign_balanced_oversample" if oversample else "pairwise_sign_balanced_undersample",
        "seed": int(seed),
        "requested_rows": int(requested_rows),
        "selected_rows": int(2 * final_per_class),
        "positive_available": int(len(positive_indices)),
        "negative_available": int(len(negative_indices)),
        "tie_available": int(len(tied_indices)),
        "invalid_available": int(invalid_rows),
        "available_balanced_rows_without_replacement": int(available_balanced_rows),
        "positive_selected": int(final_per_class),
        "negative_selected": int(final_per_class),
        "positive_used_replacement": bool(positive_used_replacement),
        "negative_used_replacement": bool(negative_used_replacement),
    }
    return dataset.select(selected_indices), summary


def _sample_train_dataset(
    dataset: Any,
    max_rows: int | None,
    seed: int,
    strategy: str,
) -> tuple[Any, dict[str, Any]]:
    if strategy == "random":
        sampled = _shuffle_and_truncate_dataset(dataset, max_rows=max_rows, seed=seed, split_name="train")
        requested_rows = min(int(max_rows), len(dataset)) if max_rows else len(dataset)
        return sampled, {
            "strategy": "random",
            "seed": int(seed),
            "requested_rows": int(requested_rows),
            "selected_rows": int(len(sampled)),
        }
    if strategy == "pairwise_sign_balanced_undersample":
        return _balanced_pairwise_sign_dataset(
            dataset,
            max_rows=max_rows,
            seed=seed,
            split_name="train",
            oversample=False,
        )
    if strategy == "pairwise_sign_balanced_oversample":
        return _balanced_pairwise_sign_dataset(
            dataset,
            max_rows=max_rows,
            seed=seed,
            split_name="train",
            oversample=True,
        )
    raise ValueError(f"Unsupported offline_router.train.train_sampling_strategy: {strategy}")


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _normalize_utility_lambdas(values: Any) -> list[float]:
    if values is None:
        return [float(value) for value in DEFAULT_UTILITY_LAMBDAS]
    if isinstance(values, (int, float)):
        return [float(values)]
    normalized: list[float] = []
    for value in values:
        normalized.append(float(value))
    return normalized or [float(value) for value in DEFAULT_UTILITY_LAMBDAS]


def _mean_or_nan(total: float, count: int) -> float:
    if count <= 0:
        return math.nan
    return float(total / count)


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


def _compute_utility_report(
    eval_rows: list[dict[str, Any]],
    eval_dataset: Any,
    route_labels: list[str],
    lambdas: list[float],
) -> dict[str, Any]:
    target_dim = len(route_labels)
    eval_lookup: dict[str, dict[str, Any]] = {}
    duplicate_eval_lookup_rows = 0
    invalid_eval_lookup_rows = 0
    for row in eval_dataset:
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
    for row in eval_rows:
        pred_rewards = row.get("pred_rewards")
        if not isinstance(pred_rewards, list) or len(pred_rewards) != target_dim:
            skipped_invalid_route_stats += 1
            continue
        key = _prediction_problem_key(row)
        source_row = eval_lookup.get(key)
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
        try:
            rewards = [float(value) for value in rewards]
            prompt_tokens = [float(value) for value in prompt_tokens]
            output_tokens = [float(value) for value in output_tokens]
            pred_rewards = [float(value) for value in pred_rewards]
        except (TypeError, ValueError):
            skipped_invalid_route_stats += 1
            continue
        valid_examples.append(
            {
                "problem_id": row.get("problem_id"),
                "dataset": row.get("dataset"),
                "rewards": rewards,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "router_choice_idx": _argmax_index(pred_rewards),
                "oracle_choice_idx": _argmax_index(rewards),
            }
        )

    policy_defs = [
        {
            "policy": "router",
            "policy_type": "router",
            "route_idx": None,
            "route_label": None,
        }
    ]
    for route_idx, route_label in enumerate(route_labels):
        policy_defs.append(
            {
                "policy": f"always::{route_label}",
                "policy_type": "always_route",
                "route_idx": int(route_idx),
                "route_label": str(route_label),
            }
        )
    policy_defs.append(
        {
            "policy": "oracle",
            "policy_type": "oracle",
            "route_idx": None,
            "route_label": None,
        }
    )

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

        for example in valid_examples:
            if policy_type == "router":
                route_idx = int(example["router_choice_idx"])
            elif policy_type == "oracle":
                route_idx = int(example["oracle_choice_idx"])
            else:
                route_idx = int(fixed_route_idx)
            route_choice_counts[route_idx] += 1
            reward_sum += float(example["rewards"][route_idx])
            prompt_token_sum += float(example["prompt_tokens"][route_idx])
            output_token_sum += float(example["output_tokens"][route_idx])
            total_token_sum += float(example["prompt_tokens"][route_idx] + example["output_tokens"][route_idx])

        mean_reward = _mean_or_nan(reward_sum, valid_count)
        mean_prompt_tokens = _mean_or_nan(prompt_token_sum, valid_count)
        mean_output_tokens = _mean_or_nan(output_token_sum, valid_count)
        mean_total_tokens = _mean_or_nan(total_token_sum, valid_count)
        choice_counts_by_route = {
            str(route_label): int(route_choice_counts[idx]) for idx, route_label in enumerate(route_labels)
        }
        policy_summary = {
            "policy": policy_name,
            "policy_type": policy_type,
            "route_idx": None if fixed_route_idx is None else int(fixed_route_idx),
            "route_label": policy_def["route_label"],
            "n_examples": int(valid_count),
            "choice_counts_by_route": choice_counts_by_route,
            "mean_reward": mean_reward,
            "mean_prompt_tokens": mean_prompt_tokens,
            "mean_output_tokens": mean_output_tokens,
            "mean_total_tokens": mean_total_tokens,
        }
        policy_summaries[policy_name] = policy_summary

        for lambda_value in lambdas:
            lambda_value = float(lambda_value)
            utility_rows.append(
                {
                    "policy": policy_name,
                    "policy_type": policy_type,
                    "route_idx": policy_summary["route_idx"],
                    "route_label": policy_summary["route_label"],
                    "lambda": lambda_value,
                    "cost_metric": "output_tokens",
                    "mean_reward": mean_reward,
                    "mean_cost": mean_output_tokens,
                    "mean_utility": mean_reward - (lambda_value * mean_output_tokens),
                }
            )
            utility_rows.append(
                {
                    "policy": policy_name,
                    "policy_type": policy_type,
                    "route_idx": policy_summary["route_idx"],
                    "route_label": policy_summary["route_label"],
                    "lambda": lambda_value,
                    "cost_metric": "total_tokens",
                    "mean_reward": mean_reward,
                    "mean_cost": mean_total_tokens,
                    "mean_utility": mean_reward - (lambda_value * mean_total_tokens),
                }
            )

    return {
        "n_eval_examples": int(len(eval_rows)),
        "n_examples_with_utility": int(valid_count),
        "skipped_missing_eval_row": int(skipped_missing_eval_row),
        "skipped_invalid_route_stats": int(skipped_invalid_route_stats),
        "eval_lookup_rows": int(len(eval_lookup)),
        "duplicate_eval_lookup_rows": int(duplicate_eval_lookup_rows),
        "invalid_eval_lookup_rows": int(invalid_eval_lookup_rows),
        "lambdas": [float(value) for value in lambdas],
        "route_labels": list(route_labels),
        "policies": policy_summaries,
        "utility_rows": utility_rows,
    }


def _get_primary_output_text(row: dict[str, Any]) -> str | None:
    primary_output_text = row.get("primary_output_text")
    if not isinstance(primary_output_text, str):
        primary_output_text = row.get("policy_output_text")
    return primary_output_text if isinstance(primary_output_text, str) else None


def _prepare_representation_rows(
    dataset: Any,
    tokenizer: Any,
    max_seq_length: int | None,
    target_dim: int,
) -> list[dict[str, Any]]:
    prepared_rows: list[dict[str, Any]] = []
    for row in dataset:
        prompt_text = row.get("prompt_text")
        primary_output_text = _get_primary_output_text(row)
        targets = row.get("performance_targets")
        if not isinstance(prompt_text, str) or not isinstance(primary_output_text, str):
            continue
        if not isinstance(targets, list) or len(targets) != target_dim:
            continue
        encoded = tokenize_prompt_completion(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            output_text=primary_output_text,
            max_seq_length=max_seq_length,
        )
        if encoded is None:
            continue
        prepared_rows.append(
            {
                "problem_id": problem_id_from_item(row),
                "dataset": row.get("dataset"),
                "repo": row.get("repo"),
                "language": row.get("language"),
                "input_ids": encoded["input_ids"],
                "completion_last_index": encoded["completion_last_index"],
                "targets": [float(value) for value in targets],
            }
        )
    return prepared_rows


def _make_collate_fn(pad_token_id: int, target_dim: int):
    def _collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(row["input_ids"]) for row in rows)
        batch_size = len(rows)
        input_ids = torch.full((batch_size, max_len), fill_value=pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        last_indices = torch.zeros((batch_size,), dtype=torch.long)
        targets = torch.zeros((batch_size, target_dim), dtype=torch.float32)

        for batch_idx, row in enumerate(rows):
            ids = row["input_ids"]
            seq_len = len(ids)
            input_ids[batch_idx, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[batch_idx, :seq_len] = 1
            last_indices[batch_idx] = int(row["completion_last_index"])
            targets[batch_idx] = torch.tensor(row["targets"], dtype=torch.float32)

        return {
            "problem_ids": [row["problem_id"] for row in rows],
            "datasets": [row["dataset"] for row in rows],
            "repos": [row["repo"] for row in rows],
            "languages": [row["language"] for row in rows],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_last_indices": last_indices,
            "targets": targets,
        }

    return _collate


def _configure_training_mode(
    model: AutoModelForCausalLMWithValueHead,
    training_mode: str,
    supervision_mode: str,
) -> list[str]:
    if supervision_mode == "representation_head":
        return configure_router_training_mode(model, training_mode)
    if supervision_mode not in TEXT_LM_SUPERVISION_MODES:
        raise ValueError(f"Unsupported supervision mode: {supervision_mode}")
    if training_mode != "full_backbone":
        raise ValueError(
            f"offline_router.train.supervision_mode={supervision_mode} requires mode=full_backbone"
        )

    for param in model.parameters():
        param.requires_grad = False
    for param in model.value_head.parameters():
        param.requires_grad = False
    for param in model.performance_value_head.parameters():
        param.requires_grad = False
    for param in model.pretrained_model.parameters():
        param.requires_grad = True
    return ["pretrained_model"]


def _build_text_lora_runtime_config(
    train_cfg: Any,
    adapter_config: dict[str, Any] | None = None,
) -> SimpleNamespace:
    configured_lora = train_cfg.get("lora")
    if configured_lora is None:
        configured_lora = {}
    adapter_config = adapter_config or {}
    configured_target_modules = configured_lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    adapter_target_modules = adapter_config.get("target_modules")
    return SimpleNamespace(
        task_type=str(adapter_config.get("task_type", configured_lora.get("task_type", "CAUSAL_LM"))),
        base_model_8bit=bool(configured_lora.get("base_model_8bit", False)),
        base_model_4bit=bool(configured_lora.get("base_model_4bit", False)),
        r=int(adapter_config.get("r", configured_lora.get("r", 16))),
        alpha=int(adapter_config.get("lora_alpha", configured_lora.get("alpha", 32))),
        dropout=float(adapter_config.get("lora_dropout", configured_lora.get("dropout", 0.05))),
        bias=str(adapter_config.get("bias", configured_lora.get("bias", "none"))),
        target_modules=list(adapter_target_modules if adapter_target_modules is not None else configured_target_modules),
    )


def _serialize_text_lora_config(lora_config: SimpleNamespace) -> dict[str, Any]:
    return {
        "task_type": str(lora_config.task_type),
        "base_model_8bit": bool(lora_config.base_model_8bit),
        "base_model_4bit": bool(lora_config.base_model_4bit),
        "r": int(lora_config.r),
        "alpha": int(lora_config.alpha),
        "dropout": float(lora_config.dropout),
        "bias": str(lora_config.bias),
        "target_modules": list(lora_config.target_modules),
    }


def _load_text_lora_model(
    model_path: str,
    train_cfg: Any,
    device: torch.device,
) -> tuple[Any, Any, str, str | None, dict[str, Any]]:
    adapter_checkpoint_path: str | None = None
    adapter_config: dict[str, Any] | None = None
    tokenizer_path = model_path
    base_model_path = model_path
    model_path_obj = Path(model_path)
    if model_path_obj.is_dir() and has_lora_checkpoint(model_path_obj):
        adapter_checkpoint_path = str(model_path_obj)
        tokenizer_path = adapter_checkpoint_path
        adapter_config_path = model_path_obj / "adapter_config.json"
        adapter_config = json.loads(adapter_config_path.read_text())
        base_model_path = str(adapter_config["base_model_name_or_path"])
        logger.info(
            "Offline router text mode detected LoRA adapter checkpoint=%s base_model=%s",
            adapter_checkpoint_path,
            base_model_path,
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    loading_kwargs: dict[str, Any] = {}
    if device.type == "cuda" and bool(train_cfg.get("bf16", True)):
        loading_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **loading_kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    lora_runtime_config = _build_text_lora_runtime_config(train_cfg, adapter_config=adapter_config)
    model = prepare_lora_model(
        lora_runtime_config,
        model,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
    )
    if adapter_checkpoint_path is not None:
        model = lora_load(adapter_checkpoint_path, model)
    return (
        tokenizer,
        model,
        base_model_path,
        adapter_checkpoint_path,
        _serialize_text_lora_config(lora_runtime_config),
    )


def _route_prompt_aliases(route_labels: list[str]) -> list[str]:
    return [f"model_{route_idx + 1}" for route_idx, _ in enumerate(route_labels)]


def _format_reward_grid_values(grid_count: int, precision: int) -> str:
    if grid_count < 2:
        raise ValueError("Reward grid count must be at least 2")
    return ", ".join(f"{idx / (grid_count - 1):.{int(precision)}f}" for idx in range(grid_count))


def _build_text_reward_prompt(
    prompt_text: str,
    primary_output_text: str,
    route_aliases: list[str],
    target_grid_count: int | None = None,
    target_precision: int = 2,
) -> str:
    route_legend = ", ".join(route_aliases)
    reward_instruction = "Each score must be between 0.000 and 1.000.\n"
    if target_grid_count is not None:
        reward_instruction = (
            "Each score must be exactly one of these reward-grid values: "
            f"{_format_reward_grid_values(target_grid_count, target_precision)}.\n"
        )
    return (
        "Predict the realized reward for each model.\n"
        "Respond with only a compact JSON array of floats in the listed order.\n"
        "Do not include any keys, labels, or explanation text.\n"
        f"{reward_instruction}\n"
        "[Model Order]\n"
        f"{route_legend}\n\n"
        "[Original Repair Prompt]\n"
        f"{prompt_text}\n\n"
        "[Primary Model Attempt]\n"
        f"{primary_output_text}\n\n"
        "Answer:\n"
    )


def _build_text_scalar_reward_prompt(
    prompt_text: str,
    primary_output_text: str,
    route_aliases: list[str],
    route_idx: int,
    target_grid_count: int | None = None,
    target_precision: int = 2,
) -> str:
    route_legend = ", ".join(route_aliases)
    route_alias = route_aliases[route_idx]
    reward_instruction = "Respond with only one float between 0.000 and 1.000.\n"
    if target_grid_count is not None:
        reward_instruction = (
            "Respond with only one of these reward-grid values: "
            f"{_format_reward_grid_values(target_grid_count, target_precision)}.\n"
        )
    return (
        "Predict the realized reward for one model route.\n"
        f"{reward_instruction}"
        "Do not include any keys, labels, JSON, or explanation text.\n\n"
        "[Model Order]\n"
        f"{route_legend}\n"
        "model_1 is the primary model whose attempt is shown below.\n\n"
        "[Route To Score]\n"
        f"{route_alias}\n\n"
        "[Original Repair Prompt]\n"
        f"{prompt_text}\n\n"
        "[Primary Model Attempt]\n"
        f"{primary_output_text}\n\n"
        "[Route To Score]\n"
        f"{route_alias}\n\n"
        "Answer:\n"
    )


def _build_reward_bin_specs(
    bin_count: int,
    label_prefix: str = " ",
    value_order: str = "ascending",
) -> list[dict[str, Any]]:
    if bin_count < 2:
        raise ValueError("offline_router.train.text_reward.bin_count must be at least 2")
    if bin_count > 26:
        raise ValueError("offline_router.train.text_reward.bin_count currently supports at most 26 letter bins")
    if value_order not in {"ascending", "descending"}:
        raise ValueError(
            "offline_router.train.text_reward.bin_value_order must be 'ascending' or 'descending'"
        )
    labels = [chr(ord("A") + idx) for idx in range(bin_count)]
    specs: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        value_idx = idx if value_order == "ascending" else bin_count - 1 - idx
        specs.append(
            {
                "idx": int(idx),
                "label": label,
                "target_text": f"{label_prefix}{label}",
                "value": float(value_idx / (bin_count - 1)),
            }
        )
    return specs


def _format_reward_bin_table(bin_specs: list[dict[str, Any]], target_precision: int) -> str:
    return "\n".join(
        f"{spec['label']}: {float(spec['value']):.{int(target_precision)}f}" for spec in bin_specs
    )


def _build_text_bin_reward_prompt(
    prompt_text: str,
    primary_output_text: str,
    route_aliases: list[str],
    route_idx: int,
    bin_specs: list[dict[str, Any]],
    target_precision: int,
) -> str:
    route_legend = ", ".join(route_aliases)
    route_alias = route_aliases[route_idx]
    reward_bin_table = _format_reward_bin_table(bin_specs, target_precision)
    return (
        "Predict the realized reward bin for one model route.\n"
        "Choose exactly one label from the reward-bin table.\n"
        "Respond with only the label.\n\n"
        "[Reward Bins]\n"
        f"{reward_bin_table}\n\n"
        "[Model Order]\n"
        f"{route_legend}\n"
        "model_1 is the primary model whose attempt is shown below.\n\n"
        "[Route To Score]\n"
        f"{route_alias}\n\n"
        "[Original Repair Prompt]\n"
        f"{prompt_text}\n\n"
        "[Primary Model Attempt]\n"
        f"{primary_output_text}\n\n"
        "[Route To Score]\n"
        f"{route_alias}\n\n"
        "Answer:"
    )


def _build_pairwise_label_specs(
    label_prefix: str = " ",
    primary_better_label: str = "A",
    expert_better_label: str = "B",
) -> list[dict[str, Any]]:
    primary_better_label = str(primary_better_label).strip()
    expert_better_label = str(expert_better_label).strip()
    if not primary_better_label or not expert_better_label:
        raise ValueError("Pairwise labels must be non-empty")
    if primary_better_label == expert_better_label:
        raise ValueError("Pairwise labels must be distinct")
    return [
        {
            "winner_idx": 0,
            "label": primary_better_label,
            "target_text": f"{label_prefix}{primary_better_label}",
        },
        {
            "winner_idx": 1,
            "label": expert_better_label,
            "target_text": f"{label_prefix}{expert_better_label}",
        },
    ]


def _format_pairwise_label_table(
    label_specs: list[dict[str, Any]],
    route_aliases: list[str],
) -> str:
    lines: list[str] = []
    for spec in label_specs:
        winner_idx = int(spec["winner_idx"])
        lines.append(f"{spec['label']}: {route_aliases[winner_idx]} better")
    return "\n".join(lines)


def _build_text_pairwise_sign_prompt(
    prompt_text: str,
    primary_output_text: str,
    route_aliases: list[str],
    label_specs: list[dict[str, Any]],
) -> str:
    if len(route_aliases) != 2:
        raise ValueError("text_pairwise_sign currently requires exactly two routes")
    label_table = _format_pairwise_label_table(label_specs, route_aliases)
    route_legend = ", ".join(route_aliases)
    return (
        "Predict which model route achieved the higher realized reward.\n"
        "Choose exactly one label from the outcome table.\n"
        "Respond with only the label.\n\n"
        "[Outcome Labels]\n"
        f"{label_table}\n\n"
        "[Model Order]\n"
        f"{route_legend}\n"
        f"{route_aliases[0]} is the primary model whose attempt is shown below.\n"
        f"Only the {route_aliases[0]} attempt is shown below.\n\n"
        "[Original Repair Prompt]\n"
        f"{prompt_text}\n\n"
        "[Primary Model Attempt]\n"
        f"{primary_output_text}\n\n"
        "Answer:"
    )


def _quantize_reward_to_grid(value: float, grid_count: int | None) -> float:
    if grid_count is None:
        return float(value)
    if grid_count < 2:
        raise ValueError("offline_router.train.text_reward.target_grid_count must be at least 2")
    clipped = min(1.0, max(0.0, float(value)))
    grid_idx = int(math.floor(clipped * (grid_count - 1) + 0.5))
    return float(grid_idx / (grid_count - 1))


def _format_reward_target(values: list[float], precision: int, target_grid_count: int | None = None) -> str:
    rounded = [
        float(f"{_quantize_reward_to_grid(float(value), target_grid_count):.{int(precision)}f}")
        for value in values
    ]
    return json.dumps(rounded, separators=(",", ":"))


def _format_scalar_reward_target(value: float, precision: int, target_grid_count: int | None = None) -> str:
    return f"{_quantize_reward_to_grid(float(value), target_grid_count):.{int(precision)}f}"


def _reward_to_bin_idx(value: float, bin_specs: list[dict[str, Any]]) -> int:
    if len(bin_specs) < 2:
        raise ValueError("At least two reward bins are required")
    target_value = _quantize_reward_to_grid(float(value), len(bin_specs))
    return min(
        range(len(bin_specs)),
        key=lambda idx: abs(float(bin_specs[idx]["value"]) - target_value),
    )


def _label_target_token_ids(tokenizer: Any, specs: list[dict[str, Any]]) -> list[int]:
    token_ids: list[int] = []
    for spec in specs:
        ids = tokenizer(str(spec["target_text"]), add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(
                "Label target_text=%r tokenized to %d tokens; expected one token"
                % (spec["target_text"], len(ids))
            )
        token_ids.append(int(ids[0]))
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("Labels do not map to unique token ids")
    return token_ids


def _reward_bin_token_ids(tokenizer: Any, bin_specs: list[dict[str, Any]]) -> list[int]:
    return _label_target_token_ids(tokenizer, bin_specs)


def _truncate_from_left(token_ids: list[int], max_seq_length: int | None) -> list[int]:
    if max_seq_length is None or max_seq_length <= 0 or len(token_ids) <= max_seq_length:
        return token_ids
    return token_ids[-max_seq_length:]


def _tokenize_text_reward_target(
    tokenizer: Any,
    prompt_text: str,
    target_text: str,
    max_seq_length: int | None,
) -> tuple[dict[str, Any] | None, bool]:
    return _tokenize_text_reward_target_with_limit(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        target_text=target_text,
        max_seq_length=max_seq_length,
        drop_overlength=False,
    )


def _tokenize_text_reward_target_with_limit(
    tokenizer: Any,
    prompt_text: str,
    target_text: str,
    max_seq_length: int | None,
    drop_overlength: bool,
) -> tuple[dict[str, Any] | None, bool]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, verbose=False).input_ids
    full_ids = tokenizer(prompt_text + target_text, add_special_tokens=False, verbose=False).input_ids
    if not full_ids or len(full_ids) <= len(prompt_ids):
        return None, False

    prompt_len = len(prompt_ids)
    if max_seq_length is not None and max_seq_length > 0 and len(full_ids) > max_seq_length:
        if drop_overlength:
            return None, True
        drop = len(full_ids) - max_seq_length
        full_ids = full_ids[drop:]
        prompt_len = max(0, prompt_len - drop)

    if not full_ids or len(full_ids) <= prompt_len:
        return None, False

    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return {
        "input_ids": full_ids,
        "labels": labels,
    }, False


def _prepare_eval_prompt_ids(
    tokenizer: Any,
    prompt_text: str,
    max_seq_length: int | None,
    drop_overlength: bool,
) -> tuple[list[int], bool]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, verbose=False).input_ids
    if max_seq_length is not None and max_seq_length > 0 and len(prompt_ids) > max_seq_length:
        if drop_overlength:
            return [], True
        prompt_ids = _truncate_from_left(prompt_ids, max_seq_length)
    return prompt_ids, False


def _prepare_text_train_rows(
    dataset: Any,
    tokenizer: Any,
    max_seq_length: int | None,
    route_aliases: list[str],
    target_dim: int,
    target_precision: int,
    target_grid_count: int | None = None,
    drop_overlength: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    prepared_rows: list[dict[str, Any]] = []
    dropped_overlength_rows = 0
    for row in dataset:
        prompt_text = row.get("prompt_text")
        primary_output_text = _get_primary_output_text(row)
        targets = row.get("performance_targets")
        if not isinstance(prompt_text, str) or not isinstance(primary_output_text, str):
            continue
        if not isinstance(targets, list) or len(targets) != target_dim:
            continue
        try:
            problem_id = problem_id_from_item(row)
        except ValueError:
            continue
        target_rewards = [float(value) for value in targets]
        prompt = _build_text_reward_prompt(
            prompt_text,
            primary_output_text,
            route_aliases,
            target_grid_count=target_grid_count,
            target_precision=target_precision,
        )
        encoded, dropped_overlength = _tokenize_text_reward_target_with_limit(
            tokenizer=tokenizer,
            prompt_text=prompt,
            target_text=_format_reward_target(
                target_rewards,
                target_precision,
                target_grid_count=target_grid_count,
            ),
            max_seq_length=max_seq_length,
            drop_overlength=drop_overlength,
        )
        if dropped_overlength:
            dropped_overlength_rows += 1
        if encoded is None:
            continue
        prepared_rows.append(
            {
                "problem_id": problem_id,
                "dataset": row.get("dataset"),
                "repo": row.get("repo"),
                "language": row.get("language"),
                "route_aliases": list(route_aliases),
                "target_rewards": target_rewards,
                "input_ids": encoded["input_ids"],
                "labels": encoded["labels"],
            }
        )
    return prepared_rows, dropped_overlength_rows


def _make_text_train_collate_fn(pad_token_id: int, target_dim: int):
    def _collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(row["input_ids"]) for row in rows)
        batch_size = len(rows)
        input_ids = torch.full((batch_size, max_len), fill_value=pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), fill_value=-100, dtype=torch.long)
        target_rewards = torch.zeros((batch_size, target_dim), dtype=torch.float32)

        for batch_idx, row in enumerate(rows):
            ids = row["input_ids"]
            row_labels = row["labels"]
            seq_len = len(ids)
            input_ids[batch_idx, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[batch_idx, :seq_len] = 1
            labels[batch_idx, :seq_len] = torch.tensor(row_labels, dtype=torch.long)
            target_rewards[batch_idx] = torch.tensor(row["target_rewards"], dtype=torch.float32)

        return {
            "problem_ids": [row["problem_id"] for row in rows],
            "datasets": [row["dataset"] for row in rows],
            "repos": [row["repo"] for row in rows],
            "languages": [row["language"] for row in rows],
            "route_aliases": [row["route_aliases"] for row in rows],
            "targets": target_rewards,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return _collate


def _prepare_text_eval_rows(
    dataset: Any,
    tokenizer: Any,
    max_seq_length: int | None,
    route_aliases: list[str],
    target_dim: int,
    target_grid_count: int | None = None,
    target_precision: int = 2,
    drop_overlength: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    prepared_rows: list[dict[str, Any]] = []
    dropped_overlength_rows = 0
    for row in dataset:
        prompt_text = row.get("prompt_text")
        primary_output_text = _get_primary_output_text(row)
        targets = row.get("performance_targets")
        if not isinstance(prompt_text, str) or not isinstance(primary_output_text, str):
            continue
        if not isinstance(targets, list) or len(targets) != target_dim:
            continue
        try:
            problem_id = problem_id_from_item(row)
        except ValueError:
            continue
        prompt = _build_text_reward_prompt(
            prompt_text,
            primary_output_text,
            route_aliases,
            target_grid_count=target_grid_count,
            target_precision=target_precision,
        )
        prompt_ids, dropped_overlength = _prepare_eval_prompt_ids(
            tokenizer=tokenizer,
            prompt_text=prompt,
            max_seq_length=max_seq_length,
            drop_overlength=drop_overlength,
        )
        if dropped_overlength:
            dropped_overlength_rows += 1
        if not prompt_ids:
            continue
        prepared_rows.append(
            {
                "problem_id": problem_id,
                "dataset": row.get("dataset"),
                "repo": row.get("repo"),
                "language": row.get("language"),
                "route_aliases": list(route_aliases),
                "target_rewards": [float(value) for value in targets],
                "input_ids": prompt_ids,
            }
        )
    return prepared_rows, dropped_overlength_rows


def _make_text_eval_collate_fn(pad_token_id: int, target_dim: int):
    def _collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(row["input_ids"]) for row in rows)
        batch_size = len(rows)
        input_ids = torch.full((batch_size, max_len), fill_value=pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        target_rewards = torch.zeros((batch_size, target_dim), dtype=torch.float32)

        for batch_idx, row in enumerate(rows):
            ids = row["input_ids"]
            seq_len = len(ids)
            input_ids[batch_idx, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[batch_idx, :seq_len] = 1
            target_rewards[batch_idx] = torch.tensor(row["target_rewards"], dtype=torch.float32)

        return {
            "problem_ids": [row["problem_id"] for row in rows],
            "datasets": [row["dataset"] for row in rows],
            "repos": [row["repo"] for row in rows],
            "languages": [row["language"] for row in rows],
            "route_aliases": [row["route_aliases"] for row in rows],
            "targets": target_rewards,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    return _collate


def _prepare_text_scalar_train_rows(
    dataset: Any,
    tokenizer: Any,
    max_seq_length: int | None,
    route_aliases: list[str],
    route_labels: list[str],
    target_dim: int,
    target_precision: int,
    target_grid_count: int | None = None,
    drop_overlength: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    prepared_rows: list[dict[str, Any]] = []
    dropped_overlength_rows = 0
    for row in dataset:
        prompt_text = row.get("prompt_text")
        primary_output_text = _get_primary_output_text(row)
        targets = row.get("performance_targets")
        if not isinstance(prompt_text, str) or not isinstance(primary_output_text, str):
            continue
        if not isinstance(targets, list) or len(targets) != target_dim:
            continue
        try:
            problem_id = problem_id_from_item(row)
        except ValueError:
            continue
        target_rewards = [float(value) for value in targets]
        for route_idx, target_reward in enumerate(target_rewards):
            prompt = _build_text_scalar_reward_prompt(
                prompt_text=prompt_text,
                primary_output_text=primary_output_text,
                route_aliases=route_aliases,
                route_idx=route_idx,
                target_grid_count=target_grid_count,
                target_precision=target_precision,
            )
            encoded, dropped_overlength = _tokenize_text_reward_target_with_limit(
                tokenizer=tokenizer,
                prompt_text=prompt,
                target_text=_format_scalar_reward_target(
                    target_reward,
                    target_precision,
                    target_grid_count=target_grid_count,
                ),
                max_seq_length=max_seq_length,
                drop_overlength=drop_overlength,
            )
            if dropped_overlength:
                dropped_overlength_rows += 1
            if encoded is None:
                continue
            prepared_rows.append(
                {
                    "problem_id": problem_id,
                    "dataset": row.get("dataset"),
                    "repo": row.get("repo"),
                    "language": row.get("language"),
                    "route_aliases": list(route_aliases),
                    "route_idx": int(route_idx),
                    "route_alias": route_aliases[route_idx],
                    "route_label": route_labels[route_idx],
                    "target_reward": float(target_reward),
                    "input_ids": encoded["input_ids"],
                    "labels": encoded["labels"],
                }
            )
    return prepared_rows, dropped_overlength_rows


def _prepare_text_scalar_eval_rows(
    dataset: Any,
    tokenizer: Any,
    max_seq_length: int | None,
    route_aliases: list[str],
    route_labels: list[str],
    target_dim: int,
    target_grid_count: int | None = None,
    target_precision: int = 2,
    drop_overlength: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    prepared_rows: list[dict[str, Any]] = []
    dropped_overlength_rows = 0
    for row in dataset:
        prompt_text = row.get("prompt_text")
        primary_output_text = _get_primary_output_text(row)
        targets = row.get("performance_targets")
        if not isinstance(prompt_text, str) or not isinstance(primary_output_text, str):
            continue
        if not isinstance(targets, list) or len(targets) != target_dim:
            continue
        try:
            problem_id = problem_id_from_item(row)
        except ValueError:
            continue
        target_rewards = [float(value) for value in targets]
        for route_idx, target_reward in enumerate(target_rewards):
            prompt = _build_text_scalar_reward_prompt(
                prompt_text=prompt_text,
                primary_output_text=primary_output_text,
                route_aliases=route_aliases,
                route_idx=route_idx,
                target_grid_count=target_grid_count,
                target_precision=target_precision,
            )
            prompt_ids, dropped_overlength = _prepare_eval_prompt_ids(
                tokenizer=tokenizer,
                prompt_text=prompt,
                max_seq_length=max_seq_length,
                drop_overlength=drop_overlength,
            )
            if dropped_overlength:
                dropped_overlength_rows += 1
            if not prompt_ids:
                continue
            prepared_rows.append(
                {
                    "problem_id": problem_id,
                    "dataset": row.get("dataset"),
                    "repo": row.get("repo"),
                    "language": row.get("language"),
                    "route_aliases": list(route_aliases),
                    "route_idx": int(route_idx),
                    "route_alias": route_aliases[route_idx],
                    "route_label": route_labels[route_idx],
                    "target_reward": float(target_reward),
                    "input_ids": prompt_ids,
                }
            )
    return prepared_rows, dropped_overlength_rows


def _prepare_text_bin_train_rows(
    dataset: Any,
    tokenizer: Any,
    max_seq_length: int | None,
    route_aliases: list[str],
    route_labels: list[str],
    target_dim: int,
    target_precision: int,
    bin_specs: list[dict[str, Any]],
    drop_overlength: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    prepared_rows: list[dict[str, Any]] = []
    dropped_overlength_rows = 0
    for row in dataset:
        prompt_text = row.get("prompt_text")
        primary_output_text = _get_primary_output_text(row)
        targets = row.get("performance_targets")
        if not isinstance(prompt_text, str) or not isinstance(primary_output_text, str):
            continue
        if not isinstance(targets, list) or len(targets) != target_dim:
            continue
        try:
            problem_id = problem_id_from_item(row)
        except ValueError:
            continue
        target_rewards = [float(value) for value in targets]
        for route_idx, target_reward in enumerate(target_rewards):
            bin_idx = _reward_to_bin_idx(target_reward, bin_specs)
            bin_spec = bin_specs[bin_idx]
            prompt = _build_text_bin_reward_prompt(
                prompt_text=prompt_text,
                primary_output_text=primary_output_text,
                route_aliases=route_aliases,
                route_idx=route_idx,
                bin_specs=bin_specs,
                target_precision=target_precision,
            )
            encoded, dropped_overlength = _tokenize_text_reward_target_with_limit(
                tokenizer=tokenizer,
                prompt_text=prompt,
                target_text=str(bin_spec["target_text"]),
                max_seq_length=max_seq_length,
                drop_overlength=drop_overlength,
            )
            if dropped_overlength:
                dropped_overlength_rows += 1
            if encoded is None:
                continue
            prepared_rows.append(
                {
                    "problem_id": problem_id,
                    "dataset": row.get("dataset"),
                    "repo": row.get("repo"),
                    "language": row.get("language"),
                    "route_aliases": list(route_aliases),
                    "route_idx": int(route_idx),
                    "route_alias": route_aliases[route_idx],
                    "route_label": route_labels[route_idx],
                    "target_reward": float(target_reward),
                    "target_bin_idx": int(bin_idx),
                    "target_bin_label": str(bin_spec["label"]),
                    "target_bin_value": float(bin_spec["value"]),
                    "input_ids": encoded["input_ids"],
                    "labels": encoded["labels"],
                }
            )
    return prepared_rows, dropped_overlength_rows


def _prepare_text_bin_eval_rows(
    dataset: Any,
    tokenizer: Any,
    max_seq_length: int | None,
    route_aliases: list[str],
    route_labels: list[str],
    target_dim: int,
    target_precision: int,
    bin_specs: list[dict[str, Any]],
    drop_overlength: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    prepared_rows: list[dict[str, Any]] = []
    dropped_overlength_rows = 0
    for row in dataset:
        prompt_text = row.get("prompt_text")
        primary_output_text = _get_primary_output_text(row)
        targets = row.get("performance_targets")
        if not isinstance(prompt_text, str) or not isinstance(primary_output_text, str):
            continue
        if not isinstance(targets, list) or len(targets) != target_dim:
            continue
        try:
            problem_id = problem_id_from_item(row)
        except ValueError:
            continue
        target_rewards = [float(value) for value in targets]
        for route_idx, target_reward in enumerate(target_rewards):
            target_bin_idx = _reward_to_bin_idx(target_reward, bin_specs)
            target_bin_spec = bin_specs[target_bin_idx]
            prompt = _build_text_bin_reward_prompt(
                prompt_text=prompt_text,
                primary_output_text=primary_output_text,
                route_aliases=route_aliases,
                route_idx=route_idx,
                bin_specs=bin_specs,
                target_precision=target_precision,
            )
            prompt_ids, dropped_overlength = _prepare_eval_prompt_ids(
                tokenizer=tokenizer,
                prompt_text=prompt,
                max_seq_length=max_seq_length,
                drop_overlength=drop_overlength,
            )
            if dropped_overlength:
                dropped_overlength_rows += 1
            if not prompt_ids:
                continue
            prepared_rows.append(
                {
                    "problem_id": problem_id,
                    "dataset": row.get("dataset"),
                    "repo": row.get("repo"),
                    "language": row.get("language"),
                    "route_aliases": list(route_aliases),
                    "route_idx": int(route_idx),
                    "route_alias": route_aliases[route_idx],
                    "route_label": route_labels[route_idx],
                    "target_reward": float(target_reward),
                    "target_bin_idx": int(target_bin_idx),
                    "target_bin_label": str(target_bin_spec["label"]),
                    "target_bin_value": float(target_bin_spec["value"]),
                    "input_ids": prompt_ids,
                }
            )
    return prepared_rows, dropped_overlength_rows


def _pairwise_target_winner_idx(target_rewards: list[float], tie_margin: float) -> int | None:
    if len(target_rewards) != 2:
        raise ValueError("text_pairwise_sign currently requires exactly two target rewards")
    delta = float(target_rewards[0]) - float(target_rewards[1])
    if abs(delta) <= float(tie_margin):
        return None
    return 0 if delta > 0.0 else 1


def _pairwise_target_vector(target_dim: int, winner_idx: int) -> list[float]:
    target = [0.0] * int(target_dim)
    target[int(winner_idx)] = 1.0
    return target


def _prepare_text_pairwise_sign_train_rows(
    dataset: Any,
    tokenizer: Any,
    max_seq_length: int | None,
    route_aliases: list[str],
    route_labels: list[str],
    target_dim: int,
    label_specs: list[dict[str, Any]],
    tie_margin: float,
    drop_overlength: bool = False,
) -> tuple[list[dict[str, Any]], int, int]:
    if target_dim != 2:
        raise ValueError("text_pairwise_sign currently requires exactly two routes")
    prepared_rows: list[dict[str, Any]] = []
    dropped_overlength_rows = 0
    dropped_tie_rows = 0
    label_specs_by_winner = {int(spec["winner_idx"]): spec for spec in label_specs}
    for row in dataset:
        prompt_text = row.get("prompt_text")
        primary_output_text = _get_primary_output_text(row)
        targets = row.get("performance_targets")
        if not isinstance(prompt_text, str) or not isinstance(primary_output_text, str):
            continue
        if not isinstance(targets, list) or len(targets) != target_dim:
            continue
        try:
            problem_id = problem_id_from_item(row)
        except ValueError:
            continue
        target_rewards = [float(value) for value in targets]
        winner_idx = _pairwise_target_winner_idx(target_rewards, tie_margin=tie_margin)
        if winner_idx is None:
            dropped_tie_rows += 1
            continue
        prompt = _build_text_pairwise_sign_prompt(
            prompt_text=prompt_text,
            primary_output_text=primary_output_text,
            route_aliases=route_aliases,
            label_specs=label_specs,
        )
        winner_spec = label_specs_by_winner[winner_idx]
        encoded, dropped_overlength = _tokenize_text_reward_target_with_limit(
            tokenizer=tokenizer,
            prompt_text=prompt,
            target_text=str(winner_spec["target_text"]),
            max_seq_length=max_seq_length,
            drop_overlength=drop_overlength,
        )
        if dropped_overlength:
            dropped_overlength_rows += 1
        if encoded is None:
            continue
        prepared_rows.append(
            {
                "problem_id": problem_id,
                "dataset": row.get("dataset"),
                "repo": row.get("repo"),
                "language": row.get("language"),
                "route_aliases": list(route_aliases),
                "target_rewards": target_rewards,
                "target_pairwise": _pairwise_target_vector(target_dim, winner_idx),
                "target_winner_idx": int(winner_idx),
                "target_winner_alias": route_aliases[winner_idx],
                "target_winner_label": route_labels[winner_idx],
                "target_label": str(winner_spec["label"]),
                "input_ids": encoded["input_ids"],
                "labels": encoded["labels"],
            }
        )
    return prepared_rows, dropped_overlength_rows, dropped_tie_rows


def _prepare_text_pairwise_sign_eval_rows(
    dataset: Any,
    tokenizer: Any,
    max_seq_length: int | None,
    route_aliases: list[str],
    route_labels: list[str],
    target_dim: int,
    label_specs: list[dict[str, Any]],
    tie_margin: float,
    drop_overlength: bool = False,
) -> tuple[list[dict[str, Any]], int, int]:
    if target_dim != 2:
        raise ValueError("text_pairwise_sign currently requires exactly two routes")
    prepared_rows: list[dict[str, Any]] = []
    dropped_overlength_rows = 0
    dropped_tie_rows = 0
    label_specs_by_winner = {int(spec["winner_idx"]): spec for spec in label_specs}
    for row in dataset:
        prompt_text = row.get("prompt_text")
        primary_output_text = _get_primary_output_text(row)
        targets = row.get("performance_targets")
        if not isinstance(prompt_text, str) or not isinstance(primary_output_text, str):
            continue
        if not isinstance(targets, list) or len(targets) != target_dim:
            continue
        try:
            problem_id = problem_id_from_item(row)
        except ValueError:
            continue
        target_rewards = [float(value) for value in targets]
        winner_idx = _pairwise_target_winner_idx(target_rewards, tie_margin=tie_margin)
        if winner_idx is None:
            dropped_tie_rows += 1
            continue
        prompt = _build_text_pairwise_sign_prompt(
            prompt_text=prompt_text,
            primary_output_text=primary_output_text,
            route_aliases=route_aliases,
            label_specs=label_specs,
        )
        prompt_ids, dropped_overlength = _prepare_eval_prompt_ids(
            tokenizer=tokenizer,
            prompt_text=prompt,
            max_seq_length=max_seq_length,
            drop_overlength=drop_overlength,
        )
        if dropped_overlength:
            dropped_overlength_rows += 1
        if not prompt_ids:
            continue
        winner_spec = label_specs_by_winner[winner_idx]
        prepared_rows.append(
            {
                "problem_id": problem_id,
                "dataset": row.get("dataset"),
                "repo": row.get("repo"),
                "language": row.get("language"),
                "route_aliases": list(route_aliases),
                "target_rewards": target_rewards,
                "target_pairwise": _pairwise_target_vector(target_dim, winner_idx),
                "target_winner_idx": int(winner_idx),
                "target_winner_alias": route_aliases[winner_idx],
                "target_winner_label": route_labels[winner_idx],
                "target_label": str(winner_spec["label"]),
                "input_ids": prompt_ids,
            }
        )
    return prepared_rows, dropped_overlength_rows, dropped_tie_rows


def _make_text_scalar_train_collate_fn(pad_token_id: int):
    def _collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(row["input_ids"]) for row in rows)
        batch_size = len(rows)
        input_ids = torch.full((batch_size, max_len), fill_value=pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), fill_value=-100, dtype=torch.long)
        targets = torch.zeros((batch_size,), dtype=torch.float32)
        route_indices = torch.zeros((batch_size,), dtype=torch.long)

        for batch_idx, row in enumerate(rows):
            ids = row["input_ids"]
            row_labels = row["labels"]
            seq_len = len(ids)
            input_ids[batch_idx, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[batch_idx, :seq_len] = 1
            labels[batch_idx, :seq_len] = torch.tensor(row_labels, dtype=torch.long)
            targets[batch_idx] = float(row["target_reward"])
            route_indices[batch_idx] = int(row["route_idx"])

        return {
            "problem_ids": [row["problem_id"] for row in rows],
            "datasets": [row["dataset"] for row in rows],
            "repos": [row["repo"] for row in rows],
            "languages": [row["language"] for row in rows],
            "route_aliases": [row["route_aliases"] for row in rows],
            "route_indices": route_indices,
            "route_alias": [row["route_alias"] for row in rows],
            "route_label": [row["route_label"] for row in rows],
            "targets": targets,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return _collate


def _make_text_scalar_eval_collate_fn(pad_token_id: int):
    def _collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(row["input_ids"]) for row in rows)
        batch_size = len(rows)
        input_ids = torch.full((batch_size, max_len), fill_value=pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        targets = torch.zeros((batch_size,), dtype=torch.float32)
        route_indices = torch.zeros((batch_size,), dtype=torch.long)

        for batch_idx, row in enumerate(rows):
            ids = row["input_ids"]
            seq_len = len(ids)
            input_ids[batch_idx, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[batch_idx, :seq_len] = 1
            targets[batch_idx] = float(row["target_reward"])
            route_indices[batch_idx] = int(row["route_idx"])

        return {
            "problem_ids": [row["problem_id"] for row in rows],
            "datasets": [row["dataset"] for row in rows],
            "repos": [row["repo"] for row in rows],
            "languages": [row["language"] for row in rows],
            "route_aliases": [row["route_aliases"] for row in rows],
            "route_indices": route_indices,
            "route_alias": [row["route_alias"] for row in rows],
            "route_label": [row["route_label"] for row in rows],
            "targets": targets,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    return _collate


def _make_text_pairwise_train_collate_fn(pad_token_id: int, target_dim: int):
    def _collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(row["input_ids"]) for row in rows)
        batch_size = len(rows)
        input_ids = torch.full((batch_size, max_len), fill_value=pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), fill_value=-100, dtype=torch.long)
        targets = torch.zeros((batch_size, target_dim), dtype=torch.float32)
        true_rewards = torch.zeros((batch_size, target_dim), dtype=torch.float32)
        winner_indices = torch.zeros((batch_size,), dtype=torch.long)

        for batch_idx, row in enumerate(rows):
            ids = row["input_ids"]
            row_labels = row["labels"]
            seq_len = len(ids)
            input_ids[batch_idx, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[batch_idx, :seq_len] = 1
            labels[batch_idx, :seq_len] = torch.tensor(row_labels, dtype=torch.long)
            targets[batch_idx] = torch.tensor(row["target_pairwise"], dtype=torch.float32)
            true_rewards[batch_idx] = torch.tensor(row["target_rewards"], dtype=torch.float32)
            winner_indices[batch_idx] = int(row["target_winner_idx"])

        return {
            "problem_ids": [row["problem_id"] for row in rows],
            "datasets": [row["dataset"] for row in rows],
            "repos": [row["repo"] for row in rows],
            "languages": [row["language"] for row in rows],
            "route_aliases": [row["route_aliases"] for row in rows],
            "winner_indices": winner_indices,
            "winner_alias": [row["target_winner_alias"] for row in rows],
            "winner_label": [row["target_winner_label"] for row in rows],
            "target_label": [row["target_label"] for row in rows],
            "targets": targets,
            "true_rewards": true_rewards,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return _collate


def _make_text_pairwise_eval_collate_fn(pad_token_id: int, target_dim: int):
    def _collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(row["input_ids"]) for row in rows)
        batch_size = len(rows)
        input_ids = torch.full((batch_size, max_len), fill_value=pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        targets = torch.zeros((batch_size, target_dim), dtype=torch.float32)
        true_rewards = torch.zeros((batch_size, target_dim), dtype=torch.float32)
        winner_indices = torch.zeros((batch_size,), dtype=torch.long)

        for batch_idx, row in enumerate(rows):
            ids = row["input_ids"]
            seq_len = len(ids)
            input_ids[batch_idx, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[batch_idx, :seq_len] = 1
            targets[batch_idx] = torch.tensor(row["target_pairwise"], dtype=torch.float32)
            true_rewards[batch_idx] = torch.tensor(row["target_rewards"], dtype=torch.float32)
            winner_indices[batch_idx] = int(row["target_winner_idx"])

        return {
            "problem_ids": [row["problem_id"] for row in rows],
            "datasets": [row["dataset"] for row in rows],
            "repos": [row["repo"] for row in rows],
            "languages": [row["language"] for row in rows],
            "route_aliases": [row["route_aliases"] for row in rows],
            "winner_indices": winner_indices,
            "winner_alias": [row["target_winner_alias"] for row in rows],
            "winner_label": [row["target_winner_label"] for row in rows],
            "target_label": [row["target_label"] for row in rows],
            "targets": targets,
            "true_rewards": true_rewards,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    return _collate


def _forward_predictions(model: AutoModelForCausalLMWithValueHead, batch: dict[str, Any]) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        return_dict=True,
    )
    performance_values = outputs.performance_value
    if performance_values is None:
        raise ValueError("Model did not return performance_value outputs")
    batch_indices = torch.arange(performance_values.shape[0], device=performance_values.device)
    preds = performance_values[batch_indices, batch["completion_last_indices"]]
    return preds.float()


def _forward_text_reward_loss(model: Any, batch: dict[str, Any]) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        return_dict=True,
    )
    if outputs.loss is None:
        raise ValueError("Model did not return language-model loss for text reward supervision")
    return outputs.loss.float()


def _run_representation_eval(
    model: AutoModelForCausalLMWithValueHead,
    loader: DataLoader,
) -> tuple[list[dict[str, Any]], float, int]:
    model.eval()
    prediction_rows: list[dict[str, Any]] = []
    local_squared_error_sum = 0.0
    local_value_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval offline router", unit="batch", disable=not is_main_process()):
            preds = _forward_predictions(model, batch)

            preds_np = preds.cpu().numpy()
            targets_np = batch["targets"].cpu().numpy()
            local_squared_error_sum += float(((preds - batch["targets"]) ** 2).sum().item())
            local_value_count += int(batch["targets"].numel())
            for row_idx, problem_id in enumerate(batch["problem_ids"]):
                prediction_rows.append(
                    {
                        "problem_id": problem_id,
                        "dataset": batch["datasets"][row_idx],
                        "repo": batch["repos"][row_idx],
                        "language": batch["languages"][row_idx],
                        "true_rewards": targets_np[row_idx].tolist(),
                        "pred_rewards": preds_np[row_idx].tolist(),
                    }
                )
    return prediction_rows, local_squared_error_sum, local_value_count

def _parse_generated_reward(text: str, target_dim: int) -> tuple[list[float] | None, str | None]:
    def _coerce_reward_list(parsed: Any) -> tuple[list[float] | None, str | None]:
        if not isinstance(parsed, list):
            return None, f"json_type_error:{type(parsed).__name__}"
        if len(parsed) != target_dim:
            return None, f"json_length_error:{len(parsed)}"

        rewards: list[float] = []
        for idx, value in enumerate(parsed):
            try:
                rewards.append(float(value))
            except (TypeError, ValueError):
                return None, f"json_value_error:index={idx}"
        return rewards, None

    if not isinstance(text, str):
        return None, "generated_text_not_string"
    stripped = text.strip()
    if not stripped:
        return None, "generated_text_empty"

    decoder = json.JSONDecoder()

    try:
        parsed = json.loads(stripped)
        return _coerce_reward_list(parsed)
    except json.JSONDecodeError:
        pass

    # Be tolerant of fenced code blocks or extra prose by extracting the first
    # decodable JSON array from the generated text.
    for start_idx, char in enumerate(stripped):
        if char != "[":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[start_idx:])
        except json.JSONDecodeError:
            continue
        rewards, error = _coerce_reward_list(parsed)
        if rewards is not None:
            return rewards, None
        if error is not None and not error.startswith("json_length_error"):
            return None, error

    return None, "json_array_not_found"


def _parse_generated_scalar_reward(text: str) -> tuple[float | None, str | None]:
    if not isinstance(text, str):
        return None, "generated_text_not_string"
    stripped = text.strip()
    if not stripped:
        return None, "generated_text_empty"

    try:
        return float(stripped), None
    except ValueError:
        pass

    matches = list(
        re.finditer(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            stripped,
        )
    )
    if not matches:
        return None, "float_not_found"
    for match in matches:
        try:
            value = float(match.group(0))
        except ValueError:
            continue
        if 0.0 <= value <= 1.0:
            return value, None
    try:
        return float(matches[0].group(0)), None
    except ValueError:
        return None, "float_parse_error"


def _generate_text_reward_tokens(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    do_sample: bool,
    tokenizer: Any,
) -> torch.Tensor:
    generated_ids = input_ids
    generated_attention_mask = attention_mask
    batch_size = int(input_ids.shape[0])
    stop_ids = {token_id for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id) if token_id is not None}
    stop_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    if stop_token_id is None:
        stop_token_id = 0
    finished = torch.zeros((batch_size,), dtype=torch.bool, device=input_ids.device)
    generated_tokens: list[torch.Tensor] = []

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated_ids,
            attention_mask=generated_attention_mask,
            return_dict=True,
        )
        next_logits = outputs.logits[:, -1, :]
        if do_sample:
            probs = torch.softmax(next_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_logits, dim=-1)
        next_tokens = torch.where(
            finished,
            torch.full_like(next_tokens, fill_value=stop_token_id),
            next_tokens,
        )
        generated_tokens.append(next_tokens)
        generated_ids = torch.cat((generated_ids, next_tokens.unsqueeze(1)), dim=1)
        generated_attention_mask = torch.cat(
            (
                generated_attention_mask,
                torch.ones((batch_size, 1), dtype=generated_attention_mask.dtype, device=generated_attention_mask.device),
            ),
            dim=1,
        )
        if stop_ids:
            just_finished = torch.zeros_like(finished)
            for stop_id in stop_ids:
                just_finished |= next_tokens == stop_id
            finished |= just_finished
            if bool(torch.all(finished)):
                break

    if not generated_tokens:
        return torch.empty((batch_size, 0), dtype=torch.long, device=input_ids.device)
    return torch.stack(generated_tokens, dim=1)


def _run_text_reward_eval(
    model: Any,
    loader: DataLoader,
    tokenizer: Any,
    max_new_tokens: int,
    do_sample: bool,
    parse_failure_value: float,
    clip_predictions: bool,
    target_dim: int,
) -> tuple[list[dict[str, Any]], float, int]:
    model.eval()
    prediction_rows: list[dict[str, Any]] = []
    local_squared_error_sum = 0.0
    local_value_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval offline router", unit="batch", disable=not is_main_process()):
            generated_tokens = _generate_text_reward_tokens(
                model=model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                tokenizer=tokenizer,
            )
            if generated_tokens.shape[1] == 0:
                generated_texts = [""] * int(batch["input_ids"].shape[0])
            else:
                generated_texts = tokenizer.batch_decode(generated_tokens.detach().cpu(), skip_special_tokens=True)
            for row_idx, problem_id in enumerate(batch["problem_ids"]):
                parsed_rewards, parse_error = _parse_generated_reward(generated_texts[row_idx], target_dim)
                parse_success = parsed_rewards is not None
                if not parse_success:
                    parsed_rewards = [float(parse_failure_value)] * target_dim
                if clip_predictions:
                    parsed_rewards = [float(min(1.0, max(0.0, value))) for value in parsed_rewards]
                target_rewards = [float(value) for value in batch["targets"][row_idx].tolist()]
                local_squared_error_sum += float(
                    sum((pred - target) ** 2 for pred, target in zip(parsed_rewards, target_rewards))
                )
                local_value_count += target_dim
                prediction_row = {
                    "problem_id": problem_id,
                    "dataset": batch["datasets"][row_idx],
                    "repo": batch["repos"][row_idx],
                    "language": batch["languages"][row_idx],
                    "route_aliases": list(batch["route_aliases"][row_idx]),
                    "true_rewards": target_rewards,
                    "generated_text": generated_texts[row_idx],
                    "parsed_rewards": [float(value) for value in parsed_rewards],
                    "pred_rewards": [float(value) for value in parsed_rewards],
                    "parse_success": bool(parse_success),
                }
                if parse_error is not None:
                    prediction_row["parse_error"] = parse_error
                prediction_rows.append(prediction_row)
    return prediction_rows, local_squared_error_sum, local_value_count


def _run_text_scalar_reward_eval(
    model: Any,
    loader: DataLoader,
    tokenizer: Any,
    max_new_tokens: int,
    do_sample: bool,
    parse_failure_value: float,
    clip_predictions: bool,
) -> tuple[list[dict[str, Any]], float, int]:
    model.eval()
    prediction_rows: list[dict[str, Any]] = []
    local_squared_error_sum = 0.0
    local_value_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval offline router", unit="batch", disable=not is_main_process()):
            generated_tokens = _generate_text_reward_tokens(
                model=model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                tokenizer=tokenizer,
            )
            if generated_tokens.shape[1] == 0:
                generated_texts = [""] * int(batch["input_ids"].shape[0])
            else:
                generated_texts = tokenizer.batch_decode(generated_tokens.detach().cpu(), skip_special_tokens=True)
            for row_idx, problem_id in enumerate(batch["problem_ids"]):
                parsed_reward, parse_error = _parse_generated_scalar_reward(generated_texts[row_idx])
                parse_success = parsed_reward is not None
                if not parse_success:
                    parsed_reward = float(parse_failure_value)
                if clip_predictions:
                    parsed_reward = float(min(1.0, max(0.0, parsed_reward)))
                target_reward = float(batch["targets"][row_idx].item())
                local_squared_error_sum += float((parsed_reward - target_reward) ** 2)
                local_value_count += 1
                prediction_row = {
                    "problem_id": problem_id,
                    "dataset": batch["datasets"][row_idx],
                    "repo": batch["repos"][row_idx],
                    "language": batch["languages"][row_idx],
                    "route_aliases": list(batch["route_aliases"][row_idx]),
                    "route_idx": int(batch["route_indices"][row_idx].item()),
                    "route_alias": batch["route_alias"][row_idx],
                    "route_label": batch["route_label"][row_idx],
                    "true_reward": target_reward,
                    "generated_text": generated_texts[row_idx],
                    "parsed_reward": float(parsed_reward),
                    "pred_reward": float(parsed_reward),
                    "parse_success": bool(parse_success),
                }
                if parse_error is not None:
                    prediction_row["parse_error"] = parse_error
                prediction_rows.append(prediction_row)
    return prediction_rows, local_squared_error_sum, local_value_count


def _run_text_bin_reward_eval(
    model: Any,
    loader: DataLoader,
    bin_token_ids: list[int],
    bin_specs: list[dict[str, Any]],
    clip_predictions: bool,
) -> tuple[list[dict[str, Any]], float, int]:
    model.eval()
    prediction_rows: list[dict[str, Any]] = []
    local_squared_error_sum = 0.0
    local_value_count = 0
    bin_values = [float(spec["value"]) for spec in bin_specs]
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval offline router", unit="batch", disable=not is_main_process()):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=True,
            )
            last_indices = batch["attention_mask"].sum(dim=1).to(dtype=torch.long) - 1
            batch_indices = torch.arange(outputs.logits.shape[0], device=outputs.logits.device)
            next_logits = outputs.logits[batch_indices, last_indices]
            bin_token_id_tensor = torch.tensor(bin_token_ids, dtype=torch.long, device=outputs.logits.device)
            bin_value_tensor = torch.tensor(bin_values, dtype=torch.float32, device=outputs.logits.device)
            bin_logits = next_logits.index_select(dim=1, index=bin_token_id_tensor)
            bin_probs = torch.softmax(bin_logits.float(), dim=1)
            pred_rewards = torch.matmul(bin_probs, bin_value_tensor)
            if clip_predictions:
                pred_rewards = pred_rewards.clamp(0.0, 1.0)
            top_probs, top_indices = torch.max(bin_probs, dim=1)

            pred_rewards_cpu = pred_rewards.detach().cpu().tolist()
            top_probs_cpu = top_probs.detach().cpu().tolist()
            top_indices_cpu = top_indices.detach().cpu().tolist()
            bin_probs_cpu = bin_probs.detach().cpu().tolist()

            for row_idx, problem_id in enumerate(batch["problem_ids"]):
                pred_reward = float(pred_rewards_cpu[row_idx])
                target_reward = float(batch["targets"][row_idx].item())
                local_squared_error_sum += float((pred_reward - target_reward) ** 2)
                local_value_count += 1
                top_idx = int(top_indices_cpu[row_idx])
                prediction_row = {
                    "problem_id": problem_id,
                    "dataset": batch["datasets"][row_idx],
                    "repo": batch["repos"][row_idx],
                    "language": batch["languages"][row_idx],
                    "route_aliases": list(batch["route_aliases"][row_idx]),
                    "route_idx": int(batch["route_indices"][row_idx].item()),
                    "route_alias": batch["route_alias"][row_idx],
                    "route_label": batch["route_label"][row_idx],
                    "true_reward": target_reward,
                    "generated_text": str(bin_specs[top_idx]["label"]),
                    "parsed_reward": pred_reward,
                    "pred_reward": pred_reward,
                    "parse_success": True,
                    "pred_bin_idx": int(top_idx),
                    "pred_bin_label": str(bin_specs[top_idx]["label"]),
                    "pred_bin_value": float(bin_specs[top_idx]["value"]),
                    "pred_bin_probability": float(top_probs_cpu[row_idx]),
                    "bin_probabilities": {
                        str(spec["label"]): float(prob)
                        for spec, prob in zip(bin_specs, bin_probs_cpu[row_idx])
                    },
                }
                prediction_rows.append(prediction_row)
    return prediction_rows, local_squared_error_sum, local_value_count


def _run_text_pairwise_sign_eval(
    model: Any,
    loader: DataLoader,
    label_token_ids: list[int],
    label_specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float, int]:
    model.eval()
    prediction_rows: list[dict[str, Any]] = []
    local_squared_error_sum = 0.0
    local_value_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval offline router", unit="batch", disable=not is_main_process()):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=True,
            )
            last_indices = batch["attention_mask"].sum(dim=1).to(dtype=torch.long) - 1
            batch_indices = torch.arange(outputs.logits.shape[0], device=outputs.logits.device)
            next_logits = outputs.logits[batch_indices, last_indices]
            label_token_id_tensor = torch.tensor(label_token_ids, dtype=torch.long, device=outputs.logits.device)
            label_logits = next_logits.index_select(dim=1, index=label_token_id_tensor)
            label_probs = torch.softmax(label_logits.float(), dim=1)
            top_probs, top_indices = torch.max(label_probs, dim=1)

            local_squared_error_sum += float(torch.sum((label_probs - batch["targets"]) ** 2).item())
            local_value_count += int(batch["targets"].numel())

            label_probs_cpu = label_probs.detach().cpu().tolist()
            top_probs_cpu = top_probs.detach().cpu().tolist()
            top_indices_cpu = top_indices.detach().cpu().tolist()

            for row_idx, problem_id in enumerate(batch["problem_ids"]):
                pred_pairwise_probabilities = [float(value) for value in label_probs_cpu[row_idx]]
                pred_spec = label_specs[int(top_indices_cpu[row_idx])]
                pred_winner_idx = int(pred_spec["winner_idx"])
                prediction_rows.append(
                    {
                        "problem_id": problem_id,
                        "dataset": batch["datasets"][row_idx],
                        "repo": batch["repos"][row_idx],
                        "language": batch["languages"][row_idx],
                        "route_aliases": list(batch["route_aliases"][row_idx]),
                        "true_rewards": [float(value) for value in batch["true_rewards"][row_idx].tolist()],
                        "true_pairwise_targets": [float(value) for value in batch["targets"][row_idx].tolist()],
                        "generated_text": str(pred_spec["label"]),
                        "pred_pairwise_probabilities": pred_pairwise_probabilities,
                        "pred_rewards": pred_pairwise_probabilities,
                        "parse_success": True,
                        "pred_winner_idx": pred_winner_idx,
                        "pred_winner_alias": batch["route_aliases"][row_idx][pred_winner_idx],
                        "pred_label_probability": float(top_probs_cpu[row_idx]),
                        "pred_delta": float(
                            pred_pairwise_probabilities[0] - pred_pairwise_probabilities[1]
                        ),
                        "target_winner_idx": int(batch["winner_indices"][row_idx].item()),
                        "target_winner_alias": batch["winner_alias"][row_idx],
                        "target_winner_label": batch["winner_label"][row_idx],
                        "target_label": batch["target_label"][row_idx],
                        "label_probabilities": {
                            str(spec["label"]): float(prob)
                            for spec, prob in zip(label_specs, pred_pairwise_probabilities)
                        },
                    }
                )
    return prediction_rows, local_squared_error_sum, local_value_count


def _save_offline_router_checkpoint(
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    supervision_mode: str,
) -> None:
    if supervision_mode in TEXT_LM_SUPERVISION_MODES:
        save_model_and_tokenizer(output_dir, model, tokenizer, lora=True)
        return
    save_model_only(output_dir, model)


def _save_predictions_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_predictions_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _eval_shard_dir(output_dir: Path, epoch: int) -> Path:
    return output_dir / ".eval_shards" / f"epoch_{epoch:04d}"


def _eval_shard_path(output_dir: Path, epoch: int, rank: int) -> Path:
    return _eval_shard_dir(output_dir, epoch) / f"rank_{rank:05d}.jsonl"


def _write_eval_shard(output_dir: Path, epoch: int, rows: list[dict[str, Any]]) -> Path:
    shard_path = _eval_shard_path(output_dir, epoch, get_rank())
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    _save_predictions_jsonl(shard_path, rows)
    return shard_path


def _load_eval_shard_rows(output_dir: Path, epoch: int) -> list[dict[str, Any]]:
    shard_dir = _eval_shard_dir(output_dir, epoch)
    rows: list[dict[str, Any]] = []
    for shard_path in sorted(shard_dir.glob("rank_*.jsonl")):
        rows.extend(_read_predictions_jsonl(shard_path))
    return rows


def _cleanup_eval_shards(output_dir: Path, epoch: int) -> None:
    shard_dir = _eval_shard_dir(output_dir, epoch)
    if shard_dir.exists():
        shutil.rmtree(shard_dir)


def _reduce_scalar_sum(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=get_accelerator().device)
    reduced = get_accelerator().reduce(tensor, reduction="sum")
    return float(reduced.item())


def _merge_representation_eval_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    merged_rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for row in rows:
        key = _prediction_row_key(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_rows.append(row)

    if not merged_rows:
        raise ValueError("No valid eval examples after tokenization/collation")

    y_true = np.asarray([row["true_rewards"] for row in merged_rows], dtype=np.float32)
    y_pred = np.asarray([row["pred_rewards"] for row in merged_rows], dtype=np.float32)
    return merged_rows, y_true, y_pred, {"parse_failures": 0, "parse_failure_rate": 0.0}


def _merge_text_reward_eval_rows(
    rows: list[dict[str, Any]],
    route_labels: list[str],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    merged_problem_rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    total_parse_failures = 0
    for row in rows:
        key = _prediction_row_key(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_problem_rows.append(row)
        if not bool(row.get("parse_success", False)):
            total_parse_failures += 1

    if not merged_problem_rows:
        raise ValueError("No valid eval examples after tokenization/collation")

    filtered_rows: list[dict[str, Any]] = []
    for row in merged_problem_rows:
        true_rewards = row.get("true_rewards")
        pred_rewards = row.get("pred_rewards")
        route_aliases = row.get("route_aliases")
        if (
            not isinstance(true_rewards, list)
            or not isinstance(pred_rewards, list)
            or len(true_rewards) != len(route_labels)
            or len(pred_rewards) != len(route_labels)
            or not isinstance(route_aliases, list)
            or len(route_aliases) != len(route_labels)
        ):
            continue
        normalized_row = dict(row)
        normalized_row["route_labels"] = list(route_labels)
        filtered_rows.append(normalized_row)

    if not filtered_rows:
        raise ValueError("No complete eval problems after parsing text reward predictions")

    y_true = np.asarray([row["true_rewards"] for row in filtered_rows], dtype=np.float32)
    y_pred = np.asarray([row["pred_rewards"] for row in filtered_rows], dtype=np.float32)
    parse_failure_rate = float(total_parse_failures / len(filtered_rows)) if filtered_rows else 0.0
    return (
        filtered_rows,
        y_true,
        y_pred,
        {
            "parse_failures": int(total_parse_failures),
            "parse_failure_rate": parse_failure_rate,
            "eval_problem_examples": int(len(filtered_rows)),
        },
    )


def _merge_text_scalar_reward_eval_rows(
    rows: list[dict[str, Any]],
    route_labels: list[str],
    route_aliases: list[str],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    target_dim = len(route_labels)
    seen_route_keys: set[str] = set()
    total_parse_failures = 0
    total_route_examples = 0
    grouped: dict[str, dict[str, Any]] = {}

    for row in rows:
        route_key = _prediction_row_key(row)
        if route_key in seen_route_keys:
            continue
        seen_route_keys.add(route_key)
        total_route_examples += 1
        if not bool(row.get("parse_success", False)):
            total_parse_failures += 1

        try:
            route_idx = int(row.get("route_idx"))
        except (TypeError, ValueError):
            continue
        if route_idx < 0 or route_idx >= target_dim:
            continue

        problem_key = _prediction_problem_key(row)
        entry = grouped.get(problem_key)
        if entry is None:
            entry = {
                "problem_id": row.get("problem_id"),
                "dataset": row.get("dataset"),
                "repo": row.get("repo"),
                "language": row.get("language"),
                "route_aliases": list(route_aliases),
                "route_labels": list(route_labels),
                "true_rewards": [None] * target_dim,
                "pred_rewards": [None] * target_dim,
                "route_predictions": [None] * target_dim,
            }
            grouped[problem_key] = entry

        true_reward = row.get("true_reward")
        pred_reward = row.get("pred_reward")
        if true_reward is None or pred_reward is None:
            continue
        entry["true_rewards"][route_idx] = float(true_reward)
        entry["pred_rewards"][route_idx] = float(pred_reward)
        entry["route_predictions"][route_idx] = {
            "route_idx": int(route_idx),
            "route_alias": row.get("route_alias"),
            "route_label": row.get("route_label"),
            "true_reward": float(true_reward),
            "generated_text": row.get("generated_text"),
            "parsed_reward": row.get("parsed_reward"),
            "pred_reward": float(pred_reward),
            "parse_success": bool(row.get("parse_success", False)),
            "parse_error": row.get("parse_error"),
        }
        for optional_key in (
            "pred_bin_idx",
            "pred_bin_label",
            "pred_bin_value",
            "pred_bin_probability",
            "bin_probabilities",
        ):
            if optional_key in row:
                entry["route_predictions"][route_idx][optional_key] = row.get(optional_key)

    filtered_rows: list[dict[str, Any]] = []
    for row in grouped.values():
        true_rewards = row["true_rewards"]
        pred_rewards = row["pred_rewards"]
        if any(value is None for value in true_rewards) or any(value is None for value in pred_rewards):
            continue
        normalized_row = dict(row)
        normalized_row["true_rewards"] = [float(value) for value in true_rewards]
        normalized_row["pred_rewards"] = [float(value) for value in pred_rewards]
        filtered_rows.append(normalized_row)

    if not filtered_rows:
        raise ValueError("No complete eval problems after parsing scalar text reward predictions")

    y_true = np.asarray([row["true_rewards"] for row in filtered_rows], dtype=np.float32)
    y_pred = np.asarray([row["pred_rewards"] for row in filtered_rows], dtype=np.float32)
    parse_failure_rate = float(total_parse_failures / total_route_examples) if total_route_examples else 0.0
    return (
        filtered_rows,
        y_true,
        y_pred,
        {
            "parse_failures": int(total_parse_failures),
            "parse_failure_rate": parse_failure_rate,
            "eval_problem_examples": int(len(filtered_rows)),
            "eval_route_examples": int(total_route_examples),
        },
    )


def _merge_text_pairwise_sign_eval_rows(
    rows: list[dict[str, Any]],
    route_labels: list[str],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    target_dim = len(route_labels)
    merged_problem_rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for row in rows:
        key = _prediction_row_key(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_problem_rows.append(row)

    if not merged_problem_rows:
        raise ValueError("No valid eval examples after tokenization/collation")

    filtered_rows: list[dict[str, Any]] = []
    for row in merged_problem_rows:
        true_pairwise_targets = row.get("true_pairwise_targets")
        pred_pairwise_probabilities = row.get("pred_pairwise_probabilities")
        true_rewards = row.get("true_rewards")
        route_aliases = row.get("route_aliases")
        if (
            not isinstance(true_pairwise_targets, list)
            or not isinstance(pred_pairwise_probabilities, list)
            or not isinstance(true_rewards, list)
            or not isinstance(route_aliases, list)
            or len(true_pairwise_targets) != target_dim
            or len(pred_pairwise_probabilities) != target_dim
            or len(true_rewards) != target_dim
            or len(route_aliases) != target_dim
        ):
            continue
        normalized_row = dict(row)
        normalized_row["route_labels"] = list(route_labels)
        filtered_rows.append(normalized_row)

    if not filtered_rows:
        raise ValueError("No complete eval problems after parsing pairwise text predictions")

    y_true = np.asarray([row["true_pairwise_targets"] for row in filtered_rows], dtype=np.float32)
    y_pred = np.asarray([row["pred_pairwise_probabilities"] for row in filtered_rows], dtype=np.float32)
    return (
        filtered_rows,
        y_true,
        y_pred,
        {
            "parse_failures": 0,
            "parse_failure_rate": 0.0,
            "eval_problem_examples": int(len(filtered_rows)),
            "eval_route_examples": int(len(filtered_rows) * target_dim),
        },
    )


def _flatten_dict(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, subvalue in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_dict(subvalue, next_prefix))
    else:
        result[prefix] = value
    return result


def _init_offline_wandb(cfg: DictConfig, output_dir: Path):
    if not is_main_process():
        return None
    if not bool(cfg.wandb.use_wandb):
        return None
    config_for_wandb = sanitize_for_json(OmegaConf.to_container(cfg, resolve=True))
    wandb_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    root = wandb_cfg.wandb.wandb_workspace_root
    if root and not str(output_dir).startswith(str(root) + "/"):
        wandb_cfg.wandb.wandb_workspace_root = ""
    try:
        run = init_wandb(wandb_cfg, output_dir, config_for_wandb)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to initialize W&B for offline router training: %s", exc)
        return None

    wandb_info = {
        "name": run.name[:128] if run.name else None,
        "entity": run.entity,
        "project": run.project_name(),
        "id": run.id,
    }
    with open(os.path.join(output_dir, "wandb_info.json"), "w") as handle:
        json.dump(wandb_info, handle, indent=2)
    return run


def _log_epoch_to_wandb(
    epoch: int,
    step: int,
    train_loss: float,
    eval_loss: float,
    route_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> None:
    metrics: dict[str, Any] = {
        "offline_router/train_loss": train_loss,
        "offline_router/eval_loss": eval_loss,
    }
    for row in route_rows:
        label = str(row["route_label"]).replace(":", "_").replace("/", "_")
        for key in ("pearson", "spearman", "r2", "mae", "rmse", "mean_true", "mean_pred", "std_true", "std_pred"):
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics[f"offline_router/route/{label}/{key}"] = float(value)
    for row in pair_rows:
        left = str(row["left_label"]).replace(":", "_").replace("/", "_")
        right = str(row["right_label"]).replace(":", "_").replace("/", "_")
        pair_label = f"{left}_vs_{right}"
        for key in ("pearson_delta", "spearman_delta", "delta_mae", "ranking_accuracy_sign", "roc_auc"):
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics[f"offline_router/pair/{pair_label}/{key}"] = float(value)
    wandb.log(metrics, step=step)


def _log_train_progress_to_wandb(
    epoch: int,
    step: int,
    global_step: int,
    running_train_loss: float,
    progress_fraction: float,
    examples_per_sec: float,
    eta_seconds: float,
) -> None:
    metrics = {
        "offline_router/train_loss_running": running_train_loss,
        "offline_router/train_progress_fraction": progress_fraction,
        "offline_router/examples_per_sec": examples_per_sec,
        "offline_router/eta_seconds": eta_seconds,
        "offline_router/epoch": epoch,
        "offline_router/step_in_epoch": step,
    }
    wandb.log(metrics, step=global_step)


@hydra.main(config_path="../../../../conf", config_name="offline_router_train", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    had_error = False
    try:
        train_cfg = cfg.offline_router.train
        grad_accum = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
        configure_accelerator(
            gradient_accumulation_steps=grad_accum,
            dataloader_config=DataLoaderConfiguration(even_batches=True),
        )
        base_seed = int(cfg.get("seed", 42))
        _set_training_seed(base_seed)
        logger.info("Offline router training seed=%d", base_seed)
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if is_main_process():
            write_json(output_dir / "train_config.json", sanitize_for_json(OmegaConf.to_container(cfg, resolve=True)))
        wandb_run = _init_offline_wandb(cfg, output_dir)

        dataset_dir = Path(str(train_cfg.dataset_dir))
        metadata_path = dataset_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Offline router metadata missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        route_labels = list(metadata.get("route_labels") or [])
        if not route_labels:
            raise ValueError("metadata.json is missing route_labels")
        route_aliases = _route_prompt_aliases(route_labels)
        target_dim = len(route_labels)
        supervision_mode = str(train_cfg.get("supervision_mode", "representation_head"))
        if supervision_mode not in {"representation_head", *TEXT_LM_SUPERVISION_MODES}:
            raise ValueError(f"Unsupported offline_router.train.supervision_mode: {supervision_mode}")
        text_reward_cfg = train_cfg.get("text_reward")
        if text_reward_cfg is None:
            text_reward_cfg = {}
        text_pairwise_cfg = train_cfg.get("text_pairwise")
        if text_pairwise_cfg is None:
            text_pairwise_cfg = {}
        utility_cfg = train_cfg.get("utility")
        if utility_cfg is None:
            utility_cfg = {}
        text_target_precision = int(text_reward_cfg.get("target_precision", 2))
        text_target_grid_count_raw = text_reward_cfg.get("target_grid_count")
        text_target_grid_count = (
            int(text_target_grid_count_raw) if text_target_grid_count_raw is not None else None
        )
        if text_target_grid_count is not None and text_target_grid_count < 2:
            raise ValueError("offline_router.train.text_reward.target_grid_count must be at least 2")
        text_lr = float(text_reward_cfg.get("lr", 5.0e-6))
        text_weight_decay = float(text_reward_cfg.get("weight_decay", 0.01))
        text_warmup_steps = int(text_reward_cfg.get("warmup_steps", 100))
        text_gradient_clipping = float(text_reward_cfg.get("gradient_clipping", 0.3))
        text_debug_step_logging = bool(text_reward_cfg.get("debug_step_logging", False))
        text_max_new_tokens = int(text_reward_cfg.get("max_new_tokens", 16))
        text_do_sample = bool(text_reward_cfg.get("do_sample", False))
        text_parse_failure_value = float(text_reward_cfg.get("parse_failure_value", 0.0))
        text_clip_predictions = bool(text_reward_cfg.get("clip_predictions", True))
        text_drop_overlength_rows = bool(text_reward_cfg.get("drop_overlength_rows", False))
        text_reward_bin_count = int(text_reward_cfg.get("bin_count", 21))
        text_reward_bin_label_prefix = str(text_reward_cfg.get("bin_label_prefix", " "))
        text_reward_bin_value_order = str(text_reward_cfg.get("bin_value_order", "ascending"))
        text_pairwise_tie_margin = float(text_pairwise_cfg.get("tie_margin", 0.05))
        if text_pairwise_tie_margin < 0.0:
            raise ValueError("offline_router.train.text_pairwise.tie_margin must be non-negative")
        text_pairwise_label_prefix = str(text_pairwise_cfg.get("label_prefix", " "))
        text_pairwise_primary_better_label = str(text_pairwise_cfg.get("primary_better_label", "A"))
        text_pairwise_expert_better_label = str(text_pairwise_cfg.get("expert_better_label", "B"))
        utility_enabled = bool(utility_cfg.get("enabled", True))
        utility_lambdas = _normalize_utility_lambdas(utility_cfg.get("lambdas"))
        text_reward_bin_specs = (
            _build_reward_bin_specs(
                text_reward_bin_count,
                label_prefix=text_reward_bin_label_prefix,
                value_order=text_reward_bin_value_order,
            )
            if supervision_mode == "text_reward_bin"
            else []
        )
        text_pairwise_label_specs = (
            _build_pairwise_label_specs(
                label_prefix=text_pairwise_label_prefix,
                primary_better_label=text_pairwise_primary_better_label,
                expert_better_label=text_pairwise_expert_better_label,
            )
            if supervision_mode == "text_pairwise_sign"
            else []
        )
        text_reward_bin_token_ids: list[int] = []
        text_pairwise_label_token_ids: list[int] = []
        train_sampling_strategy = str(train_cfg.get("train_sampling_strategy", "random"))

        train_dataset = _load_split(dataset_dir, "train")
        eval_dataset = _load_split(dataset_dir, "eval")

        max_train_rows = train_cfg.get("max_train_rows")
        max_eval_rows = train_cfg.get("max_eval_rows")
        train_dataset, train_sampling_summary = _sample_train_dataset(
            train_dataset,
            max_rows=int(max_train_rows) if max_train_rows else None,
            seed=base_seed,
            strategy=train_sampling_strategy,
        )
        eval_dataset = _shuffle_and_truncate_dataset(
            eval_dataset,
            max_rows=int(max_eval_rows) if max_eval_rows else None,
            seed=base_seed + 1,
            split_name="eval",
        )
        raw_eval_dataset = eval_dataset
        raw_train_rows = int(len(train_dataset))
        raw_eval_rows = int(len(eval_dataset))
        train_supervision_rows = int(len(train_dataset))
        eval_supervision_rows = int(len(eval_dataset))
        if supervision_mode in {"text_reward_scalar", "text_reward_bin"}:
            train_supervision_rows *= target_dim
            eval_supervision_rows *= target_dim

        device = _pick_device(str(train_cfg.get("device", "auto")))
        model_path = str(train_cfg.model_path)
        text_lora_base_model_path: str | None = None
        text_lora_adapter_source: str | None = None
        text_lora_config_summary: dict[str, Any] | None = None
        performance_value_hidden_dims = train_cfg.get("performance_value_hidden_dims")
        if performance_value_hidden_dims is not None:
            performance_value_hidden_dims = [int(dim) for dim in performance_value_hidden_dims]
        performance_value_activation = str(train_cfg.get("performance_value_activation", "gelu"))
        logger.info(
            "Loading tokenizer/model from %s on %s (rank=%d world_size=%d)",
            model_path,
            device,
            get_rank(),
            get_world_size(),
        )
        if supervision_mode == "representation_head":
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_path,
                performance_value_dim=target_dim,
                performance_value_hidden_dims=performance_value_hidden_dims,
                performance_value_activation=performance_value_activation,
            )

            if bool(train_cfg.get("gradient_checkpointing", False)):
                model.gradient_checkpointing_enable()
            if hasattr(model.pretrained_model.config, "use_cache"):
                model.pretrained_model.config.use_cache = False

            model_dtype = torch.bfloat16 if device.type == "cuda" and bool(train_cfg.get("bf16", True)) else None
            if model_dtype is not None:
                model = model.to(device=device, dtype=model_dtype)
            else:
                model = model.to(device=device)
            _cast_auxiliary_heads_to_fp32(model, device)
            trainable_prefixes = _configure_training_mode(
                model,
                training_mode=str(train_cfg.mode),
                supervision_mode=supervision_mode,
            )
        else:
            logger.warning(
                "offline_router.train.supervision_mode=%s always uses LoRA adapters; ignoring mode=%s",
                supervision_mode,
                train_cfg.mode,
            )
            tokenizer, model, text_lora_base_model_path, text_lora_adapter_source, text_lora_config_summary = _load_text_lora_model(
                model_path=model_path,
                train_cfg=train_cfg,
                device=device,
            )
            trainable_prefixes = [
                f"lora:{target_module}" for target_module in (text_lora_config_summary or {}).get("target_modules", [])
            ] or ["lora_adapters"]
        if supervision_mode == "text_reward_bin":
            text_reward_bin_token_ids = _reward_bin_token_ids(tokenizer, text_reward_bin_specs)
            logger.info(
                "Offline router reward-bin labels=%s token_ids=%s",
                [str(spec["label"]) for spec in text_reward_bin_specs],
                text_reward_bin_token_ids,
            )
        if supervision_mode == "text_pairwise_sign":
            if target_dim != 2:
                raise ValueError("offline_router.train.supervision_mode=text_pairwise_sign requires exactly two routes")
            text_pairwise_label_token_ids = _label_target_token_ids(tokenizer, text_pairwise_label_specs)
            logger.info(
                "Offline router pairwise labels=%s token_ids=%s",
                [str(spec["label"]) for spec in text_pairwise_label_specs],
                text_pairwise_label_token_ids,
            )
        total_parameters = count_parameters(model, trainable_only=False)
        trainable_parameters = count_parameters(model, trainable_only=True)

        logger.info(
            "Offline router mode=%s supervision_mode=%s trainable_prefixes=%s trainable_params=%d total_params=%d",
            train_cfg.mode,
            supervision_mode,
            trainable_prefixes,
            trainable_parameters,
            total_parameters,
        )
        if supervision_mode == "representation_head":
            logger.info(
                "Offline router performance head hidden_dims=%s activation=%s value_head_dtype=%s performance_value_head_dtype=%s",
                performance_value_hidden_dims or [],
                performance_value_activation,
                next(model.value_head.parameters()).dtype,
                next(model.performance_value_head.parameters()).dtype,
            )
        else:
            logger.info(
                "Offline router text LoRA settings: base_model_path=%s adapter_source=%s config=%s",
                text_lora_base_model_path,
                text_lora_adapter_source,
                text_lora_config_summary,
            )
        if supervision_mode in TEXT_LM_SUPERVISION_MODES:
            logger.info(
                "Offline router text LM settings: target_precision=%d target_grid_count=%s lr=%.2e weight_decay=%.3f warmup_steps=%d gradient_clipping=%.3f debug_step_logging=%s max_new_tokens=%d do_sample=%s parse_failure_value=%.3f clip_predictions=%s drop_overlength_rows=%s train_sampling_strategy=%s train_rows=%d train_supervision_rows=%d eval_rows=%d eval_supervision_rows=%d",
                text_target_precision,
                text_target_grid_count,
                text_lr,
                text_weight_decay,
                text_warmup_steps,
                text_gradient_clipping,
                text_debug_step_logging,
                text_max_new_tokens,
                text_do_sample,
                text_parse_failure_value,
                text_clip_predictions,
                text_drop_overlength_rows,
                train_sampling_strategy,
                raw_train_rows,
                train_supervision_rows,
                raw_eval_rows,
                eval_supervision_rows,
            )
        if supervision_mode == "text_reward_bin":
            logger.info(
                "Offline router text reward bin settings: bin_count=%d label_prefix=%r value_order=%s bin_values=%s",
                text_reward_bin_count,
                text_reward_bin_label_prefix,
                text_reward_bin_value_order,
                [float(spec["value"]) for spec in text_reward_bin_specs],
            )
        if supervision_mode == "text_pairwise_sign":
            logger.info(
                "Offline router text pairwise settings: tie_margin=%.3f label_prefix=%r labels=%s",
                text_pairwise_tie_margin,
                text_pairwise_label_prefix,
                [str(spec["label"]) for spec in text_pairwise_label_specs],
            )
        optimizer, optimizer_group_summaries = _build_optimizer(
            model,
            train_cfg,
            supervision_mode=supervision_mode,
            text_reward_cfg=text_reward_cfg,
        )
        logger.info("Offline router optimizer groups=%s", optimizer_group_summaries)

        max_seq_length = int(train_cfg.max_seq_length) if train_cfg.get("max_seq_length") else None
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id for offline router training")
        dropped_overlength_train_rows = 0
        dropped_overlength_eval_rows = 0
        dropped_tie_train_rows = 0
        dropped_tie_eval_rows = 0

        if supervision_mode == "representation_head":
            train_dataset = _prepare_representation_rows(
                train_dataset,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                target_dim=target_dim,
            )
            eval_dataset = _prepare_representation_rows(
                eval_dataset,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                target_dim=target_dim,
            )
            train_collate_fn = _make_collate_fn(
                pad_token_id=pad_token_id,
                target_dim=target_dim,
            )
            eval_collate_fn = train_collate_fn
        elif supervision_mode == "text_reward_vector":
            train_dataset, dropped_overlength_train_rows = _prepare_text_train_rows(
                train_dataset,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                route_aliases=route_aliases,
                target_dim=target_dim,
                target_precision=text_target_precision,
                target_grid_count=text_target_grid_count,
                drop_overlength=text_drop_overlength_rows,
            )
            eval_dataset, dropped_overlength_eval_rows = _prepare_text_eval_rows(
                eval_dataset,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                route_aliases=route_aliases,
                target_dim=target_dim,
                target_grid_count=text_target_grid_count,
                target_precision=text_target_precision,
                drop_overlength=text_drop_overlength_rows,
            )
            train_collate_fn = _make_text_train_collate_fn(pad_token_id=pad_token_id, target_dim=target_dim)
            eval_collate_fn = _make_text_eval_collate_fn(pad_token_id=pad_token_id, target_dim=target_dim)
        elif supervision_mode == "text_reward_scalar":
            train_dataset, dropped_overlength_train_rows = _prepare_text_scalar_train_rows(
                train_dataset,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                route_aliases=route_aliases,
                route_labels=route_labels,
                target_dim=target_dim,
                target_precision=text_target_precision,
                target_grid_count=text_target_grid_count,
                drop_overlength=text_drop_overlength_rows,
            )
            eval_dataset, dropped_overlength_eval_rows = _prepare_text_scalar_eval_rows(
                eval_dataset,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                route_aliases=route_aliases,
                route_labels=route_labels,
                target_dim=target_dim,
                target_grid_count=text_target_grid_count,
                target_precision=text_target_precision,
                drop_overlength=text_drop_overlength_rows,
            )
            train_collate_fn = _make_text_scalar_train_collate_fn(pad_token_id=pad_token_id)
            eval_collate_fn = _make_text_scalar_eval_collate_fn(pad_token_id=pad_token_id)
        elif supervision_mode == "text_reward_bin":
            train_dataset, dropped_overlength_train_rows = _prepare_text_bin_train_rows(
                train_dataset,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                route_aliases=route_aliases,
                route_labels=route_labels,
                target_dim=target_dim,
                target_precision=text_target_precision,
                bin_specs=text_reward_bin_specs,
                drop_overlength=text_drop_overlength_rows,
            )
            eval_dataset, dropped_overlength_eval_rows = _prepare_text_bin_eval_rows(
                eval_dataset,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                route_aliases=route_aliases,
                route_labels=route_labels,
                target_dim=target_dim,
                target_precision=text_target_precision,
                bin_specs=text_reward_bin_specs,
                drop_overlength=text_drop_overlength_rows,
            )
            train_collate_fn = _make_text_scalar_train_collate_fn(pad_token_id=pad_token_id)
            eval_collate_fn = _make_text_scalar_eval_collate_fn(pad_token_id=pad_token_id)
        elif supervision_mode == "text_pairwise_sign":
            train_dataset, dropped_overlength_train_rows, dropped_tie_train_rows = _prepare_text_pairwise_sign_train_rows(
                train_dataset,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                route_aliases=route_aliases,
                route_labels=route_labels,
                target_dim=target_dim,
                label_specs=text_pairwise_label_specs,
                tie_margin=text_pairwise_tie_margin,
                drop_overlength=text_drop_overlength_rows,
            )
            eval_dataset, dropped_overlength_eval_rows, dropped_tie_eval_rows = _prepare_text_pairwise_sign_eval_rows(
                eval_dataset,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                route_aliases=route_aliases,
                route_labels=route_labels,
                target_dim=target_dim,
                label_specs=text_pairwise_label_specs,
                tie_margin=text_pairwise_tie_margin,
                drop_overlength=text_drop_overlength_rows,
            )
            train_collate_fn = _make_text_pairwise_train_collate_fn(
                pad_token_id=pad_token_id,
                target_dim=target_dim,
            )
            eval_collate_fn = _make_text_pairwise_eval_collate_fn(
                pad_token_id=pad_token_id,
                target_dim=target_dim,
            )
        else:
            raise ValueError(f"Unsupported supervision mode: {supervision_mode}")

        if not train_dataset:
            raise ValueError("No valid training rows remain after preprocessing/tokenization")
        if not eval_dataset:
            raise ValueError("No valid eval rows remain after preprocessing/tokenization")

        preprocessed_train_rows = int(len(train_dataset))
        preprocessed_eval_rows = int(len(eval_dataset))

        logger.info(
            "Offline router preprocessed rows: train=%d eval=%d supervision_mode=%s",
            preprocessed_train_rows,
            preprocessed_eval_rows,
            supervision_mode,
        )
        if supervision_mode in TEXT_LM_SUPERVISION_MODES and text_drop_overlength_rows:
            logger.info(
                "Offline router dropped overlength text rows: train=%d eval=%d max_seq_length=%s supervision_mode=%s",
                dropped_overlength_train_rows,
                dropped_overlength_eval_rows,
                max_seq_length,
                supervision_mode,
            )
        if supervision_mode == "text_pairwise_sign":
            logger.info(
                "Offline router dropped near-tie pairwise rows: train=%d eval=%d tie_margin=%.3f",
                dropped_tie_train_rows,
                dropped_tie_eval_rows,
                text_pairwise_tie_margin,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(train_cfg.batch_size),
            shuffle=bool(train_cfg.get("shuffle_train", True)),
            num_workers=int(train_cfg.get("num_workers", 0)),
            collate_fn=train_collate_fn,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=int(train_cfg.eval_batch_size),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
            collate_fn=eval_collate_fn,
        )
        model, optimizer, train_loader, eval_loader = get_accelerator().prepare(
            model,
            optimizer,
            train_loader,
            eval_loader,
        )
        _log_auxiliary_head_dtypes("post_prepare", model)
        logger.info(
            "Offline router accelerator prepared: distributed=%s world_size=%d deepspeed=%s",
            is_distributed(),
            get_world_size(),
            getattr(get_accelerator().state, "deepspeed_plugin", None) is not None,
        )

        num_epochs = int(train_cfg.num_epochs)
        log_every_steps = max(1, int(train_cfg.get("log_every_steps", 100)))
        save_checkpoints = bool(train_cfg.get("save_checkpoints", True))
        text_scheduler = None
        text_trainable_params: list[torch.nn.Parameter] = []
        if supervision_mode in TEXT_LM_SUPERVISION_MODES:
            optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
            total_optimizer_steps = max(1, optimizer_steps_per_epoch * num_epochs)
            text_scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=min(text_warmup_steps, total_optimizer_steps),
            )
            text_trainable_params = [param for param in model.parameters() if param.requires_grad]
            logger.info(
                "Offline router text reward optimizer schedule: optimizer_steps_per_epoch=%d total_optimizer_steps=%d warmup_steps=%d gradient_clipping=%.3f",
                optimizer_steps_per_epoch,
                total_optimizer_steps,
                min(text_warmup_steps, total_optimizer_steps),
                text_gradient_clipping,
            )
        best_eval_loss = float("inf")
        best_epoch = -1
        history: list[dict[str, Any]] = []
        best_eval_rows: list[dict[str, Any]] = []
        best_y_true: np.ndarray | None = None
        best_y_pred: np.ndarray | None = None
        best_eval_extra: dict[str, Any] = {}

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_losses: list[float] = []
            batch_count = 0
            local_examples_seen = 0
            epoch_start_time = time.time()
            steps_per_epoch = max(1, len(train_loader))
            global_log_step_base = epoch * steps_per_epoch

            for step_idx, batch in enumerate(
                tqdm(train_loader, desc=f"Train epoch {epoch}", unit="batch", disable=not is_main_process()),
                start=1,
            ):
                with get_accelerator().accumulate(model):
                    if supervision_mode == "representation_head":
                        preds = _forward_predictions(model, batch)
                        loss = F.mse_loss(preds, batch["targets"])
                    else:
                        _log_text_train_step_debug(
                            text_debug_step_logging,
                            epoch,
                            step_idx,
                            len(train_loader),
                            "before_forward",
                            batch,
                            optimizer,
                        )
                        loss = _forward_text_reward_loss(model, batch)
                        _log_text_train_step_debug(
                            text_debug_step_logging,
                            epoch,
                            step_idx,
                            len(train_loader),
                            "after_forward",
                            batch,
                            optimizer,
                            loss=loss,
                        )
                    running_losses.append(float(loss.item()))
                    get_accelerator().backward(loss)
                    if supervision_mode in TEXT_LM_SUPERVISION_MODES:
                        _log_text_train_step_debug(
                            text_debug_step_logging,
                            epoch,
                            step_idx,
                            len(train_loader),
                            "after_backward",
                            batch,
                            optimizer,
                            loss=loss,
                        )
                    if (
                        supervision_mode in TEXT_LM_SUPERVISION_MODES
                        and get_accelerator().sync_gradients
                        and text_gradient_clipping > 0.0
                    ):
                        _log_text_train_step_debug(
                            text_debug_step_logging,
                            epoch,
                            step_idx,
                            len(train_loader),
                            "before_clip",
                            batch,
                            optimizer,
                            loss=loss,
                        )
                        get_accelerator().clip_grad_norm_(text_trainable_params, text_gradient_clipping)
                        _log_text_train_step_debug(
                            text_debug_step_logging,
                            epoch,
                            step_idx,
                            len(train_loader),
                            "after_clip",
                            batch,
                            optimizer,
                            loss=loss,
                        )
                    if supervision_mode in TEXT_LM_SUPERVISION_MODES:
                        _log_text_train_step_debug(
                            text_debug_step_logging,
                            epoch,
                            step_idx,
                            len(train_loader),
                            "before_optimizer_step",
                            batch,
                            optimizer,
                            loss=loss,
                        )
                    optimizer.step()
                    if supervision_mode in TEXT_LM_SUPERVISION_MODES:
                        _log_text_train_step_debug(
                            text_debug_step_logging,
                            epoch,
                            step_idx,
                            len(train_loader),
                            "after_optimizer_step",
                            batch,
                            optimizer,
                            loss=loss,
                        )
                    if text_scheduler is not None and get_accelerator().sync_gradients:
                        text_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if supervision_mode in TEXT_LM_SUPERVISION_MODES:
                        _log_text_train_step_debug(
                            text_debug_step_logging,
                            epoch,
                            step_idx,
                            len(train_loader),
                            "after_zero_grad",
                            batch,
                            optimizer,
                            loss=loss,
                        )
                batch_count += 1
                local_examples_seen += int(batch["input_ids"].shape[0])

                if is_main_process() and (step_idx % log_every_steps == 0 or step_idx == len(train_loader)):
                    elapsed = max(time.time() - epoch_start_time, 1e-6)
                    running_train_loss = float(np.mean(running_losses)) if running_losses else math.nan
                    progress_fraction = step_idx / max(len(train_loader), 1)
                    examples_per_sec = (local_examples_seen * get_world_size()) / elapsed
                    eta_seconds = ((len(train_loader) - step_idx) / max(step_idx, 1)) * elapsed
                    logger.info(
                        "Offline router epoch=%d step=%d/%d running_train_loss=%.6f processed_examples=%d examples_per_sec=%.2f eta_seconds=%.1f",
                        epoch,
                        step_idx,
                        len(train_loader),
                        running_train_loss,
                        local_examples_seen * get_world_size(),
                        examples_per_sec,
                        eta_seconds,
                    )
                    if wandb_run is not None:
                        try:
                            _log_train_progress_to_wandb(
                                epoch=epoch,
                                step=step_idx,
                                global_step=global_log_step_base + step_idx,
                                running_train_loss=running_train_loss,
                                progress_fraction=progress_fraction,
                                examples_per_sec=examples_per_sec,
                                eta_seconds=eta_seconds,
                            )
                        except Exception as exc:  # pylint: disable=broad-except
                            logger.warning("Failed to log offline router progress to W&B: %s", exc)

            if not running_losses:
                raise ValueError("No valid training batches after tokenization/collation")
            logger.info(
                "Offline router epoch=%d rank=%d train loop complete local_batches=%d local_examples=%d local_loss_count=%d",
                epoch,
                get_rank(),
                batch_count,
                local_examples_seen,
                len(running_losses),
            )
            local_loss_sum = float(np.sum(running_losses))
            total_loss_sum = _reduce_scalar_sum(local_loss_sum)
            total_loss_count = int(round(_reduce_scalar_sum(float(len(running_losses)))))
            train_loss = total_loss_sum / total_loss_count if total_loss_count > 0 else float(np.mean(running_losses))
            logger.info(
                "Offline router epoch=%d rank=%d post_train_loss_reduce total_loss_count=%d train_loss=%.6f",
                epoch,
                get_rank(),
                total_loss_count,
                train_loss,
            )

            logger.info("Offline router epoch=%d rank=%d entering pre-eval barrier", epoch, get_rank())
            barrier_if_distributed()
            logger.info("Offline router epoch=%d rank=%d exited pre-eval barrier", epoch, get_rank())
            logger.info("Offline router epoch=%d rank=%d entering eval", epoch, get_rank())
            if is_main_process():
                logger.info("Offline router epoch=%d starting eval", epoch)
            eval_start_time = time.time()
            if supervision_mode == "representation_head":
                local_eval_rows, local_eval_squared_error_sum, local_eval_value_count = _run_representation_eval(
                    model,
                    eval_loader,
                )
            elif supervision_mode == "text_reward_vector":
                local_eval_rows, local_eval_squared_error_sum, local_eval_value_count = _run_text_reward_eval(
                    model,
                    eval_loader,
                    tokenizer=tokenizer,
                    max_new_tokens=text_max_new_tokens,
                    do_sample=text_do_sample,
                    parse_failure_value=text_parse_failure_value,
                    clip_predictions=text_clip_predictions,
                    target_dim=target_dim,
                )
            elif supervision_mode == "text_reward_scalar":
                local_eval_rows, local_eval_squared_error_sum, local_eval_value_count = _run_text_scalar_reward_eval(
                    model,
                    eval_loader,
                    tokenizer=tokenizer,
                    max_new_tokens=text_max_new_tokens,
                    do_sample=text_do_sample,
                    parse_failure_value=text_parse_failure_value,
                    clip_predictions=text_clip_predictions,
                )
            elif supervision_mode == "text_reward_bin":
                local_eval_rows, local_eval_squared_error_sum, local_eval_value_count = _run_text_bin_reward_eval(
                    model,
                    eval_loader,
                    bin_token_ids=text_reward_bin_token_ids,
                    bin_specs=text_reward_bin_specs,
                    clip_predictions=text_clip_predictions,
                )
            elif supervision_mode == "text_pairwise_sign":
                local_eval_rows, local_eval_squared_error_sum, local_eval_value_count = _run_text_pairwise_sign_eval(
                    model,
                    eval_loader,
                    label_token_ids=text_pairwise_label_token_ids,
                    label_specs=text_pairwise_label_specs,
                )
            else:
                raise ValueError(f"Unsupported supervision mode: {supervision_mode}")
            shard_path = _write_eval_shard(output_dir, epoch, local_eval_rows)
            logger.info(
                "Offline router epoch=%d rank=%d finished local eval rows=%d values=%d shard=%s",
                epoch,
                get_rank(),
                len(local_eval_rows),
                local_eval_value_count,
                shard_path,
            )
            total_eval_squared_error_sum = _reduce_scalar_sum(local_eval_squared_error_sum)
            total_eval_value_count = int(round(_reduce_scalar_sum(float(local_eval_value_count))))
            eval_loss = (
                total_eval_squared_error_sum / total_eval_value_count
                if total_eval_value_count > 0
                else math.nan
            )
            logger.info("Offline router epoch=%d rank=%d entering post-eval barrier", epoch, get_rank())
            barrier_if_distributed()
            logger.info("Offline router epoch=%d rank=%d exited post-eval barrier", epoch, get_rank())
            eval_rows: list[dict[str, Any]] = []
            y_true = np.empty((0, 0), dtype=np.float32)
            y_pred = np.empty((0, 0), dtype=np.float32)
            eval_extra: dict[str, Any] = {}
            if is_main_process():
                logger.info("Offline router epoch=%d merging eval shards from %s", epoch, _eval_shard_dir(output_dir, epoch))
                merged_shard_rows = _load_eval_shard_rows(output_dir, epoch)
                if supervision_mode == "representation_head":
                    eval_rows, y_true, y_pred, eval_extra = _merge_representation_eval_rows(merged_shard_rows)
                elif supervision_mode == "text_reward_vector":
                    eval_rows, y_true, y_pred, eval_extra = _merge_text_reward_eval_rows(
                        merged_shard_rows,
                        route_labels=route_labels,
                    )
                elif supervision_mode in {"text_reward_scalar", "text_reward_bin"}:
                    eval_rows, y_true, y_pred, eval_extra = _merge_text_scalar_reward_eval_rows(
                        merged_shard_rows,
                        route_labels=route_labels,
                        route_aliases=route_aliases,
                    )
                elif supervision_mode == "text_pairwise_sign":
                    eval_rows, y_true, y_pred, eval_extra = _merge_text_pairwise_sign_eval_rows(
                        merged_shard_rows,
                        route_labels=route_labels,
                    )
                else:
                    raise ValueError(f"Unsupported supervision mode: {supervision_mode}")
                logger.info(
                    "Offline router epoch=%d finished metric preparation merged_rows=%d elapsed_seconds=%.1f",
                    epoch,
                    len(eval_rows),
                    time.time() - eval_start_time,
                )
                _cleanup_eval_shards(output_dir, epoch)
            barrier_if_distributed()
            if is_main_process():
                if supervision_mode == "representation_head":
                    logger.info(
                        "Offline router epoch=%d finished eval eval_loss=%.6f eval_rows=%d elapsed_seconds=%.1f",
                        epoch,
                        eval_loss,
                        len(eval_rows),
                        time.time() - eval_start_time,
                    )
                else:
                    logger.info(
                        "Offline router epoch=%d finished eval eval_loss=%.6f eval_rows=%d eval_problem_examples=%d parse_failures=%d parse_failure_rate=%.4f elapsed_seconds=%.1f",
                        epoch,
                        eval_loss,
                        len(eval_rows),
                        int(eval_extra.get("eval_problem_examples", 0)),
                        int(eval_extra.get("parse_failures", 0)),
                        float(eval_extra.get("parse_failure_rate", 0.0)),
                        time.time() - eval_start_time,
                    )

            if is_main_process():
                epoch_route_rows = compute_per_route_metrics(y_true, y_pred, route_labels)
                epoch_pair_rows = compute_pairwise_metrics(y_true, y_pred, route_labels)
                epoch_log_step = global_log_step_base + steps_per_epoch
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "eval_loss": eval_loss,
                        "global_step": epoch_log_step,
                        "eval_parse_failures": int(eval_extra.get("parse_failures", 0)),
                        "eval_parse_failure_rate": float(eval_extra.get("parse_failure_rate", 0.0)),
                    }
                )
                logger.info(
                    "Offline router epoch=%d train_loss=%.6f eval_loss=%.6f",
                    epoch,
                    train_loss,
                    eval_loss,
                )
                if wandb_run is not None:
                    try:
                        _log_epoch_to_wandb(
                            epoch=epoch,
                            step=epoch_log_step,
                            train_loss=train_loss,
                            eval_loss=eval_loss,
                            route_rows=epoch_route_rows,
                            pair_rows=epoch_pair_rows,
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.warning("Failed to log offline router epoch %d to W&B: %s", epoch, exc)

            previous_best_eval_loss = best_eval_loss
            should_save_best = eval_loss < previous_best_eval_loss
            if should_save_best:
                best_eval_loss = eval_loss
                best_epoch = epoch
            if is_main_process() and should_save_best:
                best_eval_rows = eval_rows
                best_y_true = y_true
                best_y_pred = y_pred
                best_eval_extra = dict(eval_extra)
            if should_save_best and save_checkpoints:
                best_dir = output_dir / "checkpoints" / "best"
                if is_main_process():
                    logger.info("Offline router epoch=%d starting best checkpoint save to %s", epoch, best_dir)
                    best_dir.mkdir(parents=True, exist_ok=True)
                    save_start_time = time.time()
                barrier_if_distributed()
                _save_offline_router_checkpoint(best_dir, model, tokenizer, supervision_mode)
                barrier_if_distributed()
                if is_main_process():
                    logger.info(
                        "Offline router epoch=%d finished best checkpoint save elapsed_seconds=%.1f",
                        epoch,
                        time.time() - save_start_time,
                    )
            elif should_save_best and is_main_process():
                logger.info("Offline router epoch=%d skipping best checkpoint save because save_checkpoints=false", epoch)

            barrier_if_distributed()

        last_dir = output_dir / "checkpoints" / "last"
        if save_checkpoints:
            if is_main_process():
                logger.info("Offline router starting last checkpoint save to %s", last_dir)
                last_dir.mkdir(parents=True, exist_ok=True)
                save_start_time = time.time()
            barrier_if_distributed()
            _save_offline_router_checkpoint(last_dir, model, tokenizer, supervision_mode)
            barrier_if_distributed()
            if is_main_process():
                logger.info(
                    "Offline router finished last checkpoint save elapsed_seconds=%.1f",
                    time.time() - save_start_time,
                )
        elif is_main_process():
            logger.info("Offline router skipping last checkpoint save because save_checkpoints=false")

        if is_main_process():
            if best_y_true is None or best_y_pred is None:
                raise ValueError("Best eval predictions were never recorded")

            route_rows = compute_per_route_metrics(best_y_true, best_y_pred, route_labels)
            pair_rows = compute_pairwise_metrics(best_y_true, best_y_pred, route_labels)
            _write_csv(output_dir / "route_metrics.csv", route_rows, csv_headers_for_route_metrics())
            _write_csv(output_dir / "pairwise_metrics.csv", pair_rows, csv_headers_for_pairwise_metrics())
            _save_predictions_jsonl(output_dir / "eval_predictions.jsonl", best_eval_rows)
            utility_report = None
            if utility_enabled:
                utility_report = _compute_utility_report(
                    eval_rows=best_eval_rows,
                    eval_dataset=raw_eval_dataset,
                    route_labels=route_labels,
                    lambdas=utility_lambdas,
                )
                _write_csv(
                    output_dir / "utility_vs_baselines.csv",
                    utility_report["utility_rows"],
                    [
                        "policy",
                        "policy_type",
                        "route_idx",
                        "route_label",
                        "lambda",
                        "cost_metric",
                        "mean_reward",
                        "mean_cost",
                        "mean_utility",
                    ],
                )
                write_json(output_dir / "utility_vs_baselines.json", utility_report)
                logger.info(
                    "Offline router wrote utility report: json=%s csv=%s n_examples=%d",
                    output_dir / "utility_vs_baselines.json",
                    output_dir / "utility_vs_baselines.csv",
                    int(utility_report.get("n_examples_with_utility", 0)),
                )

            summary = {
                "seed": base_seed,
                "mode": str(train_cfg.mode),
                "supervision_mode": supervision_mode,
                "dataset_dir": str(dataset_dir),
                "model_path": model_path,
                "route_labels": route_labels,
                "target_dim": target_dim,
                "train_sampling": train_sampling_summary,
                "utility": {
                    "enabled": utility_enabled,
                    "lambdas": [float(value) for value in utility_lambdas],
                },
                "train_rows": raw_train_rows,
                "eval_rows": raw_eval_rows,
                "train_supervision_rows": train_supervision_rows,
                "eval_supervision_rows": eval_supervision_rows,
                "train_preprocessed_rows": preprocessed_train_rows,
                "eval_preprocessed_rows": preprocessed_eval_rows,
                "num_epochs": num_epochs,
                "performance_value_hidden_dims": performance_value_hidden_dims or [],
                "performance_value_activation": performance_value_activation,
                "optimizer_groups": optimizer_group_summaries,
                "best_epoch": best_epoch,
                "best_eval_loss": best_eval_loss,
                "trainable_prefixes": trainable_prefixes,
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
                "history": history,
                "best_checkpoint_dir": str(output_dir / "checkpoints" / "best") if save_checkpoints else None,
                "last_checkpoint_dir": str(last_dir) if save_checkpoints else None,
                "save_checkpoints": save_checkpoints,
                "distributed": is_distributed(),
                "world_size": get_world_size(),
            }
            if utility_report is not None:
                summary["utility"]["report_json"] = str(output_dir / "utility_vs_baselines.json")
                summary["utility"]["report_csv"] = str(output_dir / "utility_vs_baselines.csv")
                summary["utility"]["n_eval_examples"] = int(utility_report.get("n_eval_examples", 0))
                summary["utility"]["n_examples_with_utility"] = int(utility_report.get("n_examples_with_utility", 0))
                summary["utility"]["skipped_missing_eval_row"] = int(
                    utility_report.get("skipped_missing_eval_row", 0)
                )
                summary["utility"]["skipped_invalid_route_stats"] = int(
                    utility_report.get("skipped_invalid_route_stats", 0)
                )
            if supervision_mode in TEXT_LM_SUPERVISION_MODES:
                summary["text_lora"] = True
                summary["text_lora_base_model_path"] = text_lora_base_model_path
                summary["text_lora_adapter_source"] = text_lora_adapter_source
                summary["text_lora_trainable_parameters"] = trainable_parameters
                summary["text_lora_config"] = text_lora_config_summary or {}
            if supervision_mode in TEXT_REWARD_SUPERVISION_MODES:
                summary["text_reward"] = {
                    "target_precision": text_target_precision,
                    "target_grid_count": text_target_grid_count,
                    "lr": text_lr,
                    "weight_decay": text_weight_decay,
                    "warmup_steps": text_warmup_steps,
                    "gradient_clipping": text_gradient_clipping,
                    "max_new_tokens": text_max_new_tokens,
                    "do_sample": text_do_sample,
                    "parse_failure_value": text_parse_failure_value,
                    "clip_predictions": text_clip_predictions,
                    "drop_overlength_rows": text_drop_overlength_rows,
                    "dropped_overlength_train_rows": int(dropped_overlength_train_rows),
                    "dropped_overlength_eval_rows": int(dropped_overlength_eval_rows),
                    "best_eval_parse_failures": int(best_eval_extra.get("parse_failures", 0)),
                    "best_eval_parse_failure_rate": float(best_eval_extra.get("parse_failure_rate", 0.0)),
                    "best_eval_problem_examples": int(best_eval_extra.get("eval_problem_examples", 0)),
                    "best_eval_route_examples": int(best_eval_extra.get("eval_route_examples", 0)),
                }
                if supervision_mode == "text_reward_bin":
                    summary["text_reward"]["bin_count"] = text_reward_bin_count
                    summary["text_reward"]["bin_label_prefix"] = text_reward_bin_label_prefix
                    summary["text_reward"]["bin_value_order"] = text_reward_bin_value_order
                    summary["text_reward"]["bin_specs"] = [
                        {
                            "idx": int(spec["idx"]),
                            "label": str(spec["label"]),
                            "target_text": str(spec["target_text"]),
                            "value": float(spec["value"]),
                            "token_id": int(token_id),
                        }
                        for spec, token_id in zip(text_reward_bin_specs, text_reward_bin_token_ids)
                    ]
            if supervision_mode == "text_pairwise_sign":
                summary["text_pairwise"] = {
                    "tie_margin": text_pairwise_tie_margin,
                    "label_prefix": text_pairwise_label_prefix,
                    "primary_better_label": text_pairwise_primary_better_label,
                    "expert_better_label": text_pairwise_expert_better_label,
                    "drop_overlength_rows": text_drop_overlength_rows,
                    "dropped_overlength_train_rows": int(dropped_overlength_train_rows),
                    "dropped_overlength_eval_rows": int(dropped_overlength_eval_rows),
                    "dropped_tie_train_rows": int(dropped_tie_train_rows),
                    "dropped_tie_eval_rows": int(dropped_tie_eval_rows),
                    "best_eval_parse_failures": int(best_eval_extra.get("parse_failures", 0)),
                    "best_eval_parse_failure_rate": float(best_eval_extra.get("parse_failure_rate", 0.0)),
                    "best_eval_problem_examples": int(best_eval_extra.get("eval_problem_examples", 0)),
                    "best_eval_route_examples": int(best_eval_extra.get("eval_route_examples", 0)),
                    "label_specs": [
                        {
                            "winner_idx": int(spec["winner_idx"]),
                            "label": str(spec["label"]),
                            "target_text": str(spec["target_text"]),
                            "token_id": int(token_id),
                        }
                        for spec, token_id in zip(text_pairwise_label_specs, text_pairwise_label_token_ids)
                    ],
                }
            write_json(output_dir / "summary.json", summary)
            if wandb_run is not None:
                try:
                    wandb_run.summary["offline_router/best_epoch"] = best_epoch
                    wandb_run.summary["offline_router/best_eval_loss"] = best_eval_loss
                    if 0 <= best_epoch < len(history):
                        wandb_run.summary["offline_router/best_train_loss"] = history[best_epoch]["train_loss"]
                    wandb_run.finish()
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Failed to finalize offline router W&B logging: %s", exc)
            logger.info("Offline router training complete: output_dir=%s", output_dir)
    except Exception:
        had_error = True
        logger.exception("Offline router training failed on rank=%s", os.environ.get("RANK", "unknown"))
        raise
    finally:
        if accelerator_is_initialized():
            try:
                get_accelerator().end_training()
            except Exception:
                logger.exception("Offline router end_training cleanup failed on rank=%s", os.environ.get("RANK", "unknown"))
                if not had_error:
                    raise


if __name__ == "__main__":
    import sys

    for i, arg in enumerate(list(sys.argv)):
        if arg.startswith("--local_rank"):
            sys.argv = sys.argv[:i] + sys.argv[i + 1 :]
            break
    main()

#!/usr/bin/env python
import csv
import glob
import json
import logging
import math
import os
import random
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

    if supervision_mode == "text_reward_vector":
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
    if supervision_mode != "text_reward_vector":
        raise ValueError(f"Unsupported supervision mode: {supervision_mode}")
    if training_mode != "full_backbone":
        raise ValueError("offline_router.train.supervision_mode=text_reward_vector requires mode=full_backbone")

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


def _build_text_reward_prompt(
    prompt_text: str,
    primary_output_text: str,
    route_aliases: list[str],
) -> str:
    route_legend = ", ".join(route_aliases)
    return (
        "Predict the realized reward for each model.\n"
        "Respond with only a compact JSON array of floats in the listed order.\n"
        "Do not include any keys, labels, or explanation text.\n"
        "Each score must be between 0.000 and 1.000.\n\n"
        "[Model Order]\n"
        f"{route_legend}\n\n"
        "[Original Repair Prompt]\n"
        f"{prompt_text}\n\n"
        "[Primary Model Attempt]\n"
        f"{primary_output_text}\n\n"
        "Answer:\n"
    )


def _format_reward_target(values: list[float], precision: int) -> str:
    rounded = [float(f"{float(value):.{int(precision)}f}") for value in values]
    return json.dumps(rounded, separators=(",", ":"))


def _truncate_from_left(token_ids: list[int], max_seq_length: int | None) -> list[int]:
    if max_seq_length is None or max_seq_length <= 0 or len(token_ids) <= max_seq_length:
        return token_ids
    return token_ids[-max_seq_length:]


def _tokenize_text_reward_target(
    tokenizer: Any,
    prompt_text: str,
    target_text: str,
    max_seq_length: int | None,
) -> dict[str, Any] | None:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt_text + target_text, add_special_tokens=False).input_ids
    if not full_ids or len(full_ids) <= len(prompt_ids):
        return None

    prompt_len = len(prompt_ids)
    if max_seq_length is not None and max_seq_length > 0 and len(full_ids) > max_seq_length:
        drop = len(full_ids) - max_seq_length
        full_ids = full_ids[drop:]
        prompt_len = max(0, prompt_len - drop)

    if not full_ids or len(full_ids) <= prompt_len:
        return None

    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return {
        "input_ids": full_ids,
        "labels": labels,
    }


def _prepare_text_train_rows(
    dataset: Any,
    tokenizer: Any,
    max_seq_length: int | None,
    route_aliases: list[str],
    target_dim: int,
    target_precision: int,
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
        try:
            problem_id = problem_id_from_item(row)
        except ValueError:
            continue
        target_rewards = [float(value) for value in targets]
        prompt = _build_text_reward_prompt(prompt_text, primary_output_text, route_aliases)
        encoded = _tokenize_text_reward_target(
            tokenizer=tokenizer,
            prompt_text=prompt,
            target_text=_format_reward_target(target_rewards, target_precision),
            max_seq_length=max_seq_length,
        )
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
    return prepared_rows


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
        try:
            problem_id = problem_id_from_item(row)
        except ValueError:
            continue
        prompt = _build_text_reward_prompt(prompt_text, primary_output_text, route_aliases)
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        prompt_ids = _truncate_from_left(prompt_ids, max_seq_length)
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
    return prepared_rows


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


def _save_offline_router_checkpoint(
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    supervision_mode: str,
) -> None:
    if supervision_mode == "text_reward_vector":
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
        if supervision_mode not in {"representation_head", "text_reward_vector"}:
            raise ValueError(f"Unsupported offline_router.train.supervision_mode: {supervision_mode}")
        text_reward_cfg = train_cfg.get("text_reward")
        if text_reward_cfg is None:
            text_reward_cfg = {}
        text_target_precision = int(text_reward_cfg.get("target_precision", 2))
        text_lr = float(text_reward_cfg.get("lr", 5.0e-6))
        text_weight_decay = float(text_reward_cfg.get("weight_decay", 0.01))
        text_warmup_steps = int(text_reward_cfg.get("warmup_steps", 100))
        text_gradient_clipping = float(text_reward_cfg.get("gradient_clipping", 0.3))
        text_debug_step_logging = bool(text_reward_cfg.get("debug_step_logging", False))
        text_max_new_tokens = int(text_reward_cfg.get("max_new_tokens", 16))
        text_do_sample = bool(text_reward_cfg.get("do_sample", False))
        text_parse_failure_value = float(text_reward_cfg.get("parse_failure_value", 0.0))
        text_clip_predictions = bool(text_reward_cfg.get("clip_predictions", True))
        train_sampling_strategy = str(train_cfg.get("train_sampling_strategy", "random"))

        train_dataset = _load_split(dataset_dir, "train")
        eval_dataset = _load_split(dataset_dir, "eval")

        max_train_rows = train_cfg.get("max_train_rows")
        max_eval_rows = train_cfg.get("max_eval_rows")
        base_seed = int(cfg.get("seed", 42))
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
        raw_train_rows = int(len(train_dataset))
        raw_eval_rows = int(len(eval_dataset))
        train_supervision_rows = int(len(train_dataset))
        eval_supervision_rows = int(len(eval_dataset))

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
                "offline_router.train.supervision_mode=text_reward_vector always uses LoRA adapters; ignoring mode=%s",
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
        if supervision_mode == "text_reward_vector":
            logger.info(
                "Offline router text reward settings: target_precision=%d lr=%.2e weight_decay=%.3f warmup_steps=%d gradient_clipping=%.3f debug_step_logging=%s max_new_tokens=%d do_sample=%s parse_failure_value=%.3f clip_predictions=%s train_sampling_strategy=%s train_rows=%d train_supervision_rows=%d eval_rows=%d eval_supervision_rows=%d",
                text_target_precision,
                text_lr,
                text_weight_decay,
                text_warmup_steps,
                text_gradient_clipping,
                text_debug_step_logging,
                text_max_new_tokens,
                text_do_sample,
                text_parse_failure_value,
                text_clip_predictions,
                train_sampling_strategy,
                raw_train_rows,
                train_supervision_rows,
                raw_eval_rows,
                eval_supervision_rows,
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
        else:
            train_dataset = _prepare_text_train_rows(
                train_dataset,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                route_aliases=route_aliases,
                target_dim=target_dim,
                target_precision=text_target_precision,
            )
            eval_dataset = _prepare_text_eval_rows(
                eval_dataset,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                route_aliases=route_aliases,
                target_dim=target_dim,
            )
            train_collate_fn = _make_text_train_collate_fn(pad_token_id=pad_token_id, target_dim=target_dim)
            eval_collate_fn = _make_text_eval_collate_fn(pad_token_id=pad_token_id, target_dim=target_dim)

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
        if supervision_mode == "text_reward_vector":
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
                    if supervision_mode == "text_reward_vector":
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
                        supervision_mode == "text_reward_vector"
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
                    if supervision_mode == "text_reward_vector":
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
                    if supervision_mode == "text_reward_vector":
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
                    if supervision_mode == "text_reward_vector":
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
            else:
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
                else:
                    eval_rows, y_true, y_pred, eval_extra = _merge_text_reward_eval_rows(
                        merged_shard_rows,
                        route_labels=route_labels,
                    )
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

            summary = {
                "mode": str(train_cfg.mode),
                "supervision_mode": supervision_mode,
                "dataset_dir": str(dataset_dir),
                "model_path": model_path,
                "route_labels": route_labels,
                "target_dim": target_dim,
                "train_sampling": train_sampling_summary,
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
            if supervision_mode == "text_reward_vector":
                summary["text_lora"] = True
                summary["text_lora_base_model_path"] = text_lora_base_model_path
                summary["text_lora_adapter_source"] = text_lora_adapter_source
                summary["text_lora_trainable_parameters"] = trainable_parameters
                summary["text_lora_config"] = text_lora_config_summary or {}
                summary["text_reward"] = {
                    "target_precision": text_target_precision,
                    "lr": text_lr,
                    "weight_decay": text_weight_decay,
                    "warmup_steps": text_warmup_steps,
                    "gradient_clipping": text_gradient_clipping,
                    "max_new_tokens": text_max_new_tokens,
                    "do_sample": text_do_sample,
                    "parse_failure_value": text_parse_failure_value,
                    "clip_predictions": text_clip_predictions,
                    "best_eval_parse_failures": int(best_eval_extra.get("parse_failures", 0)),
                    "best_eval_parse_failure_rate": float(best_eval_extra.get("parse_failure_rate", 0.0)),
                    "best_eval_problem_examples": int(best_eval_extra.get("eval_problem_examples", 0)),
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

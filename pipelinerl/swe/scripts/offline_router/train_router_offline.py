#!/usr/bin/env python
import csv
import glob
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
import wandb
import time

from pipelinerl.finetune.checkpoints import save_model_only
from pipelinerl.finetune.context import get_accelerator
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


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


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


def _all_gather_objects(local_obj: Any) -> list[Any]:
    if not is_distributed() or not _dist_ready():
        return [local_obj]
    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, local_obj)
    return gathered


def _broadcast_object(obj: Any) -> Any:
    if not is_distributed() or not _dist_ready():
        return obj
    container = [obj if is_main_process() else None]
    dist.broadcast_object_list(container, src=0)
    return container[0]


def _prediction_row_key(row: dict[str, Any]) -> str:
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


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _make_collate_fn(tokenizer: Any, max_seq_length: int | None, target_dim: int):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer needs either pad_token_id or eos_token_id for offline router batching")

    def _collate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        encoded_rows: list[dict[str, Any]] = []
        for row in rows:
            prompt_text = row.get("prompt_text")
            primary_output_text = row.get("primary_output_text")
            targets = row.get("performance_targets")
            if not isinstance(prompt_text, str):
                continue
            if not isinstance(primary_output_text, str):
                primary_output_text = row.get("policy_output_text")
            if not isinstance(primary_output_text, str):
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
            encoded_rows.append(
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

        if not encoded_rows:
            return None

        max_len = max(len(row["input_ids"]) for row in encoded_rows)
        batch_size = len(encoded_rows)
        input_ids = torch.full((batch_size, max_len), fill_value=pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        last_indices = torch.zeros((batch_size,), dtype=torch.long)
        targets = torch.zeros((batch_size, target_dim), dtype=torch.float32)

        for batch_idx, row in enumerate(encoded_rows):
            ids = row["input_ids"]
            seq_len = len(ids)
            input_ids[batch_idx, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[batch_idx, :seq_len] = 1
            last_indices[batch_idx] = int(row["completion_last_index"])
            targets[batch_idx] = torch.tensor(row["targets"], dtype=torch.float32)

        return {
            "problem_ids": [row["problem_id"] for row in encoded_rows],
            "datasets": [row["dataset"] for row in encoded_rows],
            "repos": [row["repo"] for row in encoded_rows],
            "languages": [row["language"] for row in encoded_rows],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_last_indices": last_indices,
            "targets": targets,
        }

    return _collate


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **batch,
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "completion_last_indices": batch["completion_last_indices"].to(device),
        "targets": batch["targets"].to(device),
    }


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


def _run_eval(
    model: AutoModelForCausalLMWithValueHead,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[dict[str, Any]], np.ndarray, np.ndarray]:
    model.eval()
    prediction_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval offline router", unit="batch", disable=not is_main_process()):
            if batch is None:
                continue
            batch = _move_batch_to_device(batch, device)
            preds = _forward_predictions(model, batch)

            preds_np = preds.cpu().numpy()
            targets_np = batch["targets"].cpu().numpy()
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

    gathered = _all_gather_objects(
        {
            "rows": prediction_rows,
        }
    )
    if not is_main_process():
        return math.nan, [], np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)

    merged_rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in gathered:
        if not isinstance(item, dict):
            continue
        for row in item.get("rows", []):
            key = _prediction_row_key(row)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_rows.append(row)

    if not merged_rows:
        raise ValueError("No valid eval examples after tokenization/collation")

    y_true = np.asarray([row["true_rewards"] for row in merged_rows], dtype=np.float32)
    y_pred = np.asarray([row["pred_rewards"] for row in merged_rows], dtype=np.float32)
    mean_loss = float(np.mean((y_pred - y_true) ** 2)) if y_true.size else math.nan
    return mean_loss, merged_rows, y_true, y_pred


def _save_predictions_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


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

    try:
        train_cfg = cfg.offline_router.train
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
        target_dim = len(route_labels)

        train_dataset = _load_split(dataset_dir, "train")
        eval_dataset = _load_split(dataset_dir, "eval")

        max_train_rows = train_cfg.get("max_train_rows")
        max_eval_rows = train_cfg.get("max_eval_rows")
        if max_train_rows:
            train_dataset = train_dataset.select(range(min(int(max_train_rows), len(train_dataset))))
        if max_eval_rows:
            eval_dataset = eval_dataset.select(range(min(int(max_eval_rows), len(eval_dataset))))

        device = _pick_device(str(train_cfg.get("device", "auto")))
        model_path = str(train_cfg.model_path)
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

        trainable_prefixes = configure_router_training_mode(model, str(train_cfg.mode))
        total_parameters = count_parameters(model, trainable_only=False)
        trainable_parameters = count_parameters(model, trainable_only=True)

        logger.info(
            "Offline router mode=%s trainable_prefixes=%s trainable_params=%d total_params=%d",
            train_cfg.mode,
            trainable_prefixes,
            trainable_parameters,
            total_parameters,
        )
        logger.info(
            "Offline router performance head hidden_dims=%s activation=%s",
            performance_value_hidden_dims or [],
            performance_value_activation,
        )
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(train_cfg.lr),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )

        if (ds_plugin := getattr(get_accelerator().state, "deepspeed_plugin", None)) is not None:
            ds_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = int(train_cfg.batch_size)
            ds_plugin.deepspeed_config["gradient_accumulation_steps"] = int(train_cfg.get("gradient_accumulation_steps", 1))
            ds_plugin.deepspeed_config["train_batch_size"] = (
                int(train_cfg.batch_size)
                * int(train_cfg.get("gradient_accumulation_steps", 1))
                * get_world_size()
            )

        collate_fn = _make_collate_fn(
            tokenizer=tokenizer,
            max_seq_length=int(train_cfg.max_seq_length) if train_cfg.get("max_seq_length") else None,
            target_dim=target_dim,
        )
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=bool(train_cfg.get("shuffle_train", True)),
            )
            if is_distributed()
            else None
        )
        eval_sampler = (
            DistributedSampler(
                eval_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=False,
            )
            if is_distributed()
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(train_cfg.batch_size),
            shuffle=False if train_sampler is not None else bool(train_cfg.get("shuffle_train", True)),
            sampler=train_sampler,
            num_workers=int(train_cfg.get("num_workers", 0)),
            collate_fn=collate_fn,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=int(train_cfg.eval_batch_size),
            shuffle=False,
            sampler=eval_sampler,
            num_workers=int(train_cfg.get("num_workers", 0)),
            collate_fn=collate_fn,
        )
        model, optimizer = get_accelerator().prepare(model, optimizer)
        logger.info(
            "Offline router accelerator prepared: distributed=%s world_size=%d deepspeed=%s",
            is_distributed(),
            get_world_size(),
            getattr(get_accelerator().state, "deepspeed_plugin", None) is not None,
        )

        num_epochs = int(train_cfg.num_epochs)
        grad_accum = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
        log_every_steps = max(1, int(train_cfg.get("log_every_steps", 100)))
        save_checkpoints = bool(train_cfg.get("save_checkpoints", True))
        best_eval_loss = float("inf")
        best_epoch = -1
        history: list[dict[str, Any]] = []
        best_eval_rows: list[dict[str, Any]] = []
        best_y_true: np.ndarray | None = None
        best_y_pred: np.ndarray | None = None

        for epoch in range(num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

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
                if batch is None:
                    continue
                batch = _move_batch_to_device(batch, device)
                preds = _forward_predictions(model, batch)
                loss = F.mse_loss(preds, batch["targets"])
                running_losses.append(float(loss.item()))
                get_accelerator().backward(loss / grad_accum)
                batch_count += 1
                local_examples_seen += int(batch["targets"].shape[0])

                if batch_count % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

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

            if batch_count % grad_accum != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if not running_losses:
                raise ValueError("No valid training batches after tokenization/collation")
            loss_stats = _all_gather_objects(
                {
                    "loss_sum": float(np.sum(running_losses)),
                    "loss_count": int(len(running_losses)),
                }
            )
            total_loss_sum = 0.0
            total_loss_count = 0
            for item in loss_stats:
                if not isinstance(item, dict):
                    continue
                total_loss_sum += float(item.get("loss_sum", 0.0))
                total_loss_count += int(item.get("loss_count", 0))
            train_loss = total_loss_sum / total_loss_count if total_loss_count > 0 else float(np.mean(running_losses))

            if is_main_process():
                logger.info("Offline router epoch=%d starting eval", epoch)
            eval_start_time = time.time()
            eval_loss, eval_rows, y_true, y_pred = _run_eval(model, eval_loader, device)
            if is_main_process():
                logger.info(
                    "Offline router epoch=%d finished eval eval_loss=%.6f eval_rows=%d elapsed_seconds=%.1f",
                    epoch,
                    eval_loss,
                    len(eval_rows),
                    time.time() - eval_start_time,
                )

            should_save_best = False
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

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_epoch = epoch
                    best_eval_rows = eval_rows
                    best_y_true = y_true
                    best_y_pred = y_pred
                    should_save_best = True

            should_save_best = bool(_broadcast_object(should_save_best))
            if should_save_best and save_checkpoints:
                best_dir = output_dir / "checkpoints" / "best"
                if is_main_process():
                    logger.info("Offline router epoch=%d starting best checkpoint save to %s", epoch, best_dir)
                    best_dir.mkdir(parents=True, exist_ok=True)
                    save_start_time = time.time()
                barrier_if_distributed()
                save_model_only(best_dir, model)
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
            save_model_only(last_dir, model)
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
                "dataset_dir": str(dataset_dir),
                "model_path": model_path,
                "route_labels": route_labels,
                "target_dim": target_dim,
                "train_rows": int(len(train_dataset)),
                "eval_rows": int(len(eval_dataset)),
                "num_epochs": num_epochs,
                "performance_value_hidden_dims": performance_value_hidden_dims or [],
                "performance_value_activation": performance_value_activation,
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
    finally:
        barrier_if_distributed()


if __name__ == "__main__":
    import sys

    for i, arg in enumerate(list(sys.argv)):
        if arg.startswith("--local_rank"):
            sys.argv = sys.argv[:i] + sys.argv[i + 1 :]
            break
    main()

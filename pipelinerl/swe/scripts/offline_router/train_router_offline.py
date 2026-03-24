#!/usr/bin/env python
import csv
import glob
import json
import logging
import math
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from pipelinerl.finetune.value_model import AutoModelForCausalLMWithValueHead
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


def _pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    losses: list[float] = []
    prediction_rows: list[dict[str, Any]] = []
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval offline router", unit="batch"):
            if batch is None:
                continue
            batch = _move_batch_to_device(batch, device)
            preds = _forward_predictions(model, batch)
            loss = F.mse_loss(preds, batch["targets"])
            losses.append(float(loss.item()))

            preds_np = preds.cpu().numpy()
            targets_np = batch["targets"].cpu().numpy()
            y_true_parts.append(targets_np)
            y_pred_parts.append(preds_np)
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

    if not y_true_parts:
        raise ValueError("No valid eval examples after tokenization/collation")
    mean_loss = float(np.mean(losses)) if losses else math.nan
    return mean_loss, prediction_rows, np.concatenate(y_true_parts, axis=0), np.concatenate(y_pred_parts, axis=0)


def _save_predictions_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@hydra.main(config_path="../../../../conf", config_name="offline_router_train", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    train_cfg = cfg.offline_router.train
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train_config.json", sanitize_for_json(OmegaConf.to_container(cfg, resolve=True)))

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
    logger.info("Loading tokenizer/model from %s on %s", model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, performance_value_dim=target_dim)

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

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(train_cfg.lr),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    collate_fn = _make_collate_fn(
        tokenizer=tokenizer,
        max_seq_length=int(train_cfg.max_seq_length) if train_cfg.get("max_seq_length") else None,
        target_dim=target_dim,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.batch_size),
        shuffle=bool(train_cfg.get("shuffle_train", True)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(train_cfg.eval_batch_size),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_fn,
    )

    num_epochs = int(train_cfg.num_epochs)
    grad_accum = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    best_eval_loss = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    best_eval_rows: list[dict[str, Any]] = []
    best_y_true: np.ndarray | None = None
    best_y_pred: np.ndarray | None = None

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_losses: list[float] = []
        batch_count = 0

        for step_idx, batch in enumerate(tqdm(train_loader, desc=f"Train epoch {epoch}", unit="batch"), start=1):
            if batch is None:
                continue
            batch = _move_batch_to_device(batch, device)
            preds = _forward_predictions(model, batch)
            loss = F.mse_loss(preds, batch["targets"])
            running_losses.append(float(loss.item()))
            (loss / grad_accum).backward()
            batch_count += 1

            if batch_count % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if batch_count % grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if not running_losses:
            raise ValueError("No valid training batches after tokenization/collation")
        train_loss = float(np.mean(running_losses))
        eval_loss, eval_rows, y_true, y_pred = _run_eval(model, eval_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
            }
        )
        logger.info(
            "Offline router epoch=%d train_loss=%.6f eval_loss=%.6f",
            epoch,
            train_loss,
            eval_loss,
        )

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch
            best_eval_rows = eval_rows
            best_y_true = y_true
            best_y_pred = y_pred
            best_dir = output_dir / "checkpoints" / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)

    last_dir = output_dir / "checkpoints" / "last"
    last_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(last_dir)

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
        "best_epoch": best_epoch,
        "best_eval_loss": best_eval_loss,
        "trainable_prefixes": trainable_prefixes,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "history": history,
        "best_checkpoint_dir": str(output_dir / "checkpoints" / "best"),
        "last_checkpoint_dir": str(last_dir),
    }
    write_json(output_dir / "summary.json", summary)
    logger.info("Offline router training complete: output_dir=%s", output_dir)


if __name__ == "__main__":
    main()

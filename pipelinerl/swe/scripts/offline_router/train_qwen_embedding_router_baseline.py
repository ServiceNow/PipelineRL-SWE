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
from accelerate import Accelerator
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from pipelinerl.swe.scripts.offline_router.common import (
    compute_pairwise_metrics,
    compute_per_route_metrics,
    csv_headers_for_pairwise_metrics,
    csv_headers_for_route_metrics,
    write_json,
)
from pipelinerl.swe.scripts.offline_router.train_modernbert_router_baseline import (
    DEFAULT_UTILITY_LAMBDAS,
    RouterPairDataset,
    _argmax_index,
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


def _collate_left_pad(batch: list[dict[str, Any]], pad_token_id: int, target_dim: int) -> dict[str, Any]:
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    targets = torch.zeros((len(batch), target_dim), dtype=torch.float32)
    class_targets = torch.zeros((len(batch),), dtype=torch.long)
    row_indices = torch.zeros((len(batch),), dtype=torch.long)
    for idx, row in enumerate(batch):
        seq_len = len(row["input_ids"])
        start = max_len - seq_len
        input_ids[idx, start:] = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask[idx, start:] = torch.tensor(row["attention_mask"], dtype=torch.long)
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


class QwenEmbeddingRouter(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        target_dim: int,
        dropout: float,
        mlp_hidden_size: int,
        torch_dtype: torch.dtype,
        attn_implementation: str | None,
    ) -> None:
        super().__init__()
        model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.encoder = AutoModel.from_pretrained(model_name, **model_kwargs)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        hidden_size = int(self.encoder.config.hidden_size)
        if int(mlp_hidden_size) > 0:
            self.head = torch.nn.Sequential(
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(hidden_size, int(mlp_hidden_size)),
                torch.nn.GELU(),
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(int(mlp_hidden_size), int(target_dim)),
            )
        else:
            self.head = torch.nn.Sequential(
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(hidden_size, int(target_dim)),
            )

    def train(self, mode: bool = True) -> "QwenEmbeddingRouter":
        super().train(mode)
        if hasattr(self, "encoder"):
            self.encoder.eval()
        return self

    def encode_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = _last_token_pool(outputs.last_hidden_state, attention_mask)
            return F.normalize(pooled.float(), p=2, dim=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_inputs(input_ids, attention_mask))


class PrecomputedEmbeddingDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def _collate_embeddings(batch: list[dict[str, Any]], target_dim: int) -> dict[str, Any]:
    embeddings = torch.stack([row["embedding"] for row in batch], dim=0).float()
    targets = torch.zeros((len(batch), target_dim), dtype=torch.float32)
    class_targets = torch.zeros((len(batch),), dtype=torch.long)
    row_indices = torch.zeros((len(batch),), dtype=torch.long)
    for idx, row in enumerate(batch):
        targets[idx] = torch.tensor(row["targets"], dtype=torch.float32)
        class_targets[idx] = int(row["class_target"])
        row_indices[idx] = int(row["row_idx"])
    return {
        "problem_ids": [row["problem_id"] for row in batch],
        "datasets": [row["dataset"] for row in batch],
        "repos": [row["repo"] for row in batch],
        "languages": [row["language"] for row in batch],
        "embeddings": embeddings,
        "targets": targets,
        "class_targets": class_targets,
        "row_indices": row_indices,
    }


@torch.no_grad()
def _precompute_embeddings(
    accelerator: Accelerator,
    model: QwenEmbeddingRouter,
    loader: DataLoader,
    source_dataset: RouterPairDataset,
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


def _delta_loss(preds: torch.Tensor, targets: torch.Tensor, huber_delta: float) -> torch.Tensor:
    if preds.shape[1] != 2:
        raise ValueError("Delta auxiliary loss currently expects exactly two routes")
    pred_delta = preds[:, 1] - preds[:, 0]
    true_delta = targets[:, 1] - targets[:, 0]
    if float(huber_delta) > 0.0:
        return F.huber_loss(pred_delta, true_delta, delta=float(huber_delta))
    return F.mse_loss(pred_delta, true_delta)


def _compute_train_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_targets: torch.Tensor,
    objective: str,
    reward_mse_weight: float,
    delta_aux_weight: float,
    delta_aux_huber_delta: float,
) -> torch.Tensor:
    if objective == "route_classifier":
        return F.cross_entropy(logits, class_targets.long())
    reward_loss = F.mse_loss(logits, targets.float()) * float(reward_mse_weight)
    if objective == "reward_mse_delta_aux":
        reward_loss = reward_loss + (
            float(delta_aux_weight) * _delta_loss(logits, targets.float(), float(delta_aux_huber_delta))
        )
    return reward_loss


def _predict_from_batch(model: QwenEmbeddingRouter, batch: dict[str, Any]) -> torch.Tensor:
    if "embeddings" in batch:
        return model.head(batch["embeddings"].float())
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])


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
    for batch in tqdm(loader, desc="Eval Qwen embedding router", disable=not accelerator.is_main_process):
        logits = _predict_from_batch(model, batch).float()
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
            rows.append(
                {
                    "problem_id": source_meta["problem_id"],
                    "dataset": source_meta["dataset"],
                    "repo": source_meta["repo"],
                    "language": source_meta["language"],
                    "true_rewards": [float(value) for value in gathered_targets[idx].tolist()],
                    "pred_rewards": [float(value) for value in gathered_preds[idx].tolist()],
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
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument(
        "--objective",
        choices=["reward_mse", "reward_mse_delta_aux", "route_classifier"],
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
    parser.add_argument("--reward-mse-weight", type=float, default=1.0)
    parser.add_argument("--delta-aux-weight", type=float, default=1.0)
    parser.add_argument("--delta-aux-huber-delta", type=float, default=0.0)
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
    if args.objective == "reward_mse_delta_aux" and target_dim != 2:
        raise ValueError("reward_mse_delta_aux currently expects exactly two routes")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    train_rows = _shuffle_rows(list(_load_split(dataset_dir, "train")), args.max_train_rows, args.seed)
    eval_rows_source = _shuffle_rows(list(_load_split(dataset_dir, "eval")), args.max_eval_rows, args.seed + 1)
    train_dataset = RouterPairDataset(train_rows, tokenizer, route_labels, int(args.max_seq_length))
    eval_dataset = RouterPairDataset(eval_rows_source, tokenizer, route_labels, int(args.max_seq_length))
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError(f"Prepared empty dataset train={len(train_dataset)} eval={len(eval_dataset)}")

    token_collate_fn = lambda batch: _collate_left_pad(batch, pad_token_id=int(pad_token_id), target_dim=target_dim)
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
    )
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
        embedding_collate_fn = lambda batch: _collate_embeddings(batch, target_dim=target_dim)
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
        "reward_mse_weight": float(args.reward_mse_weight),
        "delta_aux_weight": float(args.delta_aux_weight),
        "delta_aux_huber_delta": float(args.delta_aux_huber_delta),
        "encoder_frozen": True,
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
                logits = _predict_from_batch(model, batch).float()
                loss = _compute_train_loss(
                    logits=logits,
                    targets=batch["targets"].float(),
                    class_targets=batch["class_targets"],
                    objective=str(args.objective),
                    reward_mse_weight=float(args.reward_mse_weight),
                    delta_aux_weight=float(args.delta_aux_weight),
                    delta_aux_huber_delta=float(args.delta_aux_huber_delta),
                )
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
        torch.save(unwrapped.head.state_dict(), output_dir / "head.pt")
        tokenizer.save_pretrained(output_dir / "tokenizer")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Train a Qwen embedding verifier with same-instance listwise proxy supervision."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from pipelinerl.swe.scripts.offline_router.common import (
    compute_pairwise_metrics,
    compute_per_route_metrics,
    csv_headers_for_pairwise_metrics,
    csv_headers_for_route_metrics,
    problem_id_from_item,
    write_json,
)
from pipelinerl.swe.scripts.offline_router.train_modernbert_router_baseline import (
    _load_route_labels,
    _load_split,
    _shuffle_rows,
)
from pipelinerl.swe.scripts.offline_router.train_qwen_embedding_cascade_baseline import (
    _build_cascade_input_text,
    _load_model_only_checkpoint,
    _parse_int_list,
)
from pipelinerl.swe.scripts.offline_router.train_qwen_embedding_router_baseline import (
    QwenEmbeddingRouter,
    _dtype_from_name,
    _parse_route_indices,
    _safe_corr,
    _write_csv,
    _write_jsonl,
)


class ListwiseVerifierDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        route_labels: list[str],
        max_seq_length: int,
        route_order: list[int],
        source_route_idxs: list[int] | None = None,
        tie_margin: float = 0.0,
    ) -> None:
        self.groups: list[dict[str, Any]] = []
        self.candidates: list[dict[str, Any]] = []
        target_dim = len(route_labels)
        source_route_idxs = list(range(target_dim)) if source_route_idxs is None else [int(idx) for idx in source_route_idxs]
        if len(source_route_idxs) != target_dim:
            raise ValueError(
                f"source_route_idxs length {len(source_route_idxs)} does not match {target_dim} route labels"
            )
        invalid_route_idxs = [idx for idx in route_order if not 0 <= int(idx) < target_dim]
        if invalid_route_idxs:
            raise ValueError(f"route_order={invalid_route_idxs} is out of range for {target_dim} selected routes")
        max_source_route_idx = max(source_route_idxs) if source_route_idxs else -1

        for source_idx, row in enumerate(rows):
            targets = row.get("performance_targets")
            route_outputs = row.get("route_outputs")
            if not isinstance(targets, list) or len(targets) <= max_source_route_idx:
                continue
            if not isinstance(route_outputs, list) or len(route_outputs) <= max_source_route_idx:
                continue
            try:
                problem_id = problem_id_from_item(row)
                reward_targets = [float(value) for value in targets]
            except (TypeError, ValueError):
                continue

            group_candidates: list[dict[str, Any]] = []
            for local_route_idx in route_order:
                source_route_idx = int(source_route_idxs[int(local_route_idx)])
                input_text = _build_cascade_input_text(
                    row,
                    route_labels[int(local_route_idx)],
                    route_outputs[source_route_idx],
                )
                if not input_text:
                    continue
                encoded = tokenizer(
                    input_text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=int(max_seq_length),
                )
                input_ids = encoded.get("input_ids")
                attention_mask = encoded.get("attention_mask")
                if not input_ids or not attention_mask:
                    continue
                group_candidates.append(
                    {
                        "candidate_idx": len(self.candidates) + len(group_candidates),
                        "source_idx": int(source_idx),
                        "problem_id": problem_id,
                        "dataset": row.get("dataset"),
                        "repo": row.get("repo"),
                        "language": row.get("language"),
                        "route_idx": int(local_route_idx),
                        "source_route_idx": int(source_route_idx),
                        "route_label": route_labels[int(local_route_idx)],
                        "target": float(reward_targets[source_route_idx]),
                        "input_ids": [int(value) for value in input_ids],
                        "attention_mask": [int(value) for value in attention_mask],
                    }
                )

            if len(group_candidates) < 2:
                continue
            candidate_targets = np.asarray([float(candidate["target"]) for candidate in group_candidates])
            sorted_targets = np.sort(candidate_targets)
            if len(sorted_targets) >= 2 and float(sorted_targets[-1] - sorted_targets[-2]) <= float(tie_margin):
                continue
            target_local_idx = int(np.argmax(candidate_targets))
            group_idx = len(self.groups)
            first_candidate_idx = len(self.candidates)
            for local_idx, candidate in enumerate(group_candidates):
                candidate["candidate_idx"] = first_candidate_idx + local_idx
                candidate["group_idx"] = group_idx
                self.candidates.append(candidate)
            self.groups.append(
                {
                    "group_idx": group_idx,
                    "source_idx": int(source_idx),
                    "problem_id": problem_id,
                    "dataset": row.get("dataset"),
                    "candidate_indices": list(range(first_candidate_idx, first_candidate_idx + len(group_candidates))),
                    "target_local_idx": target_local_idx,
                }
            )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        group = self.groups[idx]
        return {
            "group_idx": int(group["group_idx"]),
            "target_local_idx": int(group["target_local_idx"]),
            "candidates": [self.candidates[int(candidate_idx)] for candidate_idx in group["candidate_indices"]],
        }


def _collate_groups(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    flat_candidates = [candidate for group in batch for candidate in group["candidates"]]
    max_len = max(len(row["input_ids"]) for row in flat_candidates)
    input_ids = torch.full((len(flat_candidates), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(flat_candidates), max_len), dtype=torch.long)
    targets = torch.zeros((len(flat_candidates), 1), dtype=torch.float32)
    candidate_indices = torch.zeros((len(flat_candidates),), dtype=torch.long)
    group_offsets: list[int] = []
    group_sizes: list[int] = []
    group_target_local_indices: list[int] = []

    offset = 0
    for group in batch:
        group_offsets.append(offset)
        group_sizes.append(len(group["candidates"]))
        group_target_local_indices.append(int(group["target_local_idx"]))
        for candidate in group["candidates"]:
            flat_idx = offset
            seq_len = len(candidate["input_ids"])
            start = max_len - seq_len
            input_ids[flat_idx, start:] = torch.tensor(candidate["input_ids"], dtype=torch.long)
            attention_mask[flat_idx, start:] = torch.tensor(candidate["attention_mask"], dtype=torch.long)
            targets[flat_idx, 0] = float(candidate["target"])
            candidate_indices[flat_idx] = int(candidate["candidate_idx"])
            offset += 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "targets": targets,
        "candidate_indices": candidate_indices,
        "group_offsets": torch.tensor(group_offsets, dtype=torch.long),
        "group_sizes": torch.tensor(group_sizes, dtype=torch.long),
        "group_target_local_indices": torch.tensor(group_target_local_indices, dtype=torch.long),
    }


def _listwise_ce_loss(
    logits: torch.Tensor,
    group_offsets: torch.Tensor,
    group_sizes: torch.Tensor,
    group_target_local_indices: torch.Tensor,
) -> torch.Tensor:
    flat_logits = logits.float().reshape(-1)
    losses: list[torch.Tensor] = []
    for offset, size, target_local_idx in zip(group_offsets.tolist(), group_sizes.tolist(), group_target_local_indices.tolist()):
        group_logits = flat_logits[int(offset) : int(offset) + int(size)].reshape(1, int(size))
        target = torch.tensor([int(target_local_idx)], dtype=torch.long, device=flat_logits.device)
        losses.append(F.cross_entropy(group_logits, target))
    if not losses:
        return flat_logits.sum() * 0.0
    return torch.stack(losses).mean()


def _forward_candidate_logits_sequential(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run candidate attempts one at a time to avoid 4x long-context activation peaks."""
    logits: list[torch.Tensor] = []
    for idx in range(int(input_ids.shape[0])):
        reward_logits, _, _ = model(
            input_ids=input_ids[idx : idx + 1],
            attention_mask=attention_mask[idx : idx + 1],
        )
        logits.append(reward_logits)
    return torch.cat(logits, dim=0)


def _prediction_matrix(
    prediction_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    route_labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    target_dim = len(route_labels)
    source_by_key: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        try:
            problem_id = problem_id_from_item(row)
        except ValueError:
            continue
        source_by_key[f"{row.get('dataset')}::{problem_id}"] = row

    pred_by_key: dict[str, dict[int, float]] = {}
    for row in prediction_rows:
        key = f"{row.get('dataset')}::{row.get('problem_id')}"
        pred_by_key.setdefault(key, {})[int(row["route_idx"])] = float(row["pred_score"])

    y_true: list[list[float]] = []
    y_pred: list[list[float]] = []
    for key, source_row in source_by_key.items():
        rewards = source_row.get("performance_targets")
        if not isinstance(rewards, list) or len(rewards) != target_dim:
            continue
        route_preds = pred_by_key.get(key)
        if route_preds is None or any(idx not in route_preds for idx in range(target_dim)):
            continue
        y_true.append([float(value) for value in rewards])
        y_pred.append([float(route_preds[idx]) for idx in range(target_dim)])
    return np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)


@torch.no_grad()
def _evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: ListwiseVerifierDataset,
    desc: str,
) -> tuple[float, list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_groups = 0
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc=desc, disable=not accelerator.is_main_process):
        reward_logits = _forward_candidate_logits_sequential(
            model,
            batch["input_ids"],
            batch["attention_mask"],
        )
        loss = _listwise_ce_loss(
            reward_logits,
            batch["group_offsets"],
            batch["group_sizes"],
            batch["group_target_local_indices"],
        )
        gathered_loss = accelerator.gather_for_metrics(loss.detach().reshape(1)).detach().cpu()
        gathered_logits = accelerator.gather_for_metrics(reward_logits.float().reshape(-1)).detach().cpu()
        gathered_targets = accelerator.gather_for_metrics(batch["targets"]).detach().cpu()
        gathered_indices = accelerator.gather_for_metrics(batch["candidate_indices"]).detach().cpu().tolist()
        gathered_group_count = accelerator.gather_for_metrics(
            torch.tensor([batch["group_offsets"].numel()], device=reward_logits.device, dtype=torch.long)
        ).detach().cpu()
        if accelerator.is_main_process:
            total_loss += float(torch.sum(gathered_loss).item())
            total_groups += int(torch.sum(gathered_group_count).item())
            for idx in range(gathered_logits.shape[0]):
                candidate = dataset.candidates[int(gathered_indices[idx])]
                rows.append(
                    {
                        "problem_id": candidate["problem_id"],
                        "dataset": candidate["dataset"],
                        "repo": candidate["repo"],
                        "language": candidate["language"],
                        "route_idx": int(candidate["route_idx"]),
                        "source_route_idx": int(candidate["source_route_idx"]),
                        "route_label": candidate["route_label"],
                        "true_score": float(gathered_targets[idx, 0].item()),
                        "pred_score": float(gathered_logits[idx].item()),
                    }
                )
    if not accelerator.is_main_process:
        return math.nan, []
    return (float(total_loss / total_groups) if total_groups > 0 else math.nan), rows


def _write_reports(
    output_dir: Path,
    train_pred_rows: list[dict[str, Any]],
    eval_pred_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    route_labels: list[str],
    history: list[dict[str, Any]],
    config: dict[str, Any],
    train_loss: float,
    eval_loss: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_true, eval_pred = _prediction_matrix(eval_pred_rows, eval_rows, route_labels)
    route_metrics = compute_per_route_metrics(eval_true, eval_pred, route_labels)
    pairwise_metrics = compute_pairwise_metrics(eval_true, eval_pred, route_labels)
    scalar_true = np.asarray([row["true_score"] for row in eval_pred_rows], dtype=np.float64)
    scalar_pred = np.asarray([row["pred_score"] for row in eval_pred_rows], dtype=np.float64)
    scalar_metrics = {
        "n_eval_attempts": int(scalar_true.shape[0]),
        "mse": float(np.mean((scalar_pred - scalar_true) ** 2)) if scalar_true.size else math.nan,
        "rmse": float(np.sqrt(np.mean((scalar_pred - scalar_true) ** 2))) if scalar_true.size else math.nan,
        "mae": float(np.mean(np.abs(scalar_pred - scalar_true))) if scalar_true.size else math.nan,
        "pearson": _safe_corr(scalar_pred, scalar_true),
        "mean_true": float(np.mean(scalar_true)) if scalar_true.size else math.nan,
        "mean_pred": float(np.mean(scalar_pred)) if scalar_pred.size else math.nan,
        "std_true": float(np.std(scalar_true)) if scalar_true.size else math.nan,
        "std_pred": float(np.std(scalar_pred)) if scalar_pred.size else math.nan,
    }
    _write_jsonl(output_dir / "train_attempt_predictions.jsonl", train_pred_rows)
    _write_jsonl(output_dir / "eval_attempt_predictions.jsonl", eval_pred_rows)
    _write_csv(output_dir / "route_metrics.csv", route_metrics, csv_headers_for_route_metrics())
    _write_csv(output_dir / "pairwise_metrics.csv", pairwise_metrics, csv_headers_for_pairwise_metrics())
    summary = {
        "history": history,
        "train_loss_final": float(train_loss),
        "eval_loss_final": float(eval_loss),
        "scalar_metrics": scalar_metrics,
        "route_metrics": route_metrics,
        "pairwise_metrics": pairwise_metrics,
        "config": config,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--candidate-route-order", default="0,1,2,3")
    parser.add_argument("--target-route-idxs", default=None)
    parser.add_argument("--max-seq-length", type=int, default=24000)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp-hidden-size", type=int, default=1024)
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32", "float32"], default="bf16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--encoder-frozen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    parser.add_argument("--tie-margin", type=float, default=0.0)
    parser.add_argument("--checkpoint-every-epoch", action="store_true")
    parser.add_argument("--epoch-report-every", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--init-from-model-checkpoint", default=None)
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)
    candidate_route_order = _parse_int_list(str(args.candidate_route_order))

    # static_graph=True is required: the sequential candidate forward loop calls model() N times
    # per backward pass, which causes DDP's gradient-ready hooks to fire multiple times for the
    # same LoRA parameters when gradient checkpointing is enabled (reentrant backward).
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=False, static_graph=True)]
    accelerator = Accelerator(
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        kwargs_handlers=kwargs_handlers,
    )
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    all_route_labels = _load_route_labels(dataset_dir)
    target_route_idxs = _parse_route_indices(args.target_route_idxs, len(all_route_labels))
    route_labels = [all_route_labels[int(idx)] for idx in target_route_idxs]
    target_dim = len(route_labels)
    if sorted(candidate_route_order) != list(range(target_dim)):
        raise ValueError(
            f"--candidate-route-order must contain each selected route exactly once. "
            f"Got {candidate_route_order} for {target_dim} routes"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    train_rows = _shuffle_rows(list(_load_split(dataset_dir, "train")), args.max_train_rows, args.seed)
    eval_rows = _shuffle_rows(list(_load_split(dataset_dir, "eval")), args.max_eval_rows, args.seed + 1)
    train_dataset = ListwiseVerifierDataset(
        train_rows,
        tokenizer,
        route_labels,
        int(args.max_seq_length),
        route_order=candidate_route_order,
        source_route_idxs=target_route_idxs,
        tie_margin=float(args.tie_margin),
    )
    eval_dataset = ListwiseVerifierDataset(
        eval_rows,
        tokenizer,
        route_labels,
        int(args.max_seq_length),
        route_order=candidate_route_order,
        source_route_idxs=target_route_idxs,
        tie_margin=float(args.tie_margin),
    )
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError(f"Prepared empty dataset train={len(train_dataset)} eval={len(eval_dataset)}")

    collate_fn = lambda batch: _collate_groups(batch, pad_token_id=int(pad_token_id))
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
    train_report_loader = DataLoader(
        train_dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = QwenEmbeddingRouter(
        args.model_name,
        target_dim=1,
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
        predict_costs=False,
        cost_target_dim=0,
        cost_gradient_mode="joint",
        predict_zero_reward_failure=False,
        embedding_input_layout="single",
        segment_count=1,
    )
    init_from_model_checkpoint_report = None
    if args.init_from_model_checkpoint:
        init_from_model_checkpoint_report = _load_model_only_checkpoint(model, Path(args.init_from_model_checkpoint))
        if accelerator.is_main_process:
            print(
                f"Initialized model weights from {init_from_model_checkpoint_report['checkpoint']} "
                f"({init_from_model_checkpoint_report['loaded_tensors']} tensors)",
                flush=True,
            )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(args.lr), weight_decay=float(args.weight_decay))
    update_steps_per_epoch = math.ceil(len(train_loader) / int(args.gradient_accumulation_steps))
    total_update_steps = max(1, int(args.num_epochs) * update_steps_per_epoch)
    warmup_steps = int(total_update_steps * float(args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)
    model, optimizer, train_loader, eval_loader, train_report_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, train_report_loader, scheduler
    )

    config = {
        "model_name": args.model_name,
        "dataset_dir": str(dataset_dir),
        "all_route_labels": all_route_labels,
        "target_route_idxs": [int(idx) for idx in target_route_idxs],
        "route_labels": route_labels,
        "candidate_route_order": candidate_route_order,
        "candidate_route_order_labels": [route_labels[idx] for idx in candidate_route_order],
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
        "prepared_train_groups": len(train_dataset),
        "prepared_train_attempt_rows": len(train_dataset.candidates),
        "prepared_eval_groups": len(eval_dataset),
        "prepared_eval_attempt_rows": len(eval_dataset.candidates),
        "seed": int(args.seed),
        "dropout": float(args.dropout),
        "mlp_hidden_size": int(args.mlp_hidden_size),
        "torch_dtype": str(args.torch_dtype),
        "attn_implementation": str(args.attn_implementation),
        "encoder_frozen": bool(args.encoder_frozen),
        "use_lora": bool(args.use_lora),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_target_modules": [module.strip() for module in str(args.lora_target_modules).split(",") if module.strip()],
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "tie_margin": float(args.tie_margin),
        "loss_type": "proxy_listwise_ce",
        "ddp_find_unused_parameters": bool(args.ddp_find_unused_parameters),
        "checkpoint_every_epoch": bool(args.checkpoint_every_epoch),
        "epoch_report_every": int(args.epoch_report_every),
        "resume_from_checkpoint": str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
        "init_from_model_checkpoint": str(args.init_from_model_checkpoint) if args.init_from_model_checkpoint else None,
        "init_from_model_checkpoint_report": init_from_model_checkpoint_report,
    }
    if accelerator.is_main_process:
        write_json(output_dir / "train_config.json", config)

    history: list[dict[str, Any]] = []
    start_epoch = 0
    resume_from_checkpoint = Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    if resume_from_checkpoint is not None:
        if not resume_from_checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint directory: {resume_from_checkpoint}")
        accelerator.load_state(str(resume_from_checkpoint))
        state_path = resume_from_checkpoint / "trainer_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            start_epoch = int(state.get("next_epoch", int(state.get("epoch", -1)) + 1))
            history = list(state.get("history") or [])
        else:
            try:
                start_epoch = int(str(resume_from_checkpoint.name).split("_")[-1]) + 1
            except ValueError:
                start_epoch = 0
        if accelerator.is_main_process:
            print(f"Resumed from {resume_from_checkpoint}; starting at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, int(args.num_epochs)):
        model.train()
        running_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"Train listwise verifier epoch {epoch}", disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                reward_logits = _forward_candidate_logits_sequential(
                    model,
                    batch["input_ids"],
                    batch["attention_mask"],
                )
                loss = _listwise_ce_loss(
                    reward_logits,
                    batch["group_offsets"],
                    batch["group_sizes"],
                    batch["group_target_local_indices"],
                )
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                running_losses.append(float(loss.detach().item()))
        accelerator.wait_for_everyone()
        eval_loss, _eval_pred_rows = _evaluate(accelerator, model, eval_loader, eval_dataset, "Eval listwise verifier")
        if accelerator.is_main_process:
            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(np.mean(running_losses)) if running_losses else math.nan,
                    "eval_loss": float(eval_loss),
                }
            )
        accelerator.wait_for_everyone()

        if int(args.epoch_report_every) > 0 and (int(epoch) + 1) % int(args.epoch_report_every) == 0:
            train_report_loss, train_pred_rows = _evaluate(
                accelerator, model, train_report_loader, train_dataset, f"Predict train listwise verifier epoch {epoch}"
            )
            _, eval_pred_rows = _evaluate(
                accelerator, model, eval_loader, eval_dataset, f"Predict eval listwise verifier epoch {epoch}"
            )
            if accelerator.is_main_process:
                _write_reports(
                    output_dir / "epoch_reports" / f"epoch_{epoch:04d}",
                    train_pred_rows,
                    eval_pred_rows,
                    train_rows,
                    eval_rows,
                    route_labels,
                    history,
                    config,
                    float(train_report_loss),
                    float(eval_loss),
                )
            accelerator.wait_for_everyone()

        if bool(args.checkpoint_every_epoch):
            checkpoint_dir = output_dir / "checkpoints" / f"epoch_{epoch:04d}"
            accelerator.save_state(str(checkpoint_dir))
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                write_json(
                    checkpoint_dir / "trainer_state.json",
                    {
                        "epoch": int(epoch),
                        "next_epoch": int(epoch) + 1,
                        "history": history,
                        "num_epochs": int(args.num_epochs),
                    },
                )
            accelerator.wait_for_everyone()

    train_loss, train_pred_rows = _evaluate(
        accelerator, model, train_report_loader, train_dataset, "Predict train listwise verifier"
    )
    eval_loss, eval_pred_rows = _evaluate(accelerator, model, eval_loader, eval_dataset, "Predict eval listwise verifier")
    if not accelerator.is_main_process:
        return

    _write_reports(
        output_dir,
        train_pred_rows,
        eval_pred_rows,
        train_rows,
        eval_rows,
        route_labels,
        history,
        config,
        float(train_loss),
        float(eval_loss),
    )
    if args.save_model:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.reward_head.state_dict(), output_dir / "scorer_head.pt")
        if hasattr(unwrapped, "encoder") and hasattr(unwrapped.encoder, "save_pretrained"):
            unwrapped.encoder.save_pretrained(output_dir / "encoder")
        tokenizer.save_pretrained(output_dir / "tokenizer")


if __name__ == "__main__":
    main()

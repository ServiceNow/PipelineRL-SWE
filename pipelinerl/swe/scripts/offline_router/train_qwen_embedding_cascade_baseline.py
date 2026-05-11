#!/usr/bin/env python
import argparse
import csv
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
    DEFAULT_UTILITY_LAMBDAS,
    _argmax_index,
    _load_route_labels,
    _load_split,
    _shuffle_rows,
)
from pipelinerl.swe.scripts.offline_router.train_qwen_embedding_router_baseline import (
    QwenEmbeddingRouter,
    _dtype_from_name,
    _safe_corr,
    _write_csv,
    _write_jsonl,
)


def _problem_key(dataset: Any, problem_id: str) -> str:
    return f"{dataset}::{problem_id}"


def _build_cascade_input_text(row: dict[str, Any], route_label: str, route_output: str) -> str | None:
    prompt_text = row.get("prompt_text")
    if not isinstance(prompt_text, str) or not isinstance(route_output, str):
        return None
    return (
        "Predict the realized proxy reward for this repair attempt.\n"
        "The proxy reward is computed after the repair run by comparing this route's patch against the gold patch.\n"
        "Use the original repair prompt, the route identity, and the actual generated attempt.\n\n"
        "[Route]\n"
        f"{route_label}\n\n"
        "[Original Repair Prompt]\n"
        f"{prompt_text}\n\n"
        "[Generated Repair Attempt]\n"
        f"{route_output}"
    )


class CascadeAttemptDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        route_labels: list[str],
        max_seq_length: int,
        route_idxs: list[int],
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        target_dim = len(route_labels)
        invalid_route_idxs = [idx for idx in route_idxs if not 0 <= int(idx) < target_dim]
        if invalid_route_idxs:
            raise ValueError(f"route_idxs={invalid_route_idxs} are out of range for {target_dim} routes")
        for source_idx, row in enumerate(rows):
            targets = row.get("performance_targets")
            route_outputs = row.get("route_outputs")
            if not isinstance(targets, list) or len(targets) != target_dim:
                continue
            if not isinstance(route_outputs, list) or len(route_outputs) != target_dim:
                continue
            try:
                problem_id = problem_id_from_item(row)
                reward_targets = [float(value) for value in targets]
            except (TypeError, ValueError):
                continue
            for route_idx in route_idxs:
                output_text = route_outputs[int(route_idx)]
                input_text = _build_cascade_input_text(row, route_labels[int(route_idx)], output_text)
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
                        "source_idx": int(source_idx),
                        "problem_id": problem_id,
                        "dataset": row.get("dataset"),
                        "repo": row.get("repo"),
                        "language": row.get("language"),
                        "route_idx": int(route_idx),
                        "route_label": route_labels[int(route_idx)],
                        "target": float(reward_targets[int(route_idx)]),
                        "input_ids": [int(value) for value in input_ids],
                        "attention_mask": [int(value) for value in attention_mask],
                    }
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def _collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    targets = torch.zeros((len(batch), 1), dtype=torch.float32)
    row_indices = torch.zeros((len(batch),), dtype=torch.long)
    for idx, row in enumerate(batch):
        seq_len = len(row["input_ids"])
        start = max_len - seq_len
        input_ids[idx, start:] = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask[idx, start:] = torch.tensor(row["attention_mask"], dtype=torch.long)
        targets[idx, 0] = float(row["target"])
        row_indices[idx] = int(row["row_idx"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "targets": targets,
        "row_indices": row_indices,
    }


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _candidate_thresholds(scores: np.ndarray, max_candidates: int) -> list[float]:
    if scores.size == 0:
        return [0.0]
    quantiles = np.linspace(0.0, 1.0, int(max_candidates))
    values = np.quantile(scores.astype(np.float64), quantiles)
    eps = max(1.0e-6, float(np.std(scores)) * 1.0e-6)
    candidates = [float(np.min(scores) - eps)]
    candidates.extend(float(value) for value in values)
    candidates.append(float(np.max(scores) + eps))
    return sorted(set(candidates))


def _load_eval_lookup(source_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        try:
            problem_id = problem_id_from_item(row)
        except ValueError:
            continue
        key = _problem_key(row.get("dataset"), problem_id)
        if key not in lookup:
            lookup[key] = row
    return lookup


def _prediction_matrix(
    prediction_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    route_labels: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    target_dim = len(route_labels)
    eval_lookup = _load_eval_lookup(source_rows)
    pred_by_key: dict[str, dict[int, float]] = {}
    for row in prediction_rows:
        key = _problem_key(row.get("dataset"), str(row.get("problem_id")))
        pred_by_key.setdefault(key, {})[int(row["route_idx"])] = float(row["pred_score"])

    y_true: list[list[float]] = []
    y_pred: list[list[float]] = []
    examples: list[dict[str, Any]] = []
    for key, source_row in eval_lookup.items():
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
            continue
        route_preds = pred_by_key.get(key)
        if route_preds is None or any(idx not in route_preds for idx in range(target_dim)):
            continue
        true_rewards = [float(value) for value in rewards]
        pred_scores = [float(route_preds[idx]) for idx in range(target_dim)]
        y_true.append(true_rewards)
        y_pred.append(pred_scores)
        examples.append(
            {
                "key": key,
                "dataset": source_row.get("dataset"),
                "problem_id": problem_id_from_item(source_row),
                "rewards": true_rewards,
                "pred_scores": pred_scores,
                "prompt_tokens": [float(value) for value in prompt_tokens],
                "output_tokens": [float(value) for value in output_tokens],
                "oracle_choice_idx": _argmax_index(true_rewards),
            }
        )
    return np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64), examples


def _cascade_choice(pred_scores: list[float], order: list[int], thresholds: list[float]) -> int:
    for pos, route_idx in enumerate(order[:-1]):
        if float(pred_scores[int(route_idx)]) >= float(thresholds[pos]):
            return int(route_idx)
    return int(order[-1])


def _cascade_called_routes(pred_scores: list[float], order: list[int], thresholds: list[float]) -> list[int]:
    called: list[int] = []
    for pos, route_idx in enumerate(order):
        called.append(int(route_idx))
        if pos < len(order) - 1 and float(pred_scores[int(route_idx)]) >= float(thresholds[pos]):
            break
    return called


def _summarize_policy(
    *,
    policy: str,
    policy_type: str,
    examples: list[dict[str, Any]],
    route_labels: list[str],
    lambdas: list[float],
    choices: list[int],
    called_routes: list[list[int]],
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    target_dim = len(route_labels)
    n = len(examples)
    reward_sum = 0.0
    output_cost_sum = 0.0
    total_cost_sum = 0.0
    oracle_match_sum = 0
    choice_counts = [0] * target_dim
    for example, choice, called in zip(examples, choices, called_routes):
        choice = int(choice)
        choice_counts[choice] += 1
        reward_sum += float(example["rewards"][choice])
        output_cost_sum += sum(float(example["output_tokens"][idx]) for idx in called)
        total_cost_sum += sum(float(example["prompt_tokens"][idx]) + float(example["output_tokens"][idx]) for idx in called)
        if choice == int(example["oracle_choice_idx"]):
            oracle_match_sum += 1
    mean_reward = math.nan if n == 0 else reward_sum / n
    mean_output_cost = math.nan if n == 0 else output_cost_sum / n
    mean_total_cost = math.nan if n == 0 else total_cost_sum / n
    rows: list[dict[str, Any]] = []
    for lambda_value in lambdas:
        for cost_metric, mean_cost in (("output_tokens", mean_output_cost), ("total_tokens", mean_total_cost)):
            rows.append(
                {
                    "policy": policy,
                    "policy_type": policy_type,
                    "lambda": float(lambda_value),
                    "cost_metric": cost_metric,
                    "mean_reward": mean_reward,
                    "mean_cost": mean_cost,
                    "mean_utility": mean_reward - (float(lambda_value) * mean_cost),
                    "oracle_match_rate": math.nan if n == 0 else oracle_match_sum / n,
                    "choice_counts_by_route": {
                        route_labels[idx]: int(choice_counts[idx]) for idx in range(target_dim)
                    },
                    "thresholds": list(thresholds) if thresholds is not None else None,
                }
            )
    return rows


def _optimize_thresholds(
    examples: list[dict[str, Any]],
    order: list[int],
    lambda_value: float,
    cost_metric: str,
    max_threshold_candidates: int,
) -> tuple[list[float], float]:
    if len(order) < 2:
        return [], math.nan
    candidates_by_stage = [
        _candidate_thresholds(
            np.asarray([example["pred_scores"][int(route_idx)] for example in examples], dtype=np.float64),
            max_threshold_candidates,
        )
        for route_idx in order[:-1]
    ]
    best_thresholds: list[float] | None = None
    best_utility = -float("inf")
    if len(candidates_by_stage) != 2:
        raise ValueError("This baseline currently expects a three-route cascade with two thresholds")
    for threshold_0 in candidates_by_stage[0]:
        for threshold_1 in candidates_by_stage[1]:
            thresholds = [float(threshold_0), float(threshold_1)]
            reward_sum = 0.0
            cost_sum = 0.0
            for example in examples:
                called = _cascade_called_routes(example["pred_scores"], order, thresholds)
                choice = called[-1]
                reward_sum += float(example["rewards"][choice])
                if cost_metric == "output_tokens":
                    cost_sum += sum(float(example["output_tokens"][idx]) for idx in called)
                elif cost_metric == "total_tokens":
                    cost_sum += sum(
                        float(example["prompt_tokens"][idx]) + float(example["output_tokens"][idx])
                        for idx in called
                    )
                else:
                    raise ValueError(f"Unsupported cost_metric={cost_metric}")
            utility = (reward_sum / len(examples)) - (float(lambda_value) * (cost_sum / len(examples)))
            if utility > best_utility:
                best_utility = utility
                best_thresholds = thresholds
    if best_thresholds is None:
        raise ValueError("No cascade thresholds were evaluated")
    return best_thresholds, float(best_utility)


def _compute_cascade_report(
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    route_labels: list[str],
    order: list[int],
    lambdas: list[float],
    max_threshold_candidates: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    target_dim = len(route_labels)
    for route_idx, route_label in enumerate(route_labels):
        choices = [route_idx for _ in eval_examples]
        called_routes = [[route_idx] for _ in eval_examples]
        rows.extend(
            _summarize_policy(
                policy=f"always::{route_label}",
                policy_type="always_direct",
                examples=eval_examples,
                route_labels=route_labels,
                lambdas=lambdas,
                choices=choices,
                called_routes=called_routes,
            )
        )
    oracle_choices = [int(example["oracle_choice_idx"]) for example in eval_examples]
    rows.extend(
        _summarize_policy(
            policy="oracle_direct",
            policy_type="oracle_direct",
            examples=eval_examples,
            route_labels=route_labels,
            lambdas=lambdas,
            choices=oracle_choices,
            called_routes=[[choice] for choice in oracle_choices],
        )
    )

    learned_thresholds: dict[str, dict[str, Any]] = {}
    for lambda_value in lambdas:
        for cost_metric in ("output_tokens", "total_tokens"):
            thresholds, train_utility = _optimize_thresholds(
                train_examples,
                order,
                float(lambda_value),
                cost_metric,
                max_threshold_candidates,
            )
            choices = [_cascade_choice(example["pred_scores"], order, thresholds) for example in eval_examples]
            called_routes = [
                _cascade_called_routes(example["pred_scores"], order, thresholds) for example in eval_examples
            ]
            cascade_rows = _summarize_policy(
                policy=f"cascade_train_tau::{cost_metric}::lambda={float(lambda_value):g}",
                policy_type="cascade_train_thresholds",
                examples=eval_examples,
                route_labels=route_labels,
                lambdas=[float(lambda_value)],
                choices=choices,
                called_routes=called_routes,
                thresholds=thresholds,
            )
            rows.extend([row for row in cascade_rows if row["cost_metric"] == cost_metric])
            learned_thresholds[f"{cost_metric}::{float(lambda_value):g}"] = {
                "lambda": float(lambda_value),
                "cost_metric": cost_metric,
                "thresholds": thresholds,
                "train_utility": train_utility,
            }

    return {
        "route_labels": list(route_labels),
        "cascade_order": [int(idx) for idx in order],
        "cascade_order_labels": [route_labels[int(idx)] for idx in order],
        "n_train_examples": len(train_examples),
        "n_eval_examples": len(eval_examples),
        "max_threshold_candidates": int(max_threshold_candidates),
        "learned_thresholds": learned_thresholds,
        "utility_rows": rows,
    }


@torch.no_grad()
def _evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: CascadeAttemptDataset,
    desc: str,
) -> tuple[float, list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc=desc, disable=not accelerator.is_main_process):
        reward_logits, _, _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        preds = reward_logits.float()
        targets = batch["targets"].float()
        loss = F.mse_loss(preds, targets, reduction="sum")
        gathered_loss = accelerator.gather_for_metrics(loss.detach().reshape(1)).detach().cpu()
        gathered_preds = accelerator.gather_for_metrics(preds).detach().cpu()
        gathered_targets = accelerator.gather_for_metrics(targets).detach().cpu()
        gathered_indices = accelerator.gather_for_metrics(batch["row_indices"]).detach().cpu().tolist()
        if accelerator.is_main_process:
            total_loss += float(torch.sum(gathered_loss).item())
            total_examples += int(gathered_targets.shape[0])
            for idx in range(gathered_preds.shape[0]):
                source_meta = dataset.rows[int(gathered_indices[idx])]
                rows.append(
                    {
                        "problem_id": source_meta["problem_id"],
                        "dataset": source_meta["dataset"],
                        "repo": source_meta["repo"],
                        "language": source_meta["language"],
                        "route_idx": int(source_meta["route_idx"]),
                        "route_label": source_meta["route_label"],
                        "true_score": float(gathered_targets[idx, 0].item()),
                        "pred_score": float(gathered_preds[idx, 0].item()),
                    }
                )
    if not accelerator.is_main_process:
        return math.nan, []
    return (float(total_loss / total_examples) if total_examples > 0 else math.nan), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--cascade-order", default="0,1,2")
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
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-threshold-candidates", type=int, default=51)
    parser.add_argument("--utility-lambdas", default=",".join(str(value) for value in DEFAULT_UTILITY_LAMBDAS))
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)
    order = _parse_int_list(str(args.cascade_order))
    lambdas = _parse_float_list(str(args.utility_lambdas))

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
    if sorted(order) != list(range(target_dim)):
        raise ValueError(f"--cascade-order must contain each route exactly once. Got {order} for {target_dim} routes")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    train_rows = _shuffle_rows(list(_load_split(dataset_dir, "train")), args.max_train_rows, args.seed)
    eval_rows_source = _shuffle_rows(list(_load_split(dataset_dir, "eval")), args.max_eval_rows, args.seed + 1)
    train_dataset = CascadeAttemptDataset(
        train_rows, tokenizer, route_labels, int(args.max_seq_length), route_idxs=order
    )
    eval_dataset = CascadeAttemptDataset(
        eval_rows_source, tokenizer, route_labels, int(args.max_seq_length), route_idxs=order
    )
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError(f"Prepared empty dataset train={len(train_dataset)} eval={len(eval_dataset)}")

    collate_fn = lambda batch: _collate(batch, pad_token_id=int(pad_token_id))
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
        "dataset_dir": str(dataset_dir),
        "route_labels": route_labels,
        "cascade_order": order,
        "cascade_order_labels": [route_labels[idx] for idx in order],
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
        "prepared_train_attempt_rows": len(train_dataset),
        "prepared_eval_attempt_rows": len(eval_dataset),
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
        "max_threshold_candidates": int(args.max_threshold_candidates),
        "utility_lambdas": lambdas,
        "ddp_find_unused_parameters": bool(args.ddp_find_unused_parameters),
    }
    if accelerator.is_main_process:
        write_json(output_dir / "train_config.json", config)

    history: list[dict[str, Any]] = []
    for epoch in range(int(args.num_epochs)):
        model.train()
        running_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"Train Qwen cascade scorer epoch {epoch}", disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                reward_logits, _, _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = F.mse_loss(reward_logits.float(), batch["targets"].float())
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                running_losses.append(float(loss.detach().item()))
        accelerator.wait_for_everyone()
        eval_loss, _ = _evaluate(
            accelerator, model, eval_loader, eval_dataset, desc="Eval Qwen cascade scorer"
        )
        if accelerator.is_main_process:
            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(np.mean(running_losses)) if running_losses else math.nan,
                    "eval_loss": float(eval_loss),
                }
            )
        accelerator.wait_for_everyone()

    train_pred_loader = DataLoader(
        train_dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    train_pred_loader = accelerator.prepare(train_pred_loader)
    train_loss, train_pred_rows = _evaluate(
        accelerator, model, train_pred_loader, train_dataset, desc="Predict train Qwen cascade scorer"
    )
    eval_loss, eval_pred_rows = _evaluate(
        accelerator, model, eval_loader, eval_dataset, desc="Predict eval Qwen cascade scorer"
    )
    if not accelerator.is_main_process:
        return

    train_true, train_pred, train_examples = _prediction_matrix(train_pred_rows, train_rows, route_labels)
    eval_true, eval_pred, eval_examples = _prediction_matrix(eval_pred_rows, eval_rows_source, route_labels)
    route_metrics = compute_per_route_metrics(eval_true, eval_pred, route_labels)
    pairwise_metrics = compute_pairwise_metrics(eval_true, eval_pred, route_labels)
    scalar_true = np.asarray([row["true_score"] for row in eval_pred_rows], dtype=np.float64)
    scalar_pred = np.asarray([row["pred_score"] for row in eval_pred_rows], dtype=np.float64)
    scalar_metrics = {
        "n_eval_attempts": int(scalar_true.shape[0]),
        "mse": float(np.mean((scalar_pred - scalar_true) ** 2)),
        "rmse": float(np.sqrt(np.mean((scalar_pred - scalar_true) ** 2))),
        "mae": float(np.mean(np.abs(scalar_pred - scalar_true))),
        "pearson": _safe_corr(scalar_pred, scalar_true),
        "mean_true": float(np.mean(scalar_true)),
        "mean_pred": float(np.mean(scalar_pred)),
        "std_true": float(np.std(scalar_true)),
        "std_pred": float(np.std(scalar_pred)),
    }
    cascade_report = _compute_cascade_report(
        train_examples=train_examples,
        eval_examples=eval_examples,
        route_labels=route_labels,
        order=order,
        lambdas=lambdas,
        max_threshold_candidates=int(args.max_threshold_candidates),
    )

    _write_jsonl(output_dir / "train_attempt_predictions.jsonl", train_pred_rows)
    _write_jsonl(output_dir / "eval_attempt_predictions.jsonl", eval_pred_rows)
    _write_csv(output_dir / "route_metrics.csv", route_metrics, csv_headers_for_route_metrics())
    _write_csv(output_dir / "pairwise_metrics.csv", pairwise_metrics, csv_headers_for_pairwise_metrics())
    cascade_headers = [
        "policy",
        "policy_type",
        "lambda",
        "cost_metric",
        "mean_reward",
        "mean_cost",
        "mean_utility",
        "oracle_match_rate",
        "choice_counts_by_route",
        "thresholds",
    ]
    _write_csv(output_dir / "cascade_utility_vs_baselines.csv", cascade_report["utility_rows"], cascade_headers)
    write_json(output_dir / "cascade_utility_vs_baselines.json", cascade_report)
    summary = {
        "history": history,
        "train_loss_final": float(train_loss),
        "eval_loss_final": float(eval_loss),
        "scalar_metrics": scalar_metrics,
        "route_metrics": route_metrics,
        "pairwise_metrics": pairwise_metrics,
        "cascade_utility": cascade_report,
        "config": config,
    }
    write_json(output_dir / "summary.json", summary)
    if args.save_model:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.reward_head.state_dict(), output_dir / "scorer_head.pt")
        if hasattr(unwrapped, "encoder") and hasattr(unwrapped.encoder, "save_pretrained"):
            unwrapped.encoder.save_pretrained(output_dir / "encoder")
        tokenizer.save_pretrained(output_dir / "tokenizer")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import argparse
import json
import math
import random
from itertools import combinations
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
from pipelinerl.swe.scripts.offline_router.train_qwen_embedding_router_baseline import (
    QwenEmbeddingRouter,
    _dtype_from_name,
    _parse_route_indices,
    _safe_corr,
    _write_csv,
    _write_jsonl,
)


DEFAULT_AWS_OUTPUT_COST_WEIGHTS = [2.78e-7, 1.299e-6, 4.64e-6, 1.113e-5]
DEFAULT_UTILITY_LAMBDAS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0]


def _parse_float_list(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not parsed:
        raise ValueError("Expected a non-empty comma-separated float list")
    return parsed



def _mask_key(mask: list[bool]) -> str:
    return "".join("1" if value else "0" for value in mask)


def _mask_from_indices(route_count: int, indices: list[int]) -> list[bool]:
    mask = [False] * int(route_count)
    for idx in indices:
        mask[int(idx)] = True
    return mask


def _iter_attempt_states(route_count: int, mode: str) -> list[tuple[list[bool], int]]:
    if mode == "none":
        return []
    if mode == "single":
        return [(_mask_from_indices(route_count, [idx]), idx) for idx in range(route_count)]
    if mode == "all_subsets":
        states: list[tuple[list[bool], int]] = []
        for size in range(1, route_count + 1):
            for subset in combinations(range(route_count), size):
                mask = _mask_from_indices(route_count, list(subset))
                for latest_idx in subset:
                    states.append((mask, int(latest_idx)))
        return states
    raise ValueError(f"Unsupported attempted-state mode: {mode}")


def _build_state_input_text(
    *,
    row: dict[str, Any],
    route_labels: list[str],
    source_route_idxs: list[int],
    state_kind: str,
    attempted_mask: list[bool],
    latest_route_idx: int | None,
    route_costs: list[float] | None,
) -> str | None:
    prompt_text = row.get("prompt_text")
    if not isinstance(prompt_text, str):
        return None
    route_lines = []
    for idx, label in enumerate(route_labels):
        if route_costs is None:
            route_lines.append(f"{idx}: {label}")
        else:
            route_lines.append(f"{idx}: {label} | incremental_cost={float(route_costs[idx]):.8g}")
    route_legend = "\n".join(route_lines)
    if state_kind == "bare":
        return (
            "Predict real pass probabilities for SWE repair routes.\n"
            "A route succeeds if its generated patch passes the SWE evaluation for this task.\n"
            "No repair attempt has been observed yet, so estimate each route from the task alone.\n\n"
            "[Route Order]\n"
            f"{route_legend}\n\n"
            "[State]\n"
            "No route has been attempted.\n\n"
            "[Original Repair Prompt]\n"
            f"{prompt_text}"
        )
    if state_kind != "after_attempt":
        raise ValueError(f"Unsupported state kind: {state_kind}")
    if latest_route_idx is None:
        raise ValueError("after_attempt state requires latest_route_idx")
    route_outputs = row.get("route_outputs")
    source_route_idx = int(source_route_idxs[int(latest_route_idx)])
    if not isinstance(route_outputs, list) or len(route_outputs) <= source_route_idx:
        return None
    latest_output = route_outputs[source_route_idx]
    if not isinstance(latest_output, str):
        return None
    attempted_lines = [
        f"{idx}: {route_labels[idx]}"
        for idx, attempted in enumerate(attempted_mask)
        if attempted
    ]
    attempted_text = "\n".join(attempted_lines) if attempted_lines else "none"
    return (
        "Predict real pass probabilities for SWE repair routes.\n"
        "A route succeeds if its generated patch passes the SWE evaluation for this task.\n"
        "Use the latest observed attempt as diagnostic evidence. The latest route's probability is the stop score; "
        "other route probabilities are continuation scores.\n\n"
        "[Route Order]\n"
        f"{route_legend}\n\n"
        "[State]\n"
        "At least one route has been attempted.\n\n"
        "[Attempted Routes]\n"
        f"{attempted_text}\n\n"
        "[Latest Route]\n"
        f"{latest_route_idx}: {route_labels[int(latest_route_idx)]}\n\n"
        "[Original Repair Prompt]\n"
        f"{prompt_text}\n\n"
        "[Latest Repair Attempt]\n"
        f"{latest_output}"
    )


class StatePolicyDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        route_labels: list[str],
        route_indices: list[int],
        max_seq_length: int,
        attempted_state_mode: str,
        include_bare_state: bool,
        route_costs: list[float] | None,
        route_output_cost_weights: list[float],
        sample_weighting: str = "uniform",
        regret_lambdas: list[float] | None = None,
        regret_route_costs: list[float] | None = None,
        regret_default_route_idx: int = 0,
        regret_weight_scale: float = 4.0,
        regret_weight_power: float = 1.0,
        regret_weight_min: float = 1.0,
        regret_weight_max: float = 8.0,
        normalize_sample_weights: bool = True,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        route_count = len(route_labels)
        if len(route_indices) != route_count:
            raise ValueError("route_indices length must match route_labels length")
        if len(route_output_cost_weights) != route_count:
            raise ValueError("route_output_cost_weights length must match route_labels length")
        if route_count == 0:
            raise ValueError("At least one route is required")
        max_source_idx = max(int(idx) for idx in route_indices)
        state_specs: list[tuple[str, list[bool], int | None]] = []
        if include_bare_state:
            state_specs.append(("bare", [False] * route_count, None))
        state_specs.extend(
            ("after_attempt", mask, latest_idx)
            for mask, latest_idx in _iter_attempt_states(route_count, attempted_state_mode)
        )
        regret_lambdas = [0.0] if regret_lambdas is None else [float(value) for value in regret_lambdas]
        regret_route_costs = (
            [0.0 for _ in range(route_count)]
            if regret_route_costs is None
            else [float(value) for value in regret_route_costs]
        )
        for source_idx, row in enumerate(rows):
            targets = row.get("performance_targets")
            output_tokens = row.get("route_output_tokens")
            if not isinstance(targets, list) or len(targets) <= max_source_idx:
                continue
            if not isinstance(output_tokens, list) or len(output_tokens) <= max_source_idx:
                continue
            try:
                problem_id = problem_id_from_item(row)
                target_rewards = [float(targets[int(source_idx)]) for source_idx in route_indices]
                selected_output_tokens = [float(output_tokens[int(source_idx)]) for source_idx in route_indices]
                selected_route_costs = [
                    float(selected_output_tokens[local_idx]) * float(route_output_cost_weights[local_idx])
                    for local_idx in range(route_count)
                ]
            except (TypeError, ValueError):
                continue
            for state_kind, attempted_mask, latest_route_idx in state_specs:
                input_text = _build_state_input_text(
                    row=row,
                    route_labels=route_labels,
                    source_route_idxs=route_indices,
                    state_kind=state_kind,
                    attempted_mask=attempted_mask,
                    latest_route_idx=latest_route_idx,
                    route_costs=route_costs,
                )
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
                sample_weight, routing_regret = _state_sample_weight(
                    target_rewards=target_rewards,
                    attempted_mask=list(attempted_mask),
                    latest_route_idx=latest_route_idx,
                    sample_weighting=str(sample_weighting),
                    regret_lambdas=regret_lambdas,
                    regret_route_costs=regret_route_costs,
                    regret_default_route_idx=int(regret_default_route_idx),
                    regret_weight_scale=float(regret_weight_scale),
                    regret_weight_power=float(regret_weight_power),
                    regret_weight_min=float(regret_weight_min),
                    regret_weight_max=float(regret_weight_max),
                )
                self.rows.append(
                    {
                        "row_idx": len(self.rows),
                        "source_idx": int(source_idx),
                        "problem_id": problem_id,
                        "dataset": row.get("dataset"),
                        "repo": row.get("repo"),
                        "language": row.get("language"),
                        "state_kind": state_kind,
                        "attempted_mask": list(attempted_mask),
                        "attempted_mask_key": _mask_key(list(attempted_mask)),
                        "latest_route_idx": None if latest_route_idx is None else int(latest_route_idx),
                        "input_ids": [int(value) for value in input_ids],
                        "attention_mask": [int(value) for value in attention_mask],
                        "targets": target_rewards,
                        "route_output_tokens": selected_output_tokens,
                        "route_costs": selected_route_costs,
                        "sample_weight": float(sample_weight),
                        "routing_regret": float(routing_regret),
                    }
                )
        if normalize_sample_weights and self.rows:
            mean_weight = sum(float(row.get("sample_weight", 1.0)) for row in self.rows) / len(self.rows)
            if mean_weight > 0.0:
                for row in self.rows:
                    row["sample_weight"] = float(row.get("sample_weight", 1.0)) / mean_weight

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def _collate(batch: list[dict[str, Any]], pad_token_id: int, target_dim: int) -> dict[str, Any]:
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    targets = torch.zeros((len(batch), target_dim), dtype=torch.float32)
    sample_weights = torch.ones((len(batch),), dtype=torch.float32)
    routing_regrets = torch.zeros((len(batch),), dtype=torch.float32)
    row_indices = torch.zeros((len(batch),), dtype=torch.long)
    attempted_masks = torch.zeros((len(batch), target_dim), dtype=torch.bool)
    latest_route_idxs = torch.full((len(batch),), -1, dtype=torch.long)
    route_costs = torch.zeros((len(batch), target_dim), dtype=torch.float32)
    for idx, row in enumerate(batch):
        seq_len = len(row["input_ids"])
        start = max_len - seq_len
        input_ids[idx, start:] = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask[idx, start:] = torch.tensor(row["attention_mask"], dtype=torch.long)
        targets[idx] = torch.tensor(row["targets"], dtype=torch.float32)
        sample_weights[idx] = float(row.get("sample_weight", 1.0))
        routing_regrets[idx] = float(row.get("routing_regret", 0.0))
        row_indices[idx] = int(row["row_idx"])
        attempted_masks[idx] = torch.tensor(row["attempted_mask"], dtype=torch.bool)
        latest_route_idx = row.get("latest_route_idx")
        latest_route_idxs[idx] = -1 if latest_route_idx is None else int(latest_route_idx)
        route_costs[idx] = torch.tensor(row["route_costs"], dtype=torch.float32)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "targets": targets,
        "sample_weights": sample_weights,
        "routing_regrets": routing_regrets,
        "row_indices": row_indices,
        "attempted_masks": attempted_masks,
        "latest_route_idxs": latest_route_idxs,
        "route_costs": route_costs,
    }


def _state_prediction_key(source_idx: int, latest_route_idx: int | None, attempted_mask_key: str) -> str:
    latest = -1 if latest_route_idx is None else int(latest_route_idx)
    return f"{int(source_idx)}::{latest}::{attempted_mask_key}"


def _argmax(values: list[float]) -> int:
    best_idx = 0
    best_value = float(values[0])
    for idx, value in enumerate(values[1:], start=1):
        numeric = float(value)
        if numeric > best_value:
            best_idx = idx
            best_value = numeric
    return best_idx


def _state_sample_weight(
    *,
    target_rewards: list[float],
    attempted_mask: list[bool],
    latest_route_idx: int | None,
    sample_weighting: str,
    regret_lambdas: list[float],
    regret_route_costs: list[float],
    regret_default_route_idx: int,
    regret_weight_scale: float,
    regret_weight_power: float,
    regret_weight_min: float,
    regret_weight_max: float,
) -> tuple[float, float]:
    if sample_weighting == "uniform":
        return 1.0, 0.0
    if sample_weighting not in {"regret_default", "oracle_margin"}:
        raise ValueError(f"Unsupported sample_weighting={sample_weighting}")
    if len(regret_route_costs) != len(target_rewards):
        raise ValueError("regret_route_costs length must match target rewards length")
    route_count = len(target_rewards)
    regrets: list[float] = []
    for lambda_value in regret_lambdas:
        action_values: list[float] = []
        default_value: float
        if latest_route_idx is None:
            default_idx = min(max(int(regret_default_route_idx), 0), route_count - 1)
            default_value = float(target_rewards[default_idx]) - float(lambda_value) * float(regret_route_costs[default_idx])
            for route_idx, attempted in enumerate(attempted_mask):
                if attempted:
                    continue
                action_values.append(
                    float(target_rewards[route_idx]) - float(lambda_value) * float(regret_route_costs[route_idx])
                )
        else:
            default_value = float(target_rewards[int(latest_route_idx)])
            action_values.append(default_value)
            for route_idx, attempted in enumerate(attempted_mask):
                if attempted:
                    continue
                action_values.append(
                    float(target_rewards[route_idx]) - float(lambda_value) * float(regret_route_costs[route_idx])
                )
        if not action_values:
            regrets.append(0.0)
            continue
        ordered = sorted(action_values, reverse=True)
        best_value = float(ordered[0])
        if sample_weighting == "regret_default":
            regrets.append(max(0.0, best_value - default_value))
        else:
            second_value = float(ordered[1]) if len(ordered) > 1 else best_value
            regrets.append(max(0.0, best_value - second_value))
    routing_regret = float(sum(regrets) / len(regrets)) if regrets else 0.0
    powered = routing_regret ** float(regret_weight_power) if routing_regret > 0.0 else 0.0
    sample_weight = float(regret_weight_min) + float(regret_weight_scale) * powered
    sample_weight = min(float(regret_weight_max), max(float(regret_weight_min), sample_weight))
    return sample_weight, routing_regret


def _sample_weight_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    weights = np.asarray([float(row.get("sample_weight", 1.0)) for row in rows], dtype=np.float64)
    regrets = np.asarray([float(row.get("routing_regret", 0.0)) for row in rows], dtype=np.float64)
    if weights.size == 0:
        return {"n": 0}
    return {
        "n": int(weights.size),
        "weight_mean": float(np.mean(weights)),
        "weight_std": float(np.std(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "regret_mean": float(np.mean(regrets)),
        "regret_std": float(np.std(regrets)),
        "regret_min": float(np.min(regrets)),
        "regret_max": float(np.max(regrets)),
        "regret_positive_fraction": float(np.mean(regrets > 0.0)),
    }


def _utility_decision_aux_loss(
    *,
    probs: torch.Tensor,
    targets: torch.Tensor,
    attempted_masks: torch.Tensor,
    latest_route_idxs: torch.Tensor,
    route_costs: torch.Tensor,
    lambdas: list[float],
    lambda_sampling: str,
    lambda_sample_count: int,
    lambda_min: float,
    lambda_max: float,
    temperature: float,
    sample_weights: torch.Tensor,
    stop_tie_bonus: float,
    bare_out_action: bool,
    regret_weight_mode: str,
    regret_default_route_idx: int,
    regret_weight_scale: float,
    regret_weight_power: float,
    regret_weight_min: float,
    regret_weight_max: float,
) -> torch.Tensor:
    if lambda_sampling == "none":
        sampled_lambdas = [float(value) for value in lambdas]
    else:
        if lambda_sample_count <= 0:
            raise ValueError("decision auxiliary lambda sample count must be positive")
        if lambda_min < 0.0 or lambda_max < lambda_min:
            raise ValueError("invalid decision auxiliary lambda sampling range")
        if lambda_sampling == "uniform":
            values = torch.empty((int(lambda_sample_count),), device=probs.device).uniform_(
                float(lambda_min), float(lambda_max)
            )
        elif lambda_sampling == "log_uniform":
            if lambda_min <= 0.0:
                raise ValueError("log-uniform decision auxiliary lambda sampling requires lambda_min > 0")
            values = torch.empty((int(lambda_sample_count),), device=probs.device).uniform_(
                math.log(float(lambda_min)), math.log(float(lambda_max))
            ).exp()
        else:
            raise ValueError(f"Unsupported decision auxiliary lambda sampling mode: {lambda_sampling}")
        sampled_lambdas = [float(value.detach().item()) for value in values]

    if not sampled_lambdas:
        return probs.new_tensor(0.0)
    if temperature <= 0.0:
        raise ValueError("decision auxiliary temperature must be positive")
    if regret_weight_mode not in {"none", "oracle_margin", "default_action"}:
        raise ValueError(f"Unsupported decision auxiliary regret weight mode: {regret_weight_mode}")
    targets = targets.float()
    route_costs = route_costs.float()
    attempted_masks = attempted_masks.bool()
    latest_route_idxs = latest_route_idxs.long()
    unattempted_masks = ~attempted_masks
    has_stop = latest_route_idxs >= 0
    batch_size, route_count = targets.shape
    route_positions = torch.arange(batch_size, device=targets.device)
    losses: list[torch.Tensor] = []
    for lambda_value in sampled_lambdas:
        lambda_float = float(lambda_value)
        pred_scores = probs.float() - lambda_float * route_costs
        true_scores = targets - lambda_float * route_costs
        valid_actions = unattempted_masks.clone()
        if bool(has_stop.any().item()):
            stop_rows = route_positions[has_stop]
            stop_cols = latest_route_idxs[has_stop]
            valid_actions[stop_rows, stop_cols] = True
            pred_scores[stop_rows, stop_cols] = probs.float()[stop_rows, stop_cols]
            true_scores[stop_rows, stop_cols] = targets[stop_rows, stop_cols] + float(stop_tie_bonus)
        if bool(bare_out_action):
            bare_rows = (~has_stop).view(-1, 1)
            out_pred_scores = torch.zeros((batch_size, 1), dtype=pred_scores.dtype, device=pred_scores.device)
            out_true_scores = torch.zeros((batch_size, 1), dtype=true_scores.dtype, device=true_scores.device)
            pred_scores = torch.cat([pred_scores, out_pred_scores], dim=1)
            true_scores = torch.cat([true_scores, out_true_scores], dim=1)
            valid_actions = torch.cat([valid_actions, bare_rows], dim=1)
        masked_pred_scores = pred_scores.masked_fill(~valid_actions, -1.0e9) / float(temperature)
        masked_true_scores = true_scores.masked_fill(~valid_actions, -1.0e9)
        target_actions = torch.argmax(masked_true_scores, dim=1)
        per_sample_ce = F.cross_entropy(masked_pred_scores, target_actions, reduction="none")
        if regret_weight_mode != "none":
            best_values = masked_true_scores.max(dim=1).values
            if regret_weight_mode == "oracle_margin":
                topk = torch.topk(masked_true_scores, k=min(2, masked_true_scores.shape[1]), dim=1).values
                if topk.shape[1] > 1:
                    margins = (topk[:, 0] - topk[:, 1]).clamp_min(0.0)
                else:
                    margins = torch.zeros_like(best_values)
            else:
                default_cols = torch.full(
                    (batch_size,),
                    min(max(int(regret_default_route_idx), 0), route_count - 1),
                    dtype=torch.long,
                    device=targets.device,
                )
                if bool(has_stop.any().item()):
                    default_cols = torch.where(has_stop, latest_route_idxs, default_cols)
                default_values = true_scores.gather(1, default_cols.view(-1, 1)).squeeze(1)
                margins = (best_values - default_values).clamp_min(0.0)
            powered = torch.where(
                margins > 0.0,
                margins.pow(float(regret_weight_power)),
                torch.zeros_like(margins),
            )
            regret_weights = float(regret_weight_min) + float(regret_weight_scale) * powered
            regret_weights = regret_weights.clamp(float(regret_weight_min), float(regret_weight_max))
            per_sample_ce = per_sample_ce * regret_weights
        losses.append(per_sample_ce)
    per_sample_loss = torch.stack(losses, dim=0).mean(dim=0)
    return (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)


def _selected_examples(
    rows: list[dict[str, Any]],
    route_indices: list[int],
    route_cost_weights: list[float],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    max_source_idx = max(int(idx) for idx in route_indices)
    for source_idx, row in enumerate(rows):
        targets = row.get("performance_targets")
        output_tokens = row.get("route_output_tokens")
        prompt_tokens = row.get("route_prompt_tokens")
        if not isinstance(targets, list) or len(targets) <= max_source_idx:
            continue
        if not isinstance(output_tokens, list) or len(output_tokens) <= max_source_idx:
            continue
        try:
            rewards = [float(targets[int(idx)]) for idx in route_indices]
            output_token_values = [float(output_tokens[int(idx)]) for idx in route_indices]
            prompt_token_values = (
                [float(prompt_tokens[int(idx)]) for idx in route_indices]
                if isinstance(prompt_tokens, list) and len(prompt_tokens) > max_source_idx
                else [0.0 for _ in route_indices]
            )
            costs = [
                float(output_token_values[local_idx]) * float(route_cost_weights[local_idx])
                for local_idx in range(len(route_indices))
            ]
            problem_id = problem_id_from_item(row)
        except (TypeError, ValueError):
            continue
        examples.append(
            {
                "source_idx": int(source_idx),
                "problem_id": problem_id,
                "dataset": row.get("dataset"),
                "repo": row.get("repo"),
                "language": row.get("language"),
                "rewards": rewards,
                "route_output_tokens": output_token_values,
                "route_prompt_tokens": prompt_token_values,
                "route_costs": costs,
                "oracle_choice_idx": _argmax(rewards),
            }
        )
    return examples


def _fixed_train_costs(
    train_rows: list[dict[str, Any]],
    route_indices: list[int],
    route_cost_weights: list[float],
) -> list[float]:
    examples = _selected_examples(train_rows, route_indices, route_cost_weights)
    route_count = len(route_indices)
    totals = [0.0] * route_count
    counts = [0] * route_count
    for example in examples:
        for idx, cost in enumerate(example["route_costs"]):
            totals[idx] += float(cost)
            counts[idx] += 1
    return [totals[idx] / counts[idx] if counts[idx] else 0.0 for idx in range(route_count)]


@torch.no_grad()
def _evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: StatePolicyDataset,
    route_labels: list[str],
    desc: str,
) -> tuple[float, list[dict[str, Any]], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_values = 0
    rows: list[dict[str, Any]] = []
    true_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    for batch in tqdm(loader, desc=desc, disable=not accelerator.is_main_process):
        logits, _, _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        targets = batch["targets"].float()
        loss = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="sum")
        probs = torch.sigmoid(logits.float())
        gathered_loss = accelerator.gather_for_metrics(loss.detach().reshape(1)).detach().cpu()
        gathered_probs = accelerator.gather_for_metrics(probs).detach().cpu()
        gathered_targets = accelerator.gather_for_metrics(targets).detach().cpu()
        gathered_indices = accelerator.gather_for_metrics(batch["row_indices"]).detach().cpu().tolist()
        if accelerator.is_main_process:
            total_loss += float(torch.sum(gathered_loss).item())
            total_values += int(gathered_targets.numel())
            true_chunks.append(gathered_targets.numpy())
            pred_chunks.append(gathered_probs.numpy())
            for idx in range(gathered_probs.shape[0]):
                source_meta = dataset.rows[int(gathered_indices[idx])]
                rows.append(
                    {
                        "problem_id": source_meta["problem_id"],
                        "dataset": source_meta["dataset"],
                        "repo": source_meta["repo"],
                        "language": source_meta["language"],
                        "source_idx": int(source_meta["source_idx"]),
                        "state_kind": source_meta["state_kind"],
                        "attempted_mask": list(source_meta["attempted_mask"]),
                        "attempted_mask_key": source_meta["attempted_mask_key"],
                        "latest_route_idx": source_meta["latest_route_idx"],
                        "latest_route_label": (
                            None
                            if source_meta["latest_route_idx"] is None
                            else route_labels[int(source_meta["latest_route_idx"])]
                        ),
                        "true_rewards": [float(value) for value in gathered_targets[idx].tolist()],
                        "pred_success_probs": [float(value) for value in gathered_probs[idx].tolist()],
                        "sample_weight": float(source_meta.get("sample_weight", 1.0)),
                        "routing_regret": float(source_meta.get("routing_regret", 0.0)),
                        "route_labels": list(route_labels),
                    }
                )
    if not accelerator.is_main_process:
        empty = np.empty((0, len(route_labels)))
        return math.nan, [], empty, empty
    y_true = np.concatenate(true_chunks, axis=0) if true_chunks else np.empty((0, len(route_labels)))
    y_pred = np.concatenate(pred_chunks, axis=0) if pred_chunks else np.empty((0, len(route_labels)))
    eval_loss = float(total_loss / total_values) if total_values > 0 else math.nan
    return eval_loss, rows, y_true, y_pred


def _prediction_lookup(pred_rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    lookup: dict[str, list[float]] = {}
    for row in pred_rows:
        lookup[
            _state_prediction_key(
                int(row["source_idx"]),
                row.get("latest_route_idx"),
                str(row["attempted_mask_key"]),
            )
        ] = [float(value) for value in row["pred_success_probs"]]
    return lookup


def _lookup_state_probs(
    lookup: dict[str, list[float]],
    source_idx: int,
    latest_route_idx: int | None,
    attempted_mask: list[bool],
) -> list[float] | None:
    exact = lookup.get(_state_prediction_key(source_idx, latest_route_idx, _mask_key(attempted_mask)))
    if exact is not None:
        return exact
    if latest_route_idx is None:
        return None
    single_mask = _mask_from_indices(len(attempted_mask), [int(latest_route_idx)])
    return lookup.get(_state_prediction_key(source_idx, latest_route_idx, _mask_key(single_mask)))


def _choose_run_from_probs(
    probs: list[float],
    route_costs: list[float],
    lambda_value: float,
    attempted_mask: list[bool],
) -> int | None:
    best_idx: int | None = None
    best_score = -float("inf")
    for idx, attempted in enumerate(attempted_mask):
        if attempted:
            continue
        score = float(probs[idx]) - float(lambda_value) * float(route_costs[idx])
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _simulate_direct_policy(
    *,
    examples: list[dict[str, Any]],
    pred_lookup: dict[str, list[float]],
    lambda_value: float,
    route_count: int,
    selection_route_costs: list[float],
    policy_bare_out_action: bool,
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    bare_mask = [False] * route_count
    for example in examples:
        probs = _lookup_state_probs(pred_lookup, int(example["source_idx"]), None, bare_mask)
        if probs is None:
            choice = 0
        else:
            selected = _choose_run_from_probs(probs, selection_route_costs, lambda_value, bare_mask)
            if selected is None:
                choice = -1 if bool(policy_bare_out_action) else 0
            else:
                choice = int(selected)
                if bool(policy_bare_out_action):
                    run_score = float(probs[choice]) - float(lambda_value) * float(selection_route_costs[choice])
                    if run_score <= 0.0:
                        choice = -1
        if int(choice) < 0:
            decisions.append({"choice": -1, "called_routes": [], "reward": 0.0, "cost": 0.0})
        else:
            decisions.append(
                {
                    "choice": int(choice),
                    "called_routes": [int(choice)],
                    "reward": float(example["rewards"][choice]),
                    "cost": float(example["route_costs"][choice]),
                }
            )
    return decisions


def _simulate_chain_policy(
    *,
    examples: list[dict[str, Any]],
    pred_lookup: dict[str, list[float]],
    lambda_value: float,
    route_count: int,
    max_steps: int,
    forced_first_route: int | None,
    selection_route_costs: list[float],
    policy_bare_out_action: bool,
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for example in examples:
        attempted_mask = [False] * route_count
        called_routes: list[int] = []
        total_cost = 0.0
        latest_route_idx: int | None = None
        abstained = False
        if forced_first_route is not None:
            first = int(forced_first_route)
            attempted_mask[first] = True
            latest_route_idx = first
            called_routes.append(first)
            total_cost += float(example["route_costs"][first])
        for _ in range(max(1, int(max_steps))):
            if latest_route_idx is None:
                probs = _lookup_state_probs(pred_lookup, int(example["source_idx"]), None, attempted_mask)
                if probs is None:
                    next_route = 0
                else:
                    selected = _choose_run_from_probs(probs, selection_route_costs, lambda_value, attempted_mask)
                    if selected is None:
                        if bool(policy_bare_out_action):
                            abstained = True
                            break
                        next_route = 0
                    else:
                        next_route = int(selected)
                        if bool(policy_bare_out_action):
                            run_score = float(probs[next_route]) - float(lambda_value) * float(selection_route_costs[next_route])
                            if run_score <= 0.0:
                                abstained = True
                                break
                attempted_mask[next_route] = True
                latest_route_idx = next_route
                called_routes.append(next_route)
                total_cost += float(example["route_costs"][next_route])
                if len(called_routes) >= max(1, int(max_steps)):
                    break
                continue
            probs = _lookup_state_probs(pred_lookup, int(example["source_idx"]), latest_route_idx, attempted_mask)
            if probs is None:
                break
            stop_score = float(probs[int(latest_route_idx)])
            next_route = _choose_run_from_probs(probs, selection_route_costs, lambda_value, attempted_mask)
            if next_route is None:
                break
            run_score = float(probs[next_route]) - float(lambda_value) * float(selection_route_costs[next_route])
            if stop_score >= run_score:
                break
            attempted_mask[next_route] = True
            latest_route_idx = int(next_route)
            called_routes.append(int(next_route))
            total_cost += float(example["route_costs"][next_route])
            if len(called_routes) >= max(1, int(max_steps)):
                break
        if abstained:
            decisions.append({"choice": -1, "called_routes": [], "reward": 0.0, "cost": 0.0})
            continue
        if latest_route_idx is None:
            latest_route_idx = 0
            called_routes = [0]
            total_cost = float(example["route_costs"][0])
        decisions.append(
            {
                "choice": int(latest_route_idx),
                "called_routes": [int(value) for value in called_routes],
                "reward": float(example["rewards"][int(latest_route_idx)]),
                "cost": float(total_cost),
            }
        )
    return decisions


def _summarize_decisions(
    *,
    policy: str,
    policy_type: str,
    lambda_value: float,
    examples: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    route_labels: list[str],
    selection_cost_mode: str,
) -> dict[str, Any]:
    route_count = len(route_labels)
    n = len(decisions)
    choice_counts = [0] * route_count
    call_counts = [0] * route_count
    first_call_counts = [0] * route_count
    out_count = 0
    oracle_matches = 0
    total_reward = 0.0
    total_cost = 0.0
    total_calls = 0.0
    for example, decision in zip(examples, decisions):
        choice = int(decision["choice"])
        if 0 <= choice < route_count:
            choice_counts[choice] += 1
        else:
            out_count += 1
        total_reward += float(decision["reward"])
        total_cost += float(decision["cost"])
        called = [int(value) for value in decision["called_routes"]]
        total_calls += len(called)
        if called and 0 <= int(called[0]) < route_count:
            first_call_counts[int(called[0])] += 1
        for route_idx in called:
            if 0 <= int(route_idx) < route_count:
                call_counts[int(route_idx)] += 1
        if 0 <= choice < route_count and choice == int(example["oracle_choice_idx"]):
            oracle_matches += 1
    mean_reward = math.nan if n == 0 else total_reward / n
    mean_cost = math.nan if n == 0 else total_cost / n
    choice_counts_by_route = {route_labels[idx]: int(choice_counts[idx]) for idx in range(route_count)}
    choice_counts_by_route["OUT"] = int(out_count)
    return {
        "policy": policy,
        "policy_type": policy_type,
        "selection_cost_mode": str(selection_cost_mode),
        "lambda": float(lambda_value),
        "mean_reward": mean_reward,
        "mean_cost": mean_cost,
        "mean_utility": mean_reward - float(lambda_value) * mean_cost,
        "oracle_match_rate": math.nan if n == 0 else oracle_matches / n,
        "mean_called_routes": math.nan if n == 0 else total_calls / n,
        "out_rate": math.nan if n == 0 else out_count / n,
        "choice_counts_by_route": choice_counts_by_route,
        "call_counts_by_route": {route_labels[idx]: int(call_counts[idx]) for idx in range(route_count)},
        "first_call_counts_by_route": {route_labels[idx]: int(first_call_counts[idx]) for idx in range(route_count)},
    }


def _simulate_policies(
    *,
    pred_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    route_labels: list[str],
    lambdas: list[float],
    scout_route_idx: int,
    max_policy_steps: int,
    selection_route_costs: list[float],
    selection_cost_mode: str,
    policy_bare_out_action: bool,
) -> list[dict[str, Any]]:
    route_count = len(route_labels)
    pred_lookup = _prediction_lookup(pred_rows)
    rows: list[dict[str, Any]] = []
    for lambda_value in lambdas:
        if bool(policy_bare_out_action):
            decisions = [{"choice": -1, "called_routes": [], "reward": 0.0, "cost": 0.0} for _ in examples]
            rows.append(
                _summarize_decisions(
                    policy="always::OUT",
                    policy_type="always_out",
                    lambda_value=float(lambda_value),
                    examples=examples,
                    decisions=decisions,
                    route_labels=route_labels,
                    selection_cost_mode="none",
                )
            )
        for route_idx, route_label in enumerate(route_labels):
            decisions = [
                {
                    "choice": int(route_idx),
                    "called_routes": [int(route_idx)],
                    "reward": float(example["rewards"][route_idx]),
                    "cost": float(example["route_costs"][route_idx]),
                }
                for example in examples
            ]
            rows.append(
                _summarize_decisions(
                    policy=f"always::{route_label}",
                    policy_type="always_direct",
                    lambda_value=float(lambda_value),
                    examples=examples,
                    decisions=decisions,
                    route_labels=route_labels,
                    selection_cost_mode="none",
                )
            )
        oracle_decisions = []
        for example in examples:
            scores = [
                float(example["rewards"][idx]) - float(lambda_value) * float(example["route_costs"][idx])
                for idx in range(route_count)
            ]
            if bool(policy_bare_out_action) and max(scores) <= 0.0:
                oracle_decisions.append({"choice": -1, "called_routes": [], "reward": 0.0, "cost": 0.0})
            else:
                choice = _argmax(scores)
                oracle_decisions.append(
                    {
                        "choice": int(choice),
                        "called_routes": [int(choice)],
                        "reward": float(example["rewards"][choice]),
                        "cost": float(example["route_costs"][choice]),
                    }
                )
        rows.append(
            _summarize_decisions(
                policy="oracle_direct",
                policy_type="oracle_direct",
                lambda_value=float(lambda_value),
                examples=examples,
                decisions=oracle_decisions,
                route_labels=route_labels,
                selection_cost_mode="actual_realized",
            )
        )
        oracle_fixed_decisions = []
        for example in examples:
            fixed_scores = [
                float(example["rewards"][idx]) - float(lambda_value) * float(selection_route_costs[idx])
                for idx in range(route_count)
            ]
            if bool(policy_bare_out_action) and max(fixed_scores) <= 0.0:
                oracle_fixed_decisions.append({"choice": -1, "called_routes": [], "reward": 0.0, "cost": 0.0})
            else:
                fixed_choice = _argmax(fixed_scores)
                oracle_fixed_decisions.append(
                    {
                        "choice": int(fixed_choice),
                        "called_routes": [int(fixed_choice)],
                        "reward": float(example["rewards"][fixed_choice]),
                        "cost": float(example["route_costs"][fixed_choice]),
                    }
                )
        rows.append(
            _summarize_decisions(
                policy="oracle_direct_fixed_cost_selection",
                policy_type="oracle_direct_fixed_cost_selection",
                lambda_value=float(lambda_value),
                examples=examples,
                decisions=oracle_fixed_decisions,
                route_labels=route_labels,
                selection_cost_mode=str(selection_cost_mode),
            )
        )
        rows.append(
            _summarize_decisions(
                policy="state_policy::input_only_direct_with_out" if bool(policy_bare_out_action) else "state_policy::input_only_direct",
                policy_type="input_only_direct",
                lambda_value=float(lambda_value),
                examples=examples,
                decisions=_simulate_direct_policy(
                    examples=examples,
                    pred_lookup=pred_lookup,
                    lambda_value=float(lambda_value),
                    route_count=route_count,
                    selection_route_costs=selection_route_costs,
                    policy_bare_out_action=bool(policy_bare_out_action),
                ),
                route_labels=route_labels,
                selection_cost_mode=str(selection_cost_mode),
            )
        )
        rows.append(
            _summarize_decisions(
                policy=(
                    f"state_policy::flexible_bare_max{int(max_policy_steps)}_with_out"
                    if bool(policy_bare_out_action)
                    else f"state_policy::flexible_bare_max{int(max_policy_steps)}"
                ),
                policy_type="flexible_bare_chain",
                lambda_value=float(lambda_value),
                examples=examples,
                decisions=_simulate_chain_policy(
                    examples=examples,
                    pred_lookup=pred_lookup,
                    lambda_value=float(lambda_value),
                    route_count=route_count,
                    max_steps=int(max_policy_steps),
                    forced_first_route=None,
                    selection_route_costs=selection_route_costs,
                    policy_bare_out_action=bool(policy_bare_out_action),
                ),
                route_labels=route_labels,
                selection_cost_mode=str(selection_cost_mode),
            )
        )
        rows.append(
            _summarize_decisions(
                policy=f"state_policy::forced_scout_{int(scout_route_idx)}_max{int(max_policy_steps)}",
                policy_type="forced_scout_chain",
                lambda_value=float(lambda_value),
                examples=examples,
                decisions=_simulate_chain_policy(
                    examples=examples,
                    pred_lookup=pred_lookup,
                    lambda_value=float(lambda_value),
                    route_count=route_count,
                    max_steps=int(max_policy_steps),
                    forced_first_route=int(scout_route_idx),
                    selection_route_costs=selection_route_costs,
                    policy_bare_out_action=False,
                ),
                route_labels=route_labels,
                selection_cost_mode=str(selection_cost_mode),
            )
        )
    return rows


def _state_metrics(
    pred_rows: list[dict[str, Any]],
    route_labels: list[str],
    state_filter: str,
) -> dict[str, Any]:
    filtered = [row for row in pred_rows if row["state_kind"] == state_filter]
    if not filtered:
        return {"state_filter": state_filter, "n_states": 0}
    y_true = np.asarray([row["true_rewards"] for row in filtered], dtype=np.float64)
    y_pred = np.asarray([row["pred_success_probs"] for row in filtered], dtype=np.float64)
    return {
        "state_filter": state_filter,
        "n_states": int(len(filtered)),
        "mean_true": float(np.mean(y_true)) if y_true.size else math.nan,
        "mean_pred": float(np.mean(y_pred)) if y_pred.size else math.nan,
        "std_true": float(np.std(y_true)) if y_true.size else math.nan,
        "std_pred": float(np.std(y_pred)) if y_pred.size else math.nan,
        "pearson_flat": _safe_corr(y_pred.reshape(-1), y_true.reshape(-1)),
        "route_metrics": compute_per_route_metrics(y_true, y_pred, route_labels),
        "pairwise_metrics": compute_pairwise_metrics(y_true, y_pred, route_labels),
    }


def _write_reports(
    *,
    output_dir: Path,
    train_pred_rows: list[dict[str, Any]],
    eval_pred_rows: list[dict[str, Any]],
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    route_labels: list[str],
    lambdas: list[float],
    scout_route_idx: int,
    max_policy_steps: int,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    train_loss: float,
    eval_loss: float,
    selection_route_costs: list[float],
    selection_cost_mode: str,
    policy_bare_out_action: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "train_state_predictions.jsonl", train_pred_rows)
    _write_jsonl(output_dir / "eval_state_predictions.jsonl", eval_pred_rows)
    eval_bare_rows = [row for row in eval_pred_rows if row["state_kind"] == "bare"]
    if eval_bare_rows:
        bare_true = np.asarray([row["true_rewards"] for row in eval_bare_rows], dtype=np.float64)
        bare_pred = np.asarray([row["pred_success_probs"] for row in eval_bare_rows], dtype=np.float64)
        _write_csv(output_dir / "bare_route_metrics.csv", compute_per_route_metrics(bare_true, bare_pred, route_labels), csv_headers_for_route_metrics())
        _write_csv(output_dir / "bare_pairwise_metrics.csv", compute_pairwise_metrics(bare_true, bare_pred, route_labels), csv_headers_for_pairwise_metrics())
    utility_rows = _simulate_policies(
        pred_rows=eval_pred_rows,
        examples=eval_examples,
        route_labels=route_labels,
        lambdas=lambdas,
        scout_route_idx=int(scout_route_idx),
        max_policy_steps=int(max_policy_steps),
        selection_route_costs=selection_route_costs,
        selection_cost_mode=str(selection_cost_mode),
        policy_bare_out_action=bool(policy_bare_out_action),
    )
    utility_headers = [
        "policy",
        "policy_type",
        "selection_cost_mode",
        "lambda",
        "mean_reward",
        "mean_cost",
        "mean_utility",
        "oracle_match_rate",
        "mean_called_routes",
        "out_rate",
        "choice_counts_by_route",
        "call_counts_by_route",
        "first_call_counts_by_route",
    ]
    _write_csv(output_dir / "state_policy_utility.csv", utility_rows, utility_headers)
    state_metrics = {
        "all": _state_metrics(eval_pred_rows, route_labels, "all"),
        "bare": _state_metrics(eval_pred_rows, route_labels, "bare"),
        "after_attempt": _state_metrics(eval_pred_rows, route_labels, "after_attempt"),
    }
    # The "all" entry above is filled manually because _state_metrics filters by exact state_kind.
    y_true_all = np.asarray([row["true_rewards"] for row in eval_pred_rows], dtype=np.float64)
    y_pred_all = np.asarray([row["pred_success_probs"] for row in eval_pred_rows], dtype=np.float64)
    state_metrics["all"] = {
        "state_filter": "all",
        "n_states": int(len(eval_pred_rows)),
        "mean_true": float(np.mean(y_true_all)) if y_true_all.size else math.nan,
        "mean_pred": float(np.mean(y_pred_all)) if y_pred_all.size else math.nan,
        "std_true": float(np.std(y_true_all)) if y_true_all.size else math.nan,
        "std_pred": float(np.std(y_pred_all)) if y_pred_all.size else math.nan,
        "pearson_flat": _safe_corr(y_pred_all.reshape(-1), y_true_all.reshape(-1)),
        "route_metrics": compute_per_route_metrics(y_true_all, y_pred_all, route_labels),
        "pairwise_metrics": compute_pairwise_metrics(y_true_all, y_pred_all, route_labels),
    }
    summary = {
        "history": history,
        "train_loss_final": float(train_loss),
        "eval_loss_final": float(eval_loss),
        "state_metrics": state_metrics,
        "policy_utility": {
            "route_labels": route_labels,
            "lambdas": lambdas,
            "scout_route_idx": int(scout_route_idx),
            "max_policy_steps": int(max_policy_steps),
            "policy_bare_out_action": bool(policy_bare_out_action),
            "selection_cost_mode": str(selection_cost_mode),
            "selection_route_costs": [float(value) for value in selection_route_costs],
            "n_train_examples": len(train_examples),
            "n_eval_examples": len(eval_examples),
            "rows": utility_rows,
        },
        "config": config,
    }
    write_json(output_dir / "state_policy_utility.json", summary["policy_utility"])
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--target-route-idxs", default=None)
    parser.add_argument("--max-seq-length", type=int, default=24000)
    parser.add_argument("--attempted-state-mode", choices=["none", "single", "all_subsets"], default="single")
    parser.add_argument("--include-bare-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-costs-in-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--route-output-cost-weights", default=",".join(str(value) for value in DEFAULT_AWS_OUTPUT_COST_WEIGHTS))
    parser.add_argument("--utility-lambdas", default=",".join(str(value) for value in DEFAULT_UTILITY_LAMBDAS))
    parser.add_argument("--scout-route-idx", type=int, default=0)
    parser.add_argument("--max-policy-steps", type=int, default=2)
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
    parser.add_argument("--delta-aux-weight", type=float, default=0.0)
    parser.add_argument("--delta-aux-huber-delta", type=float, default=0.0)
    parser.add_argument("--decision-aux-weight", type=float, default=0.0)
    parser.add_argument("--decision-aux-lambdas", default=None)
    parser.add_argument("--decision-aux-lambda-sampling", choices=["none", "uniform", "log_uniform"], default="none")
    parser.add_argument("--decision-aux-lambda-sample-count", type=int, default=1)
    parser.add_argument("--decision-aux-lambda-min", type=float, default=0.0)
    parser.add_argument("--decision-aux-lambda-max", type=float, default=0.0)
    parser.add_argument("--decision-aux-temperature", type=float, default=0.1)
    parser.add_argument("--decision-aux-cost-mode", choices=["actual", "fixed_train_mean"], default="fixed_train_mean")
    parser.add_argument("--decision-aux-stop-tie-bonus", type=float, default=1.0e-4)
    parser.add_argument("--decision-aux-bare-out-action", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decision-aux-regret-weight-mode", choices=["none", "oracle_margin", "default_action"], default="none")
    parser.add_argument("--decision-aux-regret-weight-scale", type=float, default=0.0)
    parser.add_argument("--decision-aux-regret-weight-power", type=float, default=1.0)
    parser.add_argument("--decision-aux-regret-weight-min", type=float, default=1.0)
    parser.add_argument("--decision-aux-regret-weight-max", type=float, default=8.0)
    parser.add_argument("--policy-bare-out-action", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-weighting", choices=["uniform", "regret_default", "oracle_margin"], default="uniform")
    parser.add_argument("--regret-lambdas", default=None)
    parser.add_argument("--regret-default-route-idx", type=int, default=0)
    parser.add_argument("--regret-weight-scale", type=float, default=4.0)
    parser.add_argument("--regret-weight-power", type=float, default=1.0)
    parser.add_argument("--regret-weight-min", type=float, default=1.0)
    parser.add_argument("--regret-weight-max", type=float, default=8.0)
    parser.add_argument("--normalize-sample-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    parser.add_argument("--checkpoint-every-epoch", action="store_true")
    parser.add_argument("--epoch-report-every", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)
    lambdas = _parse_float_list(args.utility_lambdas)
    regret_lambdas = _parse_float_list(args.regret_lambdas) if args.regret_lambdas else list(lambdas)
    decision_aux_lambdas = _parse_float_list(args.decision_aux_lambdas) if args.decision_aux_lambdas else list(lambdas)
    route_cost_weights = _parse_float_list(args.route_output_cost_weights)

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

    all_route_labels = _load_route_labels(dataset_dir)
    target_route_idxs = _parse_route_indices(args.target_route_idxs, len(all_route_labels))
    route_labels = [all_route_labels[int(idx)] for idx in target_route_idxs]
    target_dim = len(route_labels)
    if len(route_cost_weights) != target_dim:
        raise ValueError(
            f"--route-output-cost-weights has {len(route_cost_weights)} values but selected route count is {target_dim}"
        )
    if not 0 <= int(args.scout_route_idx) < target_dim:
        raise ValueError(f"--scout-route-idx={args.scout_route_idx} is out of range for {target_dim} selected routes")
    if bool(args.use_lora) and bool(args.encoder_frozen):
        raise ValueError("--use-lora requires --no-encoder-frozen")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    train_rows = _shuffle_rows(list(_load_split(dataset_dir, "train")), args.max_train_rows, args.seed)
    eval_rows_source = _shuffle_rows(list(_load_split(dataset_dir, "eval")), args.max_eval_rows, args.seed + 1)
    fixed_train_costs = _fixed_train_costs(train_rows, target_route_idxs, route_cost_weights)
    prompt_costs = fixed_train_costs if bool(args.include_costs_in_prompt) else None
    train_examples = _selected_examples(train_rows, target_route_idxs, route_cost_weights)
    eval_examples = _selected_examples(eval_rows_source, target_route_idxs, route_cost_weights)
    train_dataset = StatePolicyDataset(
        train_rows,
        tokenizer,
        route_labels,
        target_route_idxs,
        int(args.max_seq_length),
        attempted_state_mode=str(args.attempted_state_mode),
        include_bare_state=bool(args.include_bare_state),
        route_costs=prompt_costs,
        route_output_cost_weights=route_cost_weights,
        sample_weighting=str(args.sample_weighting),
        regret_lambdas=regret_lambdas,
        regret_route_costs=fixed_train_costs,
        regret_default_route_idx=int(args.regret_default_route_idx),
        regret_weight_scale=float(args.regret_weight_scale),
        regret_weight_power=float(args.regret_weight_power),
        regret_weight_min=float(args.regret_weight_min),
        regret_weight_max=float(args.regret_weight_max),
        normalize_sample_weights=bool(args.normalize_sample_weights),
    )
    eval_dataset = StatePolicyDataset(
        eval_rows_source,
        tokenizer,
        route_labels,
        target_route_idxs,
        int(args.max_seq_length),
        attempted_state_mode=str(args.attempted_state_mode),
        include_bare_state=bool(args.include_bare_state),
        route_costs=prompt_costs,
        route_output_cost_weights=route_cost_weights,
        sample_weighting=str(args.sample_weighting),
        regret_lambdas=regret_lambdas,
        regret_route_costs=fixed_train_costs,
        regret_default_route_idx=int(args.regret_default_route_idx),
        regret_weight_scale=float(args.regret_weight_scale),
        regret_weight_power=float(args.regret_weight_power),
        regret_weight_min=float(args.regret_weight_min),
        regret_weight_max=float(args.regret_weight_max),
        normalize_sample_weights=bool(args.normalize_sample_weights),
    )
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
    train_report_loader = DataLoader(
        train_dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        collate_fn=collate_fn,
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
    model, optimizer, train_loader, eval_loader, train_report_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, train_report_loader, scheduler
    )
    fixed_train_cost_tensor = torch.tensor(fixed_train_costs, dtype=torch.float32, device=accelerator.device).view(1, -1)

    config = {
        "model_name": args.model_name,
        "dataset_dir": str(dataset_dir),
        "all_route_labels": all_route_labels,
        "target_route_idxs": [int(idx) for idx in target_route_idxs],
        "route_labels": route_labels,
        "attempted_state_mode": str(args.attempted_state_mode),
        "include_bare_state": bool(args.include_bare_state),
        "include_costs_in_prompt": bool(args.include_costs_in_prompt),
        "route_output_cost_weights": route_cost_weights,
        "fixed_train_route_costs": fixed_train_costs,
        "policy_selection_cost_mode": "fixed_train_mean",
        "policy_selection_route_costs": fixed_train_costs,
        "utility_lambdas": lambdas,
        "regret_lambdas": regret_lambdas,
        "decision_aux_lambdas": decision_aux_lambdas,
        "decision_aux_weight": float(args.decision_aux_weight),
        "decision_aux_lambda_sampling": str(args.decision_aux_lambda_sampling),
        "decision_aux_lambda_sample_count": int(args.decision_aux_lambda_sample_count),
        "decision_aux_lambda_min": float(args.decision_aux_lambda_min),
        "decision_aux_lambda_max": float(args.decision_aux_lambda_max),
        "decision_aux_temperature": float(args.decision_aux_temperature),
        "decision_aux_cost_mode": str(args.decision_aux_cost_mode),
        "decision_aux_stop_tie_bonus": float(args.decision_aux_stop_tie_bonus),
        "decision_aux_bare_out_action": bool(args.decision_aux_bare_out_action),
        "decision_aux_regret_weight_mode": str(args.decision_aux_regret_weight_mode),
        "decision_aux_regret_weight_scale": float(args.decision_aux_regret_weight_scale),
        "decision_aux_regret_weight_power": float(args.decision_aux_regret_weight_power),
        "decision_aux_regret_weight_min": float(args.decision_aux_regret_weight_min),
        "decision_aux_regret_weight_max": float(args.decision_aux_regret_weight_max),
        "policy_bare_out_action": bool(args.policy_bare_out_action),
        "sample_weighting": str(args.sample_weighting),
        "regret_default_route_idx": int(args.regret_default_route_idx),
        "regret_weight_scale": float(args.regret_weight_scale),
        "regret_weight_power": float(args.regret_weight_power),
        "regret_weight_min": float(args.regret_weight_min),
        "regret_weight_max": float(args.regret_weight_max),
        "normalize_sample_weights": bool(args.normalize_sample_weights),
        "train_sample_weight_stats": _sample_weight_stats(train_dataset.rows),
        "eval_sample_weight_stats": _sample_weight_stats(eval_dataset.rows),
        "scout_route_idx": int(args.scout_route_idx),
        "max_policy_steps": int(args.max_policy_steps),
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
        "prepared_train_state_rows": len(train_dataset),
        "prepared_eval_state_rows": len(eval_dataset),
        "selected_train_examples": len(train_examples),
        "selected_eval_examples": len(eval_examples),
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
        "delta_aux_weight": float(args.delta_aux_weight),
        "delta_aux_huber_delta": float(args.delta_aux_huber_delta),
        "ddp_find_unused_parameters": bool(args.ddp_find_unused_parameters),
        "checkpoint_every_epoch": bool(args.checkpoint_every_epoch),
        "epoch_report_every": int(args.epoch_report_every),
        "resume_from_checkpoint": str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
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
        for batch in tqdm(train_loader, desc=f"Train Qwen state policy epoch {epoch}", disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                logits, _, _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                probs = torch.sigmoid(logits.float())
                sample_weights = batch["sample_weights"].float()
                per_value_loss = F.binary_cross_entropy_with_logits(
                    logits.float(),
                    batch["targets"].float(),
                    reduction="none",
                )
                loss = (per_value_loss * sample_weights.view(-1, 1)).sum() / (
                    sample_weights.sum().clamp_min(1.0) * per_value_loss.shape[1]
                )
                if float(args.delta_aux_weight) != 0.0:
                    deltas_pred = probs.unsqueeze(2) - probs.unsqueeze(1)
                    deltas_true = batch["targets"].float().unsqueeze(2) - batch["targets"].float().unsqueeze(1)
                    if float(args.delta_aux_huber_delta) > 0.0:
                        delta_loss_per = F.huber_loss(
                            deltas_pred,
                            deltas_true,
                            delta=float(args.delta_aux_huber_delta),
                            reduction="none",
                        ).mean(dim=(1, 2))
                    else:
                        delta_loss_per = ((deltas_pred - deltas_true) ** 2).mean(dim=(1, 2))
                    delta_loss = (delta_loss_per * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
                    loss = loss + float(args.delta_aux_weight) * delta_loss
                if float(args.decision_aux_weight) != 0.0:
                    if str(args.decision_aux_cost_mode) == "fixed_train_mean":
                        decision_route_costs = fixed_train_cost_tensor.expand(batch["targets"].shape[0], -1)
                    else:
                        decision_route_costs = batch["route_costs"].float()
                    decision_loss = _utility_decision_aux_loss(
                        probs=probs,
                        targets=batch["targets"].float(),
                        attempted_masks=batch["attempted_masks"],
                        latest_route_idxs=batch["latest_route_idxs"],
                        route_costs=decision_route_costs,
                        lambdas=decision_aux_lambdas,
                        lambda_sampling=str(args.decision_aux_lambda_sampling),
                        lambda_sample_count=int(args.decision_aux_lambda_sample_count),
                        lambda_min=float(args.decision_aux_lambda_min),
                        lambda_max=float(args.decision_aux_lambda_max),
                        temperature=float(args.decision_aux_temperature),
                        sample_weights=sample_weights,
                        stop_tie_bonus=float(args.decision_aux_stop_tie_bonus),
                        bare_out_action=bool(args.decision_aux_bare_out_action),
                        regret_weight_mode=str(args.decision_aux_regret_weight_mode),
                        regret_default_route_idx=int(args.regret_default_route_idx),
                        regret_weight_scale=float(args.decision_aux_regret_weight_scale),
                        regret_weight_power=float(args.decision_aux_regret_weight_power),
                        regret_weight_min=float(args.decision_aux_regret_weight_min),
                        regret_weight_max=float(args.decision_aux_regret_weight_max),
                    )
                    loss = loss + float(args.decision_aux_weight) * decision_loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                running_losses.append(float(loss.detach().item()))
        accelerator.wait_for_everyone()
        eval_loss, eval_pred_rows, _, _ = _evaluate(
            accelerator, model, eval_loader, eval_dataset, route_labels, desc="Eval Qwen state policy"
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
        if int(args.epoch_report_every) > 0 and (int(epoch) + 1) % int(args.epoch_report_every) == 0:
            train_report_loss, train_pred_rows, _, _ = _evaluate(
                accelerator,
                model,
                train_report_loader,
                train_dataset,
                route_labels,
                desc=f"Predict train Qwen state policy epoch {epoch}",
            )
            _, eval_pred_rows, _, _ = _evaluate(
                accelerator,
                model,
                eval_loader,
                eval_dataset,
                route_labels,
                desc=f"Predict eval Qwen state policy epoch {epoch}",
            )
            if accelerator.is_main_process:
                _write_reports(
                    output_dir=output_dir / "epoch_reports" / f"epoch_{epoch:04d}",
                    train_pred_rows=train_pred_rows,
                    eval_pred_rows=eval_pred_rows,
                    train_examples=train_examples,
                    eval_examples=eval_examples,
                    route_labels=route_labels,
                    lambdas=lambdas,
                    scout_route_idx=int(args.scout_route_idx),
                    max_policy_steps=int(args.max_policy_steps),
                    history=history,
                    config=config,
                    train_loss=float(train_report_loss),
                    eval_loss=float(eval_loss),
                    selection_route_costs=fixed_train_costs,
                    selection_cost_mode="fixed_train_mean",
                    policy_bare_out_action=bool(args.policy_bare_out_action),
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

    train_loss, train_pred_rows, _, _ = _evaluate(
        accelerator, model, train_report_loader, train_dataset, route_labels, desc="Predict train Qwen state policy"
    )
    eval_loss, eval_pred_rows, _, _ = _evaluate(
        accelerator, model, eval_loader, eval_dataset, route_labels, desc="Predict eval Qwen state policy"
    )
    if not accelerator.is_main_process:
        return

    _write_reports(
        output_dir=output_dir,
        train_pred_rows=train_pred_rows,
        eval_pred_rows=eval_pred_rows,
        train_examples=train_examples,
        eval_examples=eval_examples,
        route_labels=route_labels,
        lambdas=lambdas,
        scout_route_idx=int(args.scout_route_idx),
        max_policy_steps=int(args.max_policy_steps),
        history=history,
        config=config,
        train_loss=float(train_loss),
        eval_loss=float(eval_loss),
        selection_route_costs=fixed_train_costs,
        selection_cost_mode="fixed_train_mean",
        policy_bare_out_action=bool(args.policy_bare_out_action),
    )
    if args.save_model:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.reward_head.state_dict(), output_dir / "state_policy_head.pt")
        if hasattr(unwrapped, "encoder") and hasattr(unwrapped.encoder, "save_pretrained"):
            unwrapped.encoder.save_pretrained(output_dir / "encoder")
        tokenizer.save_pretrained(output_dir / "tokenizer")


if __name__ == "__main__":
    main()

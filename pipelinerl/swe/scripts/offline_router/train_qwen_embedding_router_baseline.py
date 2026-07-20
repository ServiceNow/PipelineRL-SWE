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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
    _get_primary_output_text,
    _load_route_labels,
    _load_split,
    _shuffle_rows,
)


def _load_model_only_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    """Load model weights from a checkpoint without optimizer state."""
    model_path = checkpoint_path
    if model_path.is_dir():
        model_path = model_path / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model-only checkpoint: {model_path}")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("safetensors is required for --init-from-model-checkpoint") from exc
    state_dict = load_file(str(model_path), device="cpu")
    incompatible = model.load_state_dict(state_dict, strict=False)
    return {
        "checkpoint": str(model_path),
        "loaded_tensors": len(state_dict),
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(incompatible.unexpected_keys),
        "missing_key_sample": list(incompatible.missing_keys)[:20],
        "unexpected_key_sample": list(incompatible.unexpected_keys)[:20],
    }


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
    has_segments = "segment_input_ids" in batch[0]
    if has_segments:
        segment_count = len(batch[0]["segment_input_ids"])
        segment_input_ids: list[torch.Tensor] = []
        segment_attention_mask: list[torch.Tensor] = []
        for segment_idx in range(segment_count):
            max_len = max(len(row["segment_input_ids"][segment_idx]) for row in batch)
            input_ids_tensor = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
            attention_mask_tensor = torch.zeros((len(batch), max_len), dtype=torch.long)
            for row_idx, row in enumerate(batch):
                seq_len = len(row["segment_input_ids"][segment_idx])
                start = max_len - seq_len
                input_ids_tensor[row_idx, start:] = torch.tensor(
                    row["segment_input_ids"][segment_idx], dtype=torch.long
                )
                attention_mask_tensor[row_idx, start:] = torch.tensor(
                    row["segment_attention_mask"][segment_idx], dtype=torch.long
                )
            segment_input_ids.append(input_ids_tensor)
            segment_attention_mask.append(attention_mask_tensor)
    else:
        max_len = max(len(row["input_ids"]) for row in batch)
        input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for idx, row in enumerate(batch):
            seq_len = len(row["input_ids"])
            start = max_len - seq_len
            input_ids[idx, start:] = torch.tensor(row["input_ids"], dtype=torch.long)
            attention_mask[idx, start:] = torch.tensor(row["attention_mask"], dtype=torch.long)
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
    collated = {
        "problem_ids": [row["problem_id"] for row in batch],
        "datasets": [row["dataset"] for row in batch],
        "repos": [row["repo"] for row in batch],
        "languages": [row["language"] for row in batch],
        "targets": targets,
        "output_token_targets_log": output_token_targets_log,
        "zero_reward_targets": zero_reward_targets,
        "class_targets": class_targets,
        "row_indices": row_indices,
    }
    if has_segments:
        collated["segment_input_ids"] = segment_input_ids
        collated["segment_attention_mask"] = segment_attention_mask
    else:
        collated["input_ids"] = input_ids
        collated["attention_mask"] = attention_mask
    return collated

def _build_embedding_input_text(
    row: dict[str, Any],
    route_labels: list[str],
    input_mode: str,
    primary_output_token_count: int | None,
    input_task: str,
) -> str | None:
    if input_task == "reward":
        if primary_output_token_count is None:
            return _build_input_text(row, route_labels, input_mode=input_mode)
        return _build_input_text(
            row,
            route_labels,
            input_mode=input_mode,
            primary_output_token_count=primary_output_token_count,
        )
    if input_task not in {"cost", "policy"}:
        raise ValueError(f"Unsupported input_task={input_task!r}")
    prompt_text = row.get("prompt_text")
    if not isinstance(prompt_text, str):
        return None
    route_legend = "\n".join(f"{idx}: {label}" for idx, label in enumerate(route_labels))
    if input_task == "policy":
        if input_mode == "input_only":
            return (
                "Select the cheapest model route that is likely to produce a patch passing the SWE tests.\n"
                "If no route is likely to pass, select the cheapest route. The route order is cost-ascending.\n"
                "Only the original repair prompt is shown. Infer the cheapest sufficient route from the task context.\n\n"
                "[Route Order]\n"
                f"{route_legend}\n\n"
                "[Original Repair Prompt]\n"
                f"{prompt_text}"
            )
        if input_mode != "post_primary":
            raise ValueError(f"Unsupported input_mode={input_mode!r}")
        primary_output_text = _get_primary_output_text(row)
        if not isinstance(primary_output_text, str):
            return None
        primary_token_count_block = ""
        if primary_output_token_count is not None:
            primary_token_count_block = (
                "\n\n"
                "[Scout Attempt Output Tokens]\n"
                f"{int(primary_output_token_count)}"
            )
        return (
            "Select the cheapest model route that is likely to produce a patch passing the SWE tests.\n"
            "If no route is likely to pass, select the cheapest route. The route order is cost-ascending.\n"
            "The scout attempt has already been run and is shown as diagnostic context.\n\n"
            "[Route Order]\n"
            f"{route_legend}\n\n"
            "[Original Repair Prompt]\n"
            f"{prompt_text}"
            f"{primary_token_count_block}\n\n"
            "[Scout Attempt]\n"
            f"{primary_output_text}"
        )
    if input_mode == "input_only":
        return (
            "Predict the output-token cost for each requested model route.\n"
            "The target is the number of completion tokens emitted by each route's repair attempt.\n"
            "Only the original repair prompt is shown. Predict each route from the task context alone.\n\n"
            "[Route Order]\n"
            f"{route_legend}\n\n"
            "[Original Repair Prompt]\n"
            f"{prompt_text}"
        )
    if input_mode != "post_primary":
        raise ValueError(f"Unsupported input_mode={input_mode!r}")
    primary_output_text = _get_primary_output_text(row)
    if not isinstance(primary_output_text, str):
        return None
    primary_token_count_block = ""
    if primary_output_token_count is not None:
        primary_token_count_block = (
            "\n\n"
            "[Scout Attempt Output Tokens]\n"
            f"{int(primary_output_token_count)}"
        )
    return (
        "Predict the output-token cost for each requested model route.\n"
        "The target is the number of completion tokens emitted by each route's repair attempt.\n"
        "The scout attempt has already been run and is shown as context for the remaining route predictions.\n\n"
        "[Route Order]\n"
        f"{route_legend}\n\n"
        "[Original Repair Prompt]\n"
        f"{prompt_text}"
        f"{primary_token_count_block}\n\n"
        "[Scout Attempt]\n"
        f"{primary_output_text}"
    )



def _embedding_task_instruction(input_task: str) -> str:
    if input_task == "reward":
        return (
            "Predict the probability that each SWE repair model route produces a patch that passes the tests. "
            "Use the route identities and the available repair evidence to estimate per-route success."
        )
    if input_task == "cost":
        return (
            "Predict the output-token cost for each requested SWE repair model route. "
            "Use the route identities and the available repair evidence to estimate completion length."
        )
    if input_task == "policy":
        return (
            "Select the cheapest SWE repair model route that is likely to produce a passing patch. "
            "Use route identities and evidence to estimate the cheapest sufficient solver."
        )
    raise ValueError(f"Unsupported input_task={input_task!r}")


def _build_segmented_embedding_input_texts(
    row: dict[str, Any],
    route_labels: list[str],
    input_mode: str,
    primary_output_token_count: int | None,
    input_task: str,
    embedding_input_layout: str,
) -> list[str] | None:
    route_legend = "\n".join(f"{idx}: {label}" for idx, label in enumerate(route_labels))
    task = _embedding_task_instruction(str(input_task))
    task_segment = (
        f"Instruct: {task}\n"
        "Query: Represent the route set and prediction task for this SWE routing example.\n\n"
        "[Route Order]\n"
        f"{route_legend}"
    )

    if embedding_input_layout in {"late_fusion", "late_fusion_prompt_only", "late_fusion_scout_only"}:
        prompt_text = row.get("prompt_text")
        if not isinstance(prompt_text, str):
            return None
        prompt_segment = (
            f"Instruct: {task}\n"
            "Query: Represent the original SWE repair task. Focus on the repository, language, failing behavior, "
            "and the code/edit context that may determine which solver succeeds.\n\n"
            "[Original Repair Prompt]\n"
            f"{prompt_text}"
        )
        if embedding_input_layout in {"late_fusion", "late_fusion_prompt_only"}:
            segments = [task_segment, prompt_segment]
        else:
            segments = [task_segment]
    elif embedding_input_layout in {"semantic_late_fusion", "semantic_problem_only", "semantic_code_only"}:
        problem_statement = row.get("problem_statement")
        file_context = row.get("file_context")
        if not isinstance(problem_statement, str) or not problem_statement.strip():
            return None
        if not isinstance(file_context, str) or not file_context.strip():
            return None
        problem_segment = (
            f"Instruct: {task}\n"
            "Query: Represent the natural-language SWE issue report and reproduction details.\n\n"
            "[Problem Statement]\n"
            f"{problem_statement}"
        )
        code_segment = (
            f"Instruct: {task}\n"
            "Query: Represent the relevant repository code files independently from the issue text and solver attempt.\n\n"
            "[Code Context]\n"
            f"{file_context}"
        )
        if embedding_input_layout == "semantic_problem_only":
            segments = [task_segment, problem_segment]
        elif embedding_input_layout == "semantic_code_only":
            segments = [task_segment, code_segment]
        else:
            segments = [task_segment, problem_segment, code_segment]
    else:
        raise ValueError(f"Unsupported embedding_input_layout={embedding_input_layout!r}")

    if embedding_input_layout in {"late_fusion_prompt_only", "semantic_problem_only", "semantic_code_only"}:
        return segments
    if input_mode == "input_only":
        return segments
    if input_mode != "post_primary":
        raise ValueError(f"Unsupported input_mode={input_mode!r}")
    primary_output_text = _get_primary_output_text(row)
    if not isinstance(primary_output_text, str):
        return None
    primary_token_count_block = ""
    if primary_output_token_count is not None:
        primary_token_count_block = (
            "\n\n"
            "[Scout Attempt Output Tokens]\n"
            f"{int(primary_output_token_count)}"
        )
    scout_segment = (
        f"Instruct: {task}\n"
        "Query: Represent the scout model attempt as diagnostic evidence. Focus on whether the attempt looks valid, "
        "where it fails, and what it implies about difficulty or solver compatibility.\n"
        f"{primary_token_count_block}\n\n"
        "[Scout Attempt]\n"
        f"{primary_output_text}"
    )
    segments.append(scout_segment)
    return segments

def _compute_class_target(
    target_rewards: list[float],
    mode: str,
    success_threshold: float,
) -> int:
    if mode == "argmax":
        return _argmax_index(target_rewards)
    if mode == "cheapest_success":
        for idx, value in enumerate(target_rewards):
            if float(value) > float(success_threshold):
                return int(idx)
        return 0
    if mode == "cheapest_success_or_abstain":
        for idx, value in enumerate(target_rewards):
            if float(value) > float(success_threshold):
                return int(idx)
        return max(0, len(target_rewards) - 1)
    if mode == "joint_success_2route":
        if len(target_rewards) != 2:
            raise ValueError("class_target_mode=joint_success_2route requires exactly two routes")
        first_success = float(target_rewards[0]) > float(success_threshold)
        second_success = float(target_rewards[1]) > float(success_threshold)
        return int(first_success) + (2 * int(second_success))
    raise ValueError(f"Unsupported class_target_mode={mode!r}")


class RouterCostDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        route_labels: list[str],
        route_indices: list[int],
        max_seq_length: int,
        require_cost_targets: bool,
        cost_route_idxs: list[int],
        zero_reward_epsilon: float,
        input_mode: str,
        include_primary_output_token_count: bool,
        input_task: str = "reward",
        embedding_input_layout: str = "single",
        class_target_mode: str = "argmax",
        class_success_threshold: float = 0.5,
        append_abstain_class: bool = False,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        source_target_dim = len(route_labels)
        if len(route_indices) != source_target_dim:
            raise ValueError(
                f"route_indices length {len(route_indices)} does not match source_target_dim {source_target_dim}"
            )
        if source_target_dim == 0:
            raise ValueError("At least one target route is required")
        if embedding_input_layout not in {"single", "late_fusion", "late_fusion_prompt_only", "late_fusion_scout_only", "semantic_late_fusion", "semantic_problem_only", "semantic_code_only"}:
            raise ValueError(f"Unsupported embedding_input_layout={embedding_input_layout!r}")
        prompt_route_labels = list(route_labels)
        if bool(append_abstain_class):
            prompt_route_labels.append("ABSTAIN")
        max_source_route_idx = max(int(idx) for idx in route_indices)
        invalid_cost_route_idxs = [idx for idx in cost_route_idxs if not 0 <= int(idx) < source_target_dim]
        if require_cost_targets and invalid_cost_route_idxs:
            raise ValueError(
                f"cost_route_idxs={invalid_cost_route_idxs} are out of range for {source_target_dim} routes"
            )
        for row in rows:
            targets = row.get("performance_targets")
            output_tokens = row.get("route_output_tokens")
            if not isinstance(targets, list) or len(targets) <= max_source_route_idx:
                continue
            if require_cost_targets and (
                not isinstance(output_tokens, list) or len(output_tokens) <= max_source_route_idx
            ):
                continue
            try:
                target_rewards = [float(targets[int(source_idx)]) for source_idx in route_indices]
                zero_reward_targets = [
                    1.0 if float(value) <= float(zero_reward_epsilon) else 0.0 for value in target_rewards
                ]
                output_token_targets_log = (
                    [
                        float(math.log1p(float(output_tokens[int(route_indices[int(route_idx)])])))
                        for route_idx in cost_route_idxs
                    ]
                    if isinstance(output_tokens, list) and len(output_tokens) > max_source_route_idx
                    else [0.0 for _ in cost_route_idxs]
                )
                problem_id = str(row.get("problem_id") or row.get("instance_id") or row.get("id"))
            except (TypeError, ValueError):
                continue
            primary_output_token_count: int | None = None
            if include_primary_output_token_count and input_mode == "post_primary":
                if isinstance(output_tokens, list) and len(output_tokens) > 0:
                    primary_output_token_count = int(float(output_tokens[0]))
                else:
                    primary_output_text = _get_primary_output_text(row)
                    if isinstance(primary_output_text, str):
                        primary_output_token_count = len(
                            tokenizer(primary_output_text, add_special_tokens=False).get("input_ids") or []
                        )
            model_target_rewards = list(target_rewards)
            model_zero_reward_targets = list(zero_reward_targets)
            if bool(append_abstain_class):
                model_target_rewards.append(0.0)
                model_zero_reward_targets.append(1.0)

            encoded_payload: dict[str, Any] = {}
            if embedding_input_layout == "single":
                input_text = _build_embedding_input_text(
                    row,
                    prompt_route_labels,
                    input_mode=input_mode,
                    primary_output_token_count=primary_output_token_count,
                    input_task=str(input_task),
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
                encoded_payload["input_ids"] = [int(value) for value in input_ids]
                encoded_payload["attention_mask"] = [int(value) for value in attention_mask]
            else:
                segment_texts = _build_segmented_embedding_input_texts(
                    row,
                    prompt_route_labels,
                    input_mode=input_mode,
                    primary_output_token_count=primary_output_token_count,
                    input_task=str(input_task),
                    embedding_input_layout=str(embedding_input_layout),
                )
                if not segment_texts:
                    continue
                segment_input_ids: list[list[int]] = []
                segment_attention_mask: list[list[int]] = []
                for segment_text in segment_texts:
                    encoded = tokenizer(
                        segment_text,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=max_seq_length,
                    )
                    input_ids = encoded.get("input_ids")
                    attention_mask = encoded.get("attention_mask")
                    if not input_ids or not attention_mask:
                        break
                    segment_input_ids.append([int(value) for value in input_ids])
                    segment_attention_mask.append([int(value) for value in attention_mask])
                if len(segment_input_ids) != len(segment_texts):
                    continue
                encoded_payload["segment_input_ids"] = segment_input_ids
                encoded_payload["segment_attention_mask"] = segment_attention_mask

            dataset_row = {
                "row_idx": len(self.rows),
                "problem_id": problem_id,
                "dataset": row.get("dataset"),
                "repo": row.get("repo"),
                "language": row.get("language"),
                "targets": model_target_rewards,
                "output_token_targets_log": output_token_targets_log,
                "zero_reward_targets": model_zero_reward_targets,
                "class_target": _compute_class_target(
                    model_target_rewards,
                    mode=str(class_target_mode),
                    success_threshold=float(class_success_threshold),
                ),
            }
            dataset_row.update(encoded_payload)
            self.rows.append(dataset_row)

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
        embedding_input_layout: str,
        segment_count: int,
    ) -> None:
        super().__init__()
        if bool(use_lora) and bool(encoder_frozen):
            raise ValueError("use_lora=true requires encoder_frozen=false")
        if cost_gradient_mode not in {"joint", "detached", "separate_adapter"}:
            raise ValueError(f"Unsupported cost_gradient_mode={cost_gradient_mode}")
        if embedding_input_layout not in {"single", "late_fusion", "late_fusion_prompt_only", "late_fusion_scout_only", "semantic_late_fusion", "semantic_problem_only", "semantic_code_only"}:
            raise ValueError(f"Unsupported embedding_input_layout={embedding_input_layout!r}")
        self.encoder_frozen = bool(encoder_frozen)
        self.cost_gradient_mode = str(cost_gradient_mode)
        self.embedding_input_layout = str(embedding_input_layout)
        self.segment_count = max(1, int(segment_count))
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
        effective_hidden_size = hidden_size * self.segment_count
        self.predict_costs = bool(predict_costs)
        self.predict_zero_reward_failure = bool(predict_zero_reward_failure)
        self.reward_head = self._make_head(effective_hidden_size, target_dim, dropout, mlp_hidden_size)
        self.head = self.reward_head
        self.cost_head = (
            self._make_head(effective_hidden_size, cost_target_dim, dropout, mlp_hidden_size) if self.predict_costs else None
        )
        self.zero_reward_head = (
            self._make_head(effective_hidden_size, target_dim, dropout, mlp_hidden_size)
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
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = _last_token_pool(outputs.last_hidden_state, attention_mask)
        return F.normalize(pooled.float(), p=2, dim=1)

    def encode_segmented_inputs(
        self,
        segment_input_ids: list[torch.Tensor],
        segment_attention_mask: list[torch.Tensor],
        adapter_name: str | None = None,
    ) -> torch.Tensor:
        if len(segment_input_ids) != self.segment_count or len(segment_attention_mask) != self.segment_count:
            raise ValueError(
                f"Expected {self.segment_count} input segments, got "
                f"{len(segment_input_ids)} ids and {len(segment_attention_mask)} masks"
            )
        embeddings = []
        for input_ids, attention_mask in zip(segment_input_ids, segment_attention_mask, strict=True):
            embeddings.append(self.encode_inputs(input_ids, attention_mask, adapter_name=adapter_name))
        return torch.cat(embeddings, dim=1)

    def _encode_batch(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        segment_input_ids: list[torch.Tensor] | None,
        segment_attention_mask: list[torch.Tensor] | None,
        adapter_name: str | None,
    ) -> torch.Tensor:
        if segment_input_ids is not None or segment_attention_mask is not None:
            if segment_input_ids is None or segment_attention_mask is None:
                raise ValueError("Both segment_input_ids and segment_attention_mask are required for late-fusion inputs")
            return self.encode_segmented_inputs(
                segment_input_ids,
                segment_attention_mask,
                adapter_name=adapter_name,
            )
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask are required for single-sequence inputs")
        return self.encode_inputs(input_ids, attention_mask, adapter_name=adapter_name)

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
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        segment_input_ids: list[torch.Tensor] | None = None,
        segment_attention_mask: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.cost_head is None:
            raise ValueError("cost-only forward requires a cost head")
        adapter_name = (
            self.cost_adapter_name
            if self.cost_gradient_mode == "separate_adapter"
            else self.reward_adapter_name if hasattr(self.encoder, "set_adapter") else None
        )
        embeddings = self._encode_batch(
            input_ids,
            attention_mask,
            segment_input_ids,
            segment_attention_mask,
            adapter_name=adapter_name,
        )
        return self.predict_cost_only_from_embeddings(embeddings)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        segment_input_ids: list[torch.Tensor] | None = None,
        segment_attention_mask: list[torch.Tensor] | None = None,
        cost_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if cost_only:
            return self.forward_cost_only(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_input_ids=segment_input_ids,
                segment_attention_mask=segment_attention_mask,
            )
        reward_embeddings = self._encode_batch(
            input_ids,
            attention_mask,
            segment_input_ids,
            segment_attention_mask,
            adapter_name=self.reward_adapter_name if hasattr(self.encoder, "set_adapter") else None,
        )
        if self.cost_head is None:
            return self.predict_from_embeddings(reward_embeddings)
        if self.cost_gradient_mode == "separate_adapter":
            cost_embeddings = self._encode_batch(
                input_ids,
                attention_mask,
                segment_input_ids,
                segment_attention_mask,
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
        if "segment_input_ids" in batch:
            segment_input_ids = [tensor.to(accelerator.device) for tensor in batch["segment_input_ids"]]
            segment_attention_mask = [tensor.to(accelerator.device) for tensor in batch["segment_attention_mask"]]
            embeddings = model.encode_segmented_inputs(
                segment_input_ids=segment_input_ids,
                segment_attention_mask=segment_attention_mask,
            ).detach().cpu()
        else:
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


def _compute_cost_target_stats(dataset: Dataset, cost_target_dim: int) -> tuple[list[float], list[float]]:
    values: list[list[float]] = []
    for row in dataset:
        row_values = row.get("output_token_targets_log") if isinstance(row, dict) else None
        if isinstance(row_values, list) and len(row_values) == int(cost_target_dim):
            values.append([float(value) for value in row_values])
    if not values:
        return [0.0 for _ in range(int(cost_target_dim))], [1.0 for _ in range(int(cost_target_dim))]
    arr = np.asarray(values, dtype=np.float64)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    std = np.where(std < 1.0e-6, 1.0, std)
    return [float(value) for value in mean.tolist()], [float(value) for value in std.tolist()]


def _normalize_cost_targets(
    targets: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    normalization: str,
) -> torch.Tensor:
    if normalization == "none" or mean is None or std is None:
        return targets.float()
    if normalization != "per_route_standard":
        raise ValueError(f"Unsupported cost target normalization: {normalization}")
    return (targets.float() - mean.to(targets.device, dtype=targets.dtype)) / std.to(
        targets.device,
        dtype=targets.dtype,
    )


def _denormalize_cost_predictions(
    preds: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    normalization: str,
) -> torch.Tensor:
    if normalization == "none" or mean is None or std is None:
        return preds.float()
    if normalization != "per_route_standard":
        raise ValueError(f"Unsupported cost target normalization: {normalization}")
    return (preds.float() * std.to(preds.device, dtype=preds.dtype)) + mean.to(
        preds.device,
        dtype=preds.dtype,
    )


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


def _class_targets_from_rewards(
    y_true_rewards: np.ndarray,
    class_target_mode: str,
    class_success_threshold: float,
) -> np.ndarray:
    if y_true_rewards.size == 0:
        return np.empty((0,), dtype=np.int64)
    targets: list[int] = []
    for row in y_true_rewards.tolist():
        targets.append(
            _compute_class_target(
                [float(value) for value in row],
                mode=str(class_target_mode),
                success_threshold=float(class_success_threshold),
            )
        )
    return np.asarray(targets, dtype=np.int64)


def _joint_outcome_class_labels(route_labels: list[str]) -> list[str]:
    if len(route_labels) != 2:
        raise ValueError("Joint outcome classification currently requires exactly two routes")
    return [
        "both_fail",
        f"{route_labels[0]}_only",
        f"{route_labels[1]}_only",
        "both_pass",
    ]


def _joint_outcome_probs_to_route_probs(class_probs: torch.Tensor) -> torch.Tensor:
    if class_probs.shape[-1] != 4:
        raise ValueError("Joint outcome class probabilities must have dimension 4")
    first_route_prob = class_probs[:, 1] + class_probs[:, 3]
    second_route_prob = class_probs[:, 2] + class_probs[:, 3]
    return torch.stack([first_route_prob, second_route_prob], dim=-1)


def _compute_joint_outcome_metrics(prediction_rows: list[dict[str, Any]], class_labels: list[str]) -> dict[str, Any]:
    true_classes: list[int] = []
    pred_classes: list[int] = []
    probs: list[list[float]] = []
    for row in prediction_rows:
        true_class = row.get("true_class_target")
        pred_class = row.get("pred_joint_class_target", row.get("pred_class_target"))
        pred_probs = row.get("pred_joint_outcome_probs")
        if not isinstance(true_class, int) or not isinstance(pred_class, int):
            continue
        true_classes.append(int(true_class))
        pred_classes.append(int(pred_class))
        if isinstance(pred_probs, list) and len(pred_probs) == len(class_labels):
            probs.append([float(value) for value in pred_probs])
    if not true_classes:
        return {
            "n_eval": 0,
            "class_labels": list(class_labels),
            "accuracy": math.nan,
            "true_counts_by_class": {label: 0 for label in class_labels},
            "pred_counts_by_class": {label: 0 for label in class_labels},
        }
    y_true = np.asarray(true_classes, dtype=np.int64)
    y_pred = np.asarray(pred_classes, dtype=np.int64)
    true_counts = np.bincount(y_true, minlength=len(class_labels))
    pred_counts = np.bincount(y_pred, minlength=len(class_labels))
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true, y_pred):
        if 0 <= int(true_idx) < len(class_labels) and 0 <= int(pred_idx) < len(class_labels):
            confusion[int(true_idx), int(pred_idx)] += 1
    auc_by_class: dict[str, float | None] = {}
    if len(probs) == len(true_classes):
        prob_arr = np.asarray(probs, dtype=np.float64)
        for idx, label in enumerate(class_labels):
            auc_by_class[label] = _roc_auc_binary((y_true == idx).astype(np.int64), prob_arr[:, idx])
    disagreement_mask = np.isin(y_true, [1, 2])
    return {
        "n_eval": int(y_true.shape[0]),
        "class_labels": list(class_labels),
        "accuracy": float(np.mean(y_true == y_pred)),
        "true_counts_by_class": {class_labels[idx]: int(true_counts[idx]) for idx in range(len(class_labels))},
        "pred_counts_by_class": {class_labels[idx]: int(pred_counts[idx]) for idx in range(len(class_labels))},
        "confusion_matrix_rows_true_cols_pred": confusion.tolist(),
        "one_vs_rest_auc_by_class": auc_by_class,
        "disagreement_accuracy": (
            float(np.mean(y_true[disagreement_mask] == y_pred[disagreement_mask]))
            if bool(np.any(disagreement_mask))
            else math.nan
        ),
    }


def _compute_class_sample_weights(dataset: Dataset, class_count: int, mode: str) -> list[float] | None:
    if mode == "none":
        return None
    if mode != "inverse_freq":
        raise ValueError(f"Unsupported class_oversample_mode={mode!r}")
    counts = np.zeros((int(class_count),), dtype=np.float64)
    row_targets: list[int] = []
    for row in dataset:
        if not isinstance(row, dict):
            continue
        target = int(row["class_target"])
        if target < 0 or target >= int(class_count):
            raise ValueError(f"class target {target} is out of range for {class_count} classes")
        counts[target] += 1.0
        row_targets.append(target)
    if not row_targets:
        return None
    sample_weights = []
    for target in row_targets:
        count = counts[int(target)]
        sample_weights.append(float(1.0 / count) if count > 0.0 else 0.0)
    return sample_weights


def _compute_class_weights(
    dataset: Dataset,
    target_dim: int,
    mode: str,
) -> list[float]:
    if mode == "none":
        return [1.0 for _ in range(int(target_dim))]
    if mode != "inverse_freq":
        raise ValueError(f"Unsupported class_weight_mode={mode!r}")
    counts = np.zeros((int(target_dim),), dtype=np.float64)
    for row in dataset:
        if isinstance(row, dict):
            counts[int(row["class_target"])] += 1.0
    total = float(np.sum(counts))
    if total <= 0.0:
        return [1.0 for _ in range(int(target_dim))]
    weights = np.ones((int(target_dim),), dtype=np.float64)
    for idx, count in enumerate(counts):
        weights[idx] = total / (float(target_dim) * count) if count > 0.0 else 0.0
    positive = weights[weights > 0.0]
    if positive.size > 0:
        weights = weights / float(np.mean(positive))
    return [float(value) for value in weights.tolist()]


def _compute_route_classifier_metrics(
    y_true_rewards: np.ndarray,
    y_pred_scores: np.ndarray,
    route_labels: list[str],
    class_target_mode: str,
    class_success_threshold: float,
) -> dict[str, Any]:
    if y_true_rewards.size == 0 or y_pred_scores.size == 0:
        return {
            "n_eval": 0,
            "class_target_mode": str(class_target_mode),
            "class_success_threshold": float(class_success_threshold),
            "accuracy": math.nan,
            "mean_reward_of_predicted_class": math.nan,
            "target_counts_by_route": {label: 0 for label in route_labels},
            "pred_counts_by_route": {label: 0 for label in route_labels},
        }

    target_classes = _class_targets_from_rewards(
        y_true_rewards,
        class_target_mode=str(class_target_mode),
        class_success_threshold=float(class_success_threshold),
    )
    pred_classes = np.argmax(y_pred_scores, axis=1).astype(np.int64)
    target_counts = np.bincount(target_classes, minlength=len(route_labels))
    pred_counts = np.bincount(pred_classes, minlength=len(route_labels))
    row_indices = np.arange(y_true_rewards.shape[0])
    realized_rewards = y_true_rewards[row_indices, pred_classes]
    target_rewards = y_true_rewards[row_indices, target_classes]
    confusion = np.zeros((len(route_labels), len(route_labels)), dtype=np.int64)
    for true_idx, pred_idx in zip(target_classes, pred_classes):
        confusion[int(true_idx), int(pred_idx)] += 1
    return {
        "n_eval": int(target_classes.shape[0]),
        "class_target_mode": str(class_target_mode),
        "class_success_threshold": float(class_success_threshold),
        "accuracy": float(np.mean(target_classes == pred_classes)),
        "mean_reward_of_target_class": float(np.mean(target_rewards)),
        "mean_reward_of_predicted_class": float(np.mean(realized_rewards)),
        "target_counts_by_route": {
            route_labels[idx]: int(target_counts[idx]) for idx in range(len(route_labels))
        },
        "pred_counts_by_route": {
            route_labels[idx]: int(pred_counts[idx]) for idx in range(len(route_labels))
        },
        "confusion_matrix_rows_true_cols_pred": confusion.tolist(),
    }


def _compute_predicted_cost_utility_report(
    prediction_rows: list[dict[str, Any]],
    eval_source_rows: list[dict[str, Any]],
    route_labels: list[str],
    lambdas: list[float],
    cost_route_idxs: list[int],
    reported_cost_route_idxs: list[int] | None = None,
    reported_cost_route_labels: list[str] | None = None,
) -> dict[str, Any]:
    target_dim = len(route_labels)
    cost_route_pos = {int(route_idx): idx for idx, route_idx in enumerate(cost_route_idxs)}
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
            or len(pred_output_tokens) != len(cost_route_idxs)
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
                "pred_output_tokens_by_cost_route": [max(0.0, float(value)) for value in pred_output_tokens],
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
                    if route_idx in cost_route_pos:
                        pred_cost = example["pred_output_tokens_by_cost_route"][cost_route_pos[route_idx]]
                    else:
                        pred_cost = example["output_tokens"][route_idx]
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
        "predicted_cost_route_idxs": [int(idx) for idx in (reported_cost_route_idxs or cost_route_idxs)],
        "predicted_cost_route_labels": list(
            reported_cost_route_labels
            if reported_cost_route_labels is not None
            else [route_labels[int(idx)] for idx in cost_route_idxs]
        ),
        "predicted_cost_local_route_idxs": [int(idx) for idx in cost_route_idxs],
        "utility_rows": utility_rows,
    }


def _ranking_loss(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry pairwise ranking: for each pair (i,j) where route i succeeds and j fails,
    push -log σ(prob[i] - prob[j]). Averaged over all ordered pairs and batch."""
    total = torch.zeros(1, device=probs.device, dtype=probs.dtype)
    n_pairs = 0
    for i in range(probs.shape[1]):
        for j in range(probs.shape[1]):
            if i == j:
                continue
            mask = (targets[:, i] > 0.5) & (targets[:, j] < 0.5)
            if mask.any():
                total = total + (-torch.log(torch.sigmoid(probs[mask, i] - probs[mask, j]) + 1e-8)).mean()
                n_pairs += 1
    return total / max(n_pairs, 1)


def _delta_loss(preds: torch.Tensor, targets: torch.Tensor, huber_delta: float) -> torch.Tensor:
    if preds.shape[1] < 2:
        raise ValueError("Delta auxiliary loss expects at least two routes")
    pred_deltas = []
    true_deltas = []
    for left_idx in range(int(preds.shape[1])):
        for right_idx in range(left_idx + 1, int(preds.shape[1])):
            pred_deltas.append(preds[:, right_idx] - preds[:, left_idx])
            true_deltas.append(targets[:, right_idx] - targets[:, left_idx])
    pred_delta = torch.stack(pred_deltas, dim=1)
    true_delta = torch.stack(true_deltas, dim=1)
    if float(huber_delta) > 0.0:
        return F.huber_loss(pred_delta, true_delta, delta=float(huber_delta))
    return F.mse_loss(pred_delta, true_delta)


def _hierarchical_route_classifier_loss(
    logits: torch.Tensor,
    class_targets: torch.Tensor,
    class_weights: torch.Tensor | None,
    any_success_weight: float,
    route_weight: float,
    reduction: str,
) -> torch.Tensor:
    if logits.shape[-1] < 2:
        raise ValueError("Hierarchical route classifier requires at least one route plus ABSTAIN")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"Unsupported reduction={reduction!r}")
    abstain_idx = int(logits.shape[-1]) - 1
    targets = class_targets.long()
    is_success = targets != abstain_idx

    # Binary stop/go decision: compare aggregate non-abstain mass against abstain.
    any_success_logits = torch.logsumexp(logits[:, :abstain_idx], dim=-1) - logits[:, abstain_idx]
    any_success_losses = F.binary_cross_entropy_with_logits(
        any_success_logits,
        is_success.float(),
        reduction="none",
    )

    # Conditional route decision, only defined when at least one route succeeds.
    route_losses = torch.zeros_like(any_success_losses)
    if torch.any(is_success):
        route_class_weights = None
        if class_weights is not None:
            route_class_weights = class_weights[:abstain_idx].to(logits.device, dtype=logits.dtype)
        route_losses[is_success] = F.cross_entropy(
            logits[is_success, :abstain_idx],
            targets[is_success],
            weight=route_class_weights,
            reduction="none",
        )

    losses = (float(any_success_weight) * any_success_losses) + (float(route_weight) * route_losses)
    if reduction == "sum":
        return torch.sum(losses)
    if reduction == "mean":
        return torch.mean(losses)
    return losses


def _compute_train_loss(
    reward_logits: torch.Tensor,
    cost_logits: torch.Tensor | None,
    zero_reward_logits: torch.Tensor | None,
    targets: torch.Tensor,
    output_token_targets_log: torch.Tensor,
    zero_reward_targets: torch.Tensor,
    class_targets: torch.Tensor,
    class_weights: torch.Tensor | None,
    objective: str,
    hierarchical_any_success_weight: float,
    hierarchical_route_weight: float,
    reward_mse_weight: float,
    reward_bce_weight: float,
    delta_aux_weight: float,
    delta_aux_huber_delta: float,
    predict_costs: bool,
    cost_mse_weight: float,
    cost_delta_aux_weight: float,
    predict_zero_reward_failure: bool,
    zero_reward_bce_weight: float,
    ranking_aux_weight: float = 0.0,
    discrim_upweight: float = 1.0,
) -> torch.Tensor:
    if objective == "cost_mse":
        if not predict_costs or cost_logits is None:
            raise ValueError("objective=cost_mse requires predict_costs=true and cost logits")
        if predict_zero_reward_failure:
            raise ValueError("objective=cost_mse is incompatible with zero-reward failure prediction")
        if float(cost_delta_aux_weight) != 0.0:
            raise ValueError("cost_delta_aux_weight is not supported for expert-only cost prediction")
        return float(cost_mse_weight) * F.mse_loss(cost_logits, output_token_targets_log.float())
    if objective in {"route_classifier", "joint_outcome_2route"}:
        reward_loss = F.cross_entropy(
            reward_logits,
            class_targets.long(),
            weight=class_weights.to(reward_logits.device, dtype=reward_logits.dtype)
            if class_weights is not None
            else None,
        )
    elif objective == "route_classifier_hierarchical":
        reward_loss = _hierarchical_route_classifier_loss(
            reward_logits,
            class_targets.long(),
            class_weights=class_weights,
            any_success_weight=float(hierarchical_any_success_weight),
            route_weight=float(hierarchical_route_weight),
            reduction="mean",
        )
    elif objective in {"reward_bce", "reward_bce_delta_aux", "reward_bce_ranking"}:
        if float(discrim_upweight) > 1.0:
            # Per-sample weighted BCE: upweight tasks where routing matters
            # (at least one route succeeds but not all routes succeed)
            is_discrim = targets.bool().any(dim=1) & ~targets.bool().all(dim=1)
            w = torch.ones(targets.shape[0], device=targets.device, dtype=targets.dtype)
            w[is_discrim] = float(discrim_upweight)
            per_sample = F.binary_cross_entropy_with_logits(
                reward_logits, targets.float(), reduction="none"
            ).mean(dim=1)
            reward_loss = (per_sample * w).sum() / w.sum() * float(reward_bce_weight)
        else:
            reward_loss = (
                F.binary_cross_entropy_with_logits(reward_logits, targets.float())
                * float(reward_bce_weight)
            )
    else:
        reward_loss = F.mse_loss(reward_logits, targets.float()) * float(reward_mse_weight)
    if objective == "reward_mse_delta_aux":
        reward_loss = reward_loss + (
            float(delta_aux_weight) * _delta_loss(reward_logits, targets.float(), float(delta_aux_huber_delta))
        )
    if objective == "reward_bce_delta_aux":
        reward_probs = torch.sigmoid(reward_logits)
        reward_loss = reward_loss + (
            float(delta_aux_weight) * _delta_loss(reward_probs, targets.float(), float(delta_aux_huber_delta))
        )
    if objective == "reward_bce_ranking" or float(ranking_aux_weight) > 0.0:
        reward_probs = torch.sigmoid(reward_logits)
        reward_loss = reward_loss + float(ranking_aux_weight) * _ranking_loss(reward_probs, targets.float())
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


def _make_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    cost_head_lr: float,
) -> torch.optim.Optimizer:
    if float(cost_head_lr) <= 0.0 or not hasattr(model, "cost_head") or model.cost_head is None:
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(trainable_parameters, lr=float(lr), weight_decay=float(weight_decay))

    cost_head_params = [parameter for parameter in model.cost_head.parameters() if parameter.requires_grad]
    cost_head_param_ids = {id(parameter) for parameter in cost_head_params}
    other_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in cost_head_param_ids
    ]
    param_groups: list[dict[str, Any]] = []
    if other_params:
        param_groups.append({"params": other_params, "lr": float(lr), "weight_decay": float(weight_decay)})
    if cost_head_params:
        param_groups.append(
            {
                "params": cost_head_params,
                "lr": float(cost_head_lr),
                "weight_decay": float(weight_decay),
            }
        )
    if not param_groups:
        raise ValueError("No trainable parameters found")
    return torch.optim.AdamW(param_groups)


def _parse_route_indices(text: str | None, full_dim: int) -> list[int]:
    if text is None or str(text).strip() == "":
        return list(range(full_dim))
    indices: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= full_dim:
            raise ValueError(f"target route index {idx} is out of range for {full_dim} routes")
        indices.append(idx)
    if not indices:
        raise ValueError("--target-route-idxs resolved to an empty route set")
    if len(set(indices)) != len(indices):
        raise ValueError(f"--target-route-idxs contains duplicates: {indices}")
    return indices


def _subset_route_stats_for_utility(
    rows: list[dict[str, Any]],
    route_indices: list[int],
) -> list[dict[str, Any]]:
    if route_indices == list(range(len(route_indices))):
        return rows
    subset_rows: list[dict[str, Any]] = []
    route_stat_keys = ("performance_targets", "route_prompt_tokens", "route_output_tokens")
    max_idx = max(int(idx) for idx in route_indices)
    for row in rows:
        new_row = dict(row)
        valid = True
        for key in route_stat_keys:
            values = row.get(key)
            if not isinstance(values, list) or len(values) <= max_idx:
                valid = False
                break
            new_row[key] = [values[int(idx)] for idx in route_indices]
        if valid:
            subset_rows.append(new_row)
    return subset_rows


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
    if "segment_input_ids" in batch:
        return model(
            segment_input_ids=batch["segment_input_ids"],
            segment_attention_mask=batch["segment_attention_mask"],
            cost_only=cost_only,
        )
    if cost_only:
        return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], cost_only=True)
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])


@torch.no_grad()
def _evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loader: DataLoader,
    eval_dataset: RouterCostDataset,
    route_labels: list[str],
    route_idxs: list[int] | None,
    cost_route_idxs: list[int] | None,
    cost_route_labels: list[str] | None,
    objective: str,
    predict_costs: bool,
    predict_zero_reward_failure: bool,
    class_weights: torch.Tensor | None,
    hierarchical_any_success_weight: float,
    hierarchical_route_weight: float,
    cost_target_normalization: str,
    cost_target_mean: torch.Tensor | None,
    cost_target_std: torch.Tensor | None,
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
        cost_targets_for_loss = _normalize_cost_targets(
            batch["output_token_targets_log"].float(),
            cost_target_mean,
            cost_target_std,
            str(cost_target_normalization),
        )
        if objective == "cost_mse":
            if cost_logits is None:
                raise ValueError("objective=cost_mse requires cost logits")
            loss = F.mse_loss(cost_logits.float(), cost_targets_for_loss, reduction="sum")
            preds = logits
        elif objective == "route_classifier":
            per_example_loss = F.cross_entropy(
                logits,
                batch["class_targets"].long(),
                weight=class_weights.to(logits.device, dtype=logits.dtype)
                if class_weights is not None
                else None,
                reduction="none",
            )
            loss = per_example_loss.sum()
            preds = torch.softmax(logits, dim=-1)
            class_pred_scores = preds
        elif objective == "joint_outcome_2route":
            per_example_loss = F.cross_entropy(
                logits,
                batch["class_targets"].long(),
                weight=class_weights.to(logits.device, dtype=logits.dtype)
                if class_weights is not None
                else None,
                reduction="none",
            )
            loss = per_example_loss.sum()
            class_pred_scores = torch.softmax(logits, dim=-1)
            preds = _joint_outcome_probs_to_route_probs(class_pred_scores)
        elif objective == "route_classifier_hierarchical":
            loss = _hierarchical_route_classifier_loss(
                logits,
                batch["class_targets"].long(),
                class_weights=class_weights,
                any_success_weight=float(hierarchical_any_success_weight),
                route_weight=float(hierarchical_route_weight),
                reduction="sum",
            )
            preds = torch.softmax(logits, dim=-1)
            class_pred_scores = preds
        elif objective in {"reward_bce", "reward_bce_delta_aux", "reward_bce_ranking"}:
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
            preds = torch.sigmoid(logits)
            class_pred_scores = preds
            if predict_costs and cost_logits is not None:
                loss = loss + F.mse_loss(cost_logits.float(), cost_targets_for_loss, reduction="sum")
        else:
            loss = F.mse_loss(logits, targets, reduction="sum")
            preds = logits
            class_pred_scores = preds
            if predict_costs and cost_logits is not None:
                loss = loss + F.mse_loss(cost_logits.float(), cost_targets_for_loss, reduction="sum")
        gathered_loss = accelerator.gather_for_metrics(loss.detach().reshape(1)).detach().cpu()
        gathered_preds = accelerator.gather_for_metrics(preds).detach().cpu()
        gathered_targets = accelerator.gather_for_metrics(targets).detach().cpu()
        gathered_class_targets = accelerator.gather_for_metrics(batch["class_targets"]).detach().cpu()
        gathered_joint_scores = (
            accelerator.gather_for_metrics(class_pred_scores).detach().cpu()
            if objective == "joint_outcome_2route"
            else None
        )
        if predict_costs:
            if cost_logits is None:
                raise ValueError("predict_costs=true but model did not return cost logits")
            cost_preds_for_metrics = _denormalize_cost_predictions(
                cost_logits.float(),
                cost_target_mean,
                cost_target_std,
                str(cost_target_normalization),
            )
            gathered_cost_preds = accelerator.gather_for_metrics(cost_preds_for_metrics).detach().cpu()
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
            pred_class_target = int(torch.argmax(gathered_preds[idx]).item())
            row_extra: dict[str, Any] = {}
            if gathered_joint_scores is not None:
                pred_class_target = int(torch.argmax(gathered_joint_scores[idx]).item())
                row_extra = {
                    "pred_joint_outcome_probs": [float(value) for value in gathered_joint_scores[idx].tolist()],
                    "pred_joint_class_target": pred_class_target,
                }
            row_payload = {
                    "problem_id": source_meta["problem_id"],
                    "dataset": source_meta["dataset"],
                    "repo": source_meta["repo"],
                    "language": source_meta["language"],
                    "true_rewards": [float(value) for value in gathered_targets[idx].tolist()],
                    "pred_rewards": [float(value) for value in gathered_preds[idx].tolist()],
                    "true_class_target": int(gathered_class_targets[idx].item()),
                    "pred_class_target": pred_class_target,
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
                    "cost_route_idxs": [int(value) for value in (cost_route_idxs or [])] if predict_costs else None,
                    "cost_route_labels": list(cost_route_labels or []) if predict_costs else None,
                    "true_zero_reward_failure": [
                        float(value) for value in gathered_zero_reward_targets[idx].tolist()
                    ] if predict_zero_reward_failure else None,
                    "pred_zero_reward_failure_probs": [
                        float(value) for value in gathered_zero_reward_preds[idx].tolist()
                    ] if predict_zero_reward_failure else None,
                    "route_labels": list(route_labels),
                    "route_idxs": [int(value) for value in (route_idxs or [])],
                }
            row_payload.update(row_extra)
            rows.append(row_payload)
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
    if objective in {"route_classifier", "route_classifier_hierarchical", "joint_outcome_2route"}:
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
        choices=[
            "reward_mse",
            "reward_mse_delta_aux",
            "reward_bce",
            "reward_bce_delta_aux",
            "reward_bce_ranking",
            "route_classifier",
            "route_classifier_hierarchical",
            "joint_outcome_2route",
            "cost_mse",
        ],
        default="reward_mse",
    )
    parser.add_argument(
        "--class-target-mode",
        choices=["argmax", "cheapest_success", "cheapest_success_or_abstain", "joint_success_2route"],
        default="argmax",
        help=(
            "Target used for objective=route_classifier. "
            "argmax picks the highest reward route; cheapest_success picks the first route above threshold, "
            "falling back to route 0 if none pass."
        ),
    )
    parser.add_argument(
        "--class-success-threshold",
        type=float,
        default=0.5,
        help="Success threshold for --class-target-mode=cheapest_success.",
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=["none", "inverse_freq"],
        default="none",
        help="Optional class weighting for classifier-style objectives.",
    )
    parser.add_argument(
        "--class-oversample-mode",
        choices=["none", "inverse_freq"],
        default="none",
        help="Optional class-balanced sampling for classifier-style objectives.",
    )
    parser.add_argument(
        "--append-abstain-class",
        action="store_true",
        help="Append an explicit ABSTAIN class for route-classifier objectives.",
    )
    parser.add_argument(
        "--hierarchical-any-success-weight",
        type=float,
        default=1.0,
        help="Loss weight for the abstain-vs-any-success term in objective=route_classifier_hierarchical.",
    )
    parser.add_argument(
        "--hierarchical-route-weight",
        type=float,
        default=1.0,
        help="Loss weight for the conditional route CE term in objective=route_classifier_hierarchical.",
    )
    parser.add_argument("--max-seq-length", type=int, default=24000)
    parser.add_argument("--input-mode", choices=["post_primary", "input_only", "post_120b", "post_120b_no_think"], default="post_primary")
    parser.add_argument(
        "--embedding-input-layout",
        choices=["single", "late_fusion", "late_fusion_prompt_only", "late_fusion_scout_only", "semantic_late_fusion", "semantic_problem_only", "semantic_code_only"],
        default="single",
        help="How to present evidence to Qwen embeddings: one sequence, full-prompt late fusion, or semantic problem/code/scout fusion.",
    )
    parser.add_argument("--include-primary-output-token-count", action="store_true")
    parser.add_argument(
        "--target-route-idxs",
        default=None,
        help=(
            "Comma-separated original route indices to train/evaluate as final choices. "
            "The scout/primary attempt can still be used as context when excluded here."
        ),
    )
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument(
        "--cost-head-lr",
        type=float,
        default=0.0,
        help="Optional LR for the cost head only; <=0 uses --lr for all trainable parameters.",
    )
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
    parser.add_argument("--reward-bce-weight", type=float, default=1.0)
    parser.add_argument("--delta-aux-weight", type=float, default=1.0)
    parser.add_argument("--delta-aux-huber-delta", type=float, default=0.0)
    parser.add_argument("--ranking-aux-weight", type=float, default=0.0)
    parser.add_argument("--discrim-upweight", type=float, default=1.0)
    parser.add_argument("--predict-costs", action="store_true")
    parser.add_argument(
        "--cost-route-idx",
        type=int,
        default=1,
        help="Original dataset route index to predict costs for when --cost-route-idxs is omitted.",
    )
    parser.add_argument(
        "--cost-route-idxs",
        default=None,
        help=(
            "Comma-separated original dataset route indices to predict costs for. "
            "Every index must be included in --target-route-idxs. Defaults to --cost-route-idx."
        ),
    )
    parser.add_argument(
        "--cost-gradient-mode",
        choices=["joint", "detached", "separate_adapter"],
        default="joint",
    )
    parser.add_argument("--cost-mse-weight", type=float, default=1.0)
    parser.add_argument(
        "--cost-target-normalization",
        choices=["none", "per_route_standard"],
        default="none",
        help="Normalize log1p output-token targets during cost training; metrics are denormalized.",
    )
    parser.add_argument("--cost-delta-aux-weight", type=float, default=0.0)
    parser.add_argument("--predict-zero-reward-failure", action="store_true")
    parser.add_argument("--zero-reward-epsilon", type=float, default=0.0)
    parser.add_argument("--zero-reward-bce-weight", type=float, default=1.0)
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    parser.add_argument("--checkpoint-every-epoch", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--init-from-model-checkpoint", default=None,
                        help="Load model weights only (no optimizer state) from this checkpoint path.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--cost-normalization-config",
        default=None,
        help="Optional train_config.json/summary.json whose cost_target_mean/std should be reused for eval-only scoring.",
    )
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

    all_route_labels = _load_route_labels(dataset_dir)
    target_route_idxs = _parse_route_indices(args.target_route_idxs, len(all_route_labels))
    source_route_labels = [all_route_labels[int(idx)] for idx in target_route_idxs]
    append_abstain_class = bool(args.append_abstain_class)
    route_classifier_objectives = {"route_classifier", "route_classifier_hierarchical"}
    joint_outcome_objectives = {"joint_outcome_2route"}
    classifier_loss_objectives = route_classifier_objectives | joint_outcome_objectives
    if append_abstain_class and str(args.objective) not in route_classifier_objectives:
        raise ValueError("--append-abstain-class is only supported for route-classifier objectives")
    if str(args.objective) == "route_classifier_hierarchical":
        if not append_abstain_class:
            raise ValueError("objective=route_classifier_hierarchical requires --append-abstain-class")
        if str(args.class_target_mode) != "cheapest_success_or_abstain":
            raise ValueError("objective=route_classifier_hierarchical requires --class-target-mode=cheapest_success_or_abstain")
    route_labels = list(source_route_labels) + (["ABSTAIN"] if append_abstain_class else [])
    target_dim = len(route_labels)
    joint_outcome_class_labels: list[str] = []
    model_output_dim = target_dim
    if str(args.objective) == "joint_outcome_2route":
        if append_abstain_class:
            raise ValueError("objective=joint_outcome_2route does not support --append-abstain-class")
        if target_dim != 2:
            raise ValueError("objective=joint_outcome_2route requires exactly two target routes")
        if str(args.class_target_mode) != "joint_success_2route":
            raise ValueError("objective=joint_outcome_2route requires --class-target-mode=joint_success_2route")
        joint_outcome_class_labels = _joint_outcome_class_labels(route_labels)
        model_output_dim = len(joint_outcome_class_labels)
    if args.objective in {"reward_mse_delta_aux", "reward_bce_delta_aux", "reward_bce_ranking"} and target_dim < 2:
        raise ValueError(f"{args.objective} expects at least two routes")
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
    if args.cost_route_idxs:
        cost_route_idxs = [int(part.strip()) for part in str(args.cost_route_idxs).split(",") if part.strip()]
    else:
        cost_route_idxs = [int(args.cost_route_idx)]
    if not cost_route_idxs:
        raise ValueError("--cost-route-idxs resolved to an empty list")
    cost_route_idx = int(cost_route_idxs[0])
    target_route_idx_to_local = {int(source_idx): local_idx for local_idx, source_idx in enumerate(target_route_idxs)}
    invalid_cost_route_idxs = [idx for idx in cost_route_idxs if int(idx) not in target_route_idx_to_local]
    if bool(args.predict_costs) and invalid_cost_route_idxs:
        raise ValueError(
            "--cost-route-idxs now expects original dataset route indices, not local filtered positions. "
            f"Invalid values {invalid_cost_route_idxs}; available target route ids are {target_route_idxs}."
        )
    local_cost_route_idxs = [
        int(target_route_idx_to_local[int(idx)]) for idx in cost_route_idxs if int(idx) in target_route_idx_to_local
    ]
    if not local_cost_route_idxs:
        local_cost_route_idxs = [0]
    cost_route_labels = [all_route_labels[int(idx)] for idx in cost_route_idxs] if bool(args.predict_costs) else []
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
    if str(args.objective) == "cost_mse":
        input_task = "cost"
    elif str(args.objective) in route_classifier_objectives:
        input_task = "policy"
    else:
        input_task = "reward"
    train_dataset = RouterCostDataset(
        train_rows,
        tokenizer,
        source_route_labels,
        target_route_idxs,
        int(args.max_seq_length),
        require_cost_targets=bool(args.predict_costs),
        cost_route_idxs=local_cost_route_idxs,
        zero_reward_epsilon=float(args.zero_reward_epsilon),
        input_mode=str(args.input_mode),
        include_primary_output_token_count=bool(args.include_primary_output_token_count),
        input_task=input_task,
        embedding_input_layout=str(args.embedding_input_layout),
        class_target_mode=str(args.class_target_mode),
        class_success_threshold=float(args.class_success_threshold),
        append_abstain_class=append_abstain_class,
    )
    eval_dataset = RouterCostDataset(
        eval_rows_source,
        tokenizer,
        source_route_labels,
        target_route_idxs,
        int(args.max_seq_length),
        require_cost_targets=bool(args.predict_costs),
        cost_route_idxs=local_cost_route_idxs,
        zero_reward_epsilon=float(args.zero_reward_epsilon),
        input_mode=str(args.input_mode),
        include_primary_output_token_count=bool(args.include_primary_output_token_count),
        input_task=input_task,
        embedding_input_layout=str(args.embedding_input_layout),
        class_target_mode=str(args.class_target_mode),
        class_success_threshold=float(args.class_success_threshold),
        append_abstain_class=append_abstain_class,
    )
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError(f"Prepared empty dataset train={len(train_dataset)} eval={len(eval_dataset)}")
    train_has_segments = "segment_input_ids" in train_dataset.rows[0]
    eval_has_segments = "segment_input_ids" in eval_dataset.rows[0]
    if train_has_segments != eval_has_segments:
        raise ValueError("Train/eval embedding input layouts do not match")
    embedding_segment_count = len(train_dataset.rows[0]["segment_input_ids"]) if train_has_segments else 1
    if eval_has_segments and len(eval_dataset.rows[0]["segment_input_ids"]) != embedding_segment_count:
        raise ValueError("Train/eval segment counts do not match")

    cost_target_dim = len(local_cost_route_idxs)
    if bool(args.predict_costs):
        if str(args.cost_target_normalization) == "per_route_standard":
            cost_target_mean_values, cost_target_std_values = _compute_cost_target_stats(
                train_dataset,
                cost_target_dim,
            )
        else:
            cost_target_mean_values = [0.0 for _ in range(cost_target_dim)]
            cost_target_std_values = [1.0 for _ in range(cost_target_dim)]
    else:
        cost_target_mean_values = []
        cost_target_std_values = []
    if args.cost_normalization_config:
        cost_normalization_config = Path(args.cost_normalization_config)
        if not cost_normalization_config.exists():
            raise FileNotFoundError(f"Missing cost normalization config: {cost_normalization_config}")
        normalization_payload = json.loads(cost_normalization_config.read_text())
        normalization_config = normalization_payload.get("config", normalization_payload)
        cost_target_mean_values = [float(value) for value in normalization_config.get("cost_target_mean_log1p", [])]
        cost_target_std_values = [float(value) for value in normalization_config.get("cost_target_std_log1p", [])]
        if bool(args.predict_costs) and (
            len(cost_target_mean_values) != cost_target_dim or len(cost_target_std_values) != cost_target_dim
        ):
            raise ValueError(
                "Cost normalization config dimension mismatch: "
                f"mean={len(cost_target_mean_values)} std={len(cost_target_std_values)} expected={cost_target_dim}"
            )
    cost_target_mean_tensor = (
        torch.tensor(cost_target_mean_values, dtype=torch.float32) if cost_target_mean_values else None
    )
    cost_target_std_tensor = (
        torch.tensor(cost_target_std_values, dtype=torch.float32) if cost_target_std_values else None
    )
    class_weight_values = (
        _compute_class_weights(train_dataset, model_output_dim, str(args.class_weight_mode))
        if str(args.objective) in classifier_loss_objectives
        else [1.0 for _ in range(model_output_dim)]
    )
    class_weights_tensor = (
        torch.tensor(class_weight_values, dtype=torch.float32)
        if str(args.objective) in classifier_loss_objectives and str(args.class_weight_mode) != "none"
        else None
    )
    token_collate_fn = lambda batch: _collate_left_pad(
        batch,
        pad_token_id=int(pad_token_id),
        target_dim=target_dim,
        cost_target_dim=cost_target_dim,
    )
    train_sample_weights = (
        _compute_class_sample_weights(train_dataset, model_output_dim, str(args.class_oversample_mode))
        if str(args.objective) in classifier_loss_objectives
        else None
    )
    train_sampler = (
        WeightedRandomSampler(
            weights=torch.as_tensor(train_sample_weights, dtype=torch.double),
            num_samples=len(train_sample_weights),
            replacement=True,
        )
        if train_sample_weights is not None
        else None
    )
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=train_sampler is None,
        sampler=train_sampler,
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
        target_dim=model_output_dim,
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
        embedding_input_layout=str(args.embedding_input_layout),
        segment_count=int(embedding_segment_count),
    )
    init_from_model_checkpoint_report = None
    if args.init_from_model_checkpoint:
        init_from_model_checkpoint_report = _load_model_only_checkpoint(
            model, Path(args.init_from_model_checkpoint)
        )
        if accelerator.is_main_process:
            print(
                f"Initialized model weights from {init_from_model_checkpoint_report['checkpoint']} "
                f"({init_from_model_checkpoint_report['loaded_tensors']} tensors)",
                flush=True,
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
        train_sample_weights = (
            _compute_class_sample_weights(train_dataset, model_output_dim, str(args.class_oversample_mode))
            if str(args.objective) in classifier_loss_objectives
            else None
        )
        train_sampler = (
            WeightedRandomSampler(
                weights=torch.as_tensor(train_sample_weights, dtype=torch.double),
                num_samples=len(train_sample_weights),
                replacement=True,
            )
            if train_sample_weights is not None
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=train_sampler is None,
            sampler=train_sampler,
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

    optimizer = _make_optimizer(
        model,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        cost_head_lr=float(args.cost_head_lr),
    )
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
        "class_target_mode": str(args.class_target_mode),
        "class_success_threshold": float(args.class_success_threshold),
        "class_weight_mode": str(args.class_weight_mode),
        "class_oversample_mode": str(args.class_oversample_mode),
        "class_weights": [float(value) for value in class_weight_values],
        "class_oversampling_enabled": bool(train_sample_weights is not None),
        "model_output_dim": int(model_output_dim),
        "joint_outcome_class_labels": list(joint_outcome_class_labels),
        "hierarchical_any_success_weight": float(args.hierarchical_any_success_weight),
        "hierarchical_route_weight": float(args.hierarchical_route_weight),
        "dataset_dir": str(dataset_dir),
        "all_route_labels": all_route_labels,
        "target_route_idxs": [int(idx) for idx in target_route_idxs],
        "source_route_labels": source_route_labels,
        "route_labels": route_labels,
        "append_abstain_class": bool(append_abstain_class),
        "max_seq_length": int(args.max_seq_length),
        "input_mode": str(args.input_mode),
        "embedding_input_layout": str(args.embedding_input_layout),
        "embedding_segment_count": int(embedding_segment_count),
        "input_task": input_task,
        "include_primary_output_token_count": bool(args.include_primary_output_token_count),
        "num_epochs": int(args.num_epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "lr": float(args.lr),
        "cost_head_lr": float(args.cost_head_lr),
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
        "reward_bce_weight": float(args.reward_bce_weight),
        "delta_aux_weight": float(args.delta_aux_weight),
        "delta_aux_huber_delta": float(args.delta_aux_huber_delta),
        "predict_costs": bool(args.predict_costs),
        "cost_target": "log1p_route_output_tokens",
        "cost_route_idx": cost_route_idx,
        "cost_route_idxs": [int(idx) for idx in cost_route_idxs],
        "cost_route_label": all_route_labels[cost_route_idx] if bool(args.predict_costs) else None,
        "cost_route_labels": list(cost_route_labels),
        "local_cost_route_idx": int(local_cost_route_idxs[0]),
        "local_cost_route_idxs": [int(idx) for idx in local_cost_route_idxs],
        "cost_gradient_mode": str(args.cost_gradient_mode),
        "cost_mse_weight": float(args.cost_mse_weight),
        "cost_target_normalization": str(args.cost_target_normalization),
        "cost_target_mean_log1p": [float(value) for value in cost_target_mean_values],
        "cost_target_std_log1p": [float(value) for value in cost_target_std_values],
        "cost_delta_aux_weight": float(args.cost_delta_aux_weight),
        "predict_zero_reward_failure": bool(args.predict_zero_reward_failure),
        "zero_reward_failure_target": "reward <= zero_reward_epsilon",
        "zero_reward_epsilon": float(args.zero_reward_epsilon),
        "zero_reward_bce_weight": float(args.zero_reward_bce_weight),
        "ddp_find_unused_parameters": bool(args.ddp_find_unused_parameters),
        "checkpoint_every_epoch": bool(args.checkpoint_every_epoch),
        "resume_from_checkpoint": str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
        "init_from_model_checkpoint": str(args.init_from_model_checkpoint) if args.init_from_model_checkpoint else None,
        "init_from_model_checkpoint_report": init_from_model_checkpoint_report,
        "eval_only": bool(args.eval_only),
        "cost_normalization_config": str(args.cost_normalization_config) if args.cost_normalization_config else None,
    }
    if accelerator.is_main_process:
        write_json(output_dir / "train_config.json", config)

    history: list[dict[str, Any]] = []
    best_eval_loss = float("inf")
    best_payload: dict[str, Any] | None = None
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
            best_eval_loss = float(state.get("best_eval_loss", best_eval_loss))
        else:
            try:
                start_epoch = int(str(resume_from_checkpoint.name).split("_")[-1]) + 1
            except ValueError:
                start_epoch = 0
        if accelerator.is_main_process:
            print(f"Resumed from {resume_from_checkpoint}; starting at epoch {start_epoch}", flush=True)

    if bool(args.eval_only):
        if resume_from_checkpoint is None:
            raise ValueError("--eval-only requires --resume-from-checkpoint")
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
            route_idxs=target_route_idxs,
            cost_route_idxs=cost_route_idxs,
            cost_route_labels=cost_route_labels,
            objective=str(args.objective),
            predict_costs=bool(args.predict_costs),
            predict_zero_reward_failure=bool(args.predict_zero_reward_failure),
            class_weights=class_weights_tensor,
            hierarchical_any_success_weight=float(args.hierarchical_any_success_weight),
            hierarchical_route_weight=float(args.hierarchical_route_weight),
            cost_target_normalization=str(args.cost_target_normalization),
            cost_target_mean=cost_target_mean_tensor,
            cost_target_std=cost_target_std_tensor,
        )
        if accelerator.is_main_process:
            best_eval_loss = float(eval_loss)
            best_payload = {
                "epoch": int(start_epoch),
                "prediction_rows": pred_rows,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_cost_true": y_cost_true,
                "y_cost_pred": y_cost_pred,
                "y_zero_reward_true": y_zero_reward_true,
                "y_zero_reward_pred": y_zero_reward_pred,
            }
            history.append({"epoch": int(start_epoch), "train_loss": math.nan, "eval_loss": float(eval_loss), "eval_only": True})
        accelerator.wait_for_everyone()
    else:
        for epoch in range(start_epoch, int(args.num_epochs)):
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
                        output_token_targets_log=_normalize_cost_targets(
                            batch["output_token_targets_log"].float(),
                            cost_target_mean_tensor,
                            cost_target_std_tensor,
                            str(args.cost_target_normalization),
                        ),
                        zero_reward_targets=batch["zero_reward_targets"].float(),
                        class_targets=batch["class_targets"],
                        class_weights=class_weights_tensor,
                        objective=str(args.objective),
                        hierarchical_any_success_weight=float(args.hierarchical_any_success_weight),
                        hierarchical_route_weight=float(args.hierarchical_route_weight),
                        reward_mse_weight=float(args.reward_mse_weight),
                        reward_bce_weight=float(args.reward_bce_weight),
                        delta_aux_weight=float(args.delta_aux_weight),
                        delta_aux_huber_delta=float(args.delta_aux_huber_delta),
                        predict_costs=bool(args.predict_costs),
                        cost_mse_weight=float(args.cost_mse_weight),
                        cost_delta_aux_weight=float(args.cost_delta_aux_weight),
                        predict_zero_reward_failure=bool(args.predict_zero_reward_failure),
                        zero_reward_bce_weight=float(args.zero_reward_bce_weight),
                        ranking_aux_weight=float(args.ranking_aux_weight),
                        discrim_upweight=float(args.discrim_upweight),
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
                route_idxs=target_route_idxs,
                cost_route_idxs=cost_route_idxs,
                cost_route_labels=cost_route_labels,
                objective=str(args.objective),
                predict_costs=bool(args.predict_costs),
                predict_zero_reward_failure=bool(args.predict_zero_reward_failure),
                class_weights=class_weights_tensor,
                hierarchical_any_success_weight=float(args.hierarchical_any_success_weight),
                hierarchical_route_weight=float(args.hierarchical_route_weight),
                cost_target_normalization=str(args.cost_target_normalization),
                cost_target_mean=cost_target_mean_tensor,
                cost_target_std=cost_target_std_tensor,
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
                            "best_eval_loss": float(best_eval_loss),
                            "num_epochs": int(args.num_epochs),
                        },
                    )
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
    if str(args.objective) == "joint_outcome_2route":
        classifier_metrics = _compute_joint_outcome_metrics(pred_rows, joint_outcome_class_labels)
    else:
        classifier_metrics = _compute_route_classifier_metrics(
            y_true,
            y_pred,
            route_labels,
            class_target_mode=str(args.class_target_mode),
            class_success_threshold=float(args.class_success_threshold),
        )
    eval_rows_for_utility = _subset_route_stats_for_utility(eval_rows_source, target_route_idxs)
    if append_abstain_class:
        abstain_eval_rows = []
        for row in eval_rows_for_utility:
            new_row = dict(row)
            for key in ("performance_targets", "route_prompt_tokens", "route_output_tokens"):
                values = list(new_row.get(key) or [])
                values.append(0.0)
                new_row[key] = values
            abstain_eval_rows.append(new_row)
        eval_rows_for_utility = abstain_eval_rows
    utility_report = _compute_utility_report(pred_rows, eval_rows_for_utility, route_labels, DEFAULT_UTILITY_LAMBDAS)
    cost_route_labels = list(cost_route_labels) if args.predict_costs else []
    cost_metrics = _compute_output_token_metrics(y_cost_true, y_cost_pred, cost_route_labels) if args.predict_costs else []
    zero_reward_failure_metrics = (
        _compute_zero_reward_failure_metrics(y_zero_reward_true, y_zero_reward_pred, route_labels)
        if args.predict_zero_reward_failure
        else []
    )
    predicted_cost_utility_report = (
        _compute_predicted_cost_utility_report(
            pred_rows,
            eval_rows_for_utility,
            route_labels,
            DEFAULT_UTILITY_LAMBDAS,
            cost_route_idxs=local_cost_route_idxs,
            reported_cost_route_idxs=cost_route_idxs,
            reported_cost_route_labels=cost_route_labels,
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
        "oracle_match_rate",
        "oracle_match_rate_margin_gt_0_05",
        "oracle_match_n_margin_gt_0_05",
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

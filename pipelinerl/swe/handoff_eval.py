import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

import torch
from transformers import AutoTokenizer

from pipelinerl.finetune.value_model import AutoModelForCausalLMWithValueHead

logger = logging.getLogger(__name__)


@dataclass
class HandoffRecord:
    policy_score: float | None
    expert_score: float | None
    policy_reward: float | None
    expert_reward: float | None


class ValueScorer:
    def __init__(self, model_path: str, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
        if self.device.type == "cuda":
            self.model = self.model.to(device=self.device, dtype=torch.bfloat16)
        else:
            self.model = self.model.to(device=self.device)
        self.model.eval()

    @staticmethod
    def _response_indices(labels: list[int]) -> list[int]:
        return [idx for idx, value in enumerate(labels) if value != -100]

    def score_training_text(self, training_text) -> dict[str, float | None]:
        if not training_text.input_ids:
            return {
                "policy_value_mean": None,
                "policy_value_last": None,
                "expert_value_prompt_last": None,
            }

        input_ids = torch.tensor([training_text.input_ids], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        values = outputs.value.squeeze(0)
        expert_values = outputs.expert_value.squeeze(0) if outputs.expert_value is not None else None

        response_indices = self._response_indices(training_text.labels)
        policy_value_mean = None
        policy_value_last = None
        expert_value_prompt_last = None

        if response_indices:
            response_values = values[response_indices]
            policy_value_mean = response_values.mean().item()
            policy_value_last = response_values[-1].item()
            first_output_idx = response_indices[0]
            prompt_last_idx = first_output_idx - 1
            if expert_values is not None and prompt_last_idx >= 0:
                expert_value_prompt_last = expert_values[prompt_last_idx].item()
        elif expert_values is not None:
            prompt_last_idx = len(training_text.labels) - 1
            if prompt_last_idx >= 0:
                expert_value_prompt_last = expert_values[prompt_last_idx].item()

        return {
            "policy_value_mean": policy_value_mean,
            "policy_value_last": policy_value_last,
            "expert_value_prompt_last": expert_value_prompt_last,
        }


def compute_handoff_curve(
    records: Iterable[HandoffRecord],
    thresholds: List[float],
    handoff_margin: float,
) -> List[dict[str, Any]]:
    rows = []
    records_list = list(records)
    if not records_list:
        return rows

    valid_records = [
        rec for rec in records_list if rec.policy_score is not None and rec.expert_score is not None
    ]
    if not valid_records:
        return rows

    expert_reward_available = all(rec.expert_reward is not None for rec in valid_records)

    for threshold in thresholds:
        total_reward = 0.0
        attempted = 0
        handed_off = 0
        total = len(valid_records)

        for rec in valid_records:
            attempt = max(rec.policy_score, rec.expert_score) >= threshold
            handoff = rec.expert_score > rec.policy_score + handoff_margin

            if attempt:
                attempted += 1
                if handoff:
                    handed_off += 1
                    if expert_reward_available:
                        total_reward += rec.expert_reward or 0.0
                else:
                    total_reward += rec.policy_reward or 0.0

        avg_reward = total_reward / total if expert_reward_available else None
        rows.append(
            {
                "threshold": threshold,
                "avg_reward": avg_reward,
                "attempted": attempted,
                "handoffs": handed_off,
                "attempt_rate": attempted / total if total else 0.0,
                "handoff_rate": handed_off / total if total else 0.0,
                "expert_reward_available": expert_reward_available,
            }
        )
    return rows


def summarize_handoff_curve(curve: List[dict[str, Any]]) -> dict[str, Any]:
    if not curve:
        return {}
    valid = [row for row in curve if row.get("avg_reward") is not None]
    if not valid:
        return {
            "best_threshold": None,
            "best_avg_reward": None,
        }
    best = max(valid, key=lambda row: row["avg_reward"])
    return {
        "best_threshold": best["threshold"],
        "best_avg_reward": best["avg_reward"],
    }




def write_handoff_curve(path: Path, curve: List[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(curve, handle, indent=2)

import logging
import os
from functools import partial
from typing import Any
from pydantic import BaseModel, Field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import PreTrainedModel
from pipelinerl.finetune.types import PipelineBatchEncoding
from pipelinerl.finetune.rl.utils import per_segment_sums

from .utils import (
    sum_sum,
    mean_sum,
    replace_dataset_column,
)

# FIXME: remove a warnings, but might be worth investigating
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__name__)

RL_DATA_COLUMNS = [
    "overflow",
    "group_tokens",
    "num_labels",
    "rewards",
    "advantages",
    "old_logprobs",
    "ref_logprobs",
    "performance_targets",
]


class RLConfig(BaseModel):
    policy_loss: str = Field(
        default="ppo",
        description="Policy Loss to use for RL",
        choices=["ppo", "reinforce", "gspo"],
    )
    use_advantages: bool = Field(
        default=True,
        description="Use advantages instead of rewards to compute the loss",
    )
    epsilon_low: float = Field(default=0.2, description="Lower clip parameter for ratio of log probs")
    epsilon_high: float = Field(default=0.2, description="Upper clip parameter for ratio of log probs")
    batch_size: int = Field(default=0, description="Batch size is required for normalization")
    reward_minus_kl_coef: float = Field(
        default=0.0,
        # https://arxiv.org/abs/2402.14740
        description="Implicit KL coefficient similar to the RLOO paper",
    )
    kl_coef: float = Field(
        default=0.1,
        description="KL penalty coefficient with reference policy",
    )
    final_kl_coef: float = Field(
        default=0.1,
        description="Final KL penalty coefficient value",
    )
    entropy_bonus: float = Field(
        default=0.0,
        description="Entropy bonus coefficient",
    )
    final_entropy_bonus: float = Field(
        default=0.0,
        description="Final entropy bonus value",
    )
    relu_log_p_weights: bool = Field(
        default=False,
        description="ReLU the weights before updating the model",
    )
    clamp_log_ratio_ref_new_value: float = Field(
        default=10,
        description="Clamp the log ratio ref new value",
    )
    divide_advantage_by_std: bool = Field(
        default=True,
        description="Normalize the advantage by the standard deviation",
    )
    overlong_filtering: bool = Field(default=False, description="Filter out sequence that do not have eos_token_id")
    group_normalization: bool = Field(
        default=False,
        description="Divide the weight of each sequence by the (average) number of tokens in the group",
    )
    temperature: float = Field(
        default=1.0,
        description="Temperature for the training log probs",
    )
    filter_zero_advantage_groups: bool = Field(
        default=False,
        description="Filter out groups where all advantages are zero during preprocessing",
    )
    value_loss_coef: float = Field(
        default=0.0,
        description="Coefficient for the value loss in the final loss",
    )
    performance_value_loss_coef: float = Field(
        default=0.0,
        description="Coefficient for the performance value loss in the final loss",
    )


def make_rl_data_callback(args, current_dir, rl_config, model):
    if rl_config:
        populate_rl_data_ = partial(
            populate_rl_data,
            config=rl_config,
        )
    else:
        populate_rl_data_ = None
    return populate_rl_data_


def linear_decay_coef(current_step: int, max_step: int, initial_coef: float, final_coef: float) -> float:
    """
    Linearly decay the coefficient from initial to final value over the course of training.

    Args:
        current_step (int): Current step in the training
        max_step (int): Maximum number of steps in the training
        initial_coef (float): Initial coefficient value
        final_coef (float): Final coefficient value

    Returns:
        float: Linearly decayed coefficient value

    """
    return initial_coef + (final_coef - initial_coef) * current_step / max_step


def build_last_prompt_token_mask(
    labels: torch.Tensor,
    segments: list | None,
) -> torch.Tensor:
    """
    Build a mask that selects the last prompt token (just before the first output token)
    for each sequence. Works for both packed and unpacked batches.
    """
    mask = torch.zeros_like(labels, dtype=torch.bool)

    def mark_last_prompt(row_idx: int, start: int, end: int) -> None:
        row_labels = labels[row_idx, start:end]
        output_idx = (row_labels != -100).nonzero(as_tuple=True)[0]
        if output_idx.numel() == 0:
            return
        last_prompt = start + int(output_idx[0].item()) - 1
        if last_prompt >= start:
            mask[row_idx, last_prompt] = True

    if segments:
        # packed batches have a single row and explicit segments
        for start, end in segments:
            mark_last_prompt(0, int(start), int(end))
    else:
        for row_idx in range(labels.shape[0]):
            mark_last_prompt(row_idx, 0, labels.shape[1])

    return mask


def build_local_segments_from_segment_ids(segment_ids: torch.Tensor) -> list[tuple[int, int]]:
    """Derive local contiguous token segments from packed segment ids."""
    if segment_ids.dim() != 2 or segment_ids.shape[0] != 1:
        raise ValueError(f"Expected segment_ids shaped [1, L], got {tuple(segment_ids.shape)}")
    if segment_ids.shape[1] == 0:
        return []
    ids = segment_ids[0]
    change_points = (ids[1:] != ids[:-1]).nonzero(as_tuple=True)[0] + 1
    boundaries = torch.cat(
        [
            torch.tensor([0], device=ids.device, dtype=torch.long),
            change_points.to(dtype=torch.long),
            torch.tensor([ids.shape[0]], device=ids.device, dtype=torch.long),
        ]
    )
    return [(int(start.item()), int(end.item())) for start, end in zip(boundaries[:-1], boundaries[1:])]


def rl_step(
    model: PreTrainedModel,
    batch: PipelineBatchEncoding,
    current_step: int,
    max_step: int,
    config: RLConfig,
    seq_parallel_group=None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Perform a single RL step on the model using the given batch and config.
    Handles both packed and unpacked sequences.

    Args:
        model (PreTrainedModel): The model to train
        batch (PipelineBatchEncoding): Batch of data containing rewards, advantages, masks, input_ids etc.
        current_step (int): Current training step
        max_step (int): Maximum number of training steps
        config (RLConfig): Configuration for the RL training

    Returns:
        tuple[torch.Tensor, dict[str, float]]: Loss tensor and metrics dictionary
    """
    # pre-compute masks
    masks = batch.labels != -100
    masks_shifted = masks[:, 1:]

    has_value_head = hasattr(model, 'value_head') and config.value_loss_coef > 0
    has_performance_value_head = hasattr(model, 'performance_value_head') and config.performance_value_loss_coef > 0

    # if we have position_ids, we are packing
    if batch.is_packed:
        if batch.segment_ids is not None:
            segments = build_local_segments_from_segment_ids(batch.segment_ids)
            num_sequences = len(segments)
            assert num_sequences > 0, "No sequences found in packed batch"
            if (
                batch.seq_boundaries is not None
                and int(batch.seq_boundaries[-1].item()) != int(batch.position_ids.shape[1])
            ):
                logger.debug(
                    "Packed batch uses global seq_boundaries (last=%s) with local shard length=%s; using local segments from segment_ids",
                    int(batch.seq_boundaries[-1].item()),
                    int(batch.position_ids.shape[1]),
                )
        elif batch.seq_boundaries is not None:
            seq_boundaries = batch.seq_boundaries.to(device=batch.position_ids.device)
            num_sequences = seq_boundaries.shape[0] - 1
            assert num_sequences > 0, "No sequences found in packed batch"
            if seq_boundaries[-1] != batch.position_ids.shape[1]:
                raise ValueError(
                    "Packed batch is missing segment_ids and has incompatible seq_boundaries for this shard: "
                    f"last={int(seq_boundaries[-1].item())} shard_len={int(batch.position_ids.shape[1])}"
                )
            segments = list(zip(seq_boundaries[:-1], seq_boundaries[1:]))
        else:
            position_ids = batch.position_ids[0]
            is_sequence_start = position_ids == 0
            # For computing the loss we will consider the first token the beginning of the sequence,
            # even if currently we are in the middle of a sequence.
            is_sequence_start[0] = True
            sequence_starts = torch.where(is_sequence_start)[0]
            seq_boundaries = torch.cat(
                [
                    sequence_starts,
                    torch.tensor([position_ids.shape[0]], device=position_ids.device),
                ]
            )
            num_sequences = len(sequence_starts)

            # ensure we have valid sequence boundaries
            assert num_sequences > 0, "No sequences found in packed batch"
            assert seq_boundaries[-1] == position_ids.shape[0], "Sequence boundaries don't match input length"

            # pre-compute segment boundaries
            segments = list(zip(seq_boundaries[:-1], seq_boundaries[1:]))
    else:
        num_sequences = masks.shape[0]
        segments = None

    model_inputs = {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "labels": batch.labels,
    }
    if batch.is_packed:
        model_inputs["position_ids"] = batch.position_ids
    
    # Add visual features if present (for multimodal models)
    if hasattr(batch, 'pixel_values') and batch.pixel_values is not None:
        model_inputs["pixel_values"] = batch.pixel_values
    if hasattr(batch, 'image_grid_thw') and batch.image_grid_thw is not None:
        model_inputs["image_grid_thw"] = batch.image_grid_thw #torch.tensor(.reshape((1, 3))
    
    outputs = model(**model_inputs)

    # compute log probs and entropy
    logits = outputs.logits[:, :-1, :]
    logits = logits / config.temperature
    logprobs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * logprobs).sum(dim=-1)
    del logits, probs
    
    # get log probs for actual tokens
    new_logprobs = torch.gather(logprobs, dim=2, index=batch.input_ids[:, 1:].unsqueeze(2)).squeeze(2)
    assert torch.isfinite(new_logprobs).all(), f"new_logprobs is not finite: {new_logprobs}"
    del logprobs

    # get shifted values and compute ratios
    rewards = batch.rewards[:, 1:]
    ref_logprobs = batch.ref_logprobs[:, 1:]
    old_logprobs = batch.old_logprobs[:, 1:]
    group_tokens = batch.group_tokens[:, 1:]
    num_labels_in_seq = batch.num_labels[:, 1:] # sequence dependent normalization
    overflow = batch.overflow[:, 1:]

    if config.group_normalization:
        # assert that group_tokens is not zero
        assert (group_tokens > 0).all(), "group_tokens must be greater than zero for group normalization"
        tokens_weights = torch.ones_like(group_tokens) / group_tokens
    else:
        tokens_weights = torch.ones_like(group_tokens) / config.batch_size

    if config.overlong_filtering:
        # filter out sequences that do not have eos_token_id
        overflow = torch.tensor(overflow, device=overflow.device)
        tokens_weights = tokens_weights * (1 - overflow)

    assert new_logprobs.shape == ref_logprobs.shape

    log_ratio_new_old = new_logprobs - old_logprobs
    ratio_new_old = torch.exp(log_ratio_new_old)
    log_ratio_ref_new = ref_logprobs - new_logprobs
    assert torch.isfinite(log_ratio_ref_new).all(), f"log_ratio_ref_new is not finite: {log_ratio_ref_new}"

    if has_value_head:
        # Get value predictions if available
        value_predictions = outputs.value[:, :-1] # no target for the last token 
        # Compute value-based advantages: A(s,a) = MC_return - V(s)
        # where MC_return is the Monte Carlo return (rewards) and V(s) is the value prediction
        #FIXME: if this works better it should be a config
        #advantages = rewards - torch.clamp(value_predictions, 0, 1)
        advantages = rewards - value_predictions
    else:
        advantages = batch.advantages[:, 1:]

    log_p_weights = advantages.detach() if config.use_advantages else rewards
    if config.relu_log_p_weights:
        log_p_weights = torch.clamp(log_p_weights, min=0)

    clamp_log_ratio_ref_new_indicators = torch.abs(log_ratio_ref_new) > config.clamp_log_ratio_ref_new_value

    log_ratio_ref_new_clamp = torch.clamp(
        log_ratio_ref_new,
        min=-config.clamp_log_ratio_ref_new_value,
        max=config.clamp_log_ratio_ref_new_value,
    )

    approx_kl = torch.exp(log_ratio_ref_new_clamp) - log_ratio_ref_new_clamp - 1  # Schulman KL approx
    approx_kl_new_old = torch.exp(log_ratio_new_old) - log_ratio_new_old - 1  # Schulman KL approx

    assert torch.isfinite(approx_kl).all(), f"approx_kl is not finite: {approx_kl}"
    entropy_bonus_coef = linear_decay_coef(current_step, max_step, config.entropy_bonus, config.final_entropy_bonus)
    kl_coef = linear_decay_coef(current_step, max_step, config.kl_coef, config.final_kl_coef)

    # compute algorithm-specific losses
    policy_loss_total = None
    match config.policy_loss:
        case "ppo":
            surr1 = ratio_new_old * log_p_weights
            clamped_ratio = torch.clamp(ratio_new_old, 1 - config.epsilon_low, 1 + config.epsilon_high)
            clamp_log_ratio_new_old_indicators = clamped_ratio != ratio_new_old
            surr2 = clamped_ratio * log_p_weights
            policy_loss = torch.min(surr1, surr2)
        case "reinforce":
            surr1 = torch.zeros_like(ratio_new_old)
            surr2 = torch.zeros_like(ratio_new_old)
            clamp_log_ratio_new_old_indicators = ratio_new_old > 1 + config.epsilon_high
            ratio_new_old = torch.clamp(ratio_new_old, 0, 1 + config.epsilon_high)
            policy_loss = new_logprobs * log_p_weights * ratio_new_old.detach()
        case "gspo":
            if segments is None:
                raise ValueError("GSPO loss requires packed sequences with segments")
            lrn_sum, adv_sum, tok_count = per_segment_sums(
                batch.segment_ids,
                masks_shifted,
                log_ratio_new_old,
                advantages,
                seq_parallel_group=seq_parallel_group,
            )
            group_ratio_new_old = torch.exp(lrn_sum / tok_count.clamp(min=1e-6)).unsqueeze(1).unsqueeze(2)
            group_advantages_t = (adv_sum / tok_count.clamp(min=1e-6)).unsqueeze(1).unsqueeze(2).detach()
            zero_weights = torch.zeros_like(tokens_weights)
            weight_sum, _, _ = per_segment_sums(
                batch.segment_ids,
                masks_shifted,
                tokens_weights,
                zero_weights,
                seq_parallel_group=seq_parallel_group,
            )
            valid_mask = (tok_count > 0) & (weight_sum > 0)
            valid_mask_3d = valid_mask.unsqueeze(1).unsqueeze(2)
            surr1 = group_ratio_new_old * group_advantages_t
            clamped_group_ratio = torch.clamp(group_ratio_new_old, 1 - config.epsilon_low, 1 + config.epsilon_high)
            clamp_log_ratio_new_old_indicators = (clamped_group_ratio != group_ratio_new_old) & valid_mask_3d
            surr2 = clamped_group_ratio * group_advantages_t
            sequence_weights = weight_sum.unsqueeze(1).unsqueeze(2)
            if batch.sentinel or surr1.numel() == 0:
                policy_loss_total = new_logprobs[..., :1].sum() * 0.0
            else:
                mask_float = valid_mask_3d.to(dtype=surr1.dtype)
                min_terms = torch.min(surr1, surr2) * mask_float * sequence_weights
                policy_loss_total = -min_terms.sum()
            expanded_indicators = torch.zeros_like(masks_shifted, dtype=torch.float)
            if batch.segment_ids is None:
                raise ValueError("segment_ids must be provided for GSPO indicator expansion")
            segment_ids_shifted = batch.segment_ids[:, 1:].to(device=expanded_indicators.device, dtype=torch.long)
            if segment_ids_shifted.numel() > 0:
                segment_indicators = clamp_log_ratio_new_old_indicators.flatten().to(dtype=expanded_indicators.dtype)
                if int(segment_ids_shifted.min().item()) < 0 or int(segment_ids_shifted.max().item()) >= int(
                    segment_indicators.shape[0]
                ):
                    raise IndexError(
                        "segment_ids out of bounds for GSPO indicators: "
                        f"min={int(segment_ids_shifted.min().item())} max={int(segment_ids_shifted.max().item())} "
                        f"num_indicators={int(segment_indicators.shape[0])}"
                    )
                expanded_indicators[:, : segment_ids_shifted.shape[1]] = segment_indicators[segment_ids_shifted]
            clamp_log_ratio_new_old_indicators = expanded_indicators
        case _:
            raise ValueError(f"Unknown algorithm {config.policy_loss}")

    # combine loss components
    if config.policy_loss != "gspo":
        loss = policy_loss - kl_coef * approx_kl + entropy_bonus_coef * entropy  # 1 x (BxL) x 1
        assert loss.shape == tokens_weights.shape, (
            f"Loss shape {loss.shape} does not match example weights shape {tokens_weights.shape}"
        )
        loss = loss * tokens_weights  # 1 x (BxL) x 1

        policy_loss_total = -sum_sum(loss, masks_shifted, segments)

    if has_value_head:
        # Get the value predictions
        values = outputs.value
        # Use the already extracted and shifted rewards as value labels
        value_labels = rewards  # This is already shifted (from line 216)
        values = values[:, :-1]
        values_labels = value_labels
        assert values.shape == tokens_weights.shape, (
            f"Values shape {values.shape} does not match example weights shape {tokens_weights.shape}"
        )
        value_loss = 0.5 * torch.square(values - values_labels) * tokens_weights
        value_loss = sum_sum(value_loss, masks_shifted, segments) 
        
        # Combine policy loss and value loss
        final_loss = policy_loss_total + config.value_loss_coef * value_loss
    else:
        final_loss = policy_loss_total

    if has_performance_value_head:
        performance_values = outputs.performance_value
        performance_targets = batch.performance_targets if hasattr(batch, "performance_targets") else None
        if performance_targets is None:
            raise ValueError("performance_targets missing from batch while performance_value_head is present")

        # Train performance head on the last prompt token only (prompt-only estimate).
        performance_prompt_mask = build_last_prompt_token_mask(batch.labels, segments)

        if config.group_normalization:
            prompt_tokens_weights = torch.ones_like(batch.group_tokens) / batch.group_tokens
        else:
            prompt_tokens_weights = torch.ones_like(batch.group_tokens) / config.batch_size

        if config.overlong_filtering:
            overflow_full = batch.overflow
            prompt_tokens_weights = prompt_tokens_weights * (1 - overflow_full)

        if performance_targets.dim() != 2:
            raise ValueError(f"performance_targets should be [num_sequences, dim], got {performance_targets.shape}")

        if performance_prompt_mask.any():
            if batch.is_packed:
                if batch.segment_ids is None:
                    raise ValueError("segment_ids missing from packed batch for performance head alignment")
                prompt_segment_ids = batch.segment_ids[performance_prompt_mask].to(
                    device=performance_targets.device, dtype=torch.long
                )
                if int(prompt_segment_ids.min().item()) < 0 or int(prompt_segment_ids.max().item()) >= int(
                    performance_targets.shape[0]
                ):
                    raise IndexError(
                        "Prompt segment ids out of bounds for performance targets: "
                        f"min={int(prompt_segment_ids.min().item())} max={int(prompt_segment_ids.max().item())} "
                        f"num_targets={int(performance_targets.shape[0])}"
                    )
                filtered_targets = performance_targets[prompt_segment_ids]
            else:
                seq_has_output = performance_prompt_mask.any(dim=1)
                filtered_targets = performance_targets[seq_has_output]
            prompt_predictions = performance_values[performance_prompt_mask]
            if prompt_predictions.shape != filtered_targets.shape:
                raise ValueError(
                    f"Prompt predictions shape {prompt_predictions.shape} does not match targets shape {filtered_targets.shape}"
                )

            per_seq_mse = torch.square(prompt_predictions - filtered_targets).mean(dim=-1)
            if batch.is_packed:
                prompt_weights = prompt_tokens_weights[performance_prompt_mask]
            else:
                prompt_weights = (prompt_tokens_weights * performance_prompt_mask).sum(dim=-1)
                prompt_weights = prompt_weights[seq_has_output]

            performance_value_loss = 0.5 * (per_seq_mse * prompt_weights).sum()
        else:
            prompt_predictions = performance_values.new_empty((0, performance_values.shape[-1]))
            performance_value_loss = performance_values.new_tensor(0.0)
        final_loss = final_loss + config.performance_value_loss_coef * performance_value_loss

    # ensure loss is valid
    assert torch.isfinite(final_loss), f"Non-finite loss detected: {final_loss}"

    if int(masks_shifted.sum().item()) == 0:
        stats_no_labels = {
            "input_size": float(batch.input_ids.numel()),
        }
        return final_loss, stats_no_labels

    # Metric aggregation behavior:
    # 1. loss: pre-multiplied by token_weights, reported as sum
    # 2. min/max values: computed across entire batch
    # 3. other statistics: averaged per sequence, then averaged across batch
    stats = {
        "loss": final_loss.item(),
        "max_loss": final_loss.item(),
        "min_loss": final_loss.item(),
        "reward": sum_sum(rewards / num_labels_in_seq, masks_shifted, segments).item(),
        "max_reward": rewards[masks_shifted].max().item(),
        "min_reward": rewards[masks_shifted].min().item(),
        "entropy": sum_sum(entropy / num_labels_in_seq, masks_shifted, segments).item(),
        "old_logprobs": sum_sum(old_logprobs / num_labels_in_seq, masks_shifted, segments).item(),
        "new_logprobs": sum_sum(new_logprobs / num_labels_in_seq, masks_shifted, segments).item(),
        "ref_logprobs": sum_sum(ref_logprobs / num_labels_in_seq, masks_shifted, segments).item(),
        "advantage": sum_sum(advantages / num_labels_in_seq, masks_shifted, segments).item(),
        "max_advantage": advantages[masks_shifted].max().item(),
        "min_advantage": advantages[masks_shifted].min().item(),
        "kl": sum_sum(approx_kl / num_labels_in_seq, masks_shifted, segments).item(),
        "kl_new_old": sum_sum(approx_kl_new_old / num_labels_in_seq, masks_shifted, segments).item(),
        "max_kl": approx_kl[masks_shifted].max().item(),
        "min_kl": approx_kl[masks_shifted].min().item(),
        "ratio_new_old": sum_sum(ratio_new_old / num_labels_in_seq, masks_shifted, segments).item(),
        "ratio_new_old_sum": sum_sum(ratio_new_old, masks_shifted, segments).item(),
        "ratio_new_old_squared_sum": sum_sum(  # useful to estimate the ESS
            ratio_new_old * ratio_new_old, masks_shifted, segments
        ).item(),
        "ratio_ref_new": sum_sum(torch.exp(log_ratio_ref_new) / num_labels_in_seq, masks_shifted, segments).item(),
        "ratio_ref_old": sum_sum(torch.exp(ref_logprobs - old_logprobs) / num_labels_in_seq, masks_shifted, segments).item(),
        "clamp_log_ratio_ref_new_indicator": sum_sum(
            clamp_log_ratio_ref_new_indicators / num_labels_in_seq, masks_shifted, segments
        ).item(),
        "clamp_log_ratio_new_old_indicator": sum_sum(
            clamp_log_ratio_new_old_indicators / num_labels_in_seq, masks_shifted, segments
        ).item(),
        "token_weight": sum_sum(tokens_weights / num_labels_in_seq, masks_shifted, segments).item(),
        "max_token_weight": tokens_weights[masks_shifted].max().item(),
        "min_token_weight": tokens_weights[masks_shifted].min().item(),
        "kl_coef": num_sequences * kl_coef,
        "entropy_bonus_coef": num_sequences * entropy_bonus_coef,
        "num_output_tokens_sum": masks_shifted.sum().item(),
        "input_size": batch.input_ids.numel(), 
    }

    if has_value_head:
        stats["value_mean"] = sum_sum(value_predictions / num_labels_in_seq, masks_shifted, segments).item()
        stats["value_max"] = value_predictions[masks_shifted].max().item() if masks_shifted.any() else 0.0
        stats["value_min"] = value_predictions[masks_shifted].min().item() if masks_shifted.any() else 0.0
        stats["value_loss"] = value_loss.item()
        stats["value_mse"] = sum_sum(
            torch.square(value_predictions - value_labels) / num_labels_in_seq, masks_shifted, segments
        ).item()
    if has_performance_value_head:
        stats["performance_value_loss"] = performance_value_loss.item()
        stats["performance_value_mean"] = prompt_predictions[:, 0].mean().item() if prompt_predictions.numel() else 0.0
        stats["performance_value_max"] = prompt_predictions.max().item() if prompt_predictions.numel() else 0.0
        stats["performance_value_min"] = prompt_predictions.min().item() if prompt_predictions.numel() else 0.0

    return final_loss, stats


def populate_rl_data(dataset: list[dict[str, Any]], eos_token_id: int, config: RLConfig) -> list[dict[str, Any]]:
    """Populate RL-specific columns (advantages, overflow, num_labels) using a leave-one-out baseline."""
    # Convert to pandas for processing
    df_init = pd.DataFrame(dataset)
    df_init["group_id"] = df_init["group_id"].astype(str) + "_" + df_init["step_index"].astype(str)
    assert isinstance(df_init, pd.DataFrame)

    # Step 1: calculate group-level statistics
    df_stats = df_init[["group_id", "rollout_index", "step_index"]].copy()
    df_stats["num_tokens"] = df_init["input_ids"].apply(len)
    # We assume that rewards for all tokens are the same
    df_stats["rollout_reward"] = df_init["rewards"].apply(lambda x: x[0])

    # Check that the reward is the same for each step in the rollout
    assert df_stats.groupby(["group_id", "rollout_index"])["rollout_reward"].nunique().max() == 1
    # Only keep step_index == 0
    #df_stats = df_stats[df_stats["step_index"] == 0].drop(columns=["step_index"])
    df_grouped = (
        df_stats.groupby("group_id")
        .agg(
            rollout_reward_sum=("rollout_reward", "sum"),
            rollout_reward_count=("rollout_reward", "count"),
            rollout_reward_std=("rollout_reward", "std"),
            group_tokens=("num_tokens", "mean"),
        )
        .reset_index()
    )
    assert df_grouped.columns.tolist() == [
        "group_id",
        "rollout_reward_sum",
        "rollout_reward_count",
        "rollout_reward_std",
        "group_tokens",
    ]

    # Debug group variance issues
    logger.info(f"Processing {len(df_grouped)} groups for RL data")
    
    # Analyze variance distribution
    zero_var_groups = df_grouped[df_grouped['rollout_reward_std'] < 1e-6]
    low_var_groups = df_grouped[(df_grouped['rollout_reward_std'] >= 1e-6) & (df_grouped['rollout_reward_std'] < 0.1)]
    high_var_groups = df_grouped[df_grouped['rollout_reward_std'] >= 0.1]
    
    logger.info(f"Group variance distribution:")
    logger.info(f"  Zero variance: {len(zero_var_groups)}/{len(df_grouped)} ({len(zero_var_groups)/len(df_grouped)*100:.1f}%)")
    logger.info(f"  Low variance: {len(low_var_groups)}/{len(df_grouped)} ({len(low_var_groups)/len(df_grouped)*100:.1f}%)")
    logger.info(f"  High variance: {len(high_var_groups)}/{len(df_grouped)} ({len(high_var_groups)/len(df_grouped)*100:.1f}%)")
    
    # Sample problematic groups for detailed analysis
    if len(zero_var_groups) > 0:
        logger.warning(f"Found {len(zero_var_groups)} groups with zero variance!")
        sample_zero_var = zero_var_groups.head(3)
        for _, row in sample_zero_var.iterrows():
            group_id = row['group_id']
            # Get individual rollout rewards for this group
            group_rewards = df_stats[df_stats['group_id'] == group_id]['rollout_reward'].tolist()
            logger.warning(f"  Group {group_id}: mean={row['rollout_reward_mean']:.6f}, "
                         f"std={row['rollout_reward_std']:.6f}, rewards={group_rewards}")
    
    # Show examples of healthy groups
    if len(high_var_groups) > 0:
        sample_good = high_var_groups.head(2)
        for _, row in sample_good.iterrows():
            group_id = row['group_id']
            group_rewards = df_stats[df_stats['group_id'] == group_id]['rollout_reward'].tolist()
            logger.info(f"  Good group {group_id}: mean={row['rollout_reward_mean']:.3f}, "
                       f"std={row['rollout_reward_std']:.3f}, rewards={group_rewards}")

    # Step 2: calculate advantages for each sample
    df_advantages = pd.merge(
        df_init[["group_id", "rollout_index", "step_index", "rewards"]],
        df_grouped,
        on="group_id",
        how="left"
    )
    assert len(df_advantages) == len(df_init)
    
    def calculate_advantages(row):
        rewards = row["rewards"]
        group_sum = row["rollout_reward_sum"]
        group_count = row["rollout_reward_count"]
        current_reward = rewards[0]
        if group_count > 1:
            loo_mean = (group_sum - current_reward) / (group_count - 1)
        else:
            loo_mean = current_reward
        std = row["rollout_reward_std"]
        if config.divide_advantage_by_std:
            return [(r - loo_mean) / (np.nan_to_num(std) + 1e-4) for r in rewards]
        return [(r - loo_mean) for r in rewards]

    df_advantages["advantages"] = df_advantages.apply(calculate_advantages, axis=1)
    df_advantages = df_advantages.drop(
        columns=["rewards", "rollout_reward_sum", "rollout_reward_count", "rollout_reward_std"]
    )
    assert df_advantages.columns.tolist() == [
        "group_id",
        "rollout_index",
        "step_index",
        "group_tokens",
        "advantages",
    ]

    # Step 3: bring advantages and group level stats back to the main df
    df = df_init.drop(columns=["advantages", "group_tokens"])
    df = pd.merge(df, df_advantages, on=["group_id", "rollout_index", "step_index"], how="left")
    assert len(df) == len(df_init)

    # Step 4: make token-level overflow and mean group length information
    def _overflow_from_finish_reason(row):
        length = len(row["overflow"])
        finish_reason = row.get("finish_reason")
        if isinstance(finish_reason, str):
            finish_reason = finish_reason.strip().lower()
            if finish_reason == "length":
                return [1.0] * length
            if finish_reason in {"stop", "content_filter"}:
                return [0.0] * length
        if row.get("finished"):
            return [0.0] * length
        return [0.0] * length if eos_token_id in row["input_ids"] else [1.0] * length

    df["overflow"] = df.apply(_overflow_from_finish_reason, axis=1)
    df["group_tokens"] = df.apply(lambda row: [row["group_tokens"]] * len(row["input_ids"]), axis=1)
    df["num_labels"] = df.apply(
        lambda row: [sum(1 for label in row["labels"] if label != -100)] * len(row["input_ids"]), axis=1
    )

    # Step 5: move the results back to the dataset
    advantages_list = df["advantages"].tolist()
    group_tokens_list = df["group_tokens"].tolist()
    overflow_list = df["overflow"].tolist()
    num_labels_list = df["num_labels"].tolist()
    
    for i, entry in enumerate(dataset):
        entry["advantages"] = advantages_list[i]
        entry["group_tokens"] = group_tokens_list[i]
        entry["overflow"] = overflow_list[i]
        entry["num_labels"] = num_labels_list[i]
    
    return dataset


def prepare_rl_fields(
    encoding: dict[str, Any],
    reward: float,
    old_logprobs: list[float],
    ref_logprobs: list[float],
    expert_reward: float | None = None,
) -> dict[str, Any]:
    """
    Convert reward per agent step to reward per token and add returns and advantages placeholders
    """
    target_tokens = [token for token in encoding["labels"] if token != -100]
    assert len(target_tokens) == len(old_logprobs), (
        f"Target tokens: {len(target_tokens)}, old logprobs: {len(old_logprobs)}"
    )

    encoding["rewards"] = [reward] * len(encoding["labels"])
    encoding["advantages"] = [0.0] * len(encoding["labels"])  # place holder
    encoding["old_logprobs"] = [0] * (len(encoding["labels"]) - len(old_logprobs)) + old_logprobs
    encoding["ref_logprobs"] = [0] * (len(encoding["labels"]) - len(ref_logprobs)) + ref_logprobs
    encoding["overflow"] = [0] * len(encoding["labels"])  # place holder
    encoding["group_tokens"] = [0] * len(encoding["labels"])  # place holder
    encoding["num_labels"] = [1 if label != -100 else 0 for label in encoding["labels"]]  # count only output tokens
    performance_targets = encoding.get("performance_targets")
    if performance_targets is None:
        if expert_reward is None:
            performance_targets = [reward]
        else:
            performance_targets = [reward, expert_reward]
    encoding["performance_targets"] = performance_targets
    return encoding

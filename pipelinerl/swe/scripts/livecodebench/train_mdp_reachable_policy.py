#!/usr/bin/env python3
"""Train the full-execution failure-region policy with clean split roles.

The model is fitted on ``train``, checkpointed by calibration BCE, and touches
``test`` only once after the selected checkpoint has been restored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from pipelinerl.swe.scripts.offline_router.train_cot_abstention_predictor import _collate
from pipelinerl.swe.scripts.offline_router.train_qwen_embedding_router_baseline import (
    QwenEmbeddingRouter,
    _dtype_from_name,
)
from pipelinerl.swe.scripts.livecodebench.structured_policy import StructuredStatePolicy
from pipelinerl.swe.scripts.livecodebench.structured_state import (
    STATE_FEATURE_NAMES,
    STATE_FEATURE_VERSION,
)


HEADS = ["scout_next", "oss20_fresh", "oss120_fresh", "nothing"]


class PolicyDataset(Dataset):
    def __init__(
        self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int, *,
        require_state_features: bool, factorized: bool = False
    ) -> None:
        if factorized and rows and "failure_counts" not in rows[0]:
            raise ValueError(
                "Factorized training needs failure_counts/remaining_counts; rebuild the "
                "reachable dataset with the current builder"
            )
        self.rows: list[dict[str, Any]] = []
        for row in rows:
            # Full state text even when factorized: the decay is imposed by the loss
            # from the raw counts, so the encoder does not need a depth-free input, and
            # dropping the latest attempt's code and feedback throws away signal the
            # baseline gets. Reading problem text alone conflated "learned decay" with
            # "less input" and cost 0.048 AUC on the scout head.
            encoded = tokenizer(
                row["text"], add_special_tokens=True, truncation=True, max_length=max_length,
            )
            if not encoded.get("input_ids"):
                continue
            features = row.get("state_features")
            if require_state_features and features is None:
                raise ValueError("Structured policy requires state_features in every dataset row")
            if features is not None and len(features) != len(STATE_FEATURE_NAMES):
                raise ValueError(
                    f"Expected {len(STATE_FEATURE_NAMES)} structured state features, got {len(features)}"
                )
            self.rows.append({
                "row_idx": len(self.rows),
                "problem_id": row["problem_id"],
                **({
                    "failure_counts": [int(v) for v in row["failure_counts"]],
                    "remaining_counts": [int(v) for v in row["remaining_counts"]],
                } if "failure_counts" in row else {}),
                "failure_depth": int(row["failure_depth"]),
                "state_key": hashlib.sha256(row["text"].encode()).hexdigest(),
                "target": [float(value) for value in row["targets"]],
                "input_ids": [int(value) for value in encoded["input_ids"]],
                "attention_mask": [int(value) for value in encoded["attention_mask"]],
                **({"state_features": [float(value) for value in features]} if features is not None else {}),
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _read(path: Path) -> list[dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _collate_policy(
    batch: list[dict[str, Any]], pad_token_id: int, *, include_state_features: bool = True
) -> dict[str, Any]:
    collated = _collate(batch, pad_token_id)
    # The dataset always carries state_features, but only StructuredStatePolicy
    # accepts them; forwarding them to the plain router is a TypeError.
    if include_state_features and "state_features" in batch[0]:
        collated["state_features"] = torch.tensor(
            [row["state_features"] for row in batch], dtype=torch.float32
        )
    for key in ("failure_counts", "remaining_counts"):
        if key in batch[0]:
            collated[key] = torch.tensor([row[key] for row in batch], dtype=torch.float32)
    return collated


FACTORIZED_DIM = 6  # 3 difficulties + 3 persistences


def factorized_probs(
    logits: torch.Tensor, failure_counts: torch.Tensor, remaining_counts: torch.Tensor
) -> torch.Tensor:
    """Derive all four head probabilities from six per-problem numbers.

    The model reads the problem statement alone and emits a difficulty and a
    persistence per route. Every depth then follows analytically:

        p_m(n) = theta_m * s_m / (s_m + n_m)

    so the decay is *learned* -- s_m(x) is an output, not the hand-set 2.0 -- while
    staying monotone in n by construction. The state-conditioned model has to
    discover that shape from data and demonstrably does not: it is flat for scout
    (-0.0041 across its own failures) and its `nothing` head never falls below 0.30
    when 83.8% of depth-10 states are doomed.

    Fitting 6 numbers per problem instead of 4 per state is also the right medicine
    for a model whose calibration loss is best at epoch 0: every state of a problem
    now contributes evidence to the same six parameters.

    The `nothing` probability is *derived* rather than predicted, as the chance that
    no remaining draw of any route succeeds. That makes it consistent with the
    per-route beliefs by construction; today the two are trained independently and
    can contradict each other.
    """
    theta = torch.sigmoid(logits[:, :3].float())
    persistence = F.softplus(logits[:, 3:].float()) + 1e-3
    counts = failure_counts.float().to(logits.device)
    remaining = remaining_counts.float().to(logits.device)
    p_next = theta * persistence / (persistence + counts)

    # P(no remaining draw of any route succeeds), stepping the decay draw by draw.
    survive = torch.ones_like(p_next[:, :1]).squeeze(-1)
    max_remaining = int(remaining.max().item()) if remaining.numel() else 0
    log_survive = torch.zeros_like(p_next[:, 0])
    for step in range(max_remaining):
        p_step = theta * persistence / (persistence + counts + step)
        active = (remaining > step).float()
        log_survive = log_survive + (
            torch.log1p(-p_step.clamp(1e-7, 1 - 1e-7)) * active
        ).sum(dim=1)
    survive = torch.exp(log_survive)
    return torch.cat([p_next, survive.unsqueeze(1)], dim=1).clamp(1e-7, 1 - 1e-7)


def _forward_policy(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    if "state_features" in batch:
        kwargs["state_features"] = batch["state_features"]
    logits, _, _ = model(**kwargs)
    return logits


def _save_policy_components(model: torch.nn.Module, output_dir: Path) -> None:
    torch.save(model.reward_head.state_dict(), output_dir / "reward_head.pt")
    if isinstance(model, StructuredStatePolicy):
        torch.save(model.state_feature_encoder.state_dict(), output_dir / "state_feature_encoder.pt")


@torch.no_grad()
def _evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: PolicyDataset,
    desc: str,
) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    model.eval()
    collected: list[tuple[int, list[float], list[float]]] = []
    for batch in tqdm(loader, desc=desc, disable=not accelerator.is_main_process):
        logits = _forward_policy(model, batch)
        probs = (
            factorized_probs(logits, batch["failure_counts"], batch["remaining_counts"])
            if "failure_counts" in batch and logits.shape[1] == FACTORIZED_DIM
            else torch.sigmoid(logits.float())
        )
        targets = batch["targets"].float().to(probs.device)
        indices = batch["row_indices"].to(probs.device)
        gathered_probs = accelerator.gather_for_metrics(probs).cpu().tolist()
        gathered_targets = accelerator.gather_for_metrics(targets).cpu().tolist()
        gathered_indices = accelerator.gather_for_metrics(indices).cpu().tolist()
        if accelerator.is_main_process:
            collected.extend(
                (int(idx), list(prob), list(target))
                for idx, prob, target in zip(
                    gathered_indices, gathered_probs, gathered_targets, strict=True
                )
            )
    if not accelerator.is_main_process:
        return float("nan"), {}, []
    by_index = {idx: (prob, target) for idx, prob, target in collected}
    ordered = [(idx, *by_index[idx]) for idx in sorted(by_index)]
    probs = np.asarray([row[1] for row in ordered], dtype=float)
    targets = np.asarray([row[2] for row in ordered], dtype=float)
    loss = float(F.binary_cross_entropy(
        torch.tensor(probs).clamp(1e-7, 1 - 1e-7), torch.tensor(targets)
    ).item())
    aucs = {
        head: float(roc_auc_score(targets[:, hi], probs[:, hi]))
        for hi, head in enumerate(HEADS)
        if len(np.unique(targets[:, hi])) == 2
    }
    prediction_rows = [
        {
            "problem_id": dataset.rows[idx]["problem_id"],
            "failure_depth": dataset.rows[idx]["failure_depth"],
            "state_key": dataset.rows[idx]["state_key"],
            "p_successes": prob,
            "targets": target,
        }
        for idx, prob, target in ordered
    ]
    return loss, aucs, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument(
        "--state-feature-mode", choices=["text_only", STATE_FEATURE_VERSION], default="text_only",
        help="text_only reproduces the existing head; structured_v1 fuses normalized state scalars.",
    )
    parser.add_argument("--state-feature-hidden-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp-hidden-size", type=int, default=1024)
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--frozen-encoder", action="store_true",
        help=(
            "Freeze the encoder and drop LoRA, training only the head. The LoRA recipe's "
            "calibration loss is best at epoch 0 while train loss halves by epoch 1, so it is "
            "memorizing 551 problems rather than learning them. If a probe on frozen features "
            "matches it, the fine-tuning contributes nothing but overfitting, and the method "
            "gets simpler. The base encoder is not saved: it is unmodified and reloadable by name."
        ),
    )
    parser.add_argument(
        "--factorized", action="store_true",
        help=(
            "Read the problem statement alone and emit 6 numbers per problem -- a difficulty "
            "and a learned persistence per route -- deriving every depth as "
            "theta_m * s_m/(s_m + n_m). The decay becomes a model output rather than the "
            "hand-set pseudo-count, is monotone by construction, and the `nothing` head is "
            "derived from the per-route beliefs instead of trained against them."
        ),
    )
    parser.add_argument(
        "--pos-weight", choices=["none", "balanced"], default="none",
        help=(
            "balanced: per-head BCE pos_weight = neg/pos from the train split. The scout head "
            "has a 1.5%% positive rate and ends up 9.28x overconfident under unweighted BCE, "
            "while the 55%% nothing head is calibrated at 1.08x -- error is monotone in base rate."
        ),
    )
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    datasets = {
        split: PolicyDataset(
            _read(Path(args.dataset_dir) / f"{split}.jsonl"), tokenizer, args.max_seq_length,
            require_state_features=args.state_feature_mode == STATE_FEATURE_VERSION,
            factorized=args.factorized,
        )
        for split in ("train", "calibration", "test")
    }
    # Per-head pos_weight from the train split only. Unweighted BCE leaves the
    # rare heads badly scaled, and because base rate is inversely ordered with
    # route cost, that error systematically favours the cheapest route.
    pos_weight = None
    if args.pos_weight == "balanced":
        train_targets = np.array(
            [row["targets"] for row in _read(Path(args.dataset_dir) / "train.jsonl")],
            dtype=float,
        )
        positives = train_targets.sum(axis=0)
        negatives = len(train_targets) - positives
        pos_weight = torch.tensor(
            negatives / np.clip(positives, 1.0, None), dtype=torch.float32
        )
        accelerator.print(f"pos_weight per head: {pos_weight.tolist()}")

    # _collate_policy wraps _collate and forwards the optional per-row fields
    # (state_features, and the raw counts the factorized loss needs). The base
    # _collate drops them, so factorized runs must take this path too.
    structured = args.state_feature_mode == STATE_FEATURE_VERSION
    collate = (
        lambda batch: _collate_policy(
            batch, pad_token_id=int(pad_token_id), include_state_features=structured
        )
        if structured or args.factorized
        else _collate(batch, pad_token_id=int(pad_token_id))
    )
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    calibration_loader = DataLoader(
        datasets["calibration"], batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate
    )

    model_kwargs = dict(
        model_name=args.model_name,
        target_dim=FACTORIZED_DIM if args.factorized else len(HEADS),
        dropout=args.dropout,
        mlp_hidden_size=args.mlp_hidden_size,
        torch_dtype=_dtype_from_name(args.torch_dtype),
        attn_implementation=args.attn_implementation,
        encoder_frozen=bool(args.frozen_encoder),
        use_lora=not args.frozen_encoder,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=[value.strip() for value in args.lora_target_modules.split(",")],
        gradient_checkpointing=True,
        predict_costs=False,
        cost_target_dim=0,
        cost_gradient_mode="joint",
        predict_zero_reward_failure=False,
        embedding_input_layout="single",
        segment_count=1,
    )
    model: torch.nn.Module
    if args.state_feature_mode == STATE_FEATURE_VERSION:
        model = StructuredStatePolicy(
            **model_kwargs,
            state_feature_dim=len(STATE_FEATURE_NAMES),
            state_feature_hidden_size=args.state_feature_hidden_size,
        )
    else:
        model = QwenEmbeddingRouter(**model_kwargs)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = max(1, args.num_epochs * steps_per_epoch)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    model, optimizer, train_loader, calibration_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, calibration_loader, test_loader, scheduler
    )

    config = vars(args) | {
        "heads": HEADS,
        "factorized": bool(args.factorized),
        "frozen_encoder": bool(args.frozen_encoder),
        "split_roles": {"fit": "train", "checkpoint": "calibration", "report": "test"},
        "n_train": len(datasets["train"]),
        "n_calibration": len(datasets["calibration"]),
        "n_test": len(datasets["test"]),
        "state_feature_names": list(STATE_FEATURE_NAMES),
        "state_feature_dim": len(STATE_FEATURE_NAMES),
    }
    if accelerator.is_main_process:
        (output_dir / "train_config.json").write_text(json.dumps(config, indent=2) + "\n")

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = -1
    best_trainable_state: dict[str, torch.Tensor] | None = None
    checkpoint_dir = output_dir / "best_calibration_checkpoint"
    for epoch in range(args.num_epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"train {epoch}", disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                logits = _forward_policy(model, batch)
                if args.factorized:
                    probs = factorized_probs(
                        logits, batch["failure_counts"], batch["remaining_counts"]
                    )
                    loss = F.binary_cross_entropy(
                        probs, batch["targets"].float().to(probs.device)
                    )
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        logits.float(), batch["targets"].float(),
                        pos_weight=None if pos_weight is None else pos_weight.to(logits.device),
                    )
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                train_losses.append(float(loss.detach().item()))
        accelerator.wait_for_everyone()
        calibration_loss, calibration_aucs, _ = _evaluate(
            accelerator, model, calibration_loader, datasets["calibration"], f"calibration {epoch}"
        )
        if accelerator.is_main_process:
            history.append({
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "calibration_loss": calibration_loss,
                "calibration_aucs": calibration_aucs,
            })
        is_best = accelerator.is_main_process and calibration_loss < best_loss
        flag = torch.tensor([int(is_best)], device=accelerator.device)
        flag = accelerator.reduce(flag, reduction="max")
        if int(flag.item()) == 1:
            unwrapped = accelerator.unwrap_model(model)
            # Keep only LoRA and policy-head parameters. Accelerator's full-state
            # save includes the frozen 8B base model and optimizer (roughly 16 GB),
            # even though neither is needed to restore the selected epoch.
            best_trainable_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in unwrapped.named_parameters()
                if parameter.requires_grad
            }
            if accelerator.is_main_process:
                best_loss = calibration_loss
                best_epoch = epoch
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                _save_policy_components(unwrapped, checkpoint_dir)
                if not args.frozen_encoder:
                    unwrapped.encoder.save_pretrained(checkpoint_dir / "encoder")
        accelerator.wait_for_everyone()

    if best_trainable_state is None:
        raise RuntimeError("No calibration checkpoint was selected")
    unwrapped = accelerator.unwrap_model(model)
    named_parameters = dict(unwrapped.named_parameters())
    with torch.no_grad():
        for name, value in best_trainable_state.items():
            parameter = named_parameters[name]
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    test_loss, test_aucs, test_predictions = _evaluate(
        accelerator, model, test_loader, datasets["test"], "test once"
    )
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        _save_policy_components(unwrapped, output_dir)
        if not args.frozen_encoder:
            unwrapped.encoder.save_pretrained(output_dir / "encoder")
        tokenizer.save_pretrained(output_dir / "tokenizer")
        with open(output_dir / "test_predictions.jsonl", "w") as handle:
            for row in test_predictions:
                handle.write(json.dumps(row) + "\n")
        summary = {
            "best_calibration_epoch": best_epoch,
            "best_calibration_loss": best_loss,
            "test_loss": test_loss,
            "test_aucs": test_aucs,
            "history": history,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

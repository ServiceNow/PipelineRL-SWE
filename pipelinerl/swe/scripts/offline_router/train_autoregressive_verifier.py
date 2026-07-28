#!/usr/bin/env python3
"""
Train an autoregressive verifier that predicts patch success from
(problem statement + task model's thinking trace + patch text).

The verifier is fine-tuned with standard SFT cross-entropy on Yes/No targets.
At inference, the logit of "Yes" is used as the routing confidence score.

Input data flow:
  - trajectories_train.jsonl: problem_id, problem_statement, thinking_text, patch_text
  - Daytona report for the same predictions (run_id from collect step)
    logs/run_evaluation/<run_id>/report.json

The script merges these on problem_id to build (input, label) pairs.

Multi-GPU: launched with `accelerate launch` or falls back to single-GPU.

Usage:
  python train_autoregressive_verifier.py \
    --trajectories-path /mnt/.../trajectories_train.jsonl \
    --daytona-report-path logs/run_evaluation/<run_id>/report.json \
    --output-dir /mnt/.../verifier_model \
    --model-name Qwen/Qwen3-4B-Thinking-2507 \
    --num-epochs 3 \
    --batch-size 4
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

VERIFIER_SYSTEM = (
    "You are a code review expert. You will be shown a bug report, the reasoning process "
    "used to produce a patch, and the patch itself. Determine whether the patch correctly "
    "fixes the bug. Respond with only 'Yes' or 'No'."
)

VERIFIER_TEMPLATE = (
    "=== BUG REPORT ===\n"
    "{problem_statement}\n\n"
    "=== REASONING TRACE ===\n"
    "{thinking_text}\n\n"
    "=== PROPOSED PATCH ===\n"
    "{patch_text}\n\n"
    "Does this patch correctly fix the bug? Answer Yes or No."
)


def load_trajectories(path: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("problem_id") or rec.get("instance_id")
            if pid:
                records[str(pid)] = rec
    logger.info("Loaded %d trajectory records from %s", len(records), path)
    return records


def load_daytona_labels(report_path: str) -> dict[str, bool]:
    """Load success labels from a Daytona summary report.json."""
    with open(report_path) as fh:
        report = json.load(fh)
    labels: dict[str, bool] = {}
    for iid in report.get("ids_resolved", []):
        labels[str(iid)] = True
    for iid in report.get("ids_unresolved", []):
        labels[str(iid)] = False
    logger.info(
        "Loaded %d labels (%d resolved) from %s",
        len(labels), sum(labels.values()), report_path,
    )
    return labels


def build_dataset(
    trajectories: dict[str, dict[str, Any]],
    labels: dict[str, bool],
    include_thinking: bool,
) -> list[dict[str, Any]]:
    """Merge trajectories and labels into SFT examples."""
    examples = []
    for pid, traj in trajectories.items():
        if pid not in labels:
            continue
        stmt = str(traj.get("problem_statement") or "").strip()
        thinking = str(traj.get("thinking_text") or "").strip()
        patch = str(traj.get("patch_text") or traj.get("model_patch") or "").strip()
        if not stmt or not patch:
            continue
        label = "Yes" if labels[pid] else "No"
        user_content = VERIFIER_TEMPLATE.format(
            problem_statement=stmt,
            thinking_text=thinking if include_thinking else "(not provided)",
            patch_text=patch,
        )
        examples.append({
            "problem_id": pid,
            "messages": [
                {"role": "system", "content": VERIFIER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            "label": label,
            "resolved": labels[pid],
        })
    pos = sum(e["resolved"] for e in examples)
    logger.info(
        "Built %d SFT examples (%d pos / %d neg)", len(examples), pos, len(examples) - pos
    )
    return examples


class VerifierDataset(Dataset):
    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer: Any,
        max_seq_length: int,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Token IDs for "Yes" and "No" (used for evaluation)
        yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
        no_ids = tokenizer.encode("No", add_special_tokens=False)
        # Take the first token if multi-token
        self.yes_token_id = yes_ids[0] if yes_ids else None
        self.no_token_id = no_ids[0] if no_ids else None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        messages = ex["messages"]
        label = ex["label"]

        # Build prompt with chat template. Disable thinking on Qwen3-Thinking
        # models so the SFT target is a clean Yes/No without a think block.
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        full_text = prompt_text + label

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids

        # Truncate from left if too long (preserve the answer token)
        if len(full_ids) > self.max_seq_length:
            drop = len(full_ids) - self.max_seq_length
            full_ids = full_ids[drop:]
            prompt_len = max(0, len(prompt_ids) - drop)
        else:
            prompt_len = len(prompt_ids)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        # Mask out the prompt tokens so loss is only on the answer
        labels[:prompt_len] = -100

        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    pad_id = 0  # will be replaced by tokenizer pad_id in train loop
    input_ids = torch.stack([
        F.pad(item["input_ids"], (0, max_len - item["input_ids"].shape[0]), value=pad_id)
        for item in batch
    ])
    labels = torch.stack([
        F.pad(item["labels"], (0, max_len - item["labels"].shape[0]), value=-100)
        for item in batch
    ])
    attention_mask = (input_ids != pad_id).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def get_accelerator():
    try:
        from accelerate import Accelerator
        return Accelerator()
    except ImportError:
        return None


def train(
    model: torch.nn.Module,
    tokenizer: Any,
    train_dataset: VerifierDataset,
    eval_dataset: VerifierDataset | None,
    args: argparse.Namespace,
    accelerator: Any,
) -> None:
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        result = collate_fn(batch)
        result["input_ids"][result["input_ids"] == 0] = pad_id
        return result

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = max(1, total_steps // 10)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if accelerator is not None:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            if accelerator is None:
                device = next(model.parameters()).device
                batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            if global_step % 50 == 0:
                logger.info(
                    "epoch=%d step=%d loss=%.4f", epoch + 1, global_step, loss.item()
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info("=== Epoch %d complete. avg_loss=%.4f ===", epoch + 1, avg_loss)

    # Save model
    is_main = accelerator is None or accelerator.is_main_process
    if is_main:
        unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
        unwrapped.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        logger.info("Saved model to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train autoregressive Yes/No verifier")
    parser.add_argument("--trajectories-path", required=True,
                        help="JSONL file with thinking trajectories (from collect_cot_trajectories.py)")
    parser.add_argument("--daytona-report-path", required=True,
                        help="Daytona report.json with ids_resolved/ids_unresolved lists")
    parser.add_argument("--eval-trajectories-path", default="",
                        help="Optional eval trajectories JSONL for held-out loss")
    parser.add_argument("--eval-daytona-report-path", default="",
                        help="Daytona report.json for eval split")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Thinking-2507",
                        help="Base model to fine-tune (default: Qwen/Qwen3-4B-Thinking-2507)")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--include-thinking", action="store_true", default=True,
                        help="Include task model's thinking trace in verifier input")
    parser.add_argument("--no-include-thinking", dest="include_thinking",
                        action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = get_accelerator()
    is_main = accelerator is None or accelerator.is_main_process

    if is_main:
        logger.info("Loading model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if accelerator is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

    # Load training data
    train_trajs = load_trajectories(args.trajectories_path)
    train_labels = load_daytona_labels(args.daytona_report_path)
    train_examples = build_dataset(train_trajs, train_labels, args.include_thinking)
    random.shuffle(train_examples)
    train_dataset = VerifierDataset(train_examples, tokenizer, args.max_seq_length)

    # Load eval data if provided
    eval_dataset: VerifierDataset | None = None
    if args.eval_trajectories_path and args.eval_daytona_report_path:
        eval_trajs = load_trajectories(args.eval_trajectories_path)
        eval_labels = load_daytona_labels(args.eval_daytona_report_path)
        eval_examples = build_dataset(eval_trajs, eval_labels, args.include_thinking)
        eval_dataset = VerifierDataset(eval_examples, tokenizer, args.max_seq_length)

    train(model, tokenizer, train_dataset, eval_dataset, args, accelerator)


if __name__ == "__main__":
    main()

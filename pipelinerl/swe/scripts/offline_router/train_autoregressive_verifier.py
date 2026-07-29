#!/usr/bin/env python3
"""
Train an autoregressive verifier (LoRA adapter) that predicts whether the strong
model (gpt-oss-120b by default) will succeed, given the cheap model's CoT trace
and patch.

Input data flow:
  - trajectories_train.jsonl: problem_id, problem_statement, thinking_text, patch_text
    (from collect_cot_trajectories.py)
  - Parquet label dir: route_successes[label_route_idx] as the binary target
    (from the existing real-label dataset)

The verifier is fine-tuned with LoRA SFT on Yes/No targets (enable_thinking=False
so the SFT target is a clean answer token; at inference, enable_thinking=True so
the verifier reasons before answering).

Multi-GPU: launched with `accelerate launch` or falls back to single-GPU.

Usage:
  python train_autoregressive_verifier.py \\
    --trajectories-path /mnt/.../trajectories_train.jsonl \\
    --labels-parquet-dir /mnt/.../collect/train \\
    --output-dir /mnt/.../verifier_lora \\
    --model-name Qwen/Qwen3-4B-Thinking-2507
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

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


def load_parquet_labels(parquet_dir: str, route_idx: int = 3) -> dict[str, bool]:
    """Load binary success labels from parquet. route_idx=3 is gpt-oss-120b."""
    import pandas as pd
    paths = sorted(Path(parquet_dir).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")
    df = pd.concat([pd.read_parquet(p) for p in paths])
    labels: dict[str, bool] = {}
    for _, row in df.iterrows():
        pid = str(row.get("problem_id") or "").strip()
        if not pid:
            continue
        successes = row.get("route_successes")
        if successes is None or len(successes) <= route_idx:
            continue
        labels[pid] = bool(successes[route_idx])
    pos = sum(labels.values())
    logger.info(
        "Loaded %d labels from parquet (route_idx=%d): %d pos / %d neg",
        len(labels), route_idx, pos, len(labels) - pos,
    )
    return labels


def build_dataset(
    trajectories: dict[str, dict[str, Any]],
    labels: dict[str, bool],
    include_thinking: bool,
) -> list[dict[str, Any]]:
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
    logger.info("Built %d SFT examples (%d pos / %d neg)", len(examples), pos, len(examples) - pos)
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

        yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
        no_ids = tokenizer.encode("No", add_special_tokens=False)
        self.yes_token_id = yes_ids[0] if yes_ids else None
        self.no_token_id = no_ids[0] if no_ids else None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
        full_text = prompt_text + ex["label"]

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids

        if len(full_ids) > self.max_seq_length:
            drop = len(full_ids) - self.max_seq_length
            full_ids = full_ids[drop:]
            prompt_len = max(0, len(prompt_ids) - drop)
        else:
            prompt_len = len(prompt_ids)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        return {"input_ids": input_ids, "labels": labels}


def make_collate_fn(pad_id: int):
    def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in batch)
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
    return _collate


@torch.no_grad()
def evaluate_auc(
    model: torch.nn.Module,
    tokenizer: Any,
    eval_dataset: VerifierDataset,
    device: torch.device,
) -> float:
    from sklearn.metrics import roc_auc_score

    model.eval()
    yes_id = eval_dataset.yes_token_id
    no_id = eval_dataset.no_token_id
    scores: list[float] = []
    true_labels: list[int] = []

    for ex in eval_dataset.examples:
        try:
            prompt_text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc.input_ids
        if input_ids.shape[1] > eval_dataset.max_seq_length:
            input_ids = input_ids[:, -eval_dataset.max_seq_length:]
        input_ids = input_ids.to(device)

        logits = model(input_ids=input_ids).logits[0, -1]
        yes_logit = logits[yes_id].item() if yes_id is not None else 0.0
        no_logit = logits[no_id].item() if no_id is not None else 0.0
        p_yes = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)[0].item()
        scores.append(p_yes)
        true_labels.append(1 if ex["resolved"] else 0)

    if len(set(true_labels)) < 2:
        logger.warning("Eval set has only one class; AUC undefined")
        return float("nan")
    return float(roc_auc_score(true_labels, scores))


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
    eval_dataset: "VerifierDataset | None",
    args: argparse.Namespace,
    accelerator: Any,
) -> None:
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(pad_id),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if accelerator is not None:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    is_main = accelerator is None or accelerator.is_main_process
    device = next(model.parameters()).device

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            if accelerator is None:
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
                logger.info("epoch=%d step=%d loss=%.4f", epoch + 1, global_step, loss.item())

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info("=== Epoch %d complete. avg_loss=%.4f ===", epoch + 1, avg_loss)

        if eval_dataset is not None and is_main:
            eval_model = accelerator.unwrap_model(model) if accelerator is not None else model
            auc = evaluate_auc(eval_model, tokenizer, eval_dataset, device)
            logger.info("=== Epoch %d eval AUC=%.4f ===", epoch + 1, auc)
            eval_model.train()

    if is_main:
        unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
        unwrapped.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        logger.info("Saved LoRA adapter to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train autoregressive Yes/No verifier (LoRA)")
    parser.add_argument("--trajectories-path", required=True,
                        help="JSONL with thinking trajectories (from collect_cot_trajectories.py)")
    parser.add_argument("--labels-parquet-dir", required=True,
                        help="Directory of parquet shards with route_successes labels")
    parser.add_argument("--label-route-idx", type=int, default=3,
                        help="Index into route_successes for the target model (default: 3 = gpt-oss-120b)")
    parser.add_argument("--eval-trajectories-path", default="",
                        help="Optional eval trajectories JSONL for held-out AUC")
    parser.add_argument("--eval-labels-parquet-dir", default="",
                        help="Parquet dir for eval labels (default: same as train if omitted)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save the LoRA adapter")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Thinking-2507",
                        help="Base model to apply LoRA to")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--include-thinking", action="store_true", default=True,
                        help="Include task model's thinking trace in verifier input")
    parser.add_argument("--no-include-thinking", dest="include_thinking", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = get_accelerator()
    is_main = accelerator is None or accelerator.is_main_process

    if is_main:
        logger.info("Loading tokenizer/model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if accelerator is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = base_model.to(device)

    from peft import LoraConfig, TaskType, get_peft_model
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    if is_main:
        model.print_trainable_parameters()

    train_labels = load_parquet_labels(args.labels_parquet_dir, args.label_route_idx)
    train_trajs = load_trajectories(args.trajectories_path)
    train_examples = build_dataset(train_trajs, train_labels, args.include_thinking)
    random.shuffle(train_examples)
    train_dataset = VerifierDataset(train_examples, tokenizer, args.max_seq_length)

    eval_dataset: "VerifierDataset | None" = None
    if args.eval_trajectories_path:
        eval_labels_dir = args.eval_labels_parquet_dir or args.labels_parquet_dir
        eval_labels = load_parquet_labels(eval_labels_dir, args.label_route_idx)
        eval_trajs = load_trajectories(args.eval_trajectories_path)
        eval_examples = build_dataset(eval_trajs, eval_labels, args.include_thinking)
        eval_dataset = VerifierDataset(eval_examples, tokenizer, args.max_seq_length)

    train(model, tokenizer, train_dataset, eval_dataset, args, accelerator)


if __name__ == "__main__":
    main()

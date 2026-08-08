#!/usr/bin/env python3
"""
Train Qwen3-Embedding-8B with LoRA to predict whether gpt-oss-120b will succeed
on a SWE repair task, given the 4B CoT scout's thinking trace + generated patch.

Reads directly from collect_cot_trajectories.py output (trajectories_*.jsonl)
and route_successes[3] labels from the existing parquet collection.

The --no-include-thinking flag ablates the thinking trace so you can isolate
how much signal the CoT adds over the patch alone.

Outputs eval_predictions.jsonl: {problem_id, p_yes, resolved}
Compatible with analyze_cot_verifier_abstention.py.

Usage:
  python train_cot_abstention_predictor.py \\
    --train-trajectories /mnt/.../trajectories_train.jsonl \\
    --eval-trajectories  /mnt/.../trajectories_eval.jsonl \\
    --train-parquet-dir  /mnt/.../collect/train \\
    --eval-parquet-dir   /mnt/.../collect/eval \\
    --output-dir         /mnt/.../cot_abstention_predictor \\
    --include-thinking
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from pipelinerl.swe.scripts.offline_router.common import write_json
from pipelinerl.swe.scripts.offline_router.train_qwen_embedding_router_baseline import (
    QwenEmbeddingRouter,
    _dtype_from_name,
    _write_jsonl,
)


# ── data loading ──────────────────────────────────────────────────────────────

def load_trajectories(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_parquet_labels(parquet_dir: str, route_idx: int = 3) -> dict[str, bool]:
    paths = sorted(Path(parquet_dir).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    df = pd.concat([pd.read_parquet(p) for p in paths])
    labels: dict[str, bool] = {}
    for _, row in df.iterrows():
        pid = str(row.get("problem_id") or "").strip()
        successes = row.get("route_successes")
        if pid and successes is not None and len(successes) > route_idx:
            labels[pid] = bool(successes[route_idx])
    return labels


# ── input text ────────────────────────────────────────────────────────────────

def _build_input_text(
    problem_statement: str,
    thinking_text: str,
    patch_text: str,
    include_thinking: bool,
    input_only: bool = False,
    test_feedback: str = "",
    include_test_feedback: bool = False,
) -> str:
    if input_only:
        return "\n".join([
            "Predict whether a strong model will successfully resolve this task.",
            "Use only the problem description.",
            "",
            "[Problem Statement]",
            problem_statement.strip(),
        ])
    parts = [
        "Predict whether a strong model will successfully resolve this software repair task.",
        "Use the problem description and the scout model's repair attempt.",
        "",
        "[Problem Statement]",
        problem_statement.strip(),
        "",
        "[Scout Repair Attempt]",
    ]
    if include_thinking and thinking_text:
        parts += ["<think>", thinking_text.strip(), "</think>"]
    parts.append(patch_text.strip())
    if include_test_feedback and test_feedback:
        parts += ["", "[Scout Test Execution Feedback]", test_feedback.strip()]
    return "\n".join(parts)


# ── dataset ───────────────────────────────────────────────────────────────────

class CoTAbstentionDataset(Dataset):
    def __init__(
        self,
        trajectories: list[dict[str, Any]],
        labels: dict[str, bool],
        tokenizer: Any,
        max_seq_length: int,
        include_thinking: bool,
        multi_task_scout: bool = False,
        input_only: bool = False,
        include_test_feedback: bool = False,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        self.multi_task_scout = multi_task_scout
        skipped = 0
        for traj in trajectories:
            pid = str(traj.get("problem_id") or traj.get("instance_id") or "").strip()
            if not pid or pid not in labels:
                skipped += 1
                continue
            problem_statement = str(traj.get("problem_statement") or "").strip()
            thinking_text = str(traj.get("thinking_text") or "").strip()
            patch_text = str(traj.get("patch_text") or "").strip()
            test_feedback = str(traj.get("test_feedback") or "").strip()
            if not problem_statement or (not input_only and not patch_text):
                skipped += 1
                continue
            text = _build_input_text(
                problem_statement, thinking_text, patch_text, include_thinking,
                input_only=input_only,
                test_feedback=test_feedback,
                include_test_feedback=include_test_feedback,
            )
            encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_seq_length)
            input_ids = encoded.get("input_ids")
            attention_mask = encoded.get("attention_mask")
            if not input_ids or not attention_mask:
                skipped += 1
                continue
            if multi_task_scout:
                # [scout_correct, strong_correct] — scout label is free from the trajectory
                scout_correct = float(bool(traj.get("scout_correct", False)))
                target = [scout_correct, float(labels[pid])]
            else:
                target = [float(labels[pid])]
            self.rows.append({
                "row_idx": len(self.rows),
                "problem_id": pid,
                "target": target,
                "input_ids": [int(x) for x in input_ids],
                "attention_mask": [int(x) for x in attention_mask],
            })
        print(f"Dataset built: {len(self.rows)} examples ({skipped} skipped — no label match or empty fields)")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def _collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    max_len = max(len(row["input_ids"]) for row in batch)
    target_dim = len(batch[0]["target"])
    input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    targets = torch.zeros((len(batch), target_dim), dtype=torch.float32)
    row_indices = torch.zeros((len(batch),), dtype=torch.long)
    for i, row in enumerate(batch):
        seq_len = len(row["input_ids"])
        start = max_len - seq_len  # left-pad to match Qwen embedding convention
        input_ids[i, start:] = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask[i, start:] = torch.tensor(row["attention_mask"], dtype=torch.long)
        targets[i] = torch.tensor(row["target"], dtype=torch.float32)
        row_indices[i] = int(row["row_idx"])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "targets": targets, "row_indices": row_indices}


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: CoTAbstentionDataset,
    desc: str,
) -> tuple[float, float, list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc=desc, disable=not accelerator.is_main_process):
        logits, _, _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        targets = batch["targets"].float().clamp(0.0, 1.0)
        loss = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="sum")
        preds = torch.sigmoid(logits.float())
        g_loss = accelerator.gather_for_metrics(loss.detach().reshape(1)).detach().cpu()
        g_preds = accelerator.gather_for_metrics(preds).detach().cpu()
        g_targets = accelerator.gather_for_metrics(targets).detach().cpu()
        g_indices = accelerator.gather_for_metrics(batch["row_indices"]).detach().cpu().tolist()
        if accelerator.is_main_process:
            total_loss += float(g_loss.sum().item())
            total_n += int(g_targets.shape[0])
            for i in range(g_preds.shape[0]):
                meta = dataset.rows[int(g_indices[i])]
                # last head = strong-model head (works for both 1-task and 2-task)
                p_yes = float(g_preds[i, -1].item())
                strong_target = float(meta["target"][-1])
                rows.append({
                    "problem_id": meta["problem_id"],
                    "p_yes": p_yes,
                    "resolved": bool(strong_target > 0.5),
                })
    if not accelerator.is_main_process:
        return math.nan, math.nan, []
    mean_loss = float(total_loss / total_n) if total_n > 0 else math.nan
    auc = math.nan
    if len(rows) >= 2:
        y_true = np.array([int(r["resolved"]) for r in rows])
        y_pred = np.array([r["p_yes"] for r in rows])  # strong-model head only
        try:
            auc = float(roc_auc_score(y_true, y_pred))
        except Exception:
            pass
    return mean_loss, auc, rows


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trajectories", required=True,
                        help="trajectories_train.jsonl from collect_cot_trajectories.py")
    parser.add_argument("--eval-trajectories", required=True,
                        help="trajectories_eval.jsonl from collect_cot_trajectories.py")
    parser.add_argument("--train-parquet-dir", required=True,
                        help="Directory of train parquet shards with route_successes")
    parser.add_argument("--eval-parquet-dir", required=True,
                        help="Directory of eval parquet shards with route_successes")
    parser.add_argument("--label-route-idx", type=int, default=3,
                        help="Index into route_successes to use as label (default: 3 = gpt-oss-120b)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--include-thinking", action="store_true", default=True)
    parser.add_argument("--no-include-thinking", dest="include_thinking", action="store_false")
    parser.add_argument("--input-only", action="store_true", default=False,
                        help="Use only the problem statement (no scout trace or patch)")
    parser.add_argument("--include-test-feedback", action="store_true", default=False,
                        help="Append test execution feedback from 'test_feedback' trajectory field to predictor input")
    parser.add_argument("--no-include-test-feedback", dest="include_test_feedback", action="store_false")
    parser.add_argument("--multi-task-scout", action="store_true", default=False,
                        help="Add scout success (route 0) as a free auxiliary task alongside the strong-model target")
    parser.add_argument("--max-seq-length", type=int, default=24000)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
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
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--checkpoint-every-epoch", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(gradient_accumulation_steps=int(args.gradient_accumulation_steps))
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    if accelerator.is_main_process:
        print(f"Loading CoT trajectories and parquet labels ...", flush=True)

    train_trajs = load_trajectories(args.train_trajectories)
    eval_trajs = load_trajectories(args.eval_trajectories)
    random.shuffle(train_trajs)

    train_labels = load_parquet_labels(args.train_parquet_dir, route_idx=int(args.label_route_idx))
    eval_labels = load_parquet_labels(args.eval_parquet_dir, route_idx=int(args.label_route_idx))

    if accelerator.is_main_process:
        print(f"Train labels: {len(train_labels)} | pos={sum(train_labels.values())}", flush=True)
        print(f"Eval labels:  {len(eval_labels)} | pos={sum(eval_labels.values())}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    train_dataset = CoTAbstentionDataset(
        train_trajs, train_labels, tokenizer, int(args.max_seq_length),
        bool(args.include_thinking), multi_task_scout=bool(args.multi_task_scout),
        input_only=bool(args.input_only),
        include_test_feedback=bool(args.include_test_feedback),
    )
    eval_dataset = CoTAbstentionDataset(
        eval_trajs, eval_labels, tokenizer, int(args.max_seq_length),
        bool(args.include_thinking), multi_task_scout=bool(args.multi_task_scout),
        input_only=bool(args.input_only),
        include_test_feedback=bool(args.include_test_feedback),
    )

    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError(f"Empty dataset: train={len(train_dataset)} eval={len(eval_dataset)}")

    collate_fn = lambda batch: _collate(batch, pad_token_id=int(pad_token_id))
    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=int(args.eval_batch_size), shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    train_report_loader = DataLoader(train_dataset, batch_size=int(args.eval_batch_size), shuffle=False,
                                     collate_fn=collate_fn, num_workers=0)

    target_dim = 2 if args.multi_task_scout else 1
    model = QwenEmbeddingRouter(
        model_name=args.model_name,
        target_dim=target_dim,
        dropout=float(args.dropout),
        mlp_hidden_size=int(args.mlp_hidden_size),
        torch_dtype=_dtype_from_name(str(args.torch_dtype)),
        attn_implementation=str(args.attn_implementation) if args.attn_implementation else None,
        encoder_frozen=False,
        use_lora=True,
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        lora_target_modules=lora_target_modules,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        predict_costs=False,
        cost_target_dim=0,
        cost_gradient_mode="joint",
        predict_zero_reward_failure=False,
        embedding_input_layout="single",
        segment_count=1,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    steps_per_epoch = math.ceil(len(train_loader) / int(args.gradient_accumulation_steps))
    total_steps = max(1, int(args.num_epochs) * steps_per_epoch)
    warmup_steps = int(total_steps * float(args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model, optimizer, train_loader, eval_loader, train_report_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, train_report_loader, scheduler
    )

    config = {
        "model_name": args.model_name,
        "train_trajectories": args.train_trajectories,
        "eval_trajectories": args.eval_trajectories,
        "train_parquet_dir": args.train_parquet_dir,
        "eval_parquet_dir": args.eval_parquet_dir,
        "label_route_idx": int(args.label_route_idx),
        "include_thinking": bool(args.include_thinking),
        "include_test_feedback": bool(args.include_test_feedback),
        "multi_task_scout": bool(args.multi_task_scout),
        "max_seq_length": int(args.max_seq_length),
        "num_epochs": int(args.num_epochs),
        "lr": float(args.lr),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_target_modules": lora_target_modules,
        "n_train": len(train_dataset),
        "n_eval": len(eval_dataset),
    }
    if accelerator.is_main_process:
        write_json(output_dir / "train_config.json", config)

    history: list[dict[str, Any]] = []

    for epoch in range(int(args.num_epochs)):
        model.train()
        running_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}", disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                logits, _, _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                targets = batch["targets"].float().clamp(0.0, 1.0)
                loss = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                running_losses.append(float(loss.detach().item()))
        accelerator.wait_for_everyone()

        eval_loss, eval_auc, eval_rows = _evaluate(
            accelerator, model, eval_loader, eval_dataset, desc=f"Eval epoch {epoch}"
        )
        if accelerator.is_main_process:
            train_loss_mean = float(np.mean(running_losses)) if running_losses else math.nan
            print(
                f"Epoch {epoch}: train_loss={train_loss_mean:.4f}  "
                f"eval_loss={eval_loss:.4f}  eval_auc={eval_auc:.4f}",
                flush=True,
            )
            history.append({"epoch": epoch, "train_loss": train_loss_mean,
                            "eval_loss": eval_loss, "eval_auc": eval_auc})

        if bool(args.checkpoint_every_epoch):
            ckpt_dir = output_dir / "checkpoints" / f"epoch_{epoch:04d}"
            accelerator.save_state(str(ckpt_dir))
            accelerator.wait_for_everyone()

    # Final eval + predictions
    _, train_auc, train_rows = _evaluate(
        accelerator, model, train_report_loader, train_dataset, desc="Final train predict"
    )
    _, eval_auc, eval_rows = _evaluate(
        accelerator, model, eval_loader, eval_dataset, desc="Final eval predict"
    )

    if not accelerator.is_main_process:
        return

    print(f"\nFinal train AUC: {train_auc:.4f}", flush=True)
    print(f"Final eval  AUC: {eval_auc:.4f}", flush=True)

    _write_jsonl(output_dir / "train_predictions.jsonl", train_rows)
    _write_jsonl(output_dir / "eval_predictions.jsonl", eval_rows)

    summary = {
        "history": history,
        "final_train_auc": train_auc,
        "final_eval_auc": eval_auc,
        "n_train": len(train_dataset),
        "n_eval": len(eval_dataset),
        "config": config,
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Outputs written to {output_dir}", flush=True)

    if args.save_model:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.reward_head.state_dict(), output_dir / "reward_head.pt")
        if hasattr(unwrapped, "encoder") and hasattr(unwrapped.encoder, "save_pretrained"):
            unwrapped.encoder.save_pretrained(output_dir / "encoder")
        tokenizer.save_pretrained(output_dir / "tokenizer")


if __name__ == "__main__":
    main()

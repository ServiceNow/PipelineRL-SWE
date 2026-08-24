#!/usr/bin/env python3
"""
Train the thread-(a) sequential MDP policy: Qwen3-Embedding-8B + LoRA, 4 BCE
heads over {P(scout next draw resolves), P(oss20 fresh), P(oss120 fresh),
P(nothing succeeds)} from depth-1 decision-point examples built by
build_mdp_sequential_dataset.py.

Usage:
  python train_mdp_sequential_policy.py --dataset-dir .../mdp_seq_dataset_v1 \
      --output-dir .../mdp_seq_policy_v1
"""

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
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from pipelinerl.swe.scripts.offline_router.common import write_json
from pipelinerl.swe.scripts.offline_router.train_cot_abstention_predictor import _collate
from pipelinerl.swe.scripts.offline_router.train_qwen_embedding_router_baseline import (
    QwenEmbeddingRouter,
    _dtype_from_name,
    _write_jsonl,
)

HEADS = ["scout_next", "oss20_fresh", "oss120_fresh", "nothing"]


class SeqPolicyDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_seq_length: int) -> None:
        self.rows: list[dict[str, Any]] = []
        skipped = 0
        for row in rows:
            encoded = tokenizer(row["text"], add_special_tokens=True, truncation=True,
                                max_length=max_seq_length)
            ids = encoded.get("input_ids")
            am = encoded.get("attention_mask")
            if not ids or not am:
                skipped += 1
                continue
            self.rows.append({
                "row_idx": len(self.rows),
                "problem_id": row["problem_id"],
                "depth": row["depth"],
                "target": [float(t) for t in row["targets"]],
                "input_ids": [int(x) for x in ids],
                "attention_mask": [int(x) for x in am],
            })
        print(f"Dataset: {len(self.rows)} examples ({skipped} skipped)", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-8B")
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
    parser.add_argument("--lora-target-modules",
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(gradient_accumulation_steps=int(args.gradient_accumulation_steps))
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    def read(split):
        return [json.loads(l) for l in open(Path(args.dataset_dir) / f"{split}.jsonl") if l.strip()]

    train_rows, test_rows = read("cal"), read("test")
    random.shuffle(train_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    train_ds = SeqPolicyDataset(train_rows, tokenizer, int(args.max_seq_length))
    test_ds = SeqPolicyDataset(test_rows, tokenizer, int(args.max_seq_length))

    collate_fn = lambda batch: _collate(batch, pad_token_id=int(pad_token_id))
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    eval_loader = DataLoader(test_ds, batch_size=int(args.eval_batch_size), shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model = QwenEmbeddingRouter(
        model_name=args.model_name, target_dim=len(HEADS), dropout=float(args.dropout),
        mlp_hidden_size=int(args.mlp_hidden_size),
        torch_dtype=_dtype_from_name(str(args.torch_dtype)),
        attn_implementation=str(args.attn_implementation) if args.attn_implementation else None,
        encoder_frozen=False, use_lora=True, lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha), lora_dropout=float(args.lora_dropout),
        lora_target_modules=[m.strip() for m in args.lora_target_modules.split(",")],
        gradient_checkpointing=True, predict_costs=False, cost_target_dim=0,
        cost_gradient_mode="joint", predict_zero_reward_failure=False,
        embedding_input_layout="single", segment_count=1,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    steps_per_epoch = math.ceil(len(train_loader) / int(args.gradient_accumulation_steps))
    total_steps = max(1, int(args.num_epochs) * steps_per_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler)

    write_json(output_dir / "train_config.json", {
        "model_name": args.model_name, "heads": HEADS, "lr": args.lr,
        "num_epochs": args.num_epochs, "seed": args.seed, "n_train": len(train_ds), "n_test": len(test_ds),
    })

    history = []
    for epoch in range(int(args.num_epochs)):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                logits, _, _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                targets = batch["targets"].float().clamp(0.0, 1.0)
                loss = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="mean")
                accelerator.backward(loss)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().item()))
        accelerator.wait_for_everyone()

        # eval AUCs on test half
        model.eval()
        probs, tgts = [], []
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"eval {epoch}", disable=not accelerator.is_main_process):
                logits, _, _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                probs.extend(accelerator.gather_for_metrics(torch.sigmoid(logits.float())).cpu().tolist())
                tgts.extend(accelerator.gather_for_metrics(batch["targets"]).cpu().tolist())
        probs = np.array(probs); tgts = np.array(tgts)
        aucs = {}
        for i, h in enumerate(HEADS):
            if len(set(tgts[:, i].tolist())) == 2:
                aucs[h] = float(roc_auc_score(tgts[:, i], probs[:, i]))
        if accelerator.is_main_process:
            msg = " ".join(f"{k}={v:.3f}" for k, v in aucs.items())
            print(f"Epoch {epoch}: loss={np.mean(losses):.4f} | {msg}", flush=True)
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), "aucs": aucs})

    # final predictions dump for replay wiring
    model.eval()
    probs, tgts = [], []
    metas = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(test_ds, batch_size=int(args.eval_batch_size), shuffle=False,
                                     collate_fn=collate_fn, num_workers=0),
                          desc="final predict"):
            logits, _, _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            probs.extend(accelerator.gather_for_metrics(torch.sigmoid(logits.float())).cpu().tolist())
            tgts.extend(accelerator.gather_for_metrics(batch["targets"]).cpu().tolist())
    if accelerator.is_main_process:
        _write_jsonl(output_dir / "test_predictions.jsonl", [
            {"problem_id": test_ds.rows[i]["problem_id"], "depth": test_ds.rows[i]["depth"],
             "p_successes": probs[i], "targets": tgts[i]}
            for i in range(min(len(probs), len(test_ds.rows)))
        ])
        write_json(output_dir / "summary.json", {"history": history})
        print(f"Outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

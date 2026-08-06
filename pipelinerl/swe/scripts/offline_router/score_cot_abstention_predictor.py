#!/usr/bin/env python3
"""
Run inference with a trained CoT abstention predictor checkpoint and save
per-instance P(Yes) scores. Used for zero-shot generalization evaluation.

Output JSONL: {problem_id, p_yes, resolved}

Labels can be provided in two ways (exactly one required):
  --parquet-dir + --label-route-idx  (proxy/real parquet with route_successes)
  --real-labels-jsonl                (Daytona .results.jsonl: {instance_id, resolved})

Usage:
  python score_cot_abstention_predictor.py \
    --checkpoint-dir /mnt/.../checkpoints/epoch_0006 \
    --train-config   /mnt/.../train_config.json \
    --trajectories   /mnt/.../trajectories_verified.jsonl \
    --parquet-dir    /mnt/.../labels_verified_dir \
    --label-route-idx 0 \
    --output-path    /mnt/.../verified_predictions.jsonl

  # or with real Daytona labels:
  python score_cot_abstention_predictor.py \
    --checkpoint-dir /mnt/.../checkpoints/epoch_0006 \
    --train-config   /mnt/.../train_config.json \
    --trajectories   /mnt/.../trajectories_verified.jsonl \
    --real-labels-jsonl /mnt/.../predictions_route_0.results.jsonl \
    --output-path    /mnt/.../verified_predictions.jsonl
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _build_input_text(problem_statement: str, thinking_text: str,
                      patch_text: str, include_thinking: bool) -> str:
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
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Accelerate checkpoint dir (contains model.safetensors)")
    parser.add_argument("--train-config", required=True,
                        help="train_config.json from the training run")
    parser.add_argument("--trajectories", required=True,
                        help="JSONL with problem_id, thinking_text, patch_text")
    parser.add_argument("--parquet-dir", default=None,
                        help="Dir with parquet files containing route_successes labels")
    parser.add_argument("--label-route-idx", type=int, default=0,
                        help="Index into route_successes for the label (default 0)")
    parser.add_argument("--real-labels-jsonl", default=None,
                        help="Daytona .results.jsonl with {instance_id, resolved} rows")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if args.real_labels_jsonl is None and args.parquet_dir is None:
        parser.error("Provide either --parquet-dir or --real-labels-jsonl")

    # Load training config
    with open(args.train_config) as f:
        cfg = json.load(f)

    model_name = cfg.get("model_name", "Qwen/Qwen3-Embedding-8B")
    max_seq_length = cfg.get("max_seq_length", 24000)
    include_thinking = cfg.get("include_thinking", True)
    lora_r = cfg.get("lora_r", 32)
    lora_alpha = cfg.get("lora_alpha", 64)
    lora_target_modules = cfg.get("lora_target_modules",
                                  ["q_proj","k_proj","v_proj","o_proj",
                                   "gate_proj","up_proj","down_proj"])
    mlp_hidden_size = cfg.get("mlp_hidden_size", 1024)
    dropout = cfg.get("dropout", 0.1)

    logger.info("Model: %s  include_thinking=%s", model_name, include_thinking)

    # Load labels
    labels: dict[str, bool] = {}
    if args.real_labels_jsonl:
        with open(args.real_labels_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                iid = str(row.get("instance_id") or "").strip()
                if iid:
                    labels[iid] = bool(row.get("resolved", False))
        logger.info("Loaded %d real labels from %s (%d positive)",
                    len(labels), args.real_labels_jsonl, sum(labels.values()))
    else:
        parquet_paths = sorted(Path(args.parquet_dir).glob("*.parquet"))
        if not parquet_paths:
            parquet_paths = [Path(args.parquet_dir)]
        df = pd.concat([pd.read_parquet(p) for p in parquet_paths])
        for _, row in df.iterrows():
            pid = str(row.get("problem_id") or "").strip()
            rs = row.get("route_successes")
            if pid and rs is not None and len(rs) > args.label_route_idx:
                labels[pid] = bool(rs[args.label_route_idx])
        logger.info("Loaded %d labels from parquet (%d positive)", len(labels), sum(labels.values()))

    # Load trajectories
    trajs: list[dict] = []
    with open(args.trajectories) as f:
        for line in f:
            try:
                r = json.loads(line)
                pid = str(r.get("problem_id") or "").strip()
                if pid and pid in labels:
                    trajs.append(r)
            except json.JSONDecodeError:
                pass
    logger.info("Loaded %d trajectories with labels", len(trajs))

    # Build model
    from pipelinerl.swe.scripts.offline_router.train_qwen_embedding_router_baseline import (
        QwenEmbeddingRouter, _dtype_from_name,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    # target_dim: check if multi-task checkpoint
    multi_task = cfg.get("multi_task_scout", False)
    target_dim = 2 if multi_task else 1

    logger.info("Loading model weights from %s", args.checkpoint_dir)
    model = QwenEmbeddingRouter(
        model_name=model_name,
        target_dim=target_dim,
        dropout=dropout,
        mlp_hidden_size=mlp_hidden_size,
        torch_dtype=_dtype_from_name("bf16"),
        attn_implementation="flash_attention_2",
        encoder_frozen=False,
        use_lora=True,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        lora_target_modules=lora_target_modules,
        gradient_checkpointing=False,
        predict_costs=False,
        cost_target_dim=0,
        cost_gradient_mode="joint",
        predict_zero_reward_failure=False,
        embedding_input_layout="single",
        segment_count=1,
    )

    ckpt = Path(args.checkpoint_dir) / "model.safetensors"
    if ckpt.exists():
        from safetensors.torch import load_file
        state = load_file(str(ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info("Loaded safetensors. Missing: %d  Unexpected: %d",
                    len(missing), len(unexpected))
    else:
        raise FileNotFoundError(f"No model.safetensors in {args.checkpoint_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    with out_path.open("w") as fh, torch.no_grad():
        for traj in tqdm(trajs, desc="Scoring"):
            pid = str(traj.get("problem_id") or "").strip()
            text = _build_input_text(
                str(traj.get("problem_statement") or "").strip(),
                str(traj.get("thinking_text") or "").strip(),
                str(traj.get("patch_text") or "").strip(),
                include_thinking,
            )
            enc = tokenizer(text, add_special_tokens=True, truncation=True,
                            max_length=max_seq_length, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            # left-pad to match training convention (already handled by tokenizer here)
            logits, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            p_yes = float(torch.sigmoid(logits[0, -1]).item())  # last head = strong model
            fh.write(json.dumps({
                "problem_id": pid,
                "p_yes": p_yes,
                "resolved": labels[pid],
            }) + "\n")
            fh.flush()
            n_done += 1

    logger.info("Wrote %d predictions to %s", n_done, out_path)

    # Quick AUC
    preds_out = []
    with out_path.open() as f:
        for line in f:
            preds_out.append(json.loads(line))
    from sklearn.metrics import roc_auc_score
    y_true = np.array([int(r["resolved"]) for r in preds_out])
    y_pred = np.array([r["p_yes"] for r in preds_out])
    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred)
        logger.info("Zero-shot AUC on verified: %.4f (n=%d, pos=%d)", auc, len(y_true), y_true.sum())
    else:
        logger.warning("Only one class in labels — AUC undefined")


if __name__ == "__main__":
    main()

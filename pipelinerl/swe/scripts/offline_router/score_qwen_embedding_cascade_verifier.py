#!/usr/bin/env python
"""Score router/verifier rows with a saved Qwen embedding cascade verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from pipelinerl.swe.scripts.offline_router.train_modernbert_router_baseline import (
    _load_route_labels,
    _load_split,
)
from pipelinerl.swe.scripts.offline_router.train_qwen_embedding_cascade_baseline import (
    CascadeAttemptDataset,
    _collate,
    _parse_int_list,
    _score_loss_and_predictions,
)
from pipelinerl.swe.scripts.offline_router.train_qwen_embedding_router_baseline import (
    QwenEmbeddingRouter,
    _dtype_from_name,
    _parse_route_indices,
    _write_jsonl,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _metadata_by_problem_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        problem_id = str(row.get("problem_id") or "")
        if problem_id:
            out[problem_id] = {
                "original_problem_id": row.get("original_problem_id") or problem_id,
                "rollout_idx": row.get("rollout_idx"),
                "route_output_tokens": row.get("route_output_tokens"),
                "route_prompt_tokens": row.get("route_prompt_tokens"),
                "route_rewards": row.get("route_rewards"),
                "route_successes": row.get("route_successes"),
                "route_failure_types": row.get("route_failure_types"),
            }
    return out


def _load_saved_verifier(checkpoint_dir: Path, device: torch.device) -> tuple[QwenEmbeddingRouter, dict[str, Any]]:
    config = _read_json(checkpoint_dir / "train_config.json")
    model = QwenEmbeddingRouter(
        str(config["model_name"]),
        target_dim=1,
        dropout=float(config.get("dropout", 0.1)),
        mlp_hidden_size=int(config.get("mlp_hidden_size", 1024)),
        torch_dtype=_dtype_from_name(str(config.get("torch_dtype", "bf16"))),
        attn_implementation=str(config.get("attn_implementation") or "") or None,
        encoder_frozen=False,
        use_lora=False,
        lora_r=0,
        lora_alpha=0,
        lora_dropout=0.0,
        lora_target_modules=[],
        gradient_checkpointing=False,
        predict_costs=False,
        cost_target_dim=0,
        cost_gradient_mode="joint",
        predict_zero_reward_failure=False,
        embedding_input_layout="single",
        segment_count=1,
    )
    adapter_dir = checkpoint_dir / "encoder"
    nested_adapter_dir = adapter_dir / model.reward_adapter_name
    if nested_adapter_dir.exists():
        adapter_dir = nested_adapter_dir
    if adapter_dir.exists():
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("PEFT is required to load a saved LoRA verifier adapter") from exc
        model.encoder = PeftModel.from_pretrained(
            model.encoder,
            str(adapter_dir),
            adapter_name=model.reward_adapter_name,
            is_trainable=False,
        )
    head_path = checkpoint_dir / "scorer_head.pt"
    if not head_path.exists():
        raise FileNotFoundError(f"Missing scorer head: {head_path}")
    model.reward_head.load_state_dict(torch.load(head_path, map_location="cpu"))
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--target-route-idxs", default=None)
    parser.add_argument("--route-order", default="0,1,2,3")
    parser.add_argument("--max-seq-length", type=int, default=24000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--loss-type", choices=["mse", "soft_bce", "proxy_listwise_ce"], default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_config = _read_json(checkpoint_dir / "train_config.json")
    loss_type = str(args.loss_type or saved_config.get("loss_type") or "mse")
    model_name = str(saved_config["model_name"])
    tokenizer_dir = checkpoint_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir if tokenizer_dir.exists() else model_name), padding_side="left")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    all_route_labels = _load_route_labels(dataset_dir)
    target_route_idxs = _parse_route_indices(args.target_route_idxs, len(all_route_labels))
    route_labels = [all_route_labels[int(idx)] for idx in target_route_idxs]
    order = _parse_int_list(str(args.route_order))
    if sorted(order) != list(range(len(route_labels))):
        raise ValueError(f"--route-order must contain selected local route ids exactly once; got {order}")

    split_rows = list(_load_split(dataset_dir, str(args.split)))
    meta_by_problem = _metadata_by_problem_id(split_rows)
    dataset = CascadeAttemptDataset(
        split_rows,
        tokenizer,
        route_labels,
        int(args.max_seq_length),
        route_idxs=order,
        source_route_idxs=target_route_idxs,
    )
    if len(dataset) == 0:
        raise ValueError("Prepared empty scoring dataset")
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=lambda batch: _collate(batch, int(pad_token_id)),
        num_workers=0,
    )

    device = torch.device(str(args.device) if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    model, model_config = _load_saved_verifier(checkpoint_dir, device)

    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="score verifier"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].float().to(device)
        logits, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        _loss, preds = _score_loss_and_predictions(logits, targets, loss_type=loss_type, reduction="none")
        preds_cpu = preds.detach().cpu()
        targets_cpu = targets.detach().cpu()
        for local_idx, row_idx in enumerate(batch["row_indices"].tolist()):
            source_meta = dataset.rows[int(row_idx)]
            problem_id = str(source_meta["problem_id"])
            route_idx = int(source_meta["route_idx"])
            row_meta = meta_by_problem.get(problem_id, {})
            output_tokens = row_meta.get("route_output_tokens")
            prompt_tokens = row_meta.get("route_prompt_tokens")
            proxy_rewards = row_meta.get("route_rewards")
            proxy_successes = row_meta.get("route_successes")
            failure_types = row_meta.get("route_failure_types")
            rows.append(
                {
                    "problem_id": problem_id,
                    "original_problem_id": row_meta.get("original_problem_id") or problem_id,
                    "rollout_idx": row_meta.get("rollout_idx"),
                    "dataset": source_meta["dataset"],
                    "repo": source_meta["repo"],
                    "language": source_meta["language"],
                    "route_idx": route_idx,
                    "source_route_idx": int(source_meta["source_route_idx"]),
                    "route_label": source_meta["route_label"],
                    "true_proxy_score": float(targets_cpu[local_idx, 0].item()),
                    "pred_score": float(preds_cpu[local_idx, 0].item()),
                    "output_tokens": None if not isinstance(output_tokens, list) else output_tokens[int(source_meta["source_route_idx"])],
                    "prompt_tokens": None if not isinstance(prompt_tokens, list) else prompt_tokens[int(source_meta["source_route_idx"])],
                    "proxy_reward": None if not isinstance(proxy_rewards, list) else proxy_rewards[int(source_meta["source_route_idx"])],
                    "proxy_success": None if not isinstance(proxy_successes, list) else proxy_successes[int(source_meta["source_route_idx"])],
                    "failure_type": None if not isinstance(failure_types, list) else failure_types[int(source_meta["source_route_idx"])],
                }
            )

    _write_jsonl(output_dir / f"{args.split}_verifier_scores.jsonl", rows)
    (output_dir / "score_config.json").write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "split": str(args.split),
                "n_scored_attempts": len(rows),
                "route_labels": route_labels,
                "target_route_idxs": [int(idx) for idx in target_route_idxs],
                "route_order": order,
                "loss_type": loss_type,
                "saved_model_config": model_config,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Wrote {len(rows)} scores to {output_dir / (str(args.split) + '_verifier_scores.jsonl')}")


if __name__ == "__main__":
    main()

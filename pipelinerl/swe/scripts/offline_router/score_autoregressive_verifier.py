#!/usr/bin/env python3
"""
Run the trained LoRA verifier on eval trajectories and output per-instance
P(Yes) scores for abstention analysis.

At inference, enable_thinking=True so the verifier CoT-reasons before
producing its Yes/No answer. The score is extracted from the logit of the
first generated token (Yes vs No) — we run with max_new_tokens=1 in the
no-thinking path to get the logit directly, or generate up to budget tokens
with thinking and read the logit of the answer token.

For simplicity we use a greedy logit approach: feed the prompt, read the
last-token logit over {Yes, No} without actually generating — this is fast
and doesn't require a thinking budget.

Output: eval_verifier_scores.jsonl with fields:
  instance_id, p_yes, label (bool), resolved (bool, from parquet)

Usage:
  python score_autoregressive_verifier.py \\
    --lora-adapter-dir /mnt/.../verifier_lora \\
    --trajectories-path /mnt/.../trajectories_eval.jsonl \\
    --labels-parquet-dir /mnt/.../collect/eval \\
    --output-path /mnt/.../eval_verifier_scores.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
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
    logger.info("Loaded %d trajectories from %s", len(records), path)
    return records


def load_parquet_labels(parquet_dir: str, route_idx: int = 3) -> dict[str, bool]:
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
    logger.info("Loaded %d labels (route_idx=%d): %d pos / %d neg", len(labels), route_idx, pos, len(labels) - pos)
    return labels


def build_prompt(traj: dict[str, Any], tokenizer: Any, include_thinking: bool, max_seq_length: int) -> torch.Tensor:
    stmt = str(traj.get("problem_statement") or "").strip()
    thinking = str(traj.get("thinking_text") or "").strip()
    patch = str(traj.get("patch_text") or traj.get("model_patch") or "").strip()

    user_content = VERIFIER_TEMPLATE.format(
        problem_statement=stmt,
        thinking_text=thinking if include_thinking else "(not provided)",
        patch_text=patch,
    )
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids
    if input_ids.shape[1] > max_seq_length:
        input_ids = input_ids[:, -max_seq_length:]
    return input_ids


@torch.no_grad()
def score_all(
    model: torch.nn.Module,
    tokenizer: Any,
    trajectories: dict[str, dict[str, Any]],
    labels: dict[str, bool],
    device: torch.device,
    include_thinking: bool,
    max_seq_length: int,
) -> list[dict[str, Any]]:
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    yes_id = yes_ids[0] if yes_ids else None
    no_id = no_ids[0] if no_ids else None

    model.eval()
    results = []
    pids = sorted(trajectories.keys())
    for i, pid in enumerate(pids):
        traj = trajectories[pid]
        resolved = labels.get(pid)
        if resolved is None:
            continue

        input_ids = build_prompt(traj, tokenizer, include_thinking, max_seq_length).to(device)
        logits = model(input_ids=input_ids).logits[0, -1]

        yes_logit = logits[yes_id].item() if yes_id is not None else 0.0
        no_logit = logits[no_id].item() if no_id is not None else 0.0
        p_yes = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)[0].item()

        results.append({
            "instance_id": pid,
            "p_yes": p_yes,
            "resolved": resolved,
        })
        if (i + 1) % 50 == 0:
            logger.info("Scored %d / %d", i + 1, len(pids))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Score eval trajectories with trained LoRA verifier")
    parser.add_argument("--lora-adapter-dir", required=True,
                        help="Directory containing the saved LoRA adapter (from train_autoregressive_verifier.py)")
    parser.add_argument("--base-model-name", default="Qwen/Qwen3-4B-Thinking-2507",
                        help="Base model (must match what was used for training)")
    parser.add_argument("--trajectories-path", required=True,
                        help="Eval trajectories JSONL (from collect_cot_trajectories.py)")
    parser.add_argument("--labels-parquet-dir", required=True,
                        help="Eval parquet dir with route_successes labels")
    parser.add_argument("--label-route-idx", type=int, default=3,
                        help="Index into route_successes for the target model (default: 3 = gpt-oss-120b)")
    parser.add_argument("--output-path", required=True,
                        help="Output JSONL path for per-instance scores")
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--include-thinking", action="store_true", default=True)
    parser.add_argument("--no-include-thinking", dest="include_thinking", action="store_false")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading tokenizer from %s", args.lora_adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.lora_adapter_dir, trust_remote_code=True)

    logger.info("Loading base model: %s", args.base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    from peft import PeftModel
    logger.info("Loading LoRA adapter from %s", args.lora_adapter_dir)
    model = PeftModel.from_pretrained(base_model, args.lora_adapter_dir)
    model = model.to(device)

    trajectories = load_trajectories(args.trajectories_path)
    labels = load_parquet_labels(args.labels_parquet_dir, args.label_route_idx)

    results = score_all(
        model, tokenizer, trajectories, labels, device,
        include_thinking=args.include_thinking,
        max_seq_length=args.max_seq_length,
    )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for rec in results:
            fh.write(json.dumps(rec) + "\n")

    pos = sum(r["resolved"] for r in results)
    logger.info(
        "Wrote %d scores to %s (%d pos / %d neg)",
        len(results), out_path, pos, len(results) - pos,
    )

    # Quick AUC report
    if len(results) >= 2:
        try:
            from sklearn.metrics import roc_auc_score
            scores = [r["p_yes"] for r in results]
            true_labels = [int(r["resolved"]) for r in results]
            auc = roc_auc_score(true_labels, scores)
            logger.info("AUC: %.4f", auc)
        except Exception as e:
            logger.warning("AUC computation failed: %s", e)


if __name__ == "__main__":
    main()

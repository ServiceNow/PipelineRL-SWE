#!/usr/bin/env python3
"""
Run inference with a trained autoregressive verifier LoRA adapter and save
per-instance P(Yes) scores.

Output JSONL: {problem_id, p_yes, resolved}  (same format as cot abstention predictor)

Usage:
  python score_autoreg_verifier.py \
    --adapter-dir /mnt/.../autoreg_verifier_train_XYZ \
    --trajectories-path /mnt/.../trajectories_eval.jsonl \
    --labels-parquet-dir /mnt/.../collect/eval \
    --output-path /mnt/.../autoreg_eval_predictions.jsonl
"""
import argparse
import json
import logging
from pathlib import Path

import torch

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--trajectories-path", required=True)
    parser.add_argument("--labels-parquet-dir", required=True)
    parser.add_argument("--label-route-idx", type=int, default=3)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--include-thinking", action="store_true", default=True)
    parser.add_argument("--no-include-thinking", dest="include_thinking", action="store_false")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import pandas as pd

    logger.info("Loading tokenizer from %s", args.adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)

    logger.info("Loading base model %s", args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    logger.info("Loading LoRA adapter from %s", args.adapter_dir)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Load labels
    parquet_paths = sorted(Path(args.labels_parquet_dir).glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parquet_paths])
    labels: dict[str, bool] = {}
    for _, row in df.iterrows():
        pid = str(row.get("problem_id") or "").strip()
        if not pid:
            continue
        s = row.get("route_successes")
        if s is not None and len(s) > args.label_route_idx:
            labels[pid] = bool(s[args.label_route_idx])
    logger.info("Loaded %d labels", len(labels))

    # Load trajectories
    trajs: dict[str, dict] = {}
    with open(args.trajectories_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = str(r.get("problem_id") or r.get("instance_id") or "").strip()
            if pid:
                trajs[pid] = r
    logger.info("Loaded %d trajectories", len(trajs))

    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    yes_id = yes_ids[0] if yes_ids else None
    no_id = no_ids[0] if no_ids else None

    common = sorted(set(trajs) & set(labels))
    logger.info("Scoring %d instances", len(common))

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    with out_path.open("w") as fh:
        for pid in common:
            traj = trajs[pid]
            stmt = str(traj.get("problem_statement") or "").strip()
            thinking = str(traj.get("thinking_text") or "").strip()
            patch = str(traj.get("patch_text") or traj.get("model_patch") or "").strip()

            user_content = VERIFIER_TEMPLATE.format(
                problem_statement=stmt,
                thinking_text=thinking if args.include_thinking else "(not provided)",
                patch_text=patch,
            )
            messages = [
                {"role": "system", "content": VERIFIER_SYSTEM},
                {"role": "user", "content": user_content},
            ]
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            input_ids = enc.input_ids
            if input_ids.shape[1] > args.max_seq_length:
                input_ids = input_ids[:, -args.max_seq_length:]
            input_ids = input_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids=input_ids).logits[0, -1]
            yes_logit = logits[yes_id].item() if yes_id is not None else 0.0
            no_logit = logits[no_id].item() if no_id is not None else 0.0
            p_yes = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)[0].item()

            fh.write(json.dumps({"problem_id": pid, "p_yes": p_yes, "resolved": labels[pid]}) + "\n")
            fh.flush()
            n_done += 1
            if n_done % 50 == 0:
                logger.info("Scored %d / %d", n_done, len(common))

    logger.info("Done. Wrote %d predictions to %s", n_done, out_path)


if __name__ == "__main__":
    main()

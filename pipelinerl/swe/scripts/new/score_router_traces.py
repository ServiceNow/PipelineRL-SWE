#!/usr/bin/env python
import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from pipelinerl.finetune.value_model import AutoModelForCausalLMWithValueHead
from pipelinerl.swe.scripts.new.router_trace_utils import load_router_traces

logger = logging.getLogger(__name__)


def _pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_prompt_last_scores(
    model: AutoModelForCausalLMWithValueHead,
    tokenizer,
    device: torch.device,
    prompt_text: str,
    output_text: str,
) -> list[float] | None:
    if not prompt_text:
        return None

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt_text + (output_text or ""), add_special_tokens=False, return_tensors="pt").input_ids

    prompt_len = int(prompt_ids.shape[1])
    full_len = int(full_ids.shape[1])
    if prompt_len <= 0 or full_len <= prompt_len:
        return None

    input_ids = full_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    performance_values = outputs.performance_value
    if performance_values is None:
        return None

    prompt_last_idx = prompt_len - 1
    if prompt_last_idx >= performance_values.shape[1]:
        return None

    prompt_last_scores = performance_values[0, prompt_last_idx]
    return prompt_last_scores.detach().float().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score router traces with the performance value head.")
    parser.add_argument(
        "--input-glob",
        action="append",
        required=True,
        help="Glob pattern for router trace JSONL files. Can be provided multiple times.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path with appended scores.")
    parser.add_argument("--model-path", required=True, help="Checkpoint path containing performance_value_head.pt")
    parser.add_argument("--score-key", default="policy_value_prompt_last_all")
    parser.add_argument("--device", default="auto", help="Torch device (e.g. cuda:0, cpu, auto)")
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--all-model-versions", action="store_true", help="Do not filter to latest model version")
    parser.add_argument("--keep-duplicates", action="store_true", help="Do not dedupe by problem_id")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    split = None if args.split == "all" else args.split
    traces = load_router_traces(
        input_globs=args.input_glob,
        split=split,
        latest_model_only=not args.all_model_versions,
        dedupe_by_problem=not args.keep_duplicates,
    )
    if args.max_records is not None:
        traces = traces[: args.max_records]
    logger.info("Loaded %d traces", len(traces))

    device = _pick_device(args.device)
    logger.info("Loading tokenizer/model from %s on %s", args.model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_path)
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)
    model.eval()

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scored = 0
    missing_prompt = 0
    failed = 0

    with output_path.open("w") as sink:
        for trace in tqdm(traces, desc="Scoring traces", unit="trace"):
            policy = trace.get("policy") or {}
            prompt_text = policy.get("prompt_text")
            output_text = policy.get("output_text")
            if not isinstance(prompt_text, str):
                trace[args.score_key] = None
                missing_prompt += 1
                sink.write(json.dumps(trace) + "\n")
                continue

            try:
                scores = _compute_prompt_last_scores(model, tokenizer, device, prompt_text, output_text or "")
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Scoring failed for problem_id=%s: %s", trace.get("problem_id"), exc)
                scores = None
                failed += 1

            if scores is not None:
                scored += 1
            trace[args.score_key] = scores
            sink.write(json.dumps(trace) + "\n")

    logger.info(
        "Done. output=%s scored=%d missing_prompt=%d failed=%d total=%d",
        output_path,
        scored,
        missing_prompt,
        failed,
        len(traces),
    )


if __name__ == "__main__":
    main()

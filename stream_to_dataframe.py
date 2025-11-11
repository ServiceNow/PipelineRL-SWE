#!/usr/bin/env python3
"""Utility to materialize actor stream entries into a pandas DataFrame.

Each row represents a (step 2, step 3) pair: the query-generator request and
its matched expert response. The resulting frame contains handy columns like
stage input, context, question, copied code blocks, and expert advice.

Example
-------
python stream_to_dataframe.py \
    --path streams/actor/0/0/0.jsonl \
    --sample 5 \
    --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

MSG_PATTERN = re.compile(r"<\|im_start\|>([^\n]+)\n(.*?)<\|im_end\|>", re.S)
CODE_PATTERN = re.compile(r"<code path=\"([^\"]+)\">(.*?)</code>", re.S)


def parse_messages(text: str) -> List[Tuple[str, str]]:
    """Split a completion `text` into (role, content) messages."""
    return [(role.strip(), content) for role, content in MSG_PATTERN.findall(text)]


def last_message(text: str, role: str) -> str:
    for r, content in reversed(parse_messages(text)):
        if r == role:
            return content.strip()
    return ""


def user_before_last_assistant(text: str) -> str:
    messages = parse_messages(text)
    last_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i][0] == "assistant":
            last_idx = i
            break
    if last_idx is None:
        return ""
    for j in range(last_idx - 1, -1, -1):
        if messages[j][0] == "user":
            return messages[j][1].strip()
    return ""


def extract_tag(block: str, tag: str) -> str:
    matches = list(re.finditer(rf"<{tag}>(.*?)</{tag}>", block, re.S))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def extract_code_blocks(block: str) -> List[Dict[str, str]]:
    return [
        {"path": path, "code": body.strip()}
        for path, body in CODE_PATTERN.findall(block)
    ]


def summarize_code_paths(code_blocks: List[Dict[str, str]]) -> str:
    return ", ".join(block["path"] for block in code_blocks)


def extract_stage_input(text: str) -> str:
    return user_before_last_assistant(text)


def parse_stage_input(stage_text: str) -> Dict[str, str]:
    stage_text = stage_text.strip()
    sections = {
        "stage_problem": "",
        "stage_name": "",
        "stage_input_body": stage_text,
    }
    if not stage_text:
        return sections

    problem_match = re.search(r"Problem:\s*(.*?)(?:\n\nStage:|\nStage:)", stage_text, re.S)
    if problem_match:
        sections["stage_problem"] = problem_match.group(1).strip()

    stage_match = re.search(r"Stage:\s*([^\n]+)", stage_text)
    if stage_match:
        sections["stage_name"] = stage_match.group(1).strip()

    input_match = re.search(r"Stage Input:\s*(.*)", stage_text, re.S)
    if input_match:
        sections["stage_input_body"] = input_match.group(1).strip()

    return sections


def extract_advice(text: str) -> str:
    user_block = last_message(text, "user")
    if not user_block:
        return ""
    marker = "Expert advice received:"
    idx = user_block.rfind(marker)
    if idx != -1:
        payload = user_block[idx + len(marker) :]
    else:
        payload = user_block
    end_marker = "=== END EXPERT GUIDANCE ==="
    end_idx = payload.rfind(end_marker)
    if end_idx != -1:
        payload = payload[:end_idx]
    return payload.strip()


def iter_entries(path: Path, max_lines: Optional[int] = None) -> Iterable[Dict]:
    with path.open() as fh:
        total_lines = sum(1 for _ in fh)
    # reopen for actual iteration
    with path.open() as fh:
        iterator = enumerate(fh, start=1)
        progress = tqdm(
            iterator,
            desc="Reading stream",
            unit="lines",
            total=min(total_lines, max_lines) if max_lines else total_lines,
        )
        for line_idx, line in progress:
            if max_lines is not None and line_idx > max_lines:
                break
            if not line.strip():
                continue
            try:
                batch = json.loads(line)
            except json.JSONDecodeError:
                continue
            for entry in batch:
                yield entry


def build_dataframe(path: Path, max_lines: Optional[int] = None) -> pd.DataFrame:
    rows: List[Dict] = []
    pending: Dict[Tuple[int, int], Dict] = {}

    for entry in iter_entries(path, max_lines):
        metadata = entry.get("metadata") or {}
        step = metadata.get("step_index")
        key = (metadata.get("model_version"), metadata.get("rollout_index"))

        if step == 2:
            assistant_block = last_message(entry["text"], "assistant")
            stage_input = extract_stage_input(entry["text"])
            code_blocks = extract_code_blocks(assistant_block)
            stage_sections = parse_stage_input(stage_input)
            rows_data = {
                "model_version": metadata.get("model_version"),
                "rollout_index": metadata.get("rollout_index"),
                "stage_input": stage_input.strip(),
                "stage_problem": stage_sections["stage_problem"],
                "stage_name": stage_sections["stage_name"],
                "stage_input_body": stage_sections["stage_input_body"],
                "context": extract_tag(assistant_block, "context"),
                "question": extract_tag(assistant_block, "question"),
                "code_paths": summarize_code_paths(code_blocks),
                "code_blocks": code_blocks,
                "assistant_block_raw": assistant_block,
                "text_raw": entry["text"],
            }
            pending[key] = rows_data
        elif step == 3 and key in pending:
            rows_data = pending.pop(key)
            rows_data.update(
                {
                    "expert_advice": extract_advice(entry["text"]),
                    "step3_model_version": metadata.get("model_version"),
                }
            )
            rows.append(rows_data)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, type=Path, help="Path to actor stream JSONL")
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Optional limit on the number of JSONL lines read (handy for sampling tails)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional number of random rows to print from the resulting DataFrame",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed used when --sample is provided"
    )
    parser.add_argument(
        "--out-pickle",
        type=Path,
        required=True,
        help="Destination .pkl file for the resulting DataFrame",
    )
    args = parser.parse_args()

    df = build_dataframe(args.path, max_lines=args.max_lines)
    if df.empty:
        print("No (step2, step3) pairs found in the provided file/segment.")
        return

    df.to_pickle(args.out_pickle)
    print(f"Saved DataFrame with {len(df)} rows to {args.out_pickle}")

    if args.sample:
        sample_df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)
    else:
        sample_df = df
    with pd.option_context("display.max_colwidth", 160):
        print(sample_df)


if __name__ == "__main__":
    main()

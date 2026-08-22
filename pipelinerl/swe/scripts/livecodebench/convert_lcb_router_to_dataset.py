#!/usr/bin/env python3
"""
Convert LCB full-routing router_data (router_{train,eval}.jsonl from
materialize_lcb_full_router.py) into the parquet dataset schema consumed by
train_qwen_embedding_router_baseline.py:

  <output>/metadata.json          {"route_labels": [...]}
  <output>/train/*.parquet
  <output>/eval/*.parquet

Row fields produced:
  problem_id, dataset, repo, language,
  prompt_text           — the LCB problem statement
  primary_output_text   — scout thinking + patch (+ test-feedback block if enabled)
  performance_targets   — [scout_success, oss20_success, oss120_success] floats
  route_prompt_tokens / route_output_tokens
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as in_f:
        return [json.loads(line) for line in in_f if line.strip()]


def _build_primary_output_text(row: dict[str, Any], input_only: bool, include_feedback: bool) -> str:
    parts: list[str] = []
    thinking = str(row.get("thinking_text") or "").strip()
    patch = str(row.get("patch_text") or "").strip()
    feedback = str(row.get("test_feedback") or "").strip()
    if not input_only:
        if thinking:
            parts.append("[Scout Thinking]\n" + thinking)
        parts.append("[Scout Patch]\n" + patch)
        if include_feedback:
            parts.append("[Scout Test Execution Feedback]\n" + feedback)
    return "\n\n".join(parts)


def _convert_split(rows: list[dict[str, Any]], input_only: bool, include_feedback: bool) -> pd.DataFrame:
    out_rows = []
    for row in rows:
        successes = row.get("route_successes")
        if not isinstance(successes, list):
            continue
        targets = [float(v) for v in successes]
        primary = _build_primary_output_text(row, input_only, include_feedback)
        if not input_only and not primary.strip():
            continue
        out_rows.append({
            "problem_id": str(row.get("problem_id")),
            "dataset": "lcb_release_v6_temporal",
            "repo": str(row.get("platform") or ""),
            "language": "python",
            "prompt_text": str(row.get("problem_statement") or ""),
            "primary_output_text": primary,
            "performance_targets": targets,
            "route_labels": list(row.get("route_labels") or []),
            "route_public_successes": [
                None if v is None else float(v) for v in row.get("route_public_successes") or []
            ],
            "route_prompt_tokens": [int(v) for v in row.get("route_prompt_tokens") or []],
            "route_output_tokens": [int(v) for v in row.get("route_completion_tokens") or []],
        })
    return pd.DataFrame(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-only", action="store_true")
    parser.add_argument("--include-test-feedback", action="store_true")
    args = parser.parse_args()

    if args.input_only and args.include_test_feedback:
        raise ValueError("--input-only is incompatible with --include-test-feedback")

    data_dir = Path(args.router_data_dir)
    output_dir = Path(args.output_dir)

    sample = _read_jsonl(data_dir / "router_train.jsonl")[0]
    route_labels = list(sample["route_labels"])

    for split in ("train", "eval"):
        rows = _read_jsonl(data_dir / f"router_{split}.jsonl")
        frame = _convert_split(rows, input_only=args.input_only, include_feedback=args.include_test_feedback)
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        # single shard; datasets loads all parquet files in the dir
        shard_path = split_dir / f"{split}-00000.parquet"
        frame.to_parquet(shard_path)
        print(f"{split}: wrote {len(frame)} rows -> {shard_path}")

    with open(output_dir / "metadata.json", "w") as out_f:
        json.dump({"route_labels": route_labels}, out_f, indent=2)
    print(f"metadata: route_labels={route_labels}")


if __name__ == "__main__":
    main()

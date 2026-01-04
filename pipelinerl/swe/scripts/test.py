#!/usr/bin/env python
"""Quick utility to inspect a PipelineRL actor stream shard."""

import argparse
import json
from pathlib import Path
from typing import Any


def describe_object(obj: Any, prefix: str = "") -> None:
    """Pretty-print the structure of one parsed JSON object."""
    obj_type = type(obj).__name__
    if isinstance(obj, list):
        print(f"{prefix}type=list len={len(obj)}")
        if obj:
            first = obj[0]
            print(f"{prefix}  first element type={type(first).__name__}")
            if isinstance(first, dict):
                print(f"{prefix}  first element keys={list(first.keys())[:10]}")
    elif isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{prefix}type=dict keys={keys[:10]}")
    else:
        print(f"{prefix}type={obj_type}: {obj}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("/mnt/llmd/results/exps/aristides/reason/self_eval_all_stages/streams/actor_test/0/0/0.jsonl"),
        help="Path to the actor JSONL shard to inspect.",
    )
    parser.add_argument(
        "--peek",
        type=int,
        default=5,
        help="How many non-empty JSON lines to inspect.",
    )
    args = parser.parse_args()

    if not args.path.exists():
        raise FileNotFoundError(f"Actor JSONL not found: {args.path}")

    total_lines = 0
    total_items = 0
    detected_stages = set()
    versions = set()

    print(f"Reading {args.path} ...")
    with args.path.open() as handle:
        for line in handle:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed line {total_lines}")
                continue

            if isinstance(parsed, list):
                total_items += len(parsed)
                for obj in parsed:
                    meta = obj.get("metadata") or {}
                    versions.add(meta.get("model_version"))
                    stage = meta.get("stage")
                    if stage:
                        detected_stages.add(stage)
            else:
                total_items += 1
                meta = parsed.get("metadata") if isinstance(parsed, dict) else {}
                if isinstance(meta, dict):
                    versions.add(meta.get("model_version"))
                    stage = meta.get("stage")
                    if stage:
                        detected_stages.add(stage)

            if args.peek > 0:
                args.peek -= 1
                print(f"\n--- line {total_lines} preview ---")
                describe_object(parsed, prefix="  ")
                if isinstance(parsed, list) and parsed:
                    first = parsed[0]
                    if isinstance(first, dict):
                        print(f"  sample metadata: {first.get('metadata')}")

    print("\n=== summary ===")
    print(f"Total lines read: {total_lines}")
    print(f"Total items parsed: {total_items}")
    versions_list = sorted(v for v in versions if v is not None)
    print(f"Unique model versions observed: {versions_list}")
    print(f"Detected stages: {sorted(detected_stages) if detected_stages else 'None'}")


if __name__ == "__main__":
    main()

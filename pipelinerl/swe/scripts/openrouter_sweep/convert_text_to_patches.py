#!/usr/bin/env python3
"""
Post-process per-model prediction JSONL files: convert raw model text output
(search/replace format) into proper unified git diffs that Daytona can apply.

Reads filtered JSONLs from --predictions-dir, loads file_contents from the
HuggingFace language datasets, applies search/replace edits, generates diffs,
and overwrites the model_patch field in-place.

Usage:
    python convert_text_to_patches.py \
        --predictions-dir /mnt/.../openrouter_sweep_collect_XYZ/filtered
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipelinerl.swe.scripts.repair_eval_utils import extract_search_replace_edits
from pipelinerl.swe.utils.repair_utils import apply_edits_to_files, generate_unified_diff

HF_LANGUAGE_DATASETS = [
    "SWE-bench/SWE-smith-py",
    "SWE-bench/SWE-smith-rs",
    "SWE-bench/SWE-smith-java",
    "SWE-bench/SWE-smith-go",
]


def load_file_contents_lookup(dataset_names: list[str]) -> dict[str, dict[str, str]]:
    from datasets import load_dataset
    lookup: dict[str, dict[str, str]] = {}
    for name in dataset_names:
        print(f"Loading {name}...")
        ds = load_dataset(name, split="train")
        for row in ds:
            iid = row.get("instance_id")
            fc = row.get("gold_file_contents") or row.get("file_contents")
            if iid and fc:
                if isinstance(fc, str):
                    try:
                        fc = json.loads(fc)
                    except Exception:
                        continue
                lookup[iid] = fc
        print(f"  → {len(lookup)} instances with file_contents so far")
    return lookup


def make_git_diff(file_contents: dict[str, str], raw_text: str) -> str:
    edits = extract_search_replace_edits(raw_text)
    if not edits:
        return ""
    try:
        new_contents = apply_edits_to_files(file_contents, edits, silent=True)
    except Exception:
        return ""

    parts = []
    for path, new_code in new_contents.items():
        old_code = file_contents.get(path, "")
        if old_code == new_code:
            continue
        hunks = generate_unified_diff(old_code, new_code)
        if not hunks:
            continue
        parts.append(
            f"diff --git a/{path} b/{path}\n"
            f"--- a/{path}\n"
            f"+++ b/{path}\n"
            f"{hunks}"
        )
    return "\n".join(parts)


def convert_jsonl(jsonl_path: Path, lookup: dict[str, dict[str, str]]) -> None:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    converted = skipped = empty = 0
    for row in rows:
        iid = row["instance_id"]
        raw_text = row.get("model_patch", "")
        fc = lookup.get(iid)
        if not fc:
            skipped += 1
            continue
        patch = make_git_diff(fc, raw_text)
        row["model_patch"] = patch
        if patch:
            converted += 1
        else:
            empty += 1

    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"  converted={converted}  empty={empty}  skipped(no fc)={skipped}  → {jsonl_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", required=True)
    args = parser.parse_args()

    pdir = Path(args.predictions_dir)
    jsonl_files = sorted(f for f in pdir.glob("*.jsonl"))
    if not jsonl_files:
        sys.exit(f"No JSONL files found in {pdir}")

    print("Loading HuggingFace file_contents...")
    lookup = load_file_contents_lookup(HF_LANGUAGE_DATASETS)
    print(f"Total instances with file_contents: {len(lookup)}\n")

    for f in jsonl_files:
        convert_jsonl(f, lookup)

    print("\nDone. Re-run launch_daytona.sh to evaluate.")


if __name__ == "__main__":
    main()

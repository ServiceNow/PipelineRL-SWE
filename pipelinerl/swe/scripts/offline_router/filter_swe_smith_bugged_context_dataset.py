#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from tqdm import tqdm

from pipelinerl.swe.scripts.offline_router.validate_swe_smith_bugged_context import (
    apply_patch,
    parse_json_map,
)


def parse_file_map(raw: Any) -> dict[str, str]:
    return parse_json_map(raw)


def keep_row(row: dict[str, Any]) -> tuple[bool, str]:
    bugged = parse_file_map(row.get("repair_file_contents") or row.get("gold_file_contents"))
    fix_patch = str(row.get("repair_target_patch") or row.get("fix_patch") or row.get("patch") or "")
    bug_patch = str(row.get("bug_introducing_patch") or row.get("bug_patch") or "")
    if not bugged:
        return False, "missing_repair_file_contents"
    if not fix_patch:
        return False, "missing_repair_target_patch"
    if not bug_patch:
        return False, "missing_bug_introducing_patch"
    if not apply_patch(bugged, fix_patch):
        return False, "repair_target_patch_apply_failed"
    if apply_patch(bugged, bug_patch):
        return False, "bug_patch_applies_to_repair_context"
    return True, "kept"


def filter_split(input_path: Path, output_path: Path) -> dict[str, Any]:
    ds = load_from_disk(str(input_path))
    kept_indices: list[int] = []
    counts: dict[str, int] = {}
    for idx, row in enumerate(tqdm(ds, desc=f"filter {input_path.name}", unit="row")):
        keep, reason = keep_row(dict(row))
        counts[reason] = counts.get(reason, 0) + 1
        if keep:
            kept_indices.append(idx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.select(kept_indices).save_to_disk(str(output_path))
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_rows": len(ds),
        "kept_rows": len(kept_indices),
        "dropped_rows": len(ds) - len(kept_indices),
        "counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop ambiguous rows from an already prepared SWE-Smith bugged-context dataset.")
    parser.add_argument("--input-root", required=True, help="Root containing ds_train and ds_test")
    parser.add_argument("--output-root", required=True, help="New root to write filtered ds_train and ds_test")
    parser.add_argument("--train-split-name", default="ds_train")
    parser.add_argument("--test-split-name", default="ds_test")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    train_out = output_root / args.train_split_name
    test_out = output_root / args.test_split_name
    if not args.overwrite and (train_out.exists() or test_out.exists()):
        raise SystemExit(f"Refusing to overwrite existing output under {output_root}; pass --overwrite intentionally")

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "splits": {},
    }
    summary["splits"][args.train_split_name] = filter_split(input_root / args.train_split_name, train_out)
    summary["splits"][args.test_split_name] = filter_split(input_root / args.test_split_name, test_out)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "filter_swe_smith_bugged_context_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

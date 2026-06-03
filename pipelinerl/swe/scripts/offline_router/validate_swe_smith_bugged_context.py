#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from tqdm import tqdm
from unidiff import PatchSet


def parse_json_map(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str) and raw.strip():
        value = json.loads(raw)
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
    return {}


def patch_paths(patch: str) -> list[str]:
    try:
        return [pf.path for pf in PatchSet(patch)]
    except Exception:
        return []


def write_files(root: Path, files: dict[str, str], paths: list[str]) -> bool:
    for path in paths:
        if path not in files:
            return False
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(files[path], encoding="utf-8", errors="surrogateescape")
    return True


def apply_patch(files: dict[str, str], patch: str) -> bool:
    paths = patch_paths(patch)
    if not paths:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        if not write_files(root, files, paths):
            return False
        patch_path = root / "candidate.patch"
        patch_path.write_text(patch, encoding="utf-8", errors="surrogateescape")
        result = subprocess.run(
            ["patch", "-p1", "--batch", "--forward", "--reject-file=-", "-i", str(patch_path)],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate corrected SWE-Smith bugged repair context.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    ds = load_from_disk(args.dataset_path)
    stats = {
        "checked": 0,
        "has_marker": 0,
        "bug_applies_to_clean": 0,
        "fix_applies_to_bugged_context": 0,
        "bug_applies_to_bugged_context": 0,
        "missing_fields": 0,
    }
    for row in tqdm(ds, total=min(len(ds), args.limit), desc="validate", unit="row"):
        if stats["checked"] >= args.limit:
            break
        stats["checked"] += 1
        if row.get("swesmith_bugged_context"):
            stats["has_marker"] += 1
        clean = parse_json_map(row.get("reference_file_contents") or row.get("clean_file_contents"))
        bugged = parse_json_map(row.get("repair_file_contents") or row.get("gold_file_contents"))
        bug_patch = str(row.get("bug_introducing_patch") or row.get("bug_patch") or "")
        fix_patch = str(row.get("repair_target_patch") or row.get("fix_patch") or row.get("patch") or "")
        if not clean or not bugged or not bug_patch or not fix_patch:
            stats["missing_fields"] += 1
            continue
        if apply_patch(clean, bug_patch):
            stats["bug_applies_to_clean"] += 1
        if apply_patch(bugged, fix_patch):
            stats["fix_applies_to_bugged_context"] += 1
        if apply_patch(bugged, bug_patch):
            stats["bug_applies_to_bugged_context"] += 1
    print(json.dumps(stats, indent=2, sort_keys=True))
    expected_fix = stats["checked"] - stats["missing_fields"]
    if expected_fix and stats["fix_applies_to_bugged_context"] != expected_fix:
        raise SystemExit("Some repair_target_patch/fix patches did not apply to bugged context")
    if stats["bug_applies_to_bugged_context"]:
        raise SystemExit(
            "Some bug_introducing_patch values still apply to bugged context; "
            "regenerate with ambiguous repair contexts filtered out"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Filter per-model prediction JSONL files to their common instance-ID intersection.

Reads all *.jsonl files in --predictions-dir (excluding slugs in --exclude),
computes the intersection of instance_ids, and writes filtered copies to
--output-dir (default: <predictions-dir>/filtered/).

Usage:
    python filter_to_intersection.py \
        --predictions-dir /mnt/.../openrouter_sweep_collect_XYZ \
        [--output-dir /mnt/.../openrouter_sweep_collect_XYZ/filtered] \
        [--exclude laguna poolside]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--output-dir", default=None,
                        help="Default: <predictions-dir>/filtered/")
    parser.add_argument("--exclude", nargs="*", default=["laguna"],
                        help="Slug substrings to exclude (default: laguna)")
    args = parser.parse_args()

    pdir = Path(args.predictions_dir)
    odir = Path(args.output_dir) if args.output_dir else pdir / "filtered"

    jsonl_files = sorted(
        f for f in pdir.glob("*.jsonl")
        if not any(ex in f.stem for ex in (args.exclude or []))
        and f.stem != "collect"
    )

    if not jsonl_files:
        raise SystemExit(f"No JSONL files found in {pdir} (after exclusions)")

    # Load all predictions keyed by model slug → {instance_id: row}
    model_preds: dict[str, dict[str, dict]] = {}
    for f in jsonl_files:
        rows = {}
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows[row["instance_id"]] = row
        model_preds[f.stem] = rows
        print(f"  {len(rows):3d}/instances  {f.stem}")

    # Intersection
    id_sets = [set(rows) for rows in model_preds.values()]
    intersection = set.intersection(*id_sets)
    print(f"\nIntersection across {len(model_preds)} models: {len(intersection)} instances")

    if not intersection:
        raise SystemExit("Empty intersection — nothing to write.")

    # Write filtered JSONLs
    odir.mkdir(parents=True, exist_ok=True)
    for slug, rows in model_preds.items():
        out_path = odir / f"{slug}.jsonl"
        kept = [rows[iid] for iid in sorted(intersection) if iid in rows]
        with open(out_path, "w") as fh:
            for row in kept:
                fh.write(json.dumps(row) + "\n")
        print(f"  wrote {len(kept)} rows → {out_path.name}")

    print(f"\nFiltered predictions written to: {odir}")


if __name__ == "__main__":
    main()

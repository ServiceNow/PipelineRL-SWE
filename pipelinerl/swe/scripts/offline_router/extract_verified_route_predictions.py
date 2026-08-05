#!/usr/bin/env python3
"""
Extract model patches from the 5-route SWE-bench Verified parquet and write
per-model predictions JSONL suitable for run_swebench_eval_daytona.py.

The parquet's route_outputs column contains raw model text; this script converts
it to git-unified-diff patches using the same search/replace extraction logic as
collect_model_discovery_candidates.py.

Output: one JSONL per route with {instance_id, model_patch, model} rows.

Usage:
  python extract_verified_route_predictions.py \
    --parquet-dir  /mnt/.../collect/eval \
    --verified-dataset-path /mnt/llmd/data/swebench_verified/all_16k/ds \
    --output-dir   /mnt/.../verified_predictions \
    --routes 3         # comma-separated route indices (default: all)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

from pipelinerl.swe.scripts.repair_eval_utils import extract_search_replace_edits
from pipelinerl.swe.utils.repair_utils import apply_edits_to_files, get_normalized_patch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _text_to_patch(output_text: str, file_contents: dict[str, str]) -> str:
    """Convert search/replace model output to a unified git diff."""
    if not output_text or not output_text.strip():
        return ""
    edits = extract_search_replace_edits(output_text)
    if not edits:
        return ""
    try:
        modified_contents = apply_edits_to_files(file_contents, edits, silent=True)
        patch_dict = get_normalized_patch(file_contents, modified_contents)
        if not patch_dict:
            return ""
        return "\n".join(patch_dict.values())
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True,
                        help="Dir with eval parquet shards from the 5-route collection")
    parser.add_argument("--verified-dataset-path", required=True,
                        help="Local HF dataset path for SWE-bench Verified (has gold_file_contents)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--routes", default=None,
                        help="Comma-separated route indices to extract (default: all)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load parquet
    paths = sorted(Path(args.parquet_dir).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquets in {args.parquet_dir}")
    df = pd.concat([pd.read_parquet(p) for p in paths])
    logger.info("Loaded %d instances from parquet", len(df))

    # Determine routes to extract
    n_routes = max(len(r) for r in df["route_outputs"])
    if args.routes:
        route_indices = [int(x.strip()) for x in args.routes.split(",")]
    else:
        route_indices = list(range(n_routes))

    # Get route labels from parquet if available
    route_labels: list[str] = []
    if "route_labels" in df.columns:
        rl = df["route_labels"].iloc[0]
        route_labels = list(rl) if rl is not None else []
    if not route_labels:
        route_labels = [f"route_{i}" for i in range(n_routes)]

    # Load file contents from the SWE-bench Verified dataset for patch construction
    logger.info("Loading SWE-bench Verified dataset from %s", args.verified_dataset_path)
    ds = load_from_disk(args.verified_dataset_path)
    file_contents_by_id: dict[str, dict[str, str]] = {}
    for row in ds:
        iid = str(row.get("id") or row.get("instance_id") or "").strip()
        if not iid:
            continue
        gc = row.get("gold_file_contents")
        if gc:
            file_contents_by_id[iid] = dict(gc) if isinstance(gc, dict) else {}
    logger.info("Loaded file contents for %d instances", len(file_contents_by_id))

    for route_idx in route_indices:
        label = route_labels[route_idx] if route_idx < len(route_labels) else f"route_{route_idx}"
        slug = label.replace(":", "_").replace("/", "_").replace(" ", "_")
        out_path = out_dir / f"predictions_{slug}.jsonl"

        n_ok = n_empty = n_fail = 0
        with out_path.open("w") as fh:
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Route {route_idx} ({label})"):
                iid = str(row.get("problem_id") or "").strip()
                outputs = row.get("route_outputs")
                if outputs is None or route_idx >= len(outputs):
                    n_fail += 1
                    continue
                raw_text = str(outputs[route_idx] or "").strip()
                if not raw_text:
                    n_empty += 1
                    patch = ""
                else:
                    file_contents = file_contents_by_id.get(iid, {})
                    patch = _text_to_patch(raw_text, file_contents)
                    if not patch:
                        # Fall back to raw text — Daytona will try git apply anyway
                        patch = raw_text
                    n_ok += 1
                fh.write(json.dumps({
                    "instance_id": iid,
                    "model_patch": patch,
                    "model": label,
                }) + "\n")

        logger.info("Route %d (%s): %d patches, %d empty, %d missing → %s",
                    route_idx, label, n_ok, n_empty, n_fail, out_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert the existing 4-route SWE-Smith real-label parquet into the
trajectories_{train,eval}.jsonl format expected by train_cot_abstention_predictor.py.

The parquet already has:
  problem_id          -- instance ID
  problem_statement   -- problem text
  primary_output_text -- Qwen3-4B-Instruct-2507 patch (route 0)

Output JSONL rows: {problem_id, problem_statement, thinking_text, patch_text}
thinking_text is always "" because Instruct doesn't produce a <think> block.

Usage:
  python extract_instruct_patches_as_trajectories.py \
    --train-parquet-dir /mnt/.../collect/train \
    --eval-parquet-dir  /mnt/.../collect/eval \
    --output-dir        /mnt/.../instruct_trajectories
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _extract_split(parquet_dir: Path, output_path: Path) -> None:
    paths = sorted(parquet_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")
    df = pd.concat([pd.read_parquet(p) for p in paths])
    logger.info("Loaded %d rows from %s", len(df), parquet_dir)

    n_ok = n_skip = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        for _, row in df.iterrows():
            pid = str(row.get("problem_id") or row.get("instance_id") or "").strip()
            problem_statement = str(row.get("problem_statement") or "").strip()
            patch_text = str(row.get("primary_output_text") or row.get("output_text") or "").strip()
            if not pid or not patch_text:
                n_skip += 1
                continue
            fh.write(json.dumps({
                "problem_id": pid,
                "problem_statement": problem_statement,
                "thinking_text": "",
                "patch_text": patch_text,
            }) + "\n")
            n_ok += 1

    logger.info("Wrote %d trajectories (%d skipped, no patch) → %s", n_ok, n_skip, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet-dir", required=True)
    parser.add_argument("--eval-parquet-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    _extract_split(Path(args.train_parquet_dir), out / "trajectories_train.jsonl")
    _extract_split(Path(args.eval_parquet_dir), out / "trajectories_eval.jsonl")
    logger.info("Done. Output dir: %s", out)


if __name__ == "__main__":
    main()

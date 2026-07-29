#!/usr/bin/env python3
"""Export problem_id list from parquet shards to a JSON file."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True, help="Directory containing *.parquet shards")
    parser.add_argument("--output-file", required=True, help="Output JSON file path")
    args = parser.parse_args()

    parquet_files = sorted(Path(args.parquet_dir).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {args.parquet_dir}")

    df = pd.concat([pd.read_parquet(p) for p in parquet_files])
    pids = sorted(set(str(x) for x in df["problem_id"] if x))

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(pids, f)

    print(f"{len(pids)} IDs -> {args.output_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Add structured file_context to an existing offline-router parquet dataset.

The current router datasets already include code context inside prompt_text.
This script joins the router rows back to the source SWE dataset by problem_id
and writes a copy with file_context and file_paths as separate fields, enabling
semantic late-fusion embedding layouts without brittle prompt parsing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from pipelinerl.swe.load_datasets import load_local_swe_dataset
from pipelinerl.swe.scripts.offline_router.common import problem_id_from_item
from pipelinerl.swe.scripts.repair_eval_utils import build_repair_messages


DEFAULT_INPUT_DIR = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect"
)
DEFAULT_OUTPUT_DIR = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_labels_4route_file_context_1780639659/collect"
)
DEFAULT_SOURCE_DATASET_PATH = "/mnt/llmd/data/swe_smith_bugged_context_strict/ds_train"
DEFAULT_SOURCE_DATASET_NAME = "swe_smith_train_bugged_context"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_source_by_id(dataset_path: Path, dataset_name: str, seed: int) -> dict[str, dict[str, Any]]:
    rows = load_local_swe_dataset(
        dataset_names=[dataset_name] if dataset_name else [],
        dataset_path=str(dataset_path),
        shuffle=False,
        seed=int(seed),
        dataset_label=dataset_name or None,
        max_samples=None,
    )
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        problem_id = problem_id_from_item(row)
        by_id.setdefault(problem_id, row)
    if not by_id:
        raise ValueError(f"No source rows loaded from {dataset_path}")
    return by_id


def _file_context_from_source(problem: dict[str, Any]) -> tuple[str, list[str]]:
    file_contents = problem.get("file_contents")
    problem_statement = problem.get("problem_statement")
    if not isinstance(file_contents, dict) or not file_contents:
        raise ValueError("source row missing normalized file_contents")
    if not isinstance(problem_statement, str) or not problem_statement.strip():
        raise ValueError("source row missing problem_statement")
    _messages, file_context = build_repair_messages(problem_statement, file_contents)
    return file_context, sorted(str(path) for path in file_contents)


def _copy_sidecar_files(input_dir: Path, output_dir: Path) -> None:
    for path in input_dir.iterdir():
        if path.is_file() and path.suffix == ".json":
            payload = _read_json(path)
            if path.name in {"metadata.json", "collection_config.json", "real_label_materialization_summary.json"}:
                if isinstance(payload, dict):
                    payload = dict(payload)
                    payload["file_context_augmented"] = True
            _write_json(output_dir / path.name, payload)


def _augment_split(
    *,
    input_dir: Path,
    output_dir: Path,
    split: str,
    source_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    in_split = input_dir / split
    out_split = output_dir / split
    out_split.mkdir(parents=True, exist_ok=True)
    shard_summaries: list[dict[str, Any]] = []
    total_rows = 0
    missing_source = 0
    for shard in sorted(in_split.glob("*.parquet")):
        rows = pq.read_table(shard).to_pylist()
        out_rows: list[dict[str, Any]] = []
        for row in tqdm(rows, desc=f"augment {split}/{shard.name}", unit="row"):
            problem_id = str(row.get("problem_id") or "")
            source = source_by_id.get(problem_id)
            if source is None:
                missing_source += 1
                continue
            file_context, file_paths = _file_context_from_source(source)
            out_row = dict(row)
            out_row["file_context"] = file_context
            out_row["file_paths"] = file_paths
            out_rows.append(out_row)
        pq.write_table(pa.Table.from_pylist(out_rows), out_split / shard.name)
        total_rows += len(out_rows)
        shard_summaries.append({"shard": shard.name, "input_rows": len(rows), "written_rows": len(out_rows)})
    return {
        "split": split,
        "written_rows": total_rows,
        "missing_source": missing_source,
        "shards": shard_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-dataset-path", default=DEFAULT_SOURCE_DATASET_PATH)
    parser.add_argument("--source-dataset-name", default=DEFAULT_SOURCE_DATASET_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not (input_dir / "metadata.json").is_file():
        raise FileNotFoundError(f"Missing router dataset metadata: {input_dir / 'metadata.json'}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_by_id = _load_source_by_id(
        Path(args.source_dataset_path),
        str(args.source_dataset_name),
        int(args.seed),
    )
    _copy_sidecar_files(input_dir, output_dir)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_dataset_path": str(args.source_dataset_path),
        "source_dataset_name": str(args.source_dataset_name),
        "source_rows": len(source_by_id),
        "splits": [],
    }
    for split in ("train", "eval"):
        summary["splits"].append(
            _augment_split(
                input_dir=input_dir,
                output_dir=output_dir,
                split=split,
                source_by_id=source_by_id,
            )
        )
    _write_json(output_dir / "file_context_augmentation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

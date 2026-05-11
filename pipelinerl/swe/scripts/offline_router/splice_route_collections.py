#!/usr/bin/env python
"""Append one route from a second offline-router collection to a base collection."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROUTE_VECTOR_FIELDS = [
    "performance_targets",
    "route_rewards",
    "route_successes",
    "route_prompt_tokens",
    "route_output_tokens",
    "route_latencies_s",
    "route_outputs",
    "route_failure_types",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_rows(split_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard_path in sorted(split_dir.glob("*.parquet")):
        rows.extend(pq.read_table(shard_path).to_pylist())
    return rows


def _index_by_problem_id(rows: list[dict[str, Any]], split: str, source: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        problem_id = row.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError(f"{source}/{split} contains a row without a string problem_id")
        if problem_id in indexed:
            raise ValueError(f"{source}/{split} contains duplicate problem_id={problem_id!r}")
        indexed[problem_id] = row
    return indexed


def _append_extra_route(base_row: dict[str, Any], extra_row: dict[str, Any], extra_route_idx: int) -> dict[str, Any]:
    merged = dict(base_row)
    for field in ROUTE_VECTOR_FIELDS:
        base_values = base_row.get(field)
        extra_values = extra_row.get(field)
        if base_values is None and extra_values is None:
            continue
        if not isinstance(base_values, list):
            raise ValueError(f"Base row {base_row.get('problem_id')} has non-list {field}")
        if not isinstance(extra_values, list):
            raise ValueError(f"Extra row {extra_row.get('problem_id')} has non-list {field}")
        if extra_route_idx >= len(extra_values):
            raise ValueError(
                f"Extra row {extra_row.get('problem_id')} has {len(extra_values)} values for {field}; "
                f"cannot take route index {extra_route_idx}"
            )
        merged[field] = list(base_values) + [extra_values[extra_route_idx]]
    return merged


def _write_shards(rows: list[dict[str, Any]], split_dir: Path, split: str, shard_size: int) -> list[int]:
    split_dir.mkdir(parents=True, exist_ok=True)
    shard_counts: list[int] = []
    for shard_idx, start in enumerate(range(0, len(rows), shard_size)):
        shard_rows = rows[start : start + shard_size]
        shard_path = split_dir / f"{split}-{shard_idx:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(shard_rows), shard_path)
        shard_counts.append(len(shard_rows))
    return shard_counts


def _route_labels(metadata: dict[str, Any], name: str) -> list[str]:
    labels = metadata.get("route_labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{name} metadata is missing route_labels")
    return [str(label) for label in labels]


def _route_model_names(metadata: dict[str, Any]) -> list[str] | None:
    names = metadata.get("route_model_names")
    if not isinstance(names, list) or not names:
        return None
    return [str(name) for name in names]


def _splice_split(
    *,
    base_dir: Path,
    extra_dir: Path,
    output_dir: Path,
    split: str,
    extra_route_idx: int,
    shard_size: int,
) -> dict[str, Any]:
    base_rows = _load_rows(base_dir / split)
    extra_rows = _load_rows(extra_dir / split)
    extra_by_id = _index_by_problem_id(extra_rows, split, extra_dir)

    merged_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for base_row in base_rows:
        problem_id = str(base_row["problem_id"])
        extra_row = extra_by_id.get(problem_id)
        if extra_row is None:
            missing.append(problem_id)
            continue
        merged_rows.append(_append_extra_route(base_row, extra_row, extra_route_idx))

    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"{split}: missing {len(missing)} extra rows. First missing: {preview}")

    shard_counts = _write_shards(merged_rows, output_dir / split, split, shard_size)
    return {
        "split": split,
        "n_base_rows": len(base_rows),
        "n_extra_rows": len(extra_rows),
        "n_written": len(merged_rows),
        "shard_row_counts": shard_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-collection-dir", required=True, help="Collection whose rows and first routes are preserved.")
    parser.add_argument("--extra-collection-dir", required=True, help="Collection containing the route to append.")
    parser.add_argument("--output-collection-dir", required=True, help="Where to write the spliced collection.")
    parser.add_argument("--extra-route-idx", type=int, default=1, help="Route index to append from the extra collection.")
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_collection_dir)
    extra_dir = Path(args.extra_collection_dir)
    output_dir = Path(args.output_collection_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    base_metadata = _read_json(base_dir / "metadata.json")
    extra_metadata = _read_json(extra_dir / "metadata.json")
    base_labels = _route_labels(base_metadata, "base")
    extra_labels = _route_labels(extra_metadata, "extra")
    if args.extra_route_idx >= len(extra_labels):
        raise ValueError(f"extra-route-idx {args.extra_route_idx} is out of range for labels {extra_labels}")

    split_summaries = []
    for split in ("train", "eval"):
        split_summaries.append(
            _splice_split(
                base_dir=base_dir,
                extra_dir=extra_dir,
                output_dir=output_dir,
                split=split,
                extra_route_idx=args.extra_route_idx,
                shard_size=args.shard_size,
            )
        )

    metadata = dict(base_metadata)
    metadata["route_labels"] = base_labels + [extra_labels[args.extra_route_idx]]
    base_names = _route_model_names(base_metadata)
    extra_names = _route_model_names(extra_metadata)
    if base_names is not None and extra_names is not None and args.extra_route_idx < len(extra_names):
        metadata["route_model_names"] = base_names + [extra_names[args.extra_route_idx]]
    metadata["split_summaries"] = split_summaries
    metadata["spliced_from"] = {
        "base_collection_dir": str(base_dir),
        "extra_collection_dir": str(extra_dir),
        "extra_route_idx": args.extra_route_idx,
        "extra_route_label": extra_labels[args.extra_route_idx],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    config = _read_json(base_dir / "collection_config.json")
    config["spliced_from"] = metadata["spliced_from"]
    (output_dir / "collection_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"output_collection_dir": str(output_dir), "route_labels": metadata["route_labels"], "splits": split_summaries}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build an offline-router parquet collection by appending discovery JSONL model outputs.

This lets us reuse an existing router collection for prompt/problem metadata and
existing routes, then add routes collected by collect_model_discovery_candidates.
It can also replace route 0 with a discovery output, which is useful for making a
model a scout/observation-only route.
"""

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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_parquet_rows(split_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard_path in sorted(split_dir.glob("*.parquet")):
        rows.extend(pq.read_table(shard_path).to_pylist())
    return rows


def _load_discovery_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instance_id = str(row.get("instance_id") or "")
            if instance_id:
                rows[instance_id] = row
    return rows


def _parse_base_route(raw: str) -> tuple[str, int, str]:
    parts = raw.split(":", 2)
    if len(parts) != 3 or parts[0] != "base":
        raise ValueError(f"Invalid base route spec {raw!r}; expected base:<idx>:<label>")
    return "base", int(parts[1]), parts[2]


def _parse_discovery_route(raw: str) -> tuple[str, str, str]:
    parts = raw.split(":", 2)
    if len(parts) != 3 or parts[0] != "discovery":
        raise ValueError(f"Invalid discovery route spec {raw!r}; expected discovery:<key>:<label>")
    return "discovery", parts[1], parts[2]


def _parse_route_spec(raw: str) -> tuple[str, int | str, str]:
    if raw.startswith("base:"):
        kind, idx, label = _parse_base_route(raw)
        return kind, idx, label
    if raw.startswith("discovery:"):
        kind, key, label = _parse_discovery_route(raw)
        return kind, key, label
    raise ValueError(f"Invalid route spec {raw!r}")


def _discovery_route_values(row: dict[str, Any], success_threshold: float) -> dict[str, Any]:
    reward = float(row.get("proxy_reward") or 0.0)
    success = bool(row.get("proxy_success")) if "proxy_success" in row else bool(reward > success_threshold)
    failure_type = str(row.get("failure_type") or ("request_error" if row.get("request_error") else "unknown"))
    return {
        "performance_targets": reward,
        "route_rewards": reward,
        "route_successes": success,
        "route_prompt_tokens": int(row.get("prompt_tokens") or 0),
        "route_output_tokens": int(row.get("output_tokens") or 0),
        "route_latencies_s": float(row.get("latency_s") or 0.0),
        "route_outputs": str(row.get("output_text") or ""),
        "route_failure_types": failure_type,
    }


def _build_row(
    *,
    base_row: dict[str, Any],
    route_specs: list[tuple[str, int | str, str]],
    discovery_by_key: dict[str, dict[str, dict[str, Any]]],
    success_threshold: float,
) -> dict[str, Any]:
    problem_id = str(base_row.get("problem_id") or "")
    if not problem_id:
        raise ValueError("Base row missing problem_id")

    out = dict(base_row)
    vector_values = {field: [] for field in ROUTE_VECTOR_FIELDS}
    route_outputs: list[str] = []

    for kind, ref, _label in route_specs:
        if kind == "base":
            idx = int(ref)
            for field in ROUTE_VECTOR_FIELDS:
                values = base_row.get(field)
                if not isinstance(values, list) or idx >= len(values):
                    raise ValueError(f"Base row {problem_id} missing {field}[{idx}]")
                vector_values[field].append(values[idx])
            route_outputs.append(str(vector_values["route_outputs"][-1] or ""))
        elif kind == "discovery":
            key = str(ref)
            rows = discovery_by_key.get(key)
            if rows is None:
                raise ValueError(f"No discovery source registered for key={key!r}")
            discovery_row = rows.get(problem_id)
            if discovery_row is None:
                raise KeyError(f"Missing discovery row key={key!r} problem_id={problem_id!r}")
            values = _discovery_route_values(discovery_row, success_threshold)
            for field in ROUTE_VECTOR_FIELDS:
                vector_values[field].append(values[field])
            route_outputs.append(str(values["route_outputs"] or ""))
        else:
            raise ValueError(f"Unsupported route kind {kind!r}")

    for field, values in vector_values.items():
        out[field] = values
    out["primary_output_text"] = route_outputs[0] if route_outputs else ""
    return out


def _write_shards(rows: list[dict[str, Any]], split_dir: Path, split: str, shard_size: int) -> list[int]:
    split_dir.mkdir(parents=True, exist_ok=True)
    counts: list[int] = []
    for shard_idx, start in enumerate(range(0, len(rows), shard_size)):
        shard_rows = rows[start : start + shard_size]
        pq.write_table(pa.Table.from_pylist(shard_rows), split_dir / f"{split}-{shard_idx:05d}.parquet")
        counts.append(len(shard_rows))
    return counts


def _copy_optional_base_files(base_dir: Path, output_dir: Path) -> None:
    for name in ("collection_config.json", "route_distribution_summary.json"):
        source = base_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-collection-dir", required=True)
    parser.add_argument("--output-collection-dir", required=True)
    parser.add_argument(
        "--route-spec",
        action="append",
        required=True,
        help="Output route spec. Use base:<idx>:<label> or discovery:<source-key>:<label>. Order matters.",
    )
    parser.add_argument(
        "--train-discovery-source",
        action="append",
        default=[],
        help="Discovery source as key=/path/to/train/outputs.jsonl",
    )
    parser.add_argument(
        "--eval-discovery-source",
        action="append",
        default=[],
        help="Discovery source as key=/path/to/eval/outputs.jsonl",
    )
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_collection_dir)
    output_dir = Path(args.output_collection_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    route_specs = [_parse_route_spec(raw) for raw in args.route_spec]
    route_labels = [label for _kind, _ref, label in route_specs]

    def parse_sources(raw_sources: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
        parsed: dict[str, dict[str, dict[str, Any]]] = {}
        for raw in raw_sources:
            if "=" not in raw:
                raise ValueError(f"Invalid discovery source {raw!r}; expected key=/path/to/outputs.jsonl")
            key, path_text = raw.split("=", 1)
            key = key.strip()
            path = Path(path_text.strip())
            if not key or not path.exists():
                raise FileNotFoundError(f"Invalid source {raw!r}")
            parsed[key] = _load_discovery_rows(path)
        return parsed

    discovery_by_split = {
        "train": parse_sources(args.train_discovery_source),
        "eval": parse_sources(args.eval_discovery_source),
    }

    split_summaries: list[dict[str, Any]] = []
    missing_by_split: dict[str, list[str]] = {}
    for split in ("train", "eval"):
        base_rows = _load_parquet_rows(base_dir / split)
        output_rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for base_row in base_rows:
            try:
                output_rows.append(
                    _build_row(
                        base_row=base_row,
                        route_specs=route_specs,
                        discovery_by_key=discovery_by_split[split],
                        success_threshold=float(args.success_threshold),
                    )
                )
            except KeyError as exc:
                missing.append(str(exc))
        if missing:
            missing_by_split[split] = missing[:25]
            raise ValueError(f"{split}: missing {len(missing)} discovery rows; first={missing[:3]}")
        shard_counts = _write_shards(output_rows, output_dir / split, split, int(args.shard_size))
        split_summaries.append(
            {
                "split": split,
                "n_base_rows": len(base_rows),
                "n_written": len(output_rows),
                "shard_row_counts": shard_counts,
            }
        )

    base_metadata = _read_json(base_dir / "metadata.json")
    metadata = dict(base_metadata)
    metadata["route_labels"] = route_labels
    metadata["route_model_names"] = route_labels
    metadata["split_summaries"] = split_summaries
    metadata["built_from_discovery"] = {
        "base_collection_dir": str(base_dir),
        "route_specs": [str(raw) for raw in args.route_spec],
        "train_discovery_sources": list(args.train_discovery_source),
        "eval_discovery_sources": list(args.eval_discovery_source),
    }
    _write_json(output_dir / "metadata.json", metadata)
    _copy_optional_base_files(base_dir, output_dir)
    print(json.dumps({"output_collection_dir": str(output_dir), "route_labels": route_labels, "splits": split_summaries}, indent=2))


if __name__ == "__main__":
    main()

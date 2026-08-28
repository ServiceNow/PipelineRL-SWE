#!/usr/bin/env python3
"""Build a versioned full-execution MDP bundle from saved LCB generations.

Unlike the legacy bundle, schema v2 treats complete execution as both the
routing verdict and final outcome. Public-test outcomes are retained only as a
weak-verifier ablation. It also stores aligned draw records, prompt text,
realized token counts, and one canonical calibration/test split manifest.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
import re
from pathlib import Path
from typing import Any

import numpy as np

from pipelinerl.swe.scripts.livecodebench.collect_lcb_expert import (
    LCB_DATASET_REVISION,
    _is_complete,
    _read_latest,
)
from pipelinerl.swe.scripts.livecodebench.mdp_utils import (
    TENSOR_SCHEMA_VERSION,
    write_source_temporal_split_manifest,
    write_split_manifest,
)


MODEL_SLOTS = ["scout", "oss20", "oss120"]
RESULT_NAMES = {
    -5: "test_runner_error",
    -4: "runtime_error",
    -3: "time_limit_exceeded",
    -2: "wrong_answer",
    -1: "global_timeout",
}


def _result_feedback(resolved: bool, result_codes: list[Any]) -> str:
    normalized = [bool(x) if isinstance(x, (bool, np.bool_)) else int(x) for x in result_codes]
    passed = sum(x is True for x in normalized)
    failures = Counter(
        RESULT_NAMES.get(x, "wrong_answer") for x in normalized if x is not True
    )
    status = "PASSED" if resolved else "FAILED"
    details = ", ".join(f"{name}={count}" for name, count in sorted(failures.items()))
    suffix = f"; failure types: {details}" if details else ""
    return f"Full execution: {status}; passed={passed}/{len(normalized)}{suffix}"


def _load_source_metadata(source_dir: Path) -> dict[str, dict[str, Any]]:
    """Load both sides of the corrected source-temporal collection."""
    required = {"problem_statement", "contest_date", "platform", "difficulty"}
    source: dict[str, dict[str, Any]] = {}
    for split in ("train", "eval"):
        source_path = source_dir / f"scout_{split}.jsonl"
        if not source_path.is_file() and split == "train":
            continue
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing corrected source collection: {source_path}")
        rows = _read_latest(source_path)
        missing = [pid for pid, row in rows.items() if not str(row.get("problem_statement") or "").strip()]
        if missing:
            raise ValueError(f"{source_path} has {len(missing)} rows without problem statements")
        overlap = set(source) & set(rows)
        if overlap:
            raise ValueError(f"Source temporal splits overlap; examples={sorted(overlap)[:5]}")
        source.update({
            pid: {
                "problem_id": pid,
                "source_temporal_split": split,
                **{key: row.get(key) for key in required},
            }
            for pid, row in rows.items()
        })
    return source


def _collection_key(path: Path) -> tuple[str, str, int] | None:
    match = re.match(r"^(scout|oss20|oss120)_(train|eval)_d(\d+)$", path.stem)
    if match is None:
        return None
    return match.group(1), match.group(2), int(match.group(3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", action="append", required=True)
    parser.add_argument("--source-collection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-draws", type=int, default=10)
    parser.add_argument(
        "--route-draw-counts", default="",
        help="Optional comma-separated per-route available draws, e.g. scout=4,oss20=10,oss120=10.",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--split-mode", choices=["random", "source_temporal"], default="random")
    parser.add_argument("--temporal-calibration-fraction", type=float, default=0.5,
                        help="Fraction of the later source-temporal split assigned to calibration.")
    parser.add_argument("--require-complete", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    route_draw_counts = {slot: int(args.num_draws) for slot in MODEL_SLOTS}
    if args.route_draw_counts:
        for item in args.route_draw_counts.split(","):
            route, separator, raw_count = item.strip().partition("=")
            if separator != "=" or route not in route_draw_counts:
                raise ValueError(f"Invalid --route-draw-counts item: {item!r}")
            count = int(raw_count)
            if not 1 <= count <= args.num_draws:
                raise ValueError(f"Draw count for {route} must be in [1, {args.num_draws}]")
            route_draw_counts[route] = count
    roots = [Path(value) for value in args.collection_dir]
    source = _load_source_metadata(Path(args.source_collection_dir))
    tables: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for root in roots:
        for path in sorted(root.glob("*_d*.jsonl")):
            parsed = _collection_key(path)
            if parsed is None:
                continue
            route, split, draw = parsed
            if draw >= route_draw_counts[route]:
                continue
            rows = _read_latest(path)
            key = (route, draw)
            table = tables.setdefault(key, {})
            overlap = set(table) & set(rows)
            if overlap:
                raise ValueError(
                    f"Duplicate {route} draw {draw} rows across collection roots; "
                    f"examples={sorted(overlap)[:5]}"
                )
            expected_source_ids = {
                pid for pid, meta in source.items() if meta["source_temporal_split"] == split
            }
            unexpected = set(rows) - expected_source_ids
            if unexpected:
                raise ValueError(
                    f"{path} contains IDs outside source {split} split; "
                    f"examples={sorted(unexpected)[:5]}"
                )
            table.update(rows)

    for slot in MODEL_SLOTS:
        for draw in range(route_draw_counts[slot]):
            if (slot, draw) not in tables:
                raise FileNotFoundError(f"Missing {slot} draw {draw}")

    common_ids = set(source)
    for table in tables.values():
        common_ids &= set(table)
    pids = sorted(common_ids)
    if not pids:
        raise ValueError("No common problem IDs across source and multidraw collections")

    shape = (len(pids), len(MODEL_SLOTS), args.num_draws)
    final = np.zeros(shape, dtype=bool)
    weak = np.zeros(shape, dtype=bool)
    valid = np.zeros(shape, dtype=bool)
    prompt_tokens = np.zeros(shape, dtype=np.float32)
    completion_tokens = np.zeros(shape, dtype=np.float32)
    records: list[dict[str, Any]] = []

    for pi, pid in enumerate(pids):
        for mi, slot in enumerate(MODEL_SLOTS):
            for draw in range(args.num_draws):
                if draw >= route_draw_counts[slot]:
                    continue
                row = tables[(slot, draw)][pid]
                ok = _is_complete(row, LCB_DATASET_REVISION)
                valid[pi, mi, draw] = ok
                if not ok:
                    continue
                final[pi, mi, draw] = bool(row["resolved"])
                weak[pi, mi, draw] = bool(row["public_resolved"])
                prompt_tokens[pi, mi, draw] = float(row.get("prompt_tokens") or 0)
                completion_tokens[pi, mi, draw] = float(row.get("completion_tokens") or 0)
                records.append({
                    "problem_id": pid,
                    "model_slot": slot,
                    "draw_index": draw,
                    "code": str(row.get("code") or row.get("full_output") or ""),
                    "full_execution_feedback": _result_feedback(
                        bool(row["resolved"]), list(row.get("result_codes") or [])
                    ),
                    "full_result_codes": list(row.get("result_codes") or []),
                    "weak_verifier_outcome": bool(row["public_resolved"]),
                    "final_outcome": bool(row["resolved"]),
                    "prompt_tokens": float(row.get("prompt_tokens") or 0),
                    "completion_tokens": float(row.get("completion_tokens") or 0),
                })

    complete = np.asarray([
        all(valid[pi, mi, :route_draw_counts[slot]].all() for mi, slot in enumerate(MODEL_SLOTS))
        for pi in range(len(pids))
    ], dtype=bool)
    if args.require_complete and not complete.all():
        bad = [pids[i] for i in np.flatnonzero(~complete)[:10]]
        raise ValueError(f"{int((~complete).sum())} problems have invalid draws; examples={bad}")
    keep = complete if args.require_complete else valid.any(axis=(1, 2))
    kept_ids = [pids[i] for i in np.flatnonzero(keep)]
    kept_set = set(kept_ids)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "tensors.npz",
        final_outcome=final[keep],
        execution_outcome=final[keep],
        weak_verifier_outcome=weak[keep],
        valid=valid[keep],
        prompt_tokens=prompt_tokens[keep],
        completion_tokens=completion_tokens[keep],
        problem_ids=np.array(kept_ids),
        model_slots=np.array(MODEL_SLOTS),
        schema_version=np.array(TENSOR_SCHEMA_VERSION),
    )
    with open(out / "problems.jsonl", "w") as handle:
        for pid in kept_ids:
            handle.write(json.dumps(source[pid]) + "\n")
    with open(out / "draw_records.jsonl", "w") as handle:
        for row in records:
            if row["problem_id"] in kept_set:
                handle.write(json.dumps(row) + "\n")
    if args.split_mode == "source_temporal":
        manifest = write_source_temporal_split_manifest(
            out / "split_manifest.json",
            [source[pid] for pid in kept_ids],
            calibration_fraction_of_later=args.temporal_calibration_fraction,
        )
    else:
        manifest = write_split_manifest(
            out / "split_manifest.json",
            kept_ids,
            seed=args.split_seed,
            train_fraction=args.train_fraction,
            calibration_fraction=args.calibration_fraction,
        )
    summary = {
        "schema_version": TENSOR_SCHEMA_VERSION,
        "protocol": "full_execution",
        "num_problems": len(kept_ids),
        "num_draws": int(args.num_draws),
        "route_draw_counts": route_draw_counts,
        "model_slots": MODEL_SLOTS,
        "source_collection_dir": str(Path(args.source_collection_dir)),
        "collection_dirs": [str(root) for root in roots],
        "split_mode": args.split_mode,
        "split_seed": int(args.split_seed),
        "temporal_calibration_fraction": float(args.temporal_calibration_fraction),
        "num_train": len(manifest["train_problem_ids"]),
        "num_calibration": len(manifest["calibration_problem_ids"]),
        "num_test": len(manifest["test_problem_ids"]),
        "solve_rates": {
            slot: float(final[keep, mi][valid[keep, mi]].mean())
            for mi, slot in enumerate(MODEL_SLOTS)
        },
        "mean_prompt_tokens": {
            slot: float(prompt_tokens[keep, mi][valid[keep, mi]].mean())
            for mi, slot in enumerate(MODEL_SLOTS)
        },
        "mean_completion_tokens": {
            slot: float(completion_tokens[keep, mi][valid[keep, mi]].mean())
            for mi, slot in enumerate(MODEL_SLOTS)
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

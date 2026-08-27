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
    source_path = source_dir / "scout_eval.jsonl"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing corrected source collection: {source_path}")
    rows = _read_latest(source_path)
    required = {"problem_statement", "contest_date", "platform", "difficulty"}
    missing = [pid for pid, row in rows.items() if not str(row.get("problem_statement") or "").strip()]
    if missing:
        raise ValueError(f"Source collection has {len(missing)} rows without problem statements")
    return {
        pid: {"problem_id": pid, **{key: row.get(key) for key in required}}
        for pid, row in rows.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", action="append", required=True)
    parser.add_argument("--source-collection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-draws", type=int, default=10)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--require-complete", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    roots = [Path(value) for value in args.collection_dir]
    source = _load_source_metadata(Path(args.source_collection_dir))
    tables: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for root in roots:
        for path in sorted(root.glob("*_eval_d*.jsonl")):
            route = path.stem.split("_eval_")[0]
            if route not in MODEL_SLOTS:
                continue
            draw = int(path.stem.rsplit("_d", 1)[1])
            if draw < args.num_draws:
                key = (route, draw)
                if key in tables:
                    raise ValueError(f"Duplicate table for {key}: {path}")
                tables[key] = _read_latest(path)

    for slot in MODEL_SLOTS:
        for draw in range(args.num_draws):
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

    complete = valid.all(axis=(1, 2))
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
        "model_slots": MODEL_SLOTS,
        "source_collection_dir": str(Path(args.source_collection_dir)),
        "collection_dirs": [str(root) for root in roots],
        "split_seed": int(args.split_seed),
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

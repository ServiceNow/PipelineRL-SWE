#!/usr/bin/env python3
"""Generate grouped rolling-origin splits from a temporal LCB tensor bundle.

Each fold fits on all earlier date blocks, calibrates on the next block, and
reports on the following block. Equal-date problems are never split across
roles. The script only writes manifests; it does not train or submit jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _blocks(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    by_date: dict[str, list[str]] = {}
    for row in rows:
        date = str(row.get("contest_date") or "")
        problem_id = str(row.get("problem_id") or "")
        if not date or not problem_id:
            raise ValueError("Every problem requires contest_date and problem_id")
        by_date.setdefault(date, []).append(problem_id)
    groups = [(date, sorted(ids)) for date, ids in sorted(by_date.items())]
    if len(groups) < count:
        raise ValueError(f"Need at least {count} distinct contest dates, found {len(groups)}")
    target = len(rows) / count
    blocks: list[dict[str, object]] = []
    cursor = 0
    for block_index in range(count):
        remaining_blocks = count - block_index
        selected: list[tuple[str, list[str]]] = []
        selected_count = 0
        while cursor < len(groups):
            groups_left = len(groups) - cursor
            if selected and groups_left <= remaining_blocks - 1:
                break
            next_count = selected_count + len(groups[cursor][1])
            if selected and abs(next_count - target) > abs(selected_count - target):
                break
            selected.append(groups[cursor])
            selected_count = next_count
            cursor += 1
        if not selected:
            selected.append(groups[cursor])
            selected_count = len(groups[cursor][1])
            cursor += 1
        ids = sorted(problem_id for _, group_ids in selected for problem_id in group_ids)
        blocks.append({
            "block_index": block_index,
            "start_date": selected[0][0],
            "end_date": selected[-1][0],
            "problem_ids": ids,
        })
    if cursor != len(groups):
        extra = [problem_id for _, group_ids in groups[cursor:] for problem_id in group_ids]
        blocks[-1]["problem_ids"] = sorted(blocks[-1]["problem_ids"] + extra)
        blocks[-1]["end_date"] = groups[-1][0]
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensors-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-blocks", type=int, default=5)
    args = parser.parse_args()
    if args.num_blocks < 3:
        raise ValueError("--num-blocks must be at least 3")

    rows = _read_jsonl(Path(args.tensors_dir) / "problems.jsonl")
    blocks = _blocks(rows, args.num_blocks)
    folds = []
    for test_block in range(2, len(blocks)):
        train_ids = sorted(
            problem_id
            for block in blocks[: test_block - 1]
            for problem_id in block["problem_ids"]
        )
        calibration_ids = list(blocks[test_block - 1]["problem_ids"])
        test_ids = list(blocks[test_block]["problem_ids"])
        folds.append({
            "fold": test_block - 2,
            "method": "rolling_origin_contiguous_date_blocks",
            "train_problem_ids": train_ids,
            "calibration_problem_ids": calibration_ids,
            "test_problem_ids": test_ids,
            "train_end_date": blocks[test_block - 2]["end_date"],
            "calibration_start_date": blocks[test_block - 1]["start_date"],
            "calibration_end_date": blocks[test_block - 1]["end_date"],
            "test_start_date": blocks[test_block]["start_date"],
            "test_end_date": blocks[test_block]["end_date"],
        })
    output = {
        "schema_version": 1,
        "source_tensors_dir": str(Path(args.tensors_dir)),
        "num_problems": len(rows),
        "num_blocks": args.num_blocks,
        "blocks": blocks,
        "folds": folds,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "num_problems": len(rows),
        "blocks": [len(block["problem_ids"]) for block in blocks],
        "folds": len(folds),
    }, indent=2))


if __name__ == "__main__":
    main()

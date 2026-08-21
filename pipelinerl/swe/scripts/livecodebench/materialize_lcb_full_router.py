#!/usr/bin/env python3
"""Build aligned LCB full-routing examples from corrected tier outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipelinerl.swe.scripts.livecodebench.collect_lcb_trajectories import (
    LCB_DATASET_REVISION,
    is_evaluator_infrastructure_error,
)

ROUTE_LABELS = ["scout", "oss20", "oss120"]


def _latest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with open(path) as in_f:
        for line in in_f:
            if line.strip():
                row = json.loads(line)
                rows[str(row["problem_id"])] = row
    return rows


def _expert_valid(row: dict[str, Any]) -> bool:
    message = str((row.get("eval_metadata") or {}).get("error_message", ""))
    completed = bool(str(row.get("full_output") or "").strip()) or message == "EmptyGeneration"
    return completed and not is_evaluator_infrastructure_error(
        row.get("result_codes"), row.get("eval_metadata")
    )


def _scout_valid(row: dict[str, Any]) -> bool:
    return not (
        is_evaluator_infrastructure_error(
            row.get("_lcb_feedback_result_codes"),
            row.get("_lcb_feedback_eval_metadata"),
            row.get("_tf_failing"),
        )
        or is_evaluator_infrastructure_error(
            row.get("_lcb_result_codes"), row.get("_lcb_eval_metadata")
        )
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as out_f:
        for row in rows:
            out_f.write(json.dumps(row) + "\n")


def materialize_split(source: Path, expert: Path, output: Path, split: str) -> dict[str, int]:
    scout = _latest(source / f"scout_{split}.jsonl")
    high = _latest(source / f"oracle_{split}.jsonl")
    mid = _latest(expert / f"oss20_{split}.jsonl")
    if set(scout) != set(high) or set(scout) != set(mid):
        raise ValueError(
            f"{split} ID mismatch: scout={len(scout)} oss20={len(mid)} oss120={len(high)}"
        )

    aligned: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for pid in sorted(scout):
        s, m, h = scout[pid], mid[pid], high[pid]
        valid = _scout_valid(s) and _expert_valid(m) and _expert_valid(h)
        if not valid:
            excluded.append(
                {
                    "problem_id": pid,
                    "scout_valid": _scout_valid(s),
                    "oss20_valid": _expert_valid(m),
                    "oss120_valid": _expert_valid(h),
                }
            )
            continue
        aligned.append(
            {
                "problem_id": pid,
                "contest_date": s["contest_date"],
                "platform": s["platform"],
                "difficulty": s["difficulty"],
                "problem_statement": s["problem_statement"],
                "thinking_text": s.get("thinking_text", ""),
                "patch_text": s.get("patch_text", ""),
                "test_feedback": s.get("test_feedback", ""),
                "route_labels": ROUTE_LABELS,
                "route_models": [s.get("model_name_or_path", ""), m["model"], h["model"]],
                "route_successes": [bool(s["scout_correct"]), bool(m["resolved"]), bool(h["resolved"])],
                "route_public_successes": [
                    bool(s.get("_tf_resolved", False)),
                    bool(m["public_resolved"]),
                    None,
                ],
                "route_prompt_tokens": [
                    int(s.get("prompt_tokens", 0)),
                    int(m.get("prompt_tokens", 0)),
                    int(h.get("prompt_tokens", 0)),
                ],
                "route_completion_tokens": [
                    int(s.get("completion_tokens", 0)),
                    int(m.get("completion_tokens", 0)),
                    int(h.get("completion_tokens", 0)),
                ],
            }
        )
    _write_jsonl(output / f"router_{split}.jsonl", aligned)
    _write_jsonl(output / f"excluded_{split}.jsonl", excluded)
    return {"aligned": len(aligned), "excluded": len(excluded)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-collection-dir", required=True)
    parser.add_argument("--expert-collection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-revision", default=LCB_DATASET_REVISION)
    args = parser.parse_args()

    source, expert, output = (
        Path(args.source_collection_dir),
        Path(args.expert_collection_dir),
        Path(args.output_dir),
    )
    output.mkdir(parents=True, exist_ok=True)
    stats = {
        split: materialize_split(source, expert, output, split)
        for split in ("train", "eval")
    }
    with open(output / "manifest.json", "w") as out_f:
        json.dump(
            {
                "route_labels": ROUTE_LABELS,
                "source_collection_dir": str(source),
                "expert_collection_dir": str(expert),
                "dataset_revision": args.dataset_revision,
                "splits": stats,
            },
            out_f,
            indent=2,
            sort_keys=True,
        )
    print(json.dumps(stats, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

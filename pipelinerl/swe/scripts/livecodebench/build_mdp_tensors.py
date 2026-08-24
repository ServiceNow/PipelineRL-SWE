#!/usr/bin/env python3
"""
Build the thread-(a) MDP correctness tensors from multi-draw LCB collections.

Reads <collection_root>/{scout,oss20,oss120}_eval_d{K}.jsonl (+ scout06_*_d{K}.jsonl
sensitivity arm) written by collect_lcb_expert.py and produces a single npz +
jsonl bundle:

  tensors.npz:
    resolved      (P, M, K) bool   full-suite success per problem/model/draw
    public_resolved (P, M, K) bool public-test success (routing-time feedback)
    completion_tokens (P, M, K) float
    valid         (P, M, K) bool   row passed _is_complete validation

  problems.jsonl: one row per problem (problem_id, contest_date, platform,
                  difficulty, problem_statement)
  models.json:    model metadata per slot

Usage:
  python build_mdp_tensors.py --collection-dir /mnt/.../lcb_multidraw_experts_1787502237 \
      --collection-dir /mnt/.../lcb_multidraw_scout_1787547502 \
      --output-dir /mnt/.../mdp_tensors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipelinerl.swe.scripts.livecodebench.collect_lcb_expert import (  # noqa: E402
    _is_complete,
    LCB_DATASET_REVISION,
)

MODEL_SLOTS = ["scout", "oss20", "oss120"]


def read_latest(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    with open(path) as in_f:
        for line in in_f:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = str(row.get("problem_id") or "").strip()
            if pid:
                latest[pid] = row
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", action="append", required=True,
                        help="Directory containing *_eval_d*.jsonl files (repeatable)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-draws", type=int, default=10)
    parser.add_argument("--max-invalid-frac", type=float, default=0.10,
                        help="Drop problems exceeding this invalid-draw fraction in any slot")
    args = parser.parse_args()

    roots = [Path(d) for d in args.collection_dir]
    K = args.num_draws

    # discover problem universe + per-(model, draw) tables
    tables: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    problems_index: dict[str, dict[str, Any]] = {}
    for root in roots:
        for f in sorted(root.glob("*_eval_d*.jsonl")):
            name = f.stem  # e.g. oss120_eval_d3 / scout06_eval_d7
            route = name.split("_eval_")[0]
            draw = int(name.split("_d")[-1])
            latest = read_latest(f)
            tables[(route, draw)] = latest
            for pid, row in latest.items():
                if pid not in problems_index:
                    problems_index[pid] = {
                        "problem_id": pid,
                        "contest_date": row.get("contest_date"),
                        "platform": row.get("platform"),
                        "difficulty": row.get("difficulty"),
                    }

    pids = sorted(problems_index)
    P = len(pids)
    print(f"problems: {P}; route/draw tables: {len(tables)}")

    M = len(MODEL_SLOTS)
    resolved = np.zeros((P, M, K), dtype=bool)
    public = np.zeros((P, M, K), dtype=bool)
    tokens = np.zeros((P, M, K), dtype=np.float32)
    valid = np.zeros((P, M, K), dtype=bool)
    texts: list[list[list[str]]] = [[[] for _ in range(K)] for _ in range(M)]
    fb_texts: list[list[list[str]]] = [[[] for _ in range(K)] for _ in range(M)]

    for mi, slot in enumerate(MODEL_SLOTS):
        for k in range(K):
            tab = tables.get((slot, k))
            if tab is None:
                raise FileNotFoundError(f"missing table for ({slot}, draw {k})")
            for pi, pid in enumerate(pids):
                row = tab.get(pid)
                if row is None:
                    continue
                ok = _is_complete(row, LCB_DATASET_REVISION)
                valid[pi, mi, k] = ok
                if ok:
                    resolved[pi, mi, k] = bool(row["resolved"])
                    public[pi, mi, k] = bool(row["public_resolved"])
                    tokens[pi, mi, k] = float(row.get("completion_tokens") or 0)
                    texts[mi][k].append(str(row.get("full_output") or ""))
                    fb_texts[mi][k].append(str(row.get("test_feedback") or ""))

    # drop problems whose validity is too low in any primary slot
    frac_valid = valid.reshape(P, -1).mean(axis=1)
    keep = frac_valid >= (1.0 - args.max_invalid_frac)
    n_drop = int((~keep).sum())
    print(f"dropping {n_drop} problems with invalid-frac > {args.max_invalid_frac}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "tensors.npz",
        resolved=resolved[keep], public=public[keep], tokens=tokens[keep],
        valid=valid[keep], problem_ids=np.array(pids)[keep],
        model_slots=np.array(MODEL_SLOTS),
    )
    with open(out / "problems.jsonl", "w") as f:
        for pi in np.nonzero(keep)[0]:
            pid = pids[pi]
            rec = problems_index[pid]
            f.write(json.dumps(rec) + "\n")

    # store draw texts keyed by position for downstream feature building
    np.savez_compressed(
        out / "draw_texts.npz",
        problem_ids=np.array(pids),
        model_slots=np.array(MODEL_SLOTS),
        # ragged text saved as json instead
    )
    with open(out / "draw_texts.json", "w") as f:
        json.dump(
            {
                slot: [
                    [
                        {"problem_id": pids[i], "code": c, "feedback": fb}
                        for i, (c, fb) in enumerate(zip(texts[mi][k], fb_texts[mi][k]))
                    ]
                    for k in range(K)
                ]
                for mi, slot in enumerate(MODEL_SLOTS)
            },
            f,
        )

    summary = {
        "num_problems": int(keep.sum()),
        "num_dropped": n_drop,
        "model_slots": MODEL_SLOTS,
        "num_draws": K,
        "solve_rates_full_suite": {
            slot: float(resolved[keep][:, mi][valid[keep][:, mi]].mean())
            for mi, slot in enumerate(MODEL_SLOTS)
        },
        "solve_rates_public": {
            slot: float(public[keep][:, mi][valid[keep][:, mi]].mean())
            for mi, slot in enumerate(MODEL_SLOTS)
        },
        "mean_completion_tokens": {
            slot: float(tokens[keep][:, mi][valid[keep][:, mi]].mean())
            for mi, slot in enumerate(MODEL_SLOTS)
        },
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Construct the depth-1 sequential-policy dataset for the thread-(a) MDP.

Each example is one decision point inside a replay episode:
  input text : problem statement + most recent scout draw (code + public-test
               feedback) if any + reroll count n + prior draw outcomes
  targets    : [p_scout_next, p_oss20_fresh, p_oss120_fresh, p_nothing]
               where p_X = binary outcome of model X's same-indexed draw
               (draws are i.i.d., so same-index is an unbiased "fresh draw")
               and p_nothing = 1 iff all three are failures.

Problems are split calibration/test by hash (50/50, seed 0) mirroring the replay
harness; policy thresholds are tuned on calibration only.

Usage:
  python build_mdp_sequential_dataset.py --tensors-dir .../mdp_tensors_v1 \
      --output-dir .../mdp_seq_dataset_v1 [--max-depth 10]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def split_of(pid: str) -> str:
    h = int(hashlib.sha256(pid.encode()).hexdigest(), 16)
    return "cal" if h % 2 == 0 else "test"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensors-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-depth", type=int, default=10,
                        help="decision points per problem (one per completed scout draw)")
    args = parser.parse_args()

    td = Path(args.tensors_dir)
    d = np.load(td / "tensors.npz", allow_pickle=True)
    R = d["resolved"].astype(bool)          # (P,M,K)
    Pub = d["public"].astype(bool)
    V = d["valid"]
    pids = [str(x) for x in d["problem_ids"]]
    with open(td / "problems.jsonl") as f:
        prob_meta = {json.loads(l)["problem_id"]: json.loads(l) for l in f}
    with open(td / "draw_texts.json") as f:
        texts = json.load(f)

    M, K = 3, args.max_depth   # slots: scout, oss20, oss120

    out_rows = []
    stats = {"cal": 0, "test": 0}
    for pi, pid in enumerate(pids):
        meta = prob_meta.get(pid, {})
        problem_statement = str(meta.get("problem_statement") or "")
        # scout draw texts for this problem: texts['scout'][k] is a list over problems
        scout_draws = []
        for k in range(K):
            entries = {e["problem_id"]: e for e in texts["scout"][k]}
            scout_draws.append(entries.get(pid))

        prior_bits = []
        for j in range(min(args.max_depth, K)):
            # decision point AFTER observing scout draws 0..j-1
            ctx = [
                "[Problem]\n" + problem_statement,
                f"[Rerolls so far] {j}",
            ]
            if j > 0:
                prev = scout_draws[j - 1]
                if prev is not None:
                    fb = str(prev.get("feedback") or "").strip()
                    code = str(prev.get("code") or "").strip()
                    passed_prev = bool(Pub[pi, 0, j - 1]) if V[pi, 0, j - 1] else False
                    ctx.append(
                        "[Most Recent Scout Draw]\n"
                        f"[Passed public tests] {'YES' if passed_prev else 'NO'}\n"
                        f"[Public test feedback]\n{fb[:4000]}\n\n"
                        f"[Scout code]\n{code[:8000]}"
                    )
                if j >= 2:
                    bits = " ".join(
                        ("PASS" if (V[pi, 0, t] and Pub[pi, 0, t]) else "FAIL")
                        for t in range(j - 1)
                    )
                    ctx.append(f"[Earlier draw outcomes] {bits}")
            text = "\n\n".join(ctx)

            # targets: same-indexed draw of each model (i.i.d. => unbiased fresh draw)
            def tgt(slot_idx: int) -> float | None:
                if not V[pi, slot_idx, j]:
                    return None
                return float(R[pi, slot_idx, j])

            t_scout, t_o20, t_o120 = tgt(0), tgt(1), tgt(2)
            if t_scout is None or t_o20 is None or t_o120 is None:
                continue
            t_nothing = float((t_scout + t_o20 + t_o120) == 0)

            out_rows.append({
                "problem_id": pid,
                "split": split_of(pid),
                "depth": j,
                "text": text,
                "targets": [t_scout, t_o20, t_o120, t_nothing],
            })
            stats[split_of(pid)] += 1

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("cal", "test"):
        with open(out / f"{split}.jsonl", "w") as f:
            for r in out_rows:
                if r["split"] == split:
                    f.write(json.dumps(r) + "\n")
    print(json.dumps({
        "examples": len(out_rows),
        "cal": stats["cal"], "test": stats["test"],
        "target_means_cal": {
            k: round(float(np.mean([r["targets"][i] for r in out_rows if r["split"] == "cal"])), 3)
            for i, k in enumerate(["scout_next", "oss20", "oss120", "nothing"])
        },
    }, indent=2))


if __name__ == "__main__":
    main()

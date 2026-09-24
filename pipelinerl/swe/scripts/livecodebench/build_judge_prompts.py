#!/usr/bin/env python3
"""Prompts for judging a PRODUCED attempt, correct or not -- the verifier-free selection problem.

Every history prompt we have is failure-only, because with a perfect verifier a success ends the
episode and there is nothing left to judge. Remove the verifier and that inverts: the policy must
decide which of the attempts it is holding to submit, so the probe has to score attempts it has no
outcome for -- including the correct ones it never saw in training.

This is the regime where the method has something no baseline can copy. RoR v1 is undefined without
a verifier (its belief is a prior decayed by OBSERVED failures, and none are observed), and a
cascade's "draw cheap, check, escalate" has no check. Measured on LCB, single-draw routing without
a verifier already gains +5.4 to +10.8pt over the best randomised fixed policy, against +0.4 to
+3.3pt with one; judged multi-sampling is the version that should do better still.

Writes <out>/judge_shard<i>.jsonl plus a manifest keyed "<pid>||<slot><draw>", so the same
extraction and head-fitting path as the history prompts applies unchanged.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from pipelinerl.swe.scripts.livecodebench.build_history_prompts import route_name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--base-prompts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-draws-per-rung", type=int, default=4)
    ap.add_argument("--max-code-chars", type=int, default=6000)
    ap.add_argument("--shards", type=int, default=8)
    a = ap.parse_args()
    T = Path(a.tensors_dir)
    t = np.load(T / "tensors.npz", allow_pickle=True)
    valid = t["valid"].astype(bool); ok = (t["final_outcome"] & t["valid"]).astype(bool)
    slots = [str(s) for s in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]
    base = {json.loads(l)["problem_id"]: json.loads(l)["prompt"]
            for l in open(a.base_prompts) if l.strip()}
    recs = {}
    for l in open(T / "draw_records.jsonl"):
        if l.strip():
            r = json.loads(l)
            recs[(r["problem_id"], r["model_slot"], int(r["draw_index"]))] = r
    rows, man, n_ok = [], [], 0
    for pi, pid in enumerate(pids):
        if pid not in base:
            continue
        for mi, s in enumerate(slots):
            drawn = [k for k in range(ok.shape[2]) if valid[pi, mi, k]][: a.max_draws_per_rung]
            for k in drawn:
                r = recs.get((pid, s, k))
                if r is None:
                    continue
                code = (r.get("code") or "")[: a.max_code_chars]
                eid = f"{pid}||{s}{k}"
                rows.append({"problem_id": eid, "prompt":
                             base[pid] + "\n\n"
                             + f"An attempt by {route_name(s)}:\n```python\n{code}\n```\n"
                             + "Is this solution correct?"})
                man.append({"example_id": eid, "problem_id": pid, "slot": s, "draw": k,
                            "correct": bool(ok[pi, mi, k])})
                n_ok += int(ok[pi, mi, k])
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for i in range(a.shards):
        with open(out / f"judge_shard{i}.jsonl", "w") as f:
            for r in rows[i::a.shards]:
                f.write(json.dumps(r) + "\n")
    with open(out / "judge_manifest.jsonl", "w") as f:
        for m in man:
            f.write(json.dumps(m) + "\n")
    print(f"{len(rows)} judge prompts ({n_ok} correct, {n_ok/max(1,len(rows))*100:.1f}%) "
          f"in {a.shards} shards -> {out}")


if __name__ == "__main__":
    main()

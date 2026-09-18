#!/usr/bin/env python3
"""Choose the linked-cap constant k (B = k*R) on the CALIBRATION split.

Criterion, fixed before any calibration result was seen: for each pool, take the accuracy range
that every candidate's hull reaches on every calibration seed, evaluate each candidate's hull cost
at 41 evenly spaced accuracies in that range, and choose the k with the lowest mean log cost,
averaged over seeds. The price-only arm (no cap, k = inf) is a candidate, so the procedure can
decline the cap altogether.

Usage: select_linked_k.py <linked_dir> <pool> <n_seeds>
Prints the per-candidate objective and the chosen k.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

FAM = "content_decay_qcost"


def load(d): return json.loads(Path(d, "replay_results.json").read_text())["results"]


def hull(rows, pol):
    keep, best = [], -1.0
    for c, a in sorted({(r["mean_realized_cost"], r["correctness"])
                        for r in rows if r["policy"] == pol}):
        if a > best: keep.append((c, a)); best = a
    v = []
    for q in keep:
        while len(v) >= 2:
            (c1, a1), (c2, a2) = v[-2], v[-1]
            if (a2 - a1) * (q[0] - c1) <= (q[1] - a1) * (c2 - c1): v.pop()
            else: break
        v.append(q)
    return v


def cost_at(v, t):
    if v[0][1] >= t: return v[0][0]
    for (c1, a1), (c2, a2) in zip(v, v[1:]):
        if a1 < t <= a2: return c1 + (c2 - c1) * (t - a1) / (a2 - a1)
    return np.nan


def main(d, pool, n):
    runs = [load(Path(d) / f"{pool}_calibration_s{s}") for s in range(n)]
    pols = sorted({r["policy"] for r in runs[0] if r["policy"].startswith(f"{FAM}_linkedcap")},
                  key=lambda p: float(p.split("linkedcap")[1]))
    cands = {p.split("linkedcap")[1]: p for p in pols}
    cands["inf"] = f"{FAM}_value"
    H = {k: [hull(r, p) for r in runs] for k, p in cands.items()}
    lo = max(h[0][1] for hs in H.values() for h in hs)
    hi = min(h[-1][1] for hs in H.values() for h in hs)
    grid = np.linspace(lo, hi, 41)
    print(f"{pool}: common calibration accuracy range {lo:.3f}-{hi:.3f}, {n} seeds")
    J = {k: float(np.mean([np.mean(np.log([cost_at(h, t) for t in grid])) for h in hs]))
         for k, hs in H.items()}
    ref = J["inf"]
    for k, j in sorted(J.items(), key=lambda kv: kv[1]):
        print(f"  k={k:>6s}  mean log cost {j:+.4f}   ({100*(1-np.exp(j-ref)):+.1f}% vs no cap, "
              f"geometric mean over the range)")
    best = min(J, key=J.get)
    print(f"CHOSEN k = {best}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))

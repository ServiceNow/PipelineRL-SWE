#!/usr/bin/env python3
"""Track B: CEILING for per-instance test-writer choice in the sequential setting (368 Verified, open only).

Before building any writer router, measure the most it could possibly win. Policy family = the fixed-writer
cascade of analyze_trackB_seq.py (ladder oss20 x3 -> qwen30 x3 -> oss120; one writer's test for every candidate;
accept on valid & pass; submit the last candidate otherwise). Frontiers, cost (cents) vs accuracy:
  best fixed writer   one writer for ALL instances, any ladder length k           (5 writers x 7 k points)
  self-verification   each candidate checked by its own model's test               (reference)
  ORACLE writer       per instance, the writer maximising correct - mu*cost chosen WITH the labels; k shared
                      across instances, mu swept                                  (the ceiling for routing
                      the writer; no real router can beat it)
  ORACLE writer + k   the same, also choosing the ladder length per instance      (context: stopping too)
Frontiers are upper hulls (mixing two operating points is allowed). Reported: cost at matched accuracy and the
ratio oracle / best-fixed, with a paired bootstrap over instances (hulls refit per replicate). In-sample by
design: this is an upper bound, not a policy.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from analyze_trackB_seq import LADDER, WRITERS, run


def hull(points):
    """Upper-left hull of (cost, acc): the cheapest cost reaching each accuracy, with mixing."""
    pts = sorted(set(points))
    out = []
    for c, a in pts:
        if out and a <= out[-1][1]:
            continue
        while len(out) >= 2:
            (c1, a1), (c2, a2) = out[-2], out[-1]
            if (a2 - a1) * (c - c1) <= (a - a1) * (c2 - c1):   # middle point under the chord
                out.pop()
            else:
                break
        out.append((c, a))
    return out


def cost_at(h, target):
    if not h or target > h[-1][1]:
        return np.nan
    if target <= h[0][1]:
        return h[0][0]
    for (c1, a1), (c2, a2) in zip(h, h[1:]):
        if a1 <= target <= a2:
            return c1 + (c2 - c1) * (target - a1) / (a2 - a1)
    return np.nan


def frontiers(A, Cst, S_A, S_C, idx, mus):
    """A, Cst: [n, W, K] correctness / cost of fixed-writer cascades. S_*: [n, K] self-verification."""
    a, c = A[idx], Cst[idx]
    fixed = [(c[:, w, k].mean(), a[:, w, k].mean()) for w in range(a.shape[1]) for k in range(a.shape[2])]
    selfv = [(S_C[idx, k].mean(), S_A[idx, k].mean()) for k in range(S_A.shape[1])]
    orw, ork = [], []
    for mu in mus:
        u = a - mu * c                                   # [n, W, K]
        for k in range(a.shape[2]):
            w = u[:, :, k].argmax(1)
            orw.append((c[np.arange(len(w)), w, k].mean(), a[np.arange(len(w)), w, k].mean()))
        flat = u.reshape(len(u), -1).argmax(1)
        ork.append((c.reshape(len(c), -1)[np.arange(len(c)), flat].mean(), a.reshape(len(a), -1)[np.arange(len(a)), flat].mean()))
    return {"best fixed writer": hull(fixed), "self-verification": hull(selfv),
            "ORACLE writer": hull(orw), "ORACLE writer + k": hull(ork)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass-matrix", required=True)
    ap.add_argument("--targets", default="0.45,0.50,0.55,0.60,0.62")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.pass_matrix)]
    n, W, K = len(recs), len(WRITERS), len(LADDER)
    A = np.zeros((n, W, K)); C = np.zeros((n, W, K)); SA = np.zeros((n, K)); SC = np.zeros((n, K))
    for i, r in enumerate(recs):
        for k in range(K):
            for w, wr in enumerate(WRITERS):
                A[i, w, k], C[i, w, k] = run(r, "fixed", k + 1, fixed=wr)
            SA[i, k], SC[i, k] = run(r, "self", k + 1)
    mus = np.r_[0.0, np.geomspace(0.05, 200, 60)]
    targets = [float(x) for x in a.targets.split(",")]
    full = frontiers(A, C, SA, SC, np.arange(n), mus)
    ceil = np.mean([any(x["correct"] for x in r["candidates"] if x["cid"] in LADDER) for r in recs])
    print(f"{n} instances; some open candidate correct: {ceil*100:.1f}%")
    print("max accuracy on each frontier: " + ", ".join(f"{k} {h[-1][1]*100:.1f}% @{h[-1][0]:.3f}c" for k, h in full.items()))
    # per-writer usage under the oracle at a mid mu, full ladder
    u = A[:, :, -1] - 2.0 * C[:, :, -1]; best = u.argmax(1)
    print("oracle writer picks (full ladder, mu=2): " + ", ".join(f"{w} {np.mean(best == j)*100:.0f}%" for j, w in enumerate(WRITERS)))
    rng = np.random.default_rng(0)
    B = []
    for _ in range(a.boot):
        ii = rng.integers(0, n, n); B.append(frontiers(A, C, SA, SC, ii, mus))
    res = {}
    print(f"\ncost (cents/instance) at matched accuracy; ratio vs best fixed writer [95% bootstrap CI], <1 = cheaper")
    for t in targets:
        base = cost_at(full["best fixed writer"], t)
        row = {}
        line = f"  {t*100:4.0f}%  best fixed {base:.3f}c"
        for name in ("self-verification", "ORACLE writer", "ORACLE writer + k"):
            v = cost_at(full[name], t)
            rb = [cost_at(b[name], t) / cost_at(b["best fixed writer"], t) for b in B]
            rb = np.array([x for x in rb if np.isfinite(x)])
            lo, hi = (np.percentile(rb, 2.5), np.percentile(rb, 97.5)) if len(rb) > 20 else (np.nan, np.nan)
            row[name] = dict(cost=v, ratio=v / base, ci=[lo, hi], n_boot=len(rb))
            line += f" | {name} {v:.3f}c x{v/base:.2f} [{lo:.2f},{hi:.2f}]"
        res[t] = dict(best_fixed=base, **row)
        print(line)
    if a.out:
        json.dump({"n": n, "frontiers": full, "matched": res}, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Track B, P1: baselines and one-shot (generator, tester) pairs on the pass matrix.

CASCADES (sequential over generators, cheapest first; truncating at k = 1..G gives a cost/accuracy curve):
generate a candidate from generator g_i; the writer policy decides which tests exist; the candidate is
ACCEPTED if the policy's rule says so -> submit; else move to g_{i+1}; at the end of the ladder submit
the last candidate. A test is written once per instance and re-run on every later candidate.
  none        no tests: k=1 is "route once, submit" (B0); longer ladders are pointless without tests
  self        each generator writes its own test (B1, self-verification)
  fixed:<t>   one writer t for every candidate (B2)
  cheapfirst  buy writers cheapest-first until a VALID test exists (SWE: fails on the unpatched repo)
  vote        every writer, accept when a majority of valid tests pass (B3)
A test that is not valid (SWE) carries no information: the candidate is accepted.
ONE-SHOT PAIRS (S1): generator g, tester t, no fallback: coverage = P(accept), precision = P(correct |
accepted), cost = gen(g) + write(t) + run.
Costs in cents per instance; paired bootstrap CIs over instances. Cost at matched accuracy is linear
interpolation along each policy's k-curve (upper hull), identical treatment for every policy.
"""
from __future__ import annotations
import argparse, json
import numpy as np

SELF = {"swe_verified": {"oss20": "oss20", "oss120": "oss120", "qwen30": "qcoder30", "gemini": "gemini", "opus": "opus"},
        "lcb": {"oss20lo": "oss20lo", "oss20md": "oss20md", "dsv4f": "dsv4f", "oss120md": "oss120md"}}


def run_cascade(rec, ladder, policy, writer_order, fixed=None):
    cands = {c["gen"]: c for c in rec["candidates"]}
    tests = {t["writer"]: t for t in rec["tests"]}
    bought, spent, submitted, n_cands = [], 0.0, None, 0
    present = [g for g in ladder if g in cands]
    last = present[-1] if present else None
    for g in ladder:
        if g not in cands:
            continue
        cand = cands[g]; spent += cand["gen_cost_c"]; n_cands += 1; submitted = cand
        # which tests exist after this step
        want = []
        if policy == "self":
            w = SELF[rec["benchmark"]].get(g)
            want = [w] if w in tests else []
        elif policy == "fixed":
            want = [fixed] if fixed in tests else []
        elif policy == "vote":
            want = [w for w in writer_order if w in tests]
        elif policy == "cheapfirst":
            want = []
            for w in writer_order:
                if w not in tests:
                    continue
                want.append(w)
                if tests[w]["valid"] is not False:        # stop at the first valid test (LCB: None -> stop)
                    break
        for w in want:
            if w not in bought:
                bought.append(w); spent += tests[w]["write_cost_c"] if np.isfinite(tests[w]["write_cost_c"]) else 0.0
                spent += rec["run_cost_c"] * (n_cands - 1)   # run the new test on earlier candidates too
        spent += rec["run_cost_c"] * len(bought)             # run every bought test on this candidate
        valid = [w for w in bought if tests[w]["valid"] is not False]
        if policy == "none":
            accept = True
        elif not valid:
            accept = g == last                               # no informative test: escalate, unless last rung
        elif policy == "vote":
            accept = sum(tests[w]["passes"].get(cand["cid"], False) for w in valid) * 2 > len(valid)
        else:
            accept = all(tests[w]["passes"].get(cand["cid"], False) for w in valid)
        if accept:
            break
    if submitted is None:
        return np.nan, np.nan
    return float(submitted["correct"]), spent


def hull(pts):
    pts = sorted(pts); h = []
    for p in pts:
        while len(h) >= 2 and (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= (p[1] - h[-2][1]) * (h[-1][0] - h[-2][0]):
            h.pop()
        h.append(p)
    return h


def cost_at(curve, target):
    h = hull([(0.0, 0.0)] + curve)
    for (c0, a0), (c1, a1) in zip(h, h[1:]):
        if a0 <= target <= a1:
            return c0 + (c1 - c0) * (target - a0) / max(a1 - a0, 1e-12)
    return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass-matrix", required=True)
    ap.add_argument("--split", default="", help="restrict to this split (LCB: test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-gens", default="", help="generators left out of every ladder, e.g. qwen4b")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    recs = [json.loads(l) for l in open(a.pass_matrix)]
    if a.split:
        recs = [r for r in recs if r["split"] == a.split]
    bench = recs[0]["benchmark"]
    gens = sorted({c["gen"] for r in recs for c in r["candidates"]},
                  key=lambda g: np.nanmean([c["gen_cost_c"] for r in recs for c in r["candidates"] if c["gen"] == g]))
    writers = sorted({t["writer"] for r in recs for t in r["tests"]},
                     key=lambda w: np.nanmean([t["write_cost_c"] for r in recs for t in r["tests"] if t["writer"] == w]))
    gens = [g for g in gens if g not in set(a.exclude_gens.split(","))]
    print(f"{bench}: {len(recs)} instances | generators by cost {gens} | writers by cost {writers}")
    print("generator accuracy:", {g: round(np.mean([c["correct"] for r in recs for c in r["candidates"] if c["gen"] == g]), 3) for g in gens})
    policies = [("none", None), ("self", None), ("cheapfirst", None), ("vote", None)] + [("fixed", w) for w in writers]
    curves, per = {}, {}
    for pol, fx in policies:
        name = pol if fx is None else f"fixed:{fx}"
        ks = [1] if pol == "none" else range(1, len(gens) + 1)
        for k in ks:
            res = [run_cascade(r, gens[:k], pol, writers, fx) for r in recs]
            A = np.array([x[0] for x in res]); C = np.array([x[1] for x in res])
            per[(name, k)] = (A, C)
            curves.setdefault(name, []).append((np.nanmean(C), np.nanmean(A)))
    # B0 for every single generator (route once, submit)
    for g in gens:
        A = np.array([next((float(c["correct"]) for c in r["candidates"] if c["gen"] == g), np.nan) for r in recs])
        C = np.array([next((c["gen_cost_c"] for c in r["candidates"] if c["gen"] == g), np.nan) for r in recs])
        curves.setdefault("none", []).append((np.nanmean(C), np.nanmean(A)))
    print("\nCASCADE curves (cheapest-first generator ladder truncated at k): accuracy @ cost (cents)")
    for name, pts in curves.items():
        print(f"   {name:<16}" + "  ".join(f"{a*100:5.1f}%@{c:.3f}" for c, a in sorted(pts)))
    lo = max(min(a for _, a in pts) for pts in curves.values() if len(pts) > 1)
    hi = min(max(a for _, a in pts) for n, pts in curves.items() if n in ("self", "cheapfirst", "vote"))
    targets = np.round(np.linspace(lo, hi, 5), 3) if hi > lo else []
    print("\nCOST at matched accuracy (cents/instance; hull over k):")
    names = [n for n in curves if n != "none"]
    print("   target  " + "".join(f"{n:>16}" for n in names))
    for t in targets:
        print(f"   {t*100:5.1f}%  " + "".join(f"{cost_at(curves[n], t):>16.4f}" for n in names))
    # bootstrap: cheapest policy vs self at each target
    if len(targets):
        print("   ratio vs self (95% CI):")
        for n in [x for x in names if x != "self"]:
            line = f"     {n:<16}"
            for t in targets:
                r0 = cost_at(curves[n], t) / cost_at(curves["self"], t)
                bs = []
                for _ in range(300):
                    idx = rng.integers(0, len(recs), len(recs))
                    cn = [(np.nanmean(per[(n, k)][1][idx]), np.nanmean(per[(n, k)][0][idx])) for k in range(1, len(gens) + 1) if (n, k) in per]
                    cs = [(np.nanmean(per[("self", k)][1][idx]), np.nanmean(per[("self", k)][0][idx])) for k in range(1, len(gens) + 1)]
                    x, y = cost_at(cn, t), cost_at(cs, t)
                    if np.isfinite(x) and np.isfinite(y) and y > 0:
                        bs.append(x / y)
                ci = f"[{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]" if len(bs) > 50 else "[n/a]"
                line += f"  {t*100:.0f}%: {r0:.2f} {ci}" if np.isfinite(r0) else f"  {t*100:.0f}%: n/a"
            print(line)
    print("\nONE-SHOT PAIRS (S1): generator x tester -> coverage / precision of accepted / cost (cents)")
    print("   gen\\tester " + "".join(f"{w:>24}" for w in writers))
    for g in gens:
        line = f"   {g:<11}"
        for w in writers:
            acc, prec, cost = [], [], []
            for r in recs:
                cand = next((c for c in r["candidates"] if c["gen"] == g), None)
                t = next((x for x in r["tests"] if x["writer"] == w), None)
                if cand is None or t is None:
                    continue
                ok = t["valid"] is not False and t["passes"].get(cand["cid"], False)
                acc.append(ok); cost.append(cand["gen_cost_c"] + (t["write_cost_c"] if np.isfinite(t["write_cost_c"]) else 0) + r["run_cost_c"])
                if ok:
                    prec.append(cand["correct"])
            tag = "*" if SELF[bench].get(g) == w else " "
            line += f"   {tag}{np.mean(acc)*100:4.0f}%/{(np.mean(prec)*100 if prec else float('nan')):4.0f}%/{np.mean(cost):.3f}"
        base = np.mean([c["correct"] for r in recs for c in r["candidates"] if c["gen"] == g])
        print(line + f"   | no test: {base*100:.0f}% correct")
    print("   (* = self-verification pair; cells: accept rate / precision of accepted / cost)")


if __name__ == "__main__":
    main()

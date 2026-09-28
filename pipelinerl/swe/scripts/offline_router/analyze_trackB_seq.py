#!/usr/bin/env python3
"""Track B, SEQUENTIAL setting on SWE-bench Verified (open models only): redraw / switch / submit with imperfect,
generated, priced tests. Pass matrix from build_pass_matrix.py --redraw-exec.

Ladder (cheapest first): oss20 x3 draws -> qwen30 x3 -> oss120 x1. A candidate that produced no patch is skipped.
Every bought test runs on every candidate (runs are ~free: 1 vCPU-second-ish, still charged). Families:
  route-once   submit the first candidate of one generator, no tests                     (B0)
  self         each candidate checked by its OWN model's test; accept on valid & pass     (B1)
  fixed:<w>    one writer's test for all candidates; accept on valid & pass              (B2)
  escalate     per-instance, label-free writer choice. `invalid`: buy writers in `order` until one's test is VALID
               (fails on the unpatched repo), use it. `confirm`: first writer's pass is checked by the second
               (bought only then). `rescue`: first writer's FAIL gets a second opinion (bought only then)
  posterior    writers S bought up front; each candidate's P(correct | verdicts) = generator base rate x
               per-writer likelihood ratios (P(pass|correct), P(pass|wrong) for VALID tests, fit on TRAIN folds);
               accept when posterior >= tau; at the end of the ladder submit the highest-posterior candidate
Non-posterior cascades submit the LAST candidate if nothing was accepted. Parameters (ladder length k; tau; S)
are chosen on the TRAIN folds as the cheapest setting reaching each target accuracy, then scored on the held-out
fold (5-fold, pooled). Paired bootstrap vs the self-verification cascade and vs route-once.
"""
from __future__ import annotations
import argparse, itertools, json
import numpy as np

LADDER = ["oss20", "oss20_d1", "oss20_d2", "qwen30", "qwen30_d1", "qwen30_d2", "oss120"]
SELF = {"oss20": "oss20", "qwen30": "qcoder30", "oss120": "oss120"}
WRITERS = ["oss20", "qcoder30", "dsv4f", "oss120", "devstral"]
ESC_ORDERS = {  # cheapest writer first (median write cost: oss20 0.030c, dsv4f 0.064c, qcoder30 0.064c, oss120 0.188c)
    "invalid": [("oss20", "dsv4f"), ("oss20", "dsv4f", "oss120"), ("oss20", "qcoder30", "dsv4f"), ("dsv4f", "oss120"),
                ("oss20", "dsv4f", "oss120", "devstral")],
    "confirm": [("oss20", "dsv4f"), ("dsv4f", "oss120"), ("oss20", "oss120")],
    "rescue":  [("oss20", "dsv4f"), ("dsv4f", "oss120"), ("oss20", "oss120")]}


def fit_reliability(recs):
    """Per writer: base P(correct) per generator; P(pass|correct), P(pass|wrong) on VALID tests (Laplace)."""
    base, rel = {}, {}
    for g in ("oss20", "qwen30", "oss120"):
        ys = [c["correct"] for r in recs for c in r["candidates"] if c["gen"] == g and c["cid"] in LADDER]
        base[g] = (sum(ys) + 1) / (len(ys) + 2)
    for w in WRITERS:
        pc = [1, 2]; pw = [1, 2]
        for r in recs:
            t = next((x for x in r["tests"] if x["writer"] == w), None)
            if t is None or t["valid"] is False:
                continue
            for c in r["candidates"]:
                if c["cid"] in LADDER and c["cid"] in t["passes"]:
                    (pc if c["correct"] else pw)[0] += t["passes"][c["cid"]]; (pc if c["correct"] else pw)[1] += 1
        rel[w] = (pc[0] / pc[1], pw[0] / pw[1])
    return base, rel


def run(r, family, k, tau=None, S=(), fixed=None, base=None, rel=None, g=None, order=(), mode=None):
    cands = {c["cid"]: c for c in r["candidates"]}
    tests = {t["writer"]: t for t in r["tests"]}
    if family == "route-once":            # the chosen generator's first draw; no patch = a wrong submission
        return (float(cands[g]["correct"]), cands[g]["gen_cost_c"]) if g in cands else (0.0, 0.0)
    ladder = [x for x in LADDER[:k] if x in cands]
    if not ladder:
        return 0.0, 0.0
    spent, bought, best, n = 0.0, [], (-1.0, None), 0
    def buy(w):
        nonlocal spent
        if w in tests and w not in bought:
            bought.append(w); spent += (tests[w]["write_cost_c"] if np.isfinite(tests[w]["write_cost_c"]) else 0.0)
            spent += r["run_cost_c"] * n                       # run on candidates already drawn
    if family == "posterior":
        for w in S:
            buy(w)
    for j, cid in enumerate(ladder):
        c = cands[cid]; spent += c["gen_cost_c"]; n += 1; spent += r["run_cost_c"] * len(bought)
        if family == "route-once":
            return float(c["correct"]), spent
        if family == "self":
            buy(SELF[c["gen"]]); ws = [SELF[c["gen"]]]
        elif family == "fixed":
            buy(fixed); ws = [fixed]
        elif family == "escalate":
            ok = lambda w: w in tests and tests[w]["valid"] is not False
            passes = lambda w: tests[w]["passes"].get(cid, False)
            if mode == "invalid":
                w = None
                for x in order:
                    buy(x)
                    if ok(x):
                        w = x; break
                if w is not None and passes(w):
                    return float(c["correct"]), spent
                continue
            a_, b_ = order
            buy(a_)
            if not ok(a_):                       # first test uninformative: fall back to the second
                buy(b_)
                if ok(b_) and passes(b_):
                    return float(c["correct"]), spent
                continue
            if mode == "confirm":
                if passes(a_):
                    buy(b_)
                    if not ok(b_) or passes(b_):
                        return float(c["correct"]), spent
                continue
            if passes(a_):                       # rescue
                return float(c["correct"]), spent
            buy(b_)
            if ok(b_) and passes(b_):
                return float(c["correct"]), spent
            continue
        else:
            ws = list(S)
        valid = [w for w in ws if w in tests and tests[w]["valid"] is not False]
        if family == "posterior":
            b = base[c["gen"]]; lo = np.log(b / (1 - b))
            for w in valid:
                pc, pw = rel[w]; ok = tests[w]["passes"].get(cid, False)
                lo += np.log(pc / pw) if ok else np.log((1 - pc) / (1 - pw))
            post = 1 / (1 + np.exp(-lo))
            if post > best[0]:
                best = (post, c)
            if post >= tau:
                return float(c["correct"]), spent
        else:
            if valid and all(tests[w]["passes"].get(cid, False) for w in valid):
                return float(c["correct"]), spent
    if family == "posterior":
        return float(best[1]["correct"]), spent
    return float(cands[ladder[-1]]["correct"]), spent


def settings(train, family):
    base, rel = fit_reliability(train)
    if family == "route-once":
        return [dict(k=1, g=g) for g in ("oss20", "qwen30", "oss120")], base, rel
    if family in ("self",) or family.startswith("fixed"):
        return [dict(k=k) for k in range(1, len(LADDER) + 1)], base, rel
    if family.startswith("escalate"):
        mode = family.split(":")[1]
        orders = ESC_ORDERS[mode]
        return [dict(k=k, order=o, mode=mode) for k in range(1, len(LADDER) + 1) for o in orders], base, rel
    out = []
    for S in [("oss20",), ("dsv4f",), ("qcoder30",), ("oss20", "dsv4f"), ("oss20", "qcoder30"), ("oss20", "qcoder30", "dsv4f")]:
        for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            for k in (3, 4, 6, 7):
                out.append(dict(k=k, tau=tau, S=S))
    return out, base, rel


def evaluate(recs, family, fold_of, targets, folds=5):
    held = {t: np.zeros((len(recs), 2)) for t in targets}; picks = {}
    for f in range(folds):
        tr = [r for r, g in zip(recs, fold_of) if g != f]; te_idx = [i for i, g in enumerate(fold_of) if g == f]
        grid, base, rel = settings(tr, family)
        fixed = family.split(":")[1] if family.startswith("fixed:") else None
        fam = "fixed" if fixed else ("escalate" if family.startswith("escalate") else family)
        pts = []
        for s in grid:
            res = np.array([run(r, fam, s["k"], s.get("tau"), s.get("S", ()), fixed, base, rel, s.get("g"),
                                s.get("order", ()), s.get("mode")) for r in tr])
            pts.append((res[:, 1].mean(), res[:, 0].mean(), s))
        for t in targets:
            ok = [p for p in pts if p[1] >= t]
            s = min(ok, key=lambda p: p[0])[2] if ok else max(pts, key=lambda p: p[1])[2]
            for i in te_idx:
                held[t][i] = run(recs[i], fam, s["k"], s.get("tau"), s.get("S", ()), fixed, base, rel, s.get("g"),
                                 s.get("order", ()), s.get("mode"))
            picks.setdefault(t, []).append(s)
    return held, picks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass-matrix", required=True)
    ap.add_argument("--targets", default="0.45,0.50,0.55,0.60")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--families", default="", help="comma subset of families (self is always kept as the reference)")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    recs = [json.loads(l) for l in open(a.pass_matrix)]
    fold_of = np.empty(len(recs), int); fold_of[rng.permutation(len(recs))] = np.arange(len(recs)) % a.folds
    targets = [float(x) for x in a.targets.split(",")]
    anyc = np.mean([any(c["correct"] for c in r["candidates"] if c["cid"] in LADDER) for r in recs])
    print(f"{len(recs)} instances; ceiling (some open candidate correct) {anyc*100:.1f}%")
    fams = ["route-once", "self", "posterior"] + [f"fixed:{w}" for w in WRITERS] + [f"escalate:{m}" for m in ESC_ORDERS]
    if a.families:
        fams = [f for f in fams if f in a.families.split(",") or f == "self"]
    out = {fam: evaluate(recs, fam, fold_of, targets, a.folds) for fam in fams}
    res = {f: v[0] for f, v in out.items()}
    for f in fams:
        if f.startswith("escalate"):
            print(f"  {f} settings chosen per target (5 folds): " + "; ".join(
                f"{t*100:.0f}%: " + ",".join(f"k{s['k']}/{'>'.join(s['order'])}" for s in out[f][1][t]) for t in targets))
    for t in targets:
        print(f"\n  target {t*100:.0f}% (chosen on train folds; held-out accuracy / cost cents):")
        ref = res["self"][t]
        for fam in fams:
            x = res[fam][t]
            d_acc = x[:, 0] - ref[:, 0]; ratio = x[:, 1].mean() / max(ref[:, 1].mean(), 1e-12)
            bs = []
            for _ in range(2000):
                ii = rng.integers(0, len(recs), len(recs))
                bs.append(((x[ii, 0] - ref[ii, 0]).mean(), x[ii, 1].mean() / max(ref[ii, 1].mean(), 1e-12)))
            bs = np.array(bs)
            print(f"     {fam:<16} {x[:,0].mean()*100:5.1f}% @ {x[:,1].mean():.4f}c   vs self: acc {d_acc.mean()*100:+.1f} "
                  f"[{np.percentile(bs[:,0],2.5)*100:+.1f},{np.percentile(bs[:,0],97.5)*100:+.1f}]  cost x{ratio:.2f} "
                  f"[{np.percentile(bs[:,1],2.5):.2f},{np.percentile(bs[:,1],97.5):.2f}]")


if __name__ == "__main__":
    main()

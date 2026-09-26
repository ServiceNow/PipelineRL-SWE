"""Distributional beliefs in the Bellman policy vs the fixed cascade and RoR-style counting: COST at matched
accuracy, free perfect verifier (v=0). Every arm is a set of calibration-selected test points; cost at a
target accuracy is interpolated on each arm's upper hull (same treatment for all arms). CI: paired bootstrap
over test problems (per-problem accuracy resampled; each point's mean cost held fixed, as in
analysis/priced_verification/cost_at_matched_accuracy.py)."""
import json, sys
import numpy as np
rng = np.random.default_rng(0)


def cost_at(pts, idx, target):
    xy = sorted((c, np.mean(a[idx])) for a, c in pts)
    h = []
    for p in [(0.0, 0.0)] + xy:
        while len(h) >= 2 and (h[-1][1]-h[-2][1])*(p[0]-h[-2][0]) <= (p[1]-h[-2][1])*(h[-1][0]-h[-2][0]):
            h.pop()
        h.append(p)
    for (c0, a0), (c1, a1) in zip(h, h[1:]):
        if a0 <= target <= a1:
            return c0 + (c1-c0)*(target-a0)/max(a1-a0, 1e-12)
    return np.nan


DEFAULT = {"lcb": [0.70, 0.75, 0.80, 0.85, 0.90], "bcb": [0.55, 0.60, 0.65, 0.68, 0.70]}
# usage: compare.py [ds[:t1/t2/...] ...]   e.g.  compare.py cc:0.5/0.6/0.7/0.8
jobs = [(x.split(":")[0], [float(v) for v in x.split(":")[1].split("/")] if ":" in x else DEFAULT[x]) for x in sys.argv[1:]] \
    or list(DEFAULT.items())
for ds, targets in jobs:
    arms = {}
    d = json.load(open(f"analysis/dist_bellman/{ds}.json"))
    n = d["test_n"]
    for r in d["results"]:
        if r["v_multiplier"] != 0:
            continue
        arms.setdefault(f"bellman {r['belief']} / {r['cost']} cost", []).append(
            (np.array(r["test_accuracy_by_problem"]), float(np.mean(r["test_cost_cents_by_problem"]))))
    pv = json.load(open(f"analysis/priced_verification/{ds}_priced_verification.json"))
    for r in pv["results"]:
        if r["v_multiplier"] == 0 and r["mode"] == "always" and r["belief"] == "counts" and r["cost"] == "global":
            arms.setdefault("RoR-style counts / global cost", []).append((np.array(r["test_accuracy_by_problem"]), r["test_cost_cents"]))
    for x in json.load(open(f"analysis/priced_verification/{ds}_best_fixed_cascade.json")):
        arms.setdefault("fixed cascade", []).append((np.array(x["test_accuracy_by_problem"]), x["test_cost"]))
    names = ["fixed cascade", "RoR-style counts / global cost"] + sorted(k for k in arms if k.startswith("bellman"))
    allidx = np.arange(n)
    print(f"\n==== {ds.upper()} ({n} test problems), free perfect verifier: cost (cents/problem) at matched accuracy")
    print("   target " + "".join(f"{k:>34}" for k in names))
    for t in targets:
        print(f"   {t*100:4.0f}%  " + "".join(f"{cost_at(arms[k], allidx, t):>34.4f}" for k in names))
    print("   ratio vs fixed cascade [95% CI]  (<1 = cheaper)")
    for k in names[1:]:
        line = f"     {k:<34}"
        for t in targets:
            r0 = cost_at(arms[k], allidx, t) / cost_at(arms["fixed cascade"], allidx, t)
            bs = []
            for _ in range(500):
                idx = rng.integers(0, n, n)
                x, y = cost_at(arms[k], idx, t), cost_at(arms["fixed cascade"], idx, t)
                if np.isfinite(x) and np.isfinite(y) and y > 0:
                    bs.append(x / y)
            ci = f"[{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]" if len(bs) > 100 else "[n/a]"
            line += f"  {t*100:.0f}%: {r0:.2f} {ci}" if np.isfinite(r0) else f"  {t*100:.0f}%: n/a"
        print(line)

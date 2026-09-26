"""Cost at matched accuracy (free perfect verifier, v=0), from Codex's calibration-selected test points.
Each arm = its test (cost, accuracy) points across calibration budgets; cost at a target accuracy is
linear interpolation between adjacent points (same treatment for every arm). CI: resample test
problems (paired across arms), recompute each point's mean accuracy/cost, re-interpolate."""
import json, numpy as np
rng = np.random.default_rng(0)
def curve_points(pts):   # pts: list of (acc_by_problem, cost_by_problem)
    return pts
def cost_at(pts, idx, target):
    xy = sorted((np.mean(c[idx]), np.mean(a[idx])) for a, c in pts)
    # upper hull in (cost, acc) so interpolation is monotone
    h = []
    for p in xy:
        while len(h) >= 2 and (h[-1][1]-h[-2][1])*(p[0]-h[-2][0]) <= (p[1]-h[-2][1])*(h[-1][0]-h[-2][0]): h.pop()
        h.append(p)
    for (c0, a0), (c1, a1) in zip(h, h[1:]):
        if a0 <= target <= a1:
            return c0 + (c1-c0)*(target-a0)/max(a1-a0, 1e-12)
    return np.nan
for ds, targets in [("lcb", [0.70, 0.75, 0.80, 0.85, 0.90]), ("bcb", [0.55, 0.60, 0.65, 0.68, 0.70])]:
    d = json.load(open(f"analysis/priced_verification/{ds}_priced_verification.json"))
    n = d["test_n"]; arms = {}
    for r in d["results"]:
        if r["v_multiplier"] != 0.0 or "test_accuracy_by_problem" not in r:
            continue
        key = f'{r["mode"]}|{r["belief"]}|{r["cost"]}'
        # per-problem cost is not stored for policy arms: hold each point's mean cost fixed in the bootstrap
        arms.setdefault(key, []).append((np.array(r["test_accuracy_by_problem"]), np.full(n, r["test_cost_cents"])))
    fc = json.load(open(f"analysis/priced_verification/{ds}_best_fixed_cascade.json"))
    arms["fixed cascade"] = [(np.array(x["test_accuracy_by_problem"]), np.full(n, x["test_cost"])) for x in fc]   # same treatment as the policy arms
    keep = ["fixed cascade", "always|counts|global", "always|counts|learned", "always|content|global", "always|content|learned", "never|content|learned"]
    keep = [k for k in keep if k in arms]
    allidx = np.arange(n)
    print(f"\n==== {ds.upper()} ({n} test problems), free perfect verifier: COST (cents/problem) at matched accuracy")
    print("   target  " + "".join(f"{k.replace('always|','').replace('|',' ')[:22]:>24}" for k in keep))
    for t in targets:
        base = [cost_at(arms[k], allidx, t) for k in keep]
        print(f"   {t*100:4.0f}%   " + "".join(f"{c:>24.3f}" if np.isfinite(c) else f"{'unreachable':>24}" for c in base))
    # ratios vs fixed cascade with CI
    print("   cost ratio vs fixed cascade (arm / cascade; <1 = cheaper), 95% CI:")
    for k in keep[1:]:
        line = f"     {k.replace('always|','').replace('|',' '):<24}"
        for t in targets:
            r0 = cost_at(arms[k], allidx, t) / cost_at(arms["fixed cascade"], allidx, t)
            bs = []
            for _ in range(400):
                idx = rng.integers(0, n, n)
                a, b = cost_at(arms[k], idx, t), cost_at(arms["fixed cascade"], idx, t)
                if np.isfinite(a) and np.isfinite(b) and b > 0: bs.append(a/b)
            ci = f"[{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]" if len(bs) > 50 else "[n/a]"
            line += f"  {t*100:.0f}%: {r0:.2f} {ci}" if np.isfinite(r0) else f"  {t*100:.0f}%: n/a"
        print(line)

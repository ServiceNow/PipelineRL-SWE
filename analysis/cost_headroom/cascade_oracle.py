"""One-shot prefill routing vs FrugalGPT-style cascades with a PERFECT judge (NEW_PATH 4.A.69), pinned, billed. Single submission, no
verifier: a cascade calls the tiers of a plan cheapest-first (by mean TRAIN cost), each call one stored draw; the judge is the true
outcome (never wrong, free), so the cascade stops at the first correct attempt, and the last tier's answer is submitted regardless.
Every non-empty subset of the 5 routes is a plan; the cascade family's frontier is the convex hull over plans (an upper bound for any
learned judge, which can only be worse). Draw replicates: problem's draw d for every tier (d = 0..min draws-1), averaged.
Compared at matched accuracy on TEST: cost saved by our router (prefill success + cost readouts, V swept) vs the perfect-judge cascade
family, and vs the median-pricing router. Paired problem bootstrap (300).
Usage: REASON_ROOT=.../reason_pinned python cascade_oracle.py LCB|Omni|MMLU-Pro
"""
import itertools, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / "tmlr_free_analyses.py").read().split("# ---------------------------------------------------------------- 1.")[0]
     .replace('print(f"===== {POOL}', 'print(f"===== cascades {POOL}'))
okd = (t["final_outcome"] & v).astype(float); K = v.shape[2]
pd_draw = t["prompt_tokens"] * rates[None, :, 0, None] + t["completion_tokens"] * rates[None, :, 1, None]   # per-draw cost (tokens x billed rate)
if POOL != "LCB":                                      # one billed draw per test problem: use the billed cost
    pd_draw[:, :, 0] = np.where(v[:, :, 0], paid, pd_draw[:, :, 0])
nd = int(min(v[ev].sum(2).min(0).min(), 3)) or 1
order = list(np.argsort([paid[tr, k].mean() for k in range(M)]))
plans = [p for r in range(1, M + 1) for p in itertools.combinations(order, r)]


def cascade_points(ii):
    pts = []
    for plan in plans:
        acc, cost = 0.0, 0.0
        for d in range(nd):
            solved = np.zeros(len(ii), bool); c = np.zeros(len(ii))
            for k in plan:
                live = ~solved & v[ii, k, d]
                c += np.where(~solved, pd_draw[ii, k, d], 0.0); solved |= live & (okd[ii, k, d] > 0)
            acc += solved.mean(); cost += c.mean()
        pts.append((cost / nd, acc / nd))
    return pts


def router_points(C, ii):
    pts = []
    for V in VALUES:
        m = (V * P[ii] - C[ii]).argmax(1); k = np.arange(len(ii)); pts.append((paid[ii][k, m].mean(), q[ii][k, m].mean()))
    return pts


def saved_h(Ha, Hb):
    lo, hi = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1])
    if hi <= lo:
        return np.nan
    T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    return 1 - float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T]))))


def all_(ii):
    Hc = hull(cascade_points(ii)); Ho = hull(router_points(C_ours, ii)); Hm = hull(router_points(C_med, ii))
    return saved_h(Ho, Hc), saved_h(Hm, Hc), Hc
g_o, g_m, Hc = all_(ev)
rng = np.random.default_rng(0); B = [all_(ev[rng.integers(0, len(ev), len(ev))])[:2] for _ in range(300)]
B = np.array(B)
best_plans = sorted(zip(cascade_points(ev), plans), key=lambda x: -x[0][1])[:3]
res = dict(pool=POOL, n_eval=int(len(ev)), draw_replicates=nd, ours_vs_cascade=[g_o, *np.nanpercentile(B[:, 0], [2.5, 97.5]).tolist()],
           median_router_vs_cascade=[g_m, *np.nanpercentile(B[:, 1], [2.5, 97.5]).tolist()],
           cascade_frontier=[list(map(float, h)) for h in Hc])
print(f"  draws per tier {nd}; plans {len(plans)}; cascade max accuracy {Hc[-1][1]*100:.1f}%", flush=True)
print(f"  OUR router saves {g_o*100:+.1f}% [{np.nanpercentile(B[:,0],2.5)*100:+.1f}, {np.nanpercentile(B[:,0],97.5)*100:+.1f}] vs the PERFECT-judge cascade", flush=True)
print(f"  median-pricing router vs the perfect-judge cascade: {g_m*100:+.1f}% [{np.nanpercentile(B[:,1],2.5)*100:+.1f}, {np.nanpercentile(B[:,1],97.5)*100:+.1f}]", flush=True)
json.dump(res, open(Path(__file__).parent / f"cascade_oracle_{POOL.replace('-', '').lower()}{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)

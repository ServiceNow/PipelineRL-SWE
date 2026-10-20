"""Where the saving comes from: share of test queries sent to each route at one matched accuracy (the middle of the shared accuracy
band), for median-length pricing, our cost probe and realized length, all with the same success predictions (the paper split).
A frontier point between two hull vertices is their mixture, so its route shares are interpolated the same way as its cost.
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned python route_shares.py [POOL ...] -> route_shares_pinned.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
POOLS_ = sys.argv[1:] or ["LCB", "Omni", "MMLU-Pro", "AIME", "APPS", "BCB", "CC", "SuperGPQA", "BBEH"]; sys.argv = [sys.argv[0], "shares", "ALL"]
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])


def policies(d, Pm, C, ii):
    """every V's policy on ii: (cost, accuracy, route shares), and the hull vertices among them"""
    m = (VALUES[:, None, None] * Pm[ii][None] - C[ii][None]).argmax(2); r = np.arange(len(ii))[None]
    cost, acc = d["paid"][ii][r, m].mean(1), d["q"][ii][r, m].mean(1); sh = np.stack([(m == k).mean(1) for k in range(d["M"])], 1)
    H = hull(list(zip(cost, acc))); at = {(c, a): j for j, (c, a) in enumerate(zip(cost, acc))}
    return [(c, a, sh[at[(c, a)]]) for c, a in H]


def at_accuracy(H, t):
    for (c1, a1, s1), (c2, a2, s2) in zip(H, H[1:]):
        if a1 <= t <= a2:
            w = (t - a1) / max(a2 - a1, 1e-12); return c1 + w * (c2 - c1), s1 + w * (s2 - s1)
    return np.nan, np.full(len(H[0][2]), np.nan)


out = {}
for pool in POOLS_:
    d = load(pool); N = len(d["ids"]); ev = d["ev"]; S, T = view(d, d["tr"]), view(d, np.arange(N))
    tk = {"median": np.repeat(d["med"][None], N, 0), "ours": fit_predict("ours", S, T), "realized": d["L"]}
    Hs = {a: policies(d, d["P"], cost_of(d, x), ev) for a, x in tk.items()}
    lo, hi = max(H[0][1] for H in Hs.values()), min(H[-1][1] for H in Hs.values()); t = (lo + hi) / 2
    o = {"slots": d["slots"], "accuracy": float(t), "band": [float(lo), float(hi)], "n_test": len(ev)}
    for a, H in Hs.items():
        c, s = at_accuracy(H, t); o[a] = dict(cost_per_1k=float(c * 1000), shares=[float(x) for x in s])
    o["saving_vs_median"] = 1 - o["ours"]["cost_per_1k"] / o["median"]["cost_per_1k"]
    out[pool] = o
    print(f"{pool:<10} acc {t:.3f} (band {lo:.3f}-{hi:.3f})  " + "  ".join(f"{a} ${o[a]['cost_per_1k']:.3f} " + "/".join(f"{x:.2f}" for x in o[a]["shares"]) for a in tk), flush=True)
print("slots", d["slots"], "rates", d["rates"].tolist())
json.dump(out, open(Path(__file__).parent / f"route_shares{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1)

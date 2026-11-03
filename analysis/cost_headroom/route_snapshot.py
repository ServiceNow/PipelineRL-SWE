"""Figure 1's route-share snapshot at a chosen accuracy (not the middle of the band, as route_shares.py / fig:shares): share of test
queries per route for median-length pricing, our cost probe and realized length, same success predictions, the paper split.
The accuracy was chosen from a scan of all nine datasets for a large split between median pricing and ours, close to realized length,
at a high accuracy (LiveCodeBench, 77% = 86% of deepseek-v4-flash alone).
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned python route_snapshot.py LCB 0.77 -> route_snapshot_pinned.json (merged)
"""
import json, sys
from pathlib import Path
POOL_, ACC = sys.argv[1], float(sys.argv[2])
_here = Path(__file__).resolve().parent; sys.argv = ["x", "shares", "ALL"]; __file__ = str(_here / "route_shares.py")
exec(open(_here / "route_shares.py").read().split("out = {}")[0])
d = load(POOL_); N = len(d["ids"]); ev = d["ev"]; S, T = view(d, d["tr"]), view(d, np.arange(N))
tk = {"median": np.repeat(d["med"][None], N, 0), "ours": fit_predict("ours", S, T), "realized": d["L"]}
Hs = {a: policies(d, d["P"], cost_of(d, x), ev) for a, x in tk.items()}
lo, hi = max(H[0][1] for H in Hs.values()), min(H[-1][1] for H in Hs.values()); assert lo <= ACC <= hi
o = {"slots": d["slots"], "accuracy": ACC, "band": [float(lo), float(hi)], "n_test": len(ev)}
for a, H in Hs.items():
    c, s = at_accuracy(H, ACC); o[a] = dict(cost_per_1k=float(c * 1000), shares=[float(x) for x in s])
o["saving_vs_median"] = 1 - o["ours"]["cost_per_1k"] / o["median"]["cost_per_1k"]
o["capture"] = (o["median"]["cost_per_1k"] - o["ours"]["cost_per_1k"]) / (o["median"]["cost_per_1k"] - o["realized"]["cost_per_1k"])
outf = _here / "route_snapshot_pinned.json"; res = json.load(open(outf)) if outf.exists() else {}; res[POOL_] = o
json.dump(res, open(outf, "w"), indent=1)
print(POOL_, ACC, {a: (round(o[a]["cost_per_1k"], 3), [round(x, 2) for x in o[a]["shares"]]) for a in tk}, "saving", round(o["saving_vs_median"], 3), "capture", round(o["capture"], 2))

"""Does a predictor-free description of the data predict headroom? (NEW_PATH 4.A.109; models fixed before looking at results)
Target: headroom (perfect per-query cost vs median pricing, 10-set mean; drop_summary_pinned.json) of every route subset of the nine
five-route pools (full, leave-one-out, pairs). Features from data_decomp.py, all computed from realized lengths and prices only:
s_b (shared problem level), s_ab (problem x route interaction), s_e (noise), ps (price spread: s.d. of log mean cost per call).
Models (least squares, intercept):
  A  H ~ ps                              price gaps alone
  B  H ~ s_b + s_ab                      query-driven length variation alone
  C  H ~ s_b + s_ab + ps                 both
  D  H ~ log((s_b + s_ab) / ps^2)        query-driven variance relative to price-gap variance (the paper's condition, noise removed)
Reported: in-sample R2 and leave-one-pool-out R2. Also the full-pool decomposition and the headroom-by-source split per pool.
Usage: python data_law.py [DIR] -> data_law_pinned.json
"""
import json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/decomp"); H = Path(__file__).parent
drop = json.load(open(H / "drop_summary_pinned.json")); dec = {json.load(open(f))["pool"]: json.load(open(f)) for f in sorted(D.glob("data_decomp_*_pinned.json"))}
out = {"pools": {}, "models": {}}
print("full pool: share of log-length variance (excluding route means) | route-mean var | price spread | headroom: both / level only / route-specific only")
for pool, r in dec.items():
    f = r["decomp"]["full"]; tot = sum(f[k] or 0 for k in ("problem", "interaction", "noise")); os_ = r["oracle_split"]
    sp = {a: float(np.mean([os_[s][a] for s in os_])) for a in ("both", "level_only", "route_specific_only")}
    out["pools"][pool] = dict(decomp=f, shares={k: (f[k] or 0) / tot for k in ("problem", "interaction", "noise")}, oracle_split=sp)
    print(f"  {pool:<10} problem {f['problem'] / tot:4.0%}  interaction {f['interaction'] / tot:4.0%}  noise " + ("  n/a" if f["noise"] is None else f"{f['noise'] / tot:4.0%}")
          + f" | {f['route']:.2f} | {f['price_spread']:.2f} | {sp['both']:5.1f} / {sp['level_only']:5.1f} / {sp['route_specific_only']:5.1f}")
rows = []
for pool, r in dec.items():
    for sp_, f in r["decomp"].items():
        o = drop.get(pool, {}).get(sp_)
        if o and o["oracle"]["vs_median"][0] is not None and f["problem"] is not None and f["price_spread"] > 0:
            rows.append(dict(pool=pool, subset=sp_, H=o["oracle"]["vs_median"][0], sb=f["problem"], sab=f["interaction"], ps=f["price_spread"]))
y = np.array([r["H"] for r in rows]); pools = np.array([r["pool"] for r in rows])
F = {"A": lambda r: [r["ps"]], "B": lambda r: [r["sb"], r["sab"]], "C": lambda r: [r["sb"], r["sab"], r["ps"]], "D": lambda r: [np.log(max(r["sb"] + r["sab"], 1e-4) / r["ps"] ** 2)]}
r2 = lambda a, b: float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())
print(f"\nheadroom of {len(rows)} route subsets from predictor-free statistics:")
for nm, fx in F.items():
    X = np.c_[np.ones(len(rows)), np.array([fx(r) for r in rows])]; b = np.linalg.lstsq(X, y, rcond=None)[0]; lo = np.zeros(len(y))
    for p in set(pools):
        m = pools == p; lo[m] = X[m] @ np.linalg.lstsq(X[~m], y[~m], rcond=None)[0]
    out["models"][nm] = dict(coef=b.tolist(), r2_in=r2(y, X @ b), r2_lopo=r2(y, lo))
    print(f"  {nm}: R2 in-sample {out['models'][nm]['r2_in']:.2f}  leave-one-pool-out {out['models'][nm]['r2_lopo']:.2f}  coef {np.round(b, 2).tolist()}")
out["rows"] = rows
json.dump(out, open(H / "data_law_pinned.json", "w"), indent=1, default=float)

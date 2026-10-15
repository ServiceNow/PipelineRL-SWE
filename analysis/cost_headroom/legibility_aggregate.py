"""Aggregate legibility.py (NEW_PATH 4.A.106) over the 10 training sets and relate it to the routing results.
  routes   per pool and route: test R2 of our readout, of pricing from success, the reliability ceiling and efficiency (R2 / ceiling)
  pairs    per pool and route pair: differential R2 (predicted vs realized log length ratio), its ceiling, mean per-route R2,
           correlation of the two routes' log lengths; joined with the pair's capture (ours vs median / headroom, drop_summary)
           -> does differential legibility explain capture better than per-route legibility?
  pilot    per pool: 5-fold CV R2 within k labelled problems vs the full-data test R2 (mean over routes)
Usage: python legibility_aggregate.py [DIR] -> legibility_summary_pinned.json
"""
import json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/legibility")
H = Path(__file__).parent; drop = json.load(open(H / "drop_summary_pinned.json")); seeds, ceil = {}, {}
for f in sorted(D.glob("legibility_*_pinned.json")):
    r = json.load(open(f)); seeds.setdefault(r["pool"], {}).update(r["seeds"]); ceil[r["pool"]] = r["ceiling"]
mn = lambda xs: float(np.mean([x for x in xs if x is not None])) if any(x is not None for x in xs) else None
f2 = lambda x: " n/a" if x is None else f"{x:.2f}"
out = {"routes": {}, "pairs": [], "pilot": {}}
for pool, S in seeds.items():
    ks = sorted(S, key=int); slots = list(S[ks[0]]["routes"])
    out["routes"][pool] = {sl: {m: mn([S[k]["routes"][sl][m] for k in ks]) for m in ("r2_ours", "r2_fromsuccess", "reliability", "efficiency")} | {"icc": ceil[pool][sl]["icc"]} for sl in slots}
    o = out["routes"][pool]
    print(f"{pool:<10} R2 / ceiling / efficiency: " + "  ".join(f"{sl} {f2(o[sl]['r2_ours'])}/{f2(o[sl]['reliability'])}/{f2(o[sl]['efficiency'])}" for sl in slots))
    for pr in S[ks[0]]["pairs"]:
        row = {m: mn([S[k]["pairs"][pr][m] for k in ks]) for m in S[ks[0]]["pairs"][pr]}
        dp = drop.get(pool, {}).get(pr)
        if dp and dp["ours"]["vs_median"][0] is not None and dp["oracle"]["vs_median"][0] is not None:
            row.update(pool=pool, pair=pr, ours=dp["ours"]["vs_median"][0], headroom=dp["oracle"]["vs_median"][0]); out["pairs"].append(row)
    ks_p = [kk for kk in ("20", "50", "100") if kk in S[ks[0]]["pilot"]]
    out["pilot"][pool] = {kk: mn([float(np.mean(S[k]["pilot"][kk]["cv_r2"])) for k in ks]) for kk in ks_p} | {"full_test": mn([float(np.mean([S[k]["routes"][sl]["r2_ours"] for sl in slots])) for k in ks])}
P = [r for r in out["pairs"] if r["headroom"] > 5]
cap = np.array([r["ours"] / r["headroom"] for r in P]); sav = np.array([r["ours"] for r in P]); hd = np.array([r["headroom"] for r in P])
rank = lambda x: np.argsort(np.argsort(x))
sp = lambda a, b: float(np.corrcoef(rank(a), rank(b))[0, 1])
print(f"\npairs with headroom > 5 pt: {len(P)}")
for m in ("r2_diff_ours", "r2_mean_routes", "corr_routes", "reliability_diff", "r2_diff_fromsuccess"):
    x = np.array([r[m] if r[m] is not None else np.nan for r in P]); ok = np.isfinite(x)
    out[f"spearman_capture_{m}"] = sp(cap[ok], x[ok]); out[f"spearman_saving_{m}"] = sp(sav[ok], x[ok])
    print(f"  Spearman with capture: {m:<20} {out[f'spearman_capture_{m}']:+.2f}   with saving {out[f'spearman_saving_{m}']:+.2f}")
X1 = np.c_[np.ones(len(P)), hd * np.array([r["r2_mean_routes"] for r in P])]; X2 = np.c_[np.ones(len(P)), hd * np.array([max(r["r2_diff_ours"], 0) for r in P])]
for nm, X in (("headroom x per-route R2", X1), ("headroom x differential R2", X2)):
    b = np.linalg.lstsq(X, sav, rcond=None)[0]; yh = X @ b; r2 = 1 - ((sav - yh) ** 2).sum() / ((sav - sav.mean()) ** 2).sum()
    out[f"fit_saving_{nm}"] = dict(coef=b.tolist(), r2=float(r2)); print(f"  saving ~ {nm}: R2 {r2:.2f}  coef {np.round(b, 2).tolist()}")
print("\npilot: 5-fold CV R2 within k labelled problems vs full-data test R2 (mean over routes)")
for pool, o in out["pilot"].items():
    print(f"  {pool:<10} " + "  ".join(f"k={kk} {f2(v)}" for kk, v in o.items() if kk != "full_test") + f"   full test {f2(o['full_test'])}")
for kk in ("20", "50", "100"):
    a = np.array([o[kk] for o in out["pilot"].values() if o.get(kk) is not None]); b = np.array([o["full_test"] for o in out["pilot"].values() if o.get(kk) is not None])
    if len(a) > 2:
        out[f"pilot_spearman_k{kk}"] = sp(a, b); print(f"  k={kk}: Spearman with full test R2 over pools {out[f'pilot_spearman_k{kk}']:+.2f}, mean pilot / full {np.mean(a):.2f} / {np.mean(b):.2f}")
json.dump(out, open(H / "legibility_summary_pinned.json", "w"), indent=1, default=float)

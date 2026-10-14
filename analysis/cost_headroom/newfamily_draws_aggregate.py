"""Aggregate newfamily_draws.py (NEW_PATH 4.A.96): per benchmark and sub-pool, mean over training sets 0..9 (point) with percentile
intervals of the test resamples pooled over the sets; the manifest split (-1) is printed as a control.
Usage: python newfamily_draws_aggregate.py [DIR] -> newfamily_draws_summary_pinned.json
"""
import json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/newfam_draws")
seeds = {}
for f in sorted(D.glob("newfamily_draws_*_pinned.json")):
    r = json.load(open(f)); seeds.setdefault(r["pool"], {}).update(r["seeds"])


def agg(lists):                                       # lists: per set [point, resamples...] -> [mean point, lo, hi], reach
    pts = np.array([x[0] for x in lists], float); bs = np.concatenate([np.array(x[1:], float) for x in lists])
    pt = float(np.nanmean(pts)) if np.isfinite(pts).any() else None
    ci = list(map(float, np.nanpercentile(bs, [2.5, 97.5]))) if np.isfinite(bs).any() else [None, None]
    return dict(v=[pt, *ci], reach_sets=int(np.isfinite(pts).sum()), n_sets=len(pts))


f_ = lambda a: "  n/a " if a["v"][0] is None else f"{a['v'][0]:+6.1f} [{a['v'][1]:+.1f}, {a['v'][2]:+.1f}]"
out = {}
for pool, S in seeds.items():
    ks = sorted(k for k in S if int(k) >= 0); out[pool] = {"n_sets": len(ks)}
    for sp in S[ks[0]]:
        g = lambda path: agg([path(S[k][sp]) for k in ks]); r0 = S[ks[0]][sp]; o = {}
        o["arms"] = {a: dict(vs_median=g(lambda x: x["arms"][a]["vs_median"]), ours_vs_it=g(lambda x: x["arms"][a]["ours_vs_it"])) for a in r0["arms"]}
        o["mix"] = {a: dict(saving=g(lambda x: x["mix"][a]["saving"]), acc_gap=g(lambda x: x["mix"][a]["acc_gap"])) for a in r0["mix"]}
        o["singles"] = {a: {sl: dict(saving=g(lambda x: x["singles"][a][sl]["saving"]), acc_gap=g(lambda x: x["singles"][a][sl]["acc_gap"])) for sl in r0["singles"][a]} for a in r0["singles"]}
        o["routes"] = {sl: {k: float(np.mean([S[s][sp]["routes"][sl][k] for s in ks])) for k in ("acc", "usd_per_1k")} for sl in r0["routes"]}
        out[pool][sp] = o
        ctl = S.get("-1", {}).get(sp)
        print(f"\n===== {pool} / {sp} ({len(ks)} training sets)" + (f"; manifest split: ours vs median {ctl['arms']['ours']['vs_median'][0]:+.1f}" if ctl else ""))
        print("  vs median:  " + "  ".join(f"{a} {o['arms'][a]['vs_median']['v'][0]:+.1f}" for a in o["arms"]))
        print("  ours vs it: " + "  ".join(f"{a} {f_(o['arms'][a]['ours_vs_it'])}" for a in o["arms"] if a != "ours"))
        print(f"  vs a random mix of single routes: ours {f_(o['mix']['ours']['saving'])} ({o['mix']['ours']['saving']['reach_sets']}/{len(ks)} sets), "
              f"median pricing {f_(o['mix']['median']['saving'])}; router max acc - best single: ours {f_(o['mix']['ours']['acc_gap'])}")
        for sl, rt in o["routes"].items():
            so = o["singles"]["ours"][sl]
            print(f"    {sl:<9} acc {rt['acc'] * 100:5.1f} ${rt['usd_per_1k']:6.2f}/1k | ours saves {f_(so['saving'])} (reached {so['saving']['reach_sets']}/{len(ks)})"
                  f" | max acc - it {f_(so['acc_gap'])} | median-pricing router saves {f_(o['singles']['median'][sl]['saving'])}")
json.dump(out, open(Path(__file__).parent / "newfamily_draws_summary_pinned.json", "w"), indent=1)

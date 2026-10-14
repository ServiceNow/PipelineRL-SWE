"""Aggregate tmlr_strengthen.py drop (NEW_PATH 4.A.98): per pool and route subset (full, leave-one-route-out, pairs), means over the
10 training sets with percentile intervals of the test resamples pooled over sets; and the practitioner trade-off (cost saved
relative to calling the best route / deepseek-v4-flash alone when giving up DELTA accuracy points).
Usage: python drop_aggregate.py [DIR] -> drop_summary_pinned.json
"""
import json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/drop")
seeds = {}
for f in sorted(D.glob("tmlr_strengthen_drop_*_pinned.json")):
    r = json.load(open(f)); seeds.setdefault(r["pool"], {}).update(r["seeds"])


def agg(lists):
    pts = np.array([x[0] for x in lists], float); bs = np.concatenate([np.array(x[1:], float) for x in lists])
    return [float(np.nanmean(pts)) if np.isfinite(pts).any() else None,
            *(map(float, np.nanpercentile(bs, [2.5, 97.5])) if np.isfinite(bs).any() else [None, None]), int(np.isfinite(pts).sum())]


fmt = lambda v: "   n/a" if v[0] is None else f"{v[0]:+6.1f} [{v[1]:+.1f}, {v[2]:+.1f}]"
out = {}
for pool, S in sorted(seeds.items()):
    ks = sorted(S, key=int); out[pool] = {"n_sets": len(ks)}; print(f"\n===== {pool} ({len(ks)} training sets)")
    for sp in S[ks[0]]:
        r0 = S[ks[0]][sp]; arms = [a for a in r0 if not a.startswith("_")]
        o = {a: {m: agg([S[k][sp][a][m] for k in ks]) for m in ("vs_median", "ours_vs_it")} for a in arms}
        o["_trade"] = {rf: {a: {dl: agg([S[k][sp]["_trade"][rf][a][dl] for k in ks]) for dl in r0["_trade"][rf][a]} for a in r0["_trade"][rf]} for rf in r0["_trade"]}
        o["_refs"] = r0["_refs"]; out[pool][sp] = o
        print(f"  {sp:<18} ours vs median {fmt(o['ours']['vs_median'])}  headroom {o['oracle']['vs_median'][0]:+5.1f}"
              f"  | ours vs fromsuccess {fmt(o['fromsuccess']['ours_vs_it'])}  vs bins {fmt(o['zr-bins']['ours_vs_it'])}")
        if sp in ("full", "-dsv4f"):
            for rf, t in o["_trade"].items():
                print(f"      trade vs {o['_refs'][rf]['slot']} alone ({o['_refs'][rf]['acc'] * 100:.1f}% at ${o['_refs'][rf]['usd_per_1k']:.2f}/1k), cost saved giving up "
                      + "  ".join(f"{dl} pt: ours {fmt(t['ours'][dl])} median {t['median'][dl][0] if t['median'][dl][0] is None else round(t['median'][dl][0], 1)} mix {t['mix'][dl][0] if t['mix'][dl][0] is None else round(t['mix'][dl][0], 1)}" for dl in t["ours"]))
json.dump(out, open(Path(__file__).parent / "drop_summary_pinned.json", "w"), indent=1)

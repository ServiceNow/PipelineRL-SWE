"""Aggregate tmlr_strengthen.py singles (NEW_PATH 4.A.95): per pool and single route, mean over the 10 training sets of the cost our
router (and the median-pricing router) saves at that route's accuracy, with percentile intervals pooled over sets; 'reach' = share of
(set, resample) pairs whose frontier reaches the route's accuracy; acc gap = router's most accurate point minus the route's accuracy.
Usage: python singles_aggregate.py [DIR] -> singles_summary_pinned.json
"""
import json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/draws")
out = {}
for f in sorted(D.glob("tmlr_strengthen_singles_*_pinned.json")):
    r = json.load(open(f)); seeds = r["seeds"]; pool = r["pool"]; slots = list(next(iter(seeds.values()))["_routes"]); res = {}
    print(f"\n===== {pool} ({len(seeds)} training sets)")
    for sl in slots:
        acc = np.mean([s["_routes"][sl]["acc"] for s in seeds.values()]); usd = np.mean([s["_routes"][sl]["usd_per_1k"] for s in seeds.values()]); row = dict(acc=acc, usd_per_1k=usd)
        for a in ("ours", "median"):
            pts = np.array([s[a][sl]["saving"][0] for s in seeds.values()], float); bs = np.concatenate([s[a][sl]["saving"][1:] for s in seeds.values()]).astype(float)
            gp = np.array([s[a][sl]["acc_gap"][0] for s in seeds.values()]); gb = np.concatenate([s[a][sl]["acc_gap"][1:] for s in seeds.values()])
            row[a] = dict(saving=[float(np.nanmean(pts)) if np.isfinite(pts).any() else None, *map(float, np.nanpercentile(bs, [2.5, 97.5]))] if np.isfinite(bs).any() else None,
                          reach_sets=int(np.isfinite(pts).sum()), reach=float(np.isfinite(bs).mean()), acc_gap=[float(gp.mean()), *map(float, np.percentile(gb, [2.5, 97.5]))])
        res[sl] = row; o = row["ours"]
        sv = "not reached" if o["saving"] is None or o["saving"][0] is None else f"{o['saving'][0]:+6.1f} [{o['saving'][1]:+.1f}, {o['saving'][2]:+.1f}]"
        print(f"  {sl:<9} acc {acc * 100:5.1f}  ${usd:6.2f}/1k | ours saves {sv} (reached in {o['reach_sets']}/10 sets, {o['reach']:.0%} of resamples)"
              f" | router max acc - route {o['acc_gap'][0]:+5.1f} [{o['acc_gap'][1]:+.1f}, {o['acc_gap'][2]:+.1f}] pt | median-pricing router saves "
              + ("n/a" if row["median"]["saving"] is None or row["median"]["saving"][0] is None else f"{row['median']['saving'][0]:+.1f}"))
    if "_mix" in next(iter(seeds.values())):
        res["_mix"] = {}
        for a in ("ours", "median"):
            pts = np.array([s["_mix"][a]["saving"][0] for s in seeds.values()], float); bs = np.concatenate([s["_mix"][a]["saving"][1:] for s in seeds.values()]).astype(float)
            gp = np.array([s["_mix"][a]["acc_gap"][0] for s in seeds.values()]); gb = np.concatenate([s["_mix"][a]["acc_gap"][1:] for s in seeds.values()])
            res["_mix"][a] = dict(saving=[float(np.nanmean(pts)), *map(float, np.nanpercentile(bs, [2.5, 97.5]))], acc_gap=[float(gp.mean()), *map(float, np.percentile(gb, [2.5, 97.5]))])
        m = res["_mix"]; print(f"  vs a random mix of single routes: ours {m['ours']['saving'][0]:+.1f} [{m['ours']['saving'][1]:+.1f}, {m['ours']['saving'][2]:+.1f}]"
                               f"  median pricing {m['median']['saving'][0]:+.1f} [{m['median']['saving'][1]:+.1f}, {m['median']['saving'][2]:+.1f}]"
                               f"  | router max acc - best route {m['ours']['acc_gap'][0]:+.1f}")
    out[pool] = res
json.dump(out, open(Path(__file__).parent / "singles_summary_pinned.json", "w"), indent=1)

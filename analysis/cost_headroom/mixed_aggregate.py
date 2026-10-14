"""Aggregate mixed_pool.py (NEW_PATH 4.A.103): mean over the 10 training sets, percentile intervals of the pool-stratified test
resamples pooled over sets, and the number of sets whose interval excludes zero.
Usage: python mixed_aggregate.py [DIR] -> mixed_summary_pinned.json
"""
import json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/mixed")
runs = [json.load(open(f)) for f in sorted(D.glob("mixed_pool_pinned_d*.json"))]; out = {"n_sets": len(runs), "arms": {}}
print(f"mixed traffic, {len(runs)} training sets, test {runs[0]['n']['test']} ({runs[0]['cap']} per pool)")
for a in runs[0]["arms"]:
    o = {}
    for m in ("vs_median", "vs_median_by_ds", "ours_vs_it"):
        pts = np.array([r["arms"][a][m][0] for r in runs]); bs = np.concatenate([np.array(r["arms"][a][m][1:], float) for r in runs])
        sig = sum(np.nanpercentile(r["arms"][a][m][1:], 2.5) > 0 for r in runs)
        o[m] = [float(np.nanmean(pts)), *map(float, np.nanpercentile(bs, [2.5, 97.5])), int(sig)]
    out["arms"][a] = o
    print(f"  {a:<13} " + "  ".join(f"{m} {v[0]:+5.1f} [{v[1]:+.1f}, {v[2]:+.1f}] ({v[3]}/{len(runs)})" for m, v in o.items()))
json.dump(out, open(Path(__file__).parent / "mixed_summary_pinned.json", "w"), indent=1)

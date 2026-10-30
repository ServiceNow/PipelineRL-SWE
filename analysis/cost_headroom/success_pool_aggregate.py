"""Sharing information across our success probes, aggregate (NEW_PATH 4.A.126): pools success_pool.py over the 10 training sets. Per
dataset and arm: saving vs median pricing (our success), ours vs the arm (paired; 95% percentiles of the pooled bootstrap replicates),
per-route AUC, pairwise route-ordering accuracy, and the chosen lam. Checks the decision rule fixed before the results: adopt "shared"
only if it is never significantly behind ours and is ahead on the mean saving over datasets.
Usage: python success_pool_aggregate.py [LOGDIR] -> success_pool_summary_pinned.json
"""
import collections, glob, json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/succpool")
ARMS = ["ours", "ours-ll", "shared", "shared-raw", "zr4b+cost"]
S = {}
for f in sorted(glob.glob(str(D / "success_pool_*_pinned.json"))):
    r = json.load(open(f)); S.setdefault(r["pool"], {}).update(r["seeds"])
out = {}
for p, s in S.items():
    v = list(s.values()); out[p] = {"n_sets": len(v), "lam": dict(collections.Counter(str(x["_cfg"]["lam"]) for x in v))}
    for a in ARMS:
        bs = np.ravel([x[a]["ours_vs_it_bs"] for x in v])
        out[p][a] = dict(vs_median=float(np.mean([x[a]["vs_median"] for x in v])), vs_mix=float(np.mean([x[a]["vs_mix"] for x in v])),
                         ours_vs_it=[float(np.mean([x[a]["ours_vs_it"] for x in v])), *map(float, np.nanpercentile(bs, [2.5, 97.5]))],
                         auc=float(np.mean([x[a]["auc"] for x in v])), pair_acc=float(np.mean([x[a]["pair_acc"] for x in v])))
    print(f"{p:9s} " + "  ".join(f"{a} {out[p][a]['vs_median']:+5.1f} ({out[p][a]['ours_vs_it'][0]:+5.1f} [{out[p][a]['ours_vs_it'][1]:+5.1f},{out[p][a]['ours_vs_it'][2]:+5.1f}])" for a in ARMS))
mean = {a: float(np.mean([out[p][a]["vs_median"] for p in out])) for a in ARMS}
behind = [p for p in out if out[p]["shared"]["ours_vs_it"][1] > 0]
out["_rule"] = dict(mean_vs_median=mean, shared_significantly_behind_ours_on=behind, adopt_shared=not behind and mean["shared"] > mean["ours"])
print("mean vs median:", {a: round(m, 1) for a, m in mean.items()}, "| shared significantly behind ours on:", behind or "none", "| adopt:", out["_rule"]["adopt_shared"])
json.dump(out, open(Path(__file__).parent / "success_pool_summary_pinned.json", "w"), indent=1)

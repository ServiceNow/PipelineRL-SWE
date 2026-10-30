"""ZeroRouter's item-response structure in our success probes, aggregate (NEW_PATH 4.A.127): pools success_mirt.py over the 10 training sets. Per
dataset and arm: saving vs median pricing (our success), ours vs the arm (paired; 95% percentiles of the pooled bootstrap replicates),
per-route AUC, pairwise route-ordering accuracy, and the chosen D. Checks the decision rule fixed before the results: adopt "mirt" (or
"auto") over "shared" only if it is never significantly behind ours and is ahead of "shared" on the mean saving over datasets.
Usage: python success_pool_aggregate.py [LOGDIR] -> success_mirt_summary_pinned.json
"""
import collections, glob, json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/succmirt")
ARMS = ["ours", "shared", "mirt", "mirt-raw", "auto", "zr4b+cost"]
S = {}
for f in sorted(glob.glob(str(D / "success_mirt_*_pinned.json"))):
    r = json.load(open(f)); S.setdefault(r["pool"], {}).update(r["seeds"])
out = {}
for p, s in S.items():
    v = list(s.values()); out[p] = {"n_sets": len(v), "mirt_D": dict(collections.Counter(str(x["_cfg"]["mirt"]["D"]) for x in v))}
    for a in ARMS:
        bs = np.ravel([x[a]["ours_vs_it_bs"] for x in v])
        out[p][a] = dict(vs_median=float(np.mean([x[a]["vs_median"] for x in v])), vs_mix=float(np.mean([x[a]["vs_mix"] for x in v])),
                         ours_vs_it=[float(np.mean([x[a]["ours_vs_it"] for x in v])), *map(float, np.nanpercentile(bs, [2.5, 97.5]))],
                         auc=float(np.mean([x[a]["auc"] for x in v])), pair_acc=float(np.mean([x[a]["pair_acc"] for x in v])))
    print(f"{p:9s} " + "  ".join(f"{a} {out[p][a]['vs_median']:+5.1f} ({out[p][a]['ours_vs_it'][0]:+5.1f} [{out[p][a]['ours_vs_it'][1]:+5.1f},{out[p][a]['ours_vs_it'][2]:+5.1f}])" for a in ARMS))
mean = {a: float(np.mean([out[p][a]["vs_median"] for p in out])) for a in ARMS}
rule = {a: dict(significantly_behind_ours_on=[p for p in out if out[p][a]["ours_vs_it"][1] > 0], ahead_of_shared=mean[a] - mean["shared"]) for a in ("mirt", "mirt-raw", "auto")}
for a in rule:
    rule[a]["passes"] = not rule[a]["significantly_behind_ours_on"] and rule[a]["ahead_of_shared"] > 0
out["_rule"] = dict(mean_vs_median=mean, candidates=rule, note="the pre-set rule names mirt and auto; mirt-raw is reported, not pre-specified")
print("mean vs median:", {a: round(m, 2) for a, m in mean.items()}); print("rule:", rule)
json.dump(out, open(Path(__file__).parent / "success_mirt_summary_pinned.json", "w"), indent=1)

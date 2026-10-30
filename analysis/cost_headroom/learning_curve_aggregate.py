"""Learning curves aggregate (NEW_PATH 4.A.121): pools learning_curve.py over the 10 training sets per dataset. Per panel (cost: estimators
with the full-set success probes; system: whole routers with their own success), per n in {50, 100, 200, all}: cost saved relative to the
fixed reference (full-set median pricing, full-set success), mean over training sets with 95% percentiles of the pooled bootstrap
replicates, and ours vs each arm at the same n (paired). Usage: python learning_curve_aggregate.py [LOGDIR] -> learning_curve_summary_pinned.json
"""
import glob, json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/lcurve")
POOLS = ["LCB", "Omni", "MMLU-Pro", "AIME", "APPS", "BCB", "CC", "SuperGPQA", "BBEH", "SWESmith"]; NS = ["50", "100", "200", "all"]
lc = lambda p: p.lower().replace("-", "")
iv = lambda pt, bs: [float(np.mean(pt)), *map(float, np.nanpercentile(np.ravel(bs), [2.5, 97.5]))]
out = {}
for pool in POOLS:
    S = {}
    for f in sorted(glob.glob(str(D / f"learning_curve_{lc(pool)}_*_pinned.json"))):
        S.update(json.load(open(f))["seeds"])
    if not S:
        continue
    v = list(S.values()); out[pool] = {"n_sets": len(v), "n_train_all": v[0]["all"]["n_train"] if "n_train" in v[0]["all"] else None}
    print(f"\n===== {pool}: {len(v)} training sets")
    for panel in ("cost", "system"):
        arms = [a for a in v[0]["all"][panel] if not a.startswith("_")]; out[pool][panel] = {}
        for a in arms:
            out[pool][panel][a] = {n: dict(vs_ref=iv([x[n][panel][a]["vs_ref"] for x in v], [x[n][panel][a]["vs_ref_bs"] for x in v]),
                                           ours_vs_it=iv([x[n][panel][a]["ours_vs_it"] for x in v], [x[n][panel][a]["ours_vs_it_bs"] for x in v]))
                                   for n in NS if all(n in x for x in v)}
        print(f"  {panel}: vs reference at n = " + " / ".join(NS) + "   (* ours significantly ahead at that n, ! behind)")
        for a in arms:
            r = out[pool][panel][a]
            cell = lambda n: f"{r[n]['vs_ref'][0]:+6.1f}" + ("" if a == "ours" else ("*" if r[n]["ours_vs_it"][1] > 0 else ("!" if r[n]["ours_vs_it"][2] < 0 else " ")))
            print(f"    {a:12s} " + "  ".join(cell(n) for n in NS if n in r))
json.dump(out, open(Path(__file__).parent / "learning_curve_summary_pinned.json", "w"), indent=1)

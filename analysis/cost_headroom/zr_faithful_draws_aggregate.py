"""Aggregate zr_faithful_draws.py (NEW_PATH 4.A.118) over the 10 training sets: per pool and arm, the mean saving vs median and of ours vs
the arm, 95% intervals of the test resamples pooled over sets, sets whose own interval excludes zero, mean success AUC.
Usage: python zr_faithful_draws_aggregate.py [LOGDIR] -> zr_faithful_draws_summary_pinned.json
"""
import glob, json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/zrfaithful")
POOLS = ["LCB", "Omni", "MMLU-Pro", "AIME", "APPS", "BCB", "CC", "SuperGPQA", "BBEH"]
lc = lambda p: p.lower().replace("-", ""); ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
out = {}
for pool in POOLS:
    S = {}
    for f in sorted(glob.glob(str(D / f"zr_faithful_draws_{lc(pool)}_*_pinned.json"))):
        S.update(json.load(open(f))["seeds"])
    if not S:
        continue
    ks = sorted(S, key=int); arms = [a for a in S[ks[0]] if not a.startswith("_")]; res = {"n_sets": len(ks)}
    for a in arms:
        r = {}
        for key in ("vs_median", "ours_vs_it"):
            pts = np.array([S[s][a][key] for s in ks]); bs = np.concatenate([S[s][a][key + "_bs"] for s in ks])
            r[key] = [float(np.nanmean(pts)), *ci(bs)]
            r[key + "_ahead"] = int(sum(ci(S[s][a][key + "_bs"])[0] > 0 for s in ks)); r[key + "_behind"] = int(sum(ci(S[s][a][key + "_bs"])[1] < 0 for s in ks))
        r["auc"] = float(np.mean([np.mean(S[s][a]["auc"]) for s in ks])); r["r2"] = float("nan")
        res[a] = r
    out[pool] = res
    print(f"\n===== {pool}: {len(ks)} training sets, test n={S[ks[0]]['_n']['test']}")
    print(f"  {'arm':<10} {'vs median':>24} {'ours vs it':>24}  ahead/behind   AUC    R2")
    for a in arms:
        vm, vd = res[a]["vs_median"], res[a]["ours_vs_it"]
        print(f"  {a:<10} {vm[0]:+6.1f} [{vm[1]:+6.1f}, {vm[2]:+6.1f}]  {vd[0]:+6.1f} [{vd[1]:+6.1f}, {vd[2]:+6.1f}]   "
              f"{res[a]['ours_vs_it_ahead']:>2}/{res[a]['ours_vs_it_behind']:<2}     {res[a]['auc']:.3f}  {res[a]['r2']:.2f}")
json.dump(out, open(Path(__file__).parent / "zr_faithful_draws_summary_pinned.json", "w"), indent=1)

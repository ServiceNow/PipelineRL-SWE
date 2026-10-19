"""Aggregate budget_draws.py (NEW_PATH 4.A.114) over the 10 training sets: per pool and cap, mean test accuracy and spend per arm, and
for each paired difference the mean of the point estimates, the 95% interval of the test resamples pooled over sets, and the number of
sets whose own interval excludes zero (ahead / behind).
Usage: python budget_draws_aggregate.py [LOGDIR] -> budget_draws_summary_pinned.json
"""
import json, sys
from pathlib import Path
import numpy as np

L = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/budgetdraws"); H = Path(__file__).parent
ARMS = ["ds_alone", "single_train", "single_test", "median", "constant", "derived", "direct_fit", "direct_joint", "oracle"]
SHOW = ["direct_fit-ds_alone", "direct_joint-ds_alone", "derived-ds_alone", "oracle-ds_alone", "direct_fit-median", "direct_fit-single_train",
        "direct_fit-derived", "direct_joint-derived"]
res = {}
for f in sorted(L.glob("budget_draws_*_pinned.json")):
    r = json.load(open(f)); x = res.setdefault(r["pool"], {"caps": r["caps"], "qs": r["qs"], "seeds": {}}); x["seeds"].update(r["seeds"])
out = {}
for pool, x in res.items():
    S = x["seeds"]; ks = sorted(S, key=int); out[pool] = {"n_sets": len(ks), "caps_usd": x["caps"], "caps": {}}
    print(f"\n{pool} ({len(ks)} training sets; test {S[ks[0]]['n_test']})")
    for q_, X in zip(x["qs"], x["caps"]):
        rows = [S[s]["caps"][str(q_)] for s in ks]; o = {}
        for a in ARMS:
            o[a] = dict(acc=float(np.mean([r[a]["acc"] for r in rows])), spend_per_1k=float(np.mean([r[a]["spend_per_1k"] for r in rows])),
                        off_ds=float(np.mean([r[a]["off_ds"] for r in rows])))
        o["auc_fit_ds"] = float(np.nanmean([r["auc_fit_ds"] for r in rows])); o["auc_derived_ds"] = float(np.nanmean([r["auc_derived_ds"] for r in rows]))
        o["single_train_route"] = sorted({r["single_train_route"] for r in rows}); o["diffs"] = {}
        for k in rows[0]["diffs"]:
            b = np.concatenate([r["diffs"][k]["bs"] for r in rows]); per = [np.percentile(r["diffs"][k]["bs"], [2.5, 97.5]) for r in rows]
            o["diffs"][k] = dict(mean=float(np.mean([r["diffs"][k]["point"] for r in rows])), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                                 ahead=int(sum(p[0] > 0 for p in per)), behind=int(sum(p[1] < 0 for p in per)))
        out[pool]["caps"][str(q_)] = o
        print(f"  cap p{q_} (${X * 1e3:.2f}/1k-equiv): " + " ".join(f"{a} {o[a]['acc']:.1f}" for a in ARMS)
              + f" | AUC fit(ds) direct {o['auc_fit_ds']:.3f} derived {o['auc_derived_ds']:.3f} | single {o['single_train_route']}")
        for k in SHOW:
            d_ = o["diffs"][k]
            print(f"      {k:<26} {d_['mean']:+5.1f} [{d_['lo']:+5.1f}, {d_['hi']:+5.1f}] ({d_['ahead']}/{d_['behind']})")
json.dump(out, open(H / "budget_draws_summary_pinned.json", "w"), indent=1)

"""Aggregate prefill_draws.py (NEW_PATH 4.A.112) and nonreason_estimators.py (4.A.113) over the 10 training sets: means of the point
estimates, 95% intervals of the test resamples pooled over sets, and the number of sets whose own interval excludes zero (ahead / behind).
Usage: python prefill_nonreason_aggregate.py [LOGDIR] -> prefill_draws_summary_pinned.json, nonreason_estimators_summary_pinned.json
"""
import json, sys
from pathlib import Path
import numpy as np

L = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs"); H = Path(__file__).parent


def agg(points, bss):
    b = np.concatenate([np.asarray(x, float) for x in bss]); lo, hi = np.nanpercentile(b, [2.5, 97.5])
    per = [np.nanpercentile(x, [2.5, 97.5]) for x in bss]
    return dict(mean=float(np.nanmean(points)), lo=float(lo), hi=float(hi), sd=float(np.nanstd(points)), n=len(points),
                ahead=int(sum(p[0] > 0 for p in per)), behind=int(sum(p[1] < 0 for p in per)))


pf = {}
for f in sorted((L / "prefilldraws").glob("prefill_draws_*_pinned.json")):
    r = json.load(open(f)); pf.setdefault(r["pool"], {}).setdefault(r["tag"], {}).update(r["seeds"])
out = {}
print("other prefills, 10 training sets: vs median (own success) | ours vs it [pooled 95%] (sets ahead / behind) | AUC | R2   [ours]")
for pool, tags in pf.items():
    out[pool] = {}
    for tag, S in tags.items():
        ks = sorted(S, key=int); g = lambda k_: [S[s][k_] for s in ks]
        o = dict(vs_median=agg(g("vs_median"), g("vs_median_bs")), ours_vs_it=agg(g("ours_vs_it"), g("ours_vs_it_bs")),
                 auc=float(np.mean(g("auc"))), r2=float(np.mean(g("r2"))), auc_ours=float(np.mean(g("auc_ours"))), r2_ours=float(np.mean(g("r2_ours"))),
                 ours_vs_median=float(np.mean(g("ours_vs_median"))), n_sets=len(ks))
        out[pool][tag] = o; v, d_ = o["vs_median"], o["ours_vs_it"]
        print(f"  {pool:<8} {tag:<10} {v['mean']:5.1f} [{v['lo']:5.1f}, {v['hi']:5.1f}] | {d_['mean']:+5.1f} [{d_['lo']:+5.1f}, {d_['hi']:+5.1f}] ({d_['ahead']}/{d_['behind']}) "
              f"| {o['auc']:.3f} | {o['r2']:.2f}   [ours {o['ours_vs_median']:.1f}, {o['auc_ours']:.3f}, {o['r2_ours']:.2f}]  n={len(ks)}")
json.dump(out, open(H / "prefill_draws_summary_pinned.json", "w"), indent=1)

nr = {}
for f in sorted((L / "nrest").glob("nonreason_estimators_*_pinned.json")):
    r = json.load(open(f)); nr.setdefault(r["pool"], {"slots": r["slots"], "seeds": {}})["seeds"].update(r["seeds"])
out = {}
print("\nnon-reasoning routes, 10 training sets: vs median [pooled 95%] | ours vs it (sets ahead / behind) | mean test R2 over routes")
for pool, o in nr.items():
    S = o["seeds"]; ks = sorted(S, key=int); arms = [a for a in S[ks[0]] if isinstance(S[ks[0]][a], dict)]; out[pool] = {"arms": {}, "slots": o["slots"]}
    for a in arms:
        g = lambda k_: [S[s][a][k_] for s in ks]
        x = dict(vs_median=agg(g("vs_median"), g("vs_median_bs")), ours_vs_it=agg(g("ours_vs_it"), g("ours_vs_it_bs")) if a != "ours" else None,
                 r2=np.mean([S[s][a]["r2"] for s in ks], 0).tolist())
        out[pool]["arms"][a] = x; v = x["vs_median"]; d_ = x["ours_vs_it"]
        print(f"  {pool:<8} {a:<12} {v['mean']:+5.1f} [{v['lo']:+5.1f}, {v['hi']:+5.1f}] (ahead of median {v['ahead']}/behind {v['behind']}) | "
              + ("" if d_ is None else f"{d_['mean']:+5.1f} [{d_['lo']:+5.1f}, {d_['hi']:+5.1f}] ({d_['ahead']}/{d_['behind']})") + f" | R2 {np.mean(x['r2']):+.2f}")
    for nm in ("other_nr_routes", "reasoning_routes"):
        out[pool][nm] = np.mean([S[s][nm] for s in ks], 0).tolist()
    pre = [a for a in arms if a != "oracle"]
    out[pool]["best_pre_r2"] = [float(max(np.mean([S[s][a]["r2"][k] for s in ks]) for a in pre)) for k in range(5)]
    print(f"  {pool:<8} per route ({', '.join(o['slots'])}): best estimator R2 {np.round(out[pool]['best_pre_r2'], 2).tolist()}  other nr routes "
          f"{np.round(out[pool]['other_nr_routes'], 2).tolist()}  reasoning routes {np.round(out[pool]['reasoning_routes'], 2).tolist()}  n={len(ks)}")
json.dump(out, open(H / "nonreason_estimators_summary_pinned.json", "w"), indent=1)

"""Aggregate tmlr_strengthen.py textemb (NEW_PATH 4.A.101): per pool and arm, mean over the 10 training sets of the saving vs median
and of ours vs it, with percentile intervals of the test resamples pooled over sets, and the number of sets in which ours vs it
excludes zero; mean success AUC of the text router vs ours.
Usage: python textemb_aggregate.py [DIR] -> textemb_summary_pinned.json
"""
import json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/textemb")
seeds = {}
for f in sorted(D.glob("tmlr_strengthen_textemb_*_pinned.json")):
    r = json.load(open(f)); seeds.setdefault(r["pool"], {}).update(r["seeds"])
out = {}
for pool in ("LCB", "Omni", "MMLU-Pro", "AIME", "APPS", "BCB", "CC", "SuperGPQA", "BBEH"):
    if pool not in seeds:
        continue
    S = seeds[pool]; ks = sorted(S, key=int); arms = [a for a in S[ks[0]] if not a.startswith("_")]; o = {"n_sets": len(ks)}
    for a in arms:
        vm = float(np.nanmean([S[k][a]["vs_median"] for k in ks])); pt = np.array([S[k][a]["ours_vs_it"] for k in ks], float)
        bs = np.concatenate([np.array(S[k][a]["ours_vs_it_bs"], float) for k in ks])
        sig = sum(np.nanpercentile(S[k][a]["ours_vs_it_bs"], 2.5) > 0 for k in ks)
        o[a] = dict(vs_median=vm, ours_vs_it=[float(np.nanmean(pt)), *map(float, np.nanpercentile(bs, [2.5, 97.5]))], sets_ours_ahead=int(sig))
    o["auc_ours"] = float(np.mean([np.mean(S[k]["_auc_ours"]) for k in ks]))
    o["auc_text"] = {z: float(np.mean([np.mean(S[k]["_auc_text"][z]) for k in ks])) for z in S[ks[0]]["_auc_text"]}
    out[pool] = o
    print(f"===== {pool} ({len(ks)} sets): success AUC ours {o['auc_ours']:.3f} text " + " ".join(f"{z} {v:.3f}" for z, v in o["auc_text"].items()))
    for a in arms:
        r = o[a]; print(f"  {a:<11} vs median {r['vs_median']:+6.1f}" + ("" if a == "ours" else
              f"   ours vs it {r['ours_vs_it'][0]:+6.1f} [{r['ours_vs_it'][1]:+.1f}, {r['ours_vs_it'][2]:+.1f}]  (ours ahead, interval > 0, in {r['sets_ours_ahead']}/{len(ks)} sets)"))
json.dump(out, open(Path(__file__).parent / "textemb_summary_pinned.json", "w"), indent=1)

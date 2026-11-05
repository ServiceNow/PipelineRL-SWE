"""ZeroRouter's success model with our cost probe: direct paired comparisons from the rerun launch_nov05_zr_hybrid.sh wrote to zrfaithful/hybrid.
Per dataset and ZeroRouter version (as published zr_pub, 4B scout zr_4b): hybrid vs the same ZeroRouter, ours vs the hybrid, ours vs ZeroRouter and
each arm vs median pricing. Mean over the ten training sets, 95% interval from the pooled paired resamples, as the paper's tables; the saving at the 12
matched accuracies, mean over sets, and the number of training sets whose own interval excludes zero on each side.
Usage: python zr_hybrid_aggregate.py -> zr_hybrid_summary_pinned.json
"""
import glob, json
from pathlib import Path
import numpy as np
D = "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/zrfaithful/hybrid/"; POOLS = "LCB Omni MMLU-Pro AIME APPS BCB CC SuperGPQA BBEH SWESmith".split()


def stat(pts, bss):
    """point estimate per set, resamples per set -> mean, pooled 95% interval, sets significantly ahead / behind"""
    lo, hi = np.percentile(np.concatenate(bss), [2.5, 97.5]); per = [np.percentile(b, [2.5, 97.5]) for b in bss]
    return dict(mean=float(np.mean(pts)), ci=[float(lo), float(hi)], sig=bool(lo > 0 or hi < 0), by_set=[float(x) for x in pts],
                sets_ahead=sum(float(a) > 0 for a, _ in per), sets_behind=sum(float(b) < 0 for _, b in per))


out = {}
for p in POOLS:
    S = {}
    for f in glob.glob(D + f"zr_faithful_draws_{p.lower().replace('-', '')}_*_pinned.json"):
        S.update(json.load(open(f))["seeds"])
    assert len(S) == 10, (p, len(S))
    ks = sorted(S, key=int); o = out[p] = {}
    o["ours_vs_median"] = stat([S[k]["ours"]["vs_median"] for k in ks], [S[k]["ours"]["vs_median_bs"] for k in ks])
    for w in ("zr_pub", "zr_4b"):
        h = w + "+cost"
        o[w] = dict(
            plus_vs_whole=stat([S[k]["_plus_vs_whole"][w]["saved"] for k in ks], [S[k]["_plus_vs_whole"][w]["saved_bs"] for k in ks]),
            ours_vs_plus=stat([S[k][h]["ours_vs_it"] for k in ks], [S[k][h]["ours_vs_it_bs"] for k in ks]),
            ours_vs_whole=stat([S[k][w]["ours_vs_it"] for k in ks], [S[k][w]["ours_vs_it_bs"] for k in ks]),
            plus_vs_median=stat([S[k][h]["vs_median"] for k in ks], [S[k][h]["vs_median_bs"] for k in ks]),
            whole_vs_median=stat([S[k][w]["vs_median"] for k in ks], [S[k][w]["vs_median_bs"] for k in ks]),
            plus_vs_whole_at=np.mean([S[k]["_plus_vs_whole"][w]["at"]["saved"] for k in ks], 0).tolist(),
            ours_vs_plus_at=np.mean([S[k][h]["ours_vs_it_at"]["saved"] for k in ks], 0).tolist(),
            ours_vs_whole_at=np.mean([S[k][w]["ours_vs_it_at"]["saved"] for k in ks], 0).tolist())
json.dump(out, open(Path(__file__).parent / "zr_hybrid_summary_pinned.json", "w"), indent=1)
f = lambda s: f"{s['mean']:6.1f} [{s['ci'][0]:6.1f},{s['ci'][1]:6.1f}]{'*' if s['sig'] else ' '} {s['sets_ahead']:2d}/{s['sets_behind']:2d}"
for w in ("zr_4b", "zr_pub"):
    print(f"\n{w}: hybrid vs ZR | ours vs hybrid | ours vs ZR | hybrid vs median | ZR vs median   (* = pooled interval excludes zero; n/m = sets ahead/behind)")
    for p in POOLS:
        o = out[p][w]; print(f"{p:10s}" + " | ".join(f(o[k]) for k in ("plus_vs_whole", "ours_vs_plus", "ours_vs_whole", "plus_vs_median", "whole_vs_median")))
print("\nours vs median: " + "  ".join(f"{p} {out[p]['ours_vs_median']['mean']:.1f}" for p in POOLS))
NINE = POOLS[:-1]
for w in ("zr_4b", "zr_pub"):
    print(f"means over nine datasets vs median, {w}: ours {np.mean([out[p]['ours_vs_median']['mean'] for p in NINE]):.1f}, hybrid {np.mean([out[p][w]['plus_vs_median']['mean'] for p in NINE]):.1f}, ZR {np.mean([out[p][w]['whole_vs_median']['mean'] for p in NINE]):.1f}")
print("\nhighest-accuracy point (index 11): hybrid vs ZR / ours vs hybrid / ours vs ZR")
for w in ("zr_4b", "zr_pub"):
    print(w, "  ".join(f"{p} {out[p][w]['plus_vs_whole_at'][11]:.0f}/{out[p][w]['ours_vs_plus_at'][11]:.0f}/{out[p][w]['ours_vs_whole_at'][11]:.0f}" for p in POOLS))

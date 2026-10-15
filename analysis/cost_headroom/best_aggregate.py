"""Aggregate best_estimators.py (NEW_PATH 4.A.108): per pool and route, mean over 10 training sets of the test log-length R2 of ours,
of the best pre-generation estimator (per set), which estimator is best most often, the two leaky references (true difficulty; other
routes' realized lengths) and the noise ceiling (legibility_summary_pinned.json): lower bound <= legibility <= ceiling.
Usage: python best_aggregate.py [DIR] -> best_estimators_summary_pinned.json
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/bestest"); H = Path(__file__).parent
leg = json.load(open(H / "legibility_summary_pinned.json"))["routes"]; seeds, slots = {}, {}
for f in sorted(D.glob("best_estimators_*_pinned.json")):
    r = json.load(open(f)); seeds.setdefault(r["pool"], {}).update(r["seeds"]); slots[r["pool"]] = r["slots"]
out = {}
print("per route: ours | best pre-generation (most often) | ceiling | true difficulty (leaky) | other routes' lengths (leaky)")
for pool in ("LCB", "Omni", "MMLU-Pro", "AIME", "APPS", "BCB", "CC", "SuperGPQA", "BBEH"):
    if pool not in seeds:
        continue
    S = seeds[pool]; ks = sorted(S, key=int); o = {}
    for k, sl in enumerate(slots[pool]):
        g = lambda a: float(np.mean([S[s]["r2"][a][k] for s in ks]))
        o[sl] = dict(ours=g("ours"), best=float(np.mean([S[s]["best_pre_r2"][k] for s in ks])), best_arm=Counter(S[s]["best_pre"][k] for s in ks).most_common(1)[0][0],
                     difficulty=g("true_difficulty"), other_routes=g("other_routes_realized"), ceiling=leg.get(pool, {}).get(sl, {}).get("reliability"),
                     all={a: g(a) for a in S[ks[0]]["r2"]})
    out[pool] = o
    print(f"{pool:<10} " + "  ".join(f"{sl} {x['ours']:.2f}|{x['best']:.2f}({x['best_arm']})|" + ("n/a" if x["ceiling"] is None else f"{x['ceiling']:.2f}")
                                       + f"|{x['difficulty']:.2f}|{x['other_routes']:.2f}" for sl, x in o.items()))
json.dump(out, open(H / "best_estimators_summary_pinned.json", "w"), indent=1)

"""Per-route output-length spread on every pool (NEW_PATH 4.A.99): is deepseek-v4-flash's length unusually variable, or unusually
hard to predict from difficulty? Manifest split, readouts refitted as in the training-set analyses (DRAW=-1). Per pool and route:
mean billed cost per call, p90/p10 of per-problem mean output length and s.d. of its log (train + calibration + test), and on the
test set the R2 of log length for the median rule (0 by construction up to level), pricing from success (difficulty) and our readout;
'beyond difficulty' = R2(ours) - R2(from success). Also the s.d. of log cost per call within the route vs the s.d. of the routes' log
mean cost (the price-gap scale of sec:when).
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned OUT_DIR=... python route_spread.py [POOLS]
"""
import json, os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault("DRAW", "-1")
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent)); res = {}
r2 = lambda y, yh: float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
for P_ in (sys.argv[1] if len(sys.argv) > 1 else "LCB,Omni,MMLU-Pro,AIME,APPS,BCB,CC,SuperGPQA,BBEH").split(","):
    d = load(P_); N = len(d["ids"]); tr, ca, ev = d["tr"], d["ca"], d["ev"]; S, T = view(d, tr), view(d, np.arange(N))
    tk = {"ours": fit_predict("ours", S, T), "fromsuccess": fit_predict("fromsuccess", S, T)}
    allp = np.concatenate([tr, ca, ev]); Y = np.log(d["L"]); lc = np.log(d["paid"])
    gap = float(np.std(np.log(d["paid"][allp].mean(0)))); out = {"between_route_sd_log_mean_cost": gap, "routes": {}}
    for k, sl in enumerate(d["slots"]):
        L = d["L"][allp, k]; y = Y[ev, k]
        o = dict(usd_per_1k=float(d["paid"][ev, k].mean() * 1e3), acc=float(d["q"][ev, k].mean()), p90_p10=float(np.percentile(L, 90) / np.percentile(L, 10)),
                 sd_log_len=float(Y[allp, k].std()), sd_log_cost=float(lc[allp, k].std()),
                 r2_fromsuccess=r2(y, np.log(tk["fromsuccess"][ev, k])), r2_ours=r2(y, np.log(tk["ours"][ev, k])))
        o["beyond_difficulty"] = o["r2_ours"] - o["r2_fromsuccess"]; out["routes"][sl] = o
    res[P_] = out
    print(f"\n===== {P_}: s.d. of log mean cost between routes {gap:.2f}", flush=True)
    for sl, o in out["routes"].items():
        print(f"  {sl:<9} acc {o['acc'] * 100:5.1f}  ${o['usd_per_1k']:5.2f}/1k  p90/p10 {o['p90_p10']:6.1f}x  sd log len {o['sd_log_len']:.2f}  sd log cost {o['sd_log_cost']:.2f}"
              f"  R2 test: from success {o['r2_fromsuccess']:+.2f}  ours {o['r2_ours']:+.2f}  beyond difficulty {o['beyond_difficulty']:+.2f}", flush=True)
    json.dump(res, open(OUT / f"route_spread{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1)
print("DONE", flush=True)

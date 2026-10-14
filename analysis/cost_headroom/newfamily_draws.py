"""The eight-route pools of newfamily_eval.py (NEW_PATH 4.A.85) on the 10 training sets (NEW_PATH 4.A.96): is a single model dominant
when deepseek-v4-flash is not in the pool, and does the estimator ranking hold? For each training set (DRAW=s of
decompose.split_manifest; -1 = the manifest split, refitted the same way) every readout of build() is refitted, then per sub-pool
  newpool   gpt-oss-20b low/medium + Qwen3-32B, GLM-4.7-Flash, Nemotron-3-Super-120B (no deepseek-v4-flash, no gpt-oss-120b)
  no_dsv4f  all eight routes except deepseek-v4-flash (four gpt-oss routes + the three new families)
  all8      all eight routes (reference)
  arms      ours / median / mean / fromsuccess / bins / oracle as newfamily_eval.routing(): saving vs median and ours vs it
  singles   per route m: 1 - C_router(acc_m) / cost_m (the frontier's cheapest point if acc_m is below it; NaN if never reached) and the
            router's most accurate point minus acc_m, for ours and median pricing (as tmlr_strengthen.py singles)
  mix       saving vs the convex hull of the single routes (any point on it is a random mix of single models, no router needed),
            and the router's max accuracy minus the best single route's
Point = the test set, then NBOOT test resamples (rng 0, the same in every set); pooled over sets by newfamily_draws_aggregate.py.
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned OUT_DIR=... python newfamily_draws.py LCB|Omni|MMLU-Pro S0 S1
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / "newfamily_eval.py").read().split("res = {}; built = {}")[0])
SUB = {"newpool": [0, 1, 5, 6, 7], "no_dsv4f": [0, 1, 3, 4, 5, 6, 7], "all8": list(range(8))}
NB = int(os.environ.get("NBOOT", "300")); P_, s0, s1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
res = {"pool": P_, "seeds": {}}; TAG = os.environ.get("RESULT_TAG", "")
for s in range(s0, s1):
    os.environ["DRAW"] = str(s); S = build(P_); tr, ev, N = S["tr"], S["ev"], S["N"]; Y = np.log(S["L"]); lg = np.log(S["P"] / (1 - S["P"]))
    rb = np.random.default_rng(0); BS = [ev] + [ev[rb.integers(0, len(ev), len(ev))] for _ in range(NB)]; r_s = {}
    for pname, cols in SUB.items():
        tk = {"ours": S["tok"], "median": np.repeat(np.median(S["L"][tr], 0)[None], N, 0), "mean": np.repeat(S["L"][tr].mean(0)[None], N, 0)}
        FS = np.c_[lg[:, cols], lg[:, cols] ** 2]; fs = np.zeros((N, 8)); bins_ = np.zeros((N, 8))
        dbar = lg[:, cols].mean(1); e = np.quantile(dbar[tr], np.linspace(0, 1, 11)[1:-1]); b = np.searchsorted(e, dbar)
        for k in range(8):
            mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-3, 1e4, 15))).fit(FS[tr], Y[tr, k]); yh = mdl.predict(FS)
            fs[:, k] = np.exp(yh) * np.mean(np.exp(Y[tr, k] - yh[tr])); fs[:, k] *= S["L"][tr, k].mean() / fs[tr, k].mean()
            bins_[:, k] = np.array([S["L"][tr][b[tr] == j, k].mean() if (b[tr] == j).any() else S["L"][tr, k].mean() for j in range(10)])[b]
        tk["fromsuccess"], tk["bins"] = fs, bins_
        C = {a: costs8(S, x) for a, x in tk.items()} | {"oracle": S["paid"]}
        r_ = {"arms": {a: dict(vs_median=[], ours_vs_it=[]) for a in C}, "singles": {a: {S["slots"][k]: dict(saving=[], acc_gap=[]) for k in cols} for a in ("ours", "median")},
              "mix": {a: dict(saving=[], acc_gap=[]) for a in ("ours", "median")}}
        for ii in BS:
            H = {a: curve8(S, S["P"], C[a], ii, cols) for a in C}
            for a in C:
                r_["arms"][a]["vs_median"].append(saved(H[a], H["median"]) * 100); r_["arms"][a]["ours_vs_it"].append(saved(H["ours"], H[a]) * 100)
            pts = [(S["paid"][ii, k].mean(), S["q"][ii, k].mean()) for k in cols]; Hm = hull(pts)
            for a in ("ours", "median"):
                for k in cols:
                    cm, am = pts[cols.index(k)]; cr = H[a][0][0] if am < H[a][0][1] else cost_at(H[a], am)
                    r_["singles"][a][S["slots"][k]]["saving"].append(float(1 - cr / cm) * 100 if np.isfinite(cr) else np.nan)
                    r_["singles"][a][S["slots"][k]]["acc_gap"].append(float(H[a][-1][1] - am) * 100)
                r_["mix"][a]["saving"].append(saved(H[a], Hm) * 100); r_["mix"][a]["acc_gap"].append(float(H[a][-1][1] - Hm[-1][1]) * 100)
        r_["routes"] = {S["slots"][k]: dict(acc=float(S["q"][ev, k].mean()), usd_per_1k=float(S["paid"][ev, k].mean() * 1e3)) for k in cols}
        r_s[pname] = r_; o = r_["singles"]["ours"]
        print(f"  set {s:2d} {pname:<8}: vs median " + " ".join(f"{a} {r_['arms'][a]['vs_median'][0]:+5.1f}" for a in C)
              + f" | vs mix of singles ours {r_['mix']['ours']['saving'][0]:+5.1f} (max acc gap {r_['mix']['ours']['acc_gap'][0]:+.1f}), median {r_['mix']['median']['saving'][0]:+5.1f}"
              + " | singles " + " ".join(f"{sl} {r_['routes'][sl]['acc']:.3f}@${r_['routes'][sl]['usd_per_1k']:.2f}: {o[sl]['saving'][0]:+.0f}%" for sl in o), flush=True)
    res["seeds"][s] = r_s
    json.dump(res, open(OUT / f"newfamily_draws_{P_.lower().replace('-', '')}_{s0}_{s1}{TAG}.json", "w"), indent=1, default=float)
print("DONE", flush=True)

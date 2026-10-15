"""Legibility drill-down (NEW_PATH 4.A.106): what limits how much of per-query cost our readout reads, and can it be predicted early?
Per pool, on the 10 training sets (DRAW=s, readouts refitted as in tmlr_strengthen.py draws):
  ceiling     per route, the share of log-output-length variance that lies between problems (one-way ICC on problems with >= 2
              valid draws, log of each draw's length), and the reliability of the TEST target (log mean length over that problem's
              n_i test draws): rel = s_b2 / (s_b2 + mean_i(s_w2 / n_i)). No predictor of a problem's expected length can exceed rel
              in R2 (up to the log-of-mean vs mean-of-log approximation). efficiency = test R2 of our readout / rel.
  pairs       differential legibility: for every pair of routes (a, b), test R2 of our predicted log(L_a / L_b) for the realized one,
              its reliability ceiling, and the same for pricing from success; the per-route R2s for comparison.
  pilot       can legibility be estimated from a small pilot? For k in KS labelled training problems (REPS draws of k): 5-fold CV
              R2 of the readout within the k problems (the pilot estimate), against the full-data test R2.
Usage: REASON_ROOT=.../reason_pinned OUT_DIR=... python legibility.py POOL [S0 S1]
"""
import json, os, sys, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
POOL_ = sys.argv[1]; S0, S1 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 10)
sys.argv = [sys.argv[0], "legibility", POOL_]
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
from sklearn.model_selection import KFold
OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent)); KS, REPS = (20, 50, 100), int(os.environ.get("REPS", "5"))
r2 = lambda y, yh: float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())

# ---- noise ceiling (split-free): raw draws
name, feat, ds = POOLS[POOL_]; F = R / "expanded_eval_20261001" / ds if ds else R / name
t = np.load(F / "tensors.npz", allow_pickle=True); v = t["valid"].astype(bool); CTd = np.where(v, t["completion_tokens"], np.nan).astype(float)
LD = np.log(np.maximum(CTd, 1)); nD = v.sum(2); M = LD.shape[1]
ceil = {}
for k in range(M):
    m = nD[:, k] >= 2; x = LD[m, k]; s_w2 = float(np.nanmean(np.nanvar(x, 1, ddof=1)))
    s_b2 = max(float(np.nanvar(np.nanmean(x, 1))) - s_w2 * float(np.mean(1 / nD[m, k])), 0)
    ceil[k] = dict(icc=s_b2 / (s_b2 + s_w2), s_b2=s_b2, s_w2=s_w2, n_problems=int(m.sum()))
res = {"pool": POOL_, "ceiling": {}, "seeds": {}}
for s in range(S0, S1):
    os.environ["DRAW"] = str(s); d = load(POOL_); N = len(d["ids"]); tr, ca, ev = d["tr"], d["ca"], d["ev"]; slots = d["slots"]
    assert d["ids"] == list(map(str, t["problem_ids"]))
    T = view(d, np.arange(N)); tok = fit_predict("ours", view(d, tr), T); fs = fit_predict("fromsuccess", view(d, tr), T)
    Y, Yh, Yf = np.log(d["L"]), np.log(np.maximum(tok, 1)), np.log(np.maximum(fs, 1))
    nte = np.maximum(nD[ev], 1)
    rel = {k: ceil[k]["s_b2"] / (ceil[k]["s_b2"] + float(np.mean(ceil[k]["s_w2"] / nte[:, k]))) for k in range(M)}
    r_ = {"routes": {}, "pairs": {}, "pilot": {}}
    for k in range(M):
        rk = r2(Y[ev, k], Yh[ev, k]); r_["routes"][slots[k]] = dict(r2_ours=rk, r2_fromsuccess=r2(Y[ev, k], Yf[ev, k]), reliability=rel[k], efficiency=rk / rel[k] if rel[k] > 0 else None)
    for a, b in itertools.combinations(range(M), 2):
        dy, dh, df = Y[ev, a] - Y[ev, b], Yh[ev, a] - Yh[ev, b], Yf[ev, a] - Yf[ev, b]
        # reliability of the difference: between-problem variance of the difference over its total, noise of the two routes independent
        s_bd = max(float(np.var(dy)) - float(np.mean(ceil[a]["s_w2"] / nte[:, a] + ceil[b]["s_w2"] / nte[:, b])), 0)
        reld = s_bd / max(float(np.var(dy)), 1e-9)
        r_["pairs"][f"{slots[a]}+{slots[b]}"] = dict(r2_diff_ours=r2(dy, dh), r2_diff_fromsuccess=r2(dy, df), reliability_diff=reld, sd_diff=float(dy.std()),
                                                      corr_routes=float(np.corrcoef(Y[ev, a], Y[ev, b])[0, 1]), r2_mean_routes=(r_["routes"][slots[a]]["r2_ours"] + r_["routes"][slots[b]]["r2_ours"]) / 2)
    rng = np.random.default_rng(100 + s)
    for kk in KS:
        if kk > len(tr):
            continue
        est = []
        for _ in range(REPS):
            sub = rng.choice(tr, kk, replace=False); cv = np.zeros((kk, M))
            for a_, b_ in KFold(5, shuffle=True, random_state=0).split(sub):
                cv[b_] = np.log(np.maximum(fit_predict("ours", view(d, sub[a_]), view(d, sub[b_])), 1))
            est.append([r2(Y[sub, k], cv[:, k]) for k in range(M)])
        r_["pilot"][str(kk)] = dict(cv_r2=np.mean(est, 0).tolist(), cv_r2_sd=np.std(est, 0).tolist())
    res["seeds"][s] = r_
    print(f"  set {s}: R2 ours / reliability / efficiency " + "  ".join(f"{sl} {r_['routes'][sl]['r2_ours']:.2f}/{r_['routes'][sl]['reliability']:.2f}/{r_['routes'][sl]['efficiency']:.2f}" for sl in slots)
          + " | pilot CV R2 (mean over routes) " + " ".join(f"k={kk} {np.mean(r_['pilot'][str(kk)]['cv_r2']):.2f}" for kk in KS if str(kk) in r_["pilot"]), flush=True)
    res["ceiling"] = {slots[k]: ceil[k] for k in range(M)}
    json.dump(res, open(OUT / f"legibility_{POOL_.lower().replace('-', '')}_{S0}_{S1}{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
print("DONE", flush=True)

"""Does per-route Platt scaling help or hurt our own success probes? (NEW_PATH 4.A.128). Without per-route Platt the ZeroRouter-style
probe saves more (4.A.127) and the shared probes save less (4.A.126). The paper's probes (success_linear: per route, L2 logistic, C by
calibration AUC, binomial over all draws) are refitted per training set (DRAW=s) and routed with the SAME cost probe (ours) in three
calibrations:
  ours        Platt per route on the calibration rows (the paper; equals d["P"])
  ours-raw    the probes' own probabilities, no recalibration
  ours-pool   one Platt map fitted on every route's calibration rows together (keeps the order of the routes' logits within a query)
Reports as success_pool.py.
Usage: (GPU) REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned NBOOT=200 OUT_DIR=... python success_platt.py POOL S0 S1
"""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).parent
_sp = open(HERE / "success_pool.py").read()
exec(_sp[:_sp.index('res = {"pool": POOL_, "seeds": {}}')])            # learning_curve's shared code, platt, pair_acc, bll


def probes(d, tr, ca):
    """success_linear's probes, returning the calibration and full-set logits before Platt"""
    S, Ca = view(d, tr), view(d, ca); sc = StandardScaler().fit(S["X"]); Xs, Xc, Xa = sc.transform(S["X"]), sc.transform(Ca["X"]), sc.transform(d["X"])
    scale = max(1, S["X"].shape[1] // 2560); M = d["M"]; LO = np.zeros((len(Xa), M))
    for k in range(M):
        yb = S["y0"][:, k]; best = (1e-3, -1.0)
        for cand in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
            if 0 < yb.mean() < 1 and 0 < Ca["y0"][:, k].mean() < 1:
                v_ = auc(Ca["y0"][:, k], LogisticRegression(max_iter=2000, C=cand / scale).fit(Xs, yb).decision_function(Xc))
                if v_ > best[1]:
                    best = (cand, v_)
        s_, n_ = S["succ"][:, k], S["n"][:, k]; w = np.r_[s_, n_ - s_]; kw = w > 0
        clf = LogisticRegression(max_iter=2000, C=best[0] / scale).fit(np.vstack([Xs, Xs])[kw], np.r_[np.ones(len(Xs)), np.zeros(len(Xs))][kw], sample_weight=w[kw])
        LO[:, k] = clf.decision_function(Xa)
    return LO


res = {"pool": POOL_, "seeds": {}}; outf = OUT / f"success_platt_{POOL_.lower().replace('-', '')}_{S0}_{S1}{TAG}.json"
for s in range(S0, S1):
    os.environ["DRAW"] = str(s); d = load(POOL_); N = len(d["ids"]); tr, ca, ev = d["tr"], d["ca"], d["ev"]; M = d["M"]
    S, T = view(d, tr), view(d, np.arange(N)); tk = fit_predict("ours", S, T); C = cost_of(d, tk)
    LO = probes(d, tr, ca)
    Pp = np.clip(np.stack([platt(LO[ca, k], d["y0"][ca, k], LO[:, k]) for k in range(M)], 1), 1e-4, 1 - 1e-4)
    pl = LogisticRegression(max_iter=2000, C=1e6).fit(LO[ca].reshape(-1, 1), d["y0"][ca].ravel())
    Ps = {"ours": d["P"], "ours-raw": np.clip(sig(LO), 1e-4, 1 - 1e-4), "ours-pool": np.clip(pl.predict_proba(LO.reshape(-1, 1))[:, 1].reshape(N, M), 1e-4, 1 - 1e-4)}
    check = float(np.abs(Pp - d["P"]).max())                                # the refit reproduces the paper's probes
    Cm = cost_of(d, np.repeat(d["med"][None], N, 0)); rb = np.random.default_rng(0); BS = [ev] + [ev[rb.integers(0, len(ev), len(ev))] for _ in range(NBOOT)]
    r_ = {a: dict(vs_median=[], vs_mix=[], ours_vs_it=[]) for a in Ps}
    for ii in BS:
        H0 = curve(d, d["P"], Cm, ii); Hm = hull([(d["paid"][ii, k].mean(), d["q"][ii, k].mean()) for k in range(M)])
        H = {a: curve(d, P, C, ii) for a, P in Ps.items()}
        for a in Ps:
            r_[a]["vs_median"].append(saved(H[a], H0) * 100); r_[a]["vs_mix"].append(saved(H[a], Hm) * 100); r_[a]["ours_vs_it"].append(saved(H["ours"], H[a]) * 100)
    out = {"_check_max_abs_diff_vs_paper_P": check}
    for a, v in r_.items():
        out[a] = {k: float(x[0]) for k, x in v.items()} | {k + "_bs": [float(y) for y in x[1:]] for k, x in v.items()}
        out[a]["auc"] = float(np.mean([auc(d["y0"][ev, m], Ps[a][ev, m]) for m in range(M)])); out[a]["pair_acc"] = pair_acc(d["q"][ev], Ps[a][ev])
        out[a]["logloss_test"] = bll(d["succ"][ev].ravel(), d["n"][ev].ravel(), Ps[a][ev].ravel())
    res["seeds"][s] = out
    print(f"  set {s}: check {check:.1e}  " + "  ".join(f"{a}: vs median {o['vs_median']:+5.1f} AUC {o['auc']:.3f} pair {o['pair_acc']:.3f} ll {o['logloss_test']:.3f}"
                                                   for a, o in out.items() if not a.startswith("_")), flush=True)
    json.dump(res, open(outf, "w"), indent=1, default=float)
print("DONE", flush=True)

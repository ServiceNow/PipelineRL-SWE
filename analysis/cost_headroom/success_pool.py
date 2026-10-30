"""Sharing information across our success probes (NEW_PATH 4.A.126). ZeroRouter's coupled item-response success routes better than our
five independent probes on MMLU-Pro (and on the point estimate on BigCodeBench, SWE-Smith, APPS) although its per-route AUC is lower
everywhere: it orders the routes within a query better (pairwise route-ordering accuracy higher on all ten datasets), while ours ranks the
queries within a route better. Our probes are fitted per route and their C is chosen by that route's calibration AUC, a criterion blind to
the order of the routes. One fixed rule for every dataset, routed with the SAME cost probe (ours) in every arm, per training set (DRAW=s):
  ours        the paper: per route, L2 logistic, C by calibration AUC, Platt per route
  ours-ll     the same, C chosen by calibration log loss (isolates the selection criterion)
  shared      multi-task shrinkage: route m's weights = w0 + lam * v_m and a free intercept per route, one logistic on (query, route) rows,
              ||w0||^2 + sum ||v_m||^2 penalized; (C, lam) chosen by calibration log loss pooled over routes; Platt per route
  shared-raw  the same without Platt (Platt is per route and may undo the shared ordering)
  zr4b+cost   ZeroRouter's success read from our 4B prefill (success_combo.py), the reference
lam = 0 is one shared slope (routes differ by intercept only); large lam approaches independent probes. The high-dimensional fit is exact
on the training rows' span: an L2 linear model's weights lie in the span of its standardized training rows, so the features are projected
onto it (X V, V from the SVD of the training rows) with no loss. Saving vs median pricing with OUR success (the tab:allest reference) and
vs a random mix of single routes, point + NBOOT test resamples (rng 0, the same in every set); per-route AUC and pairwise route-ordering
accuracy on the test set (pairs of routes whose mean success differs on a query).
Usage: (GPU) REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned NBOOT=200 OUT_DIR=... python success_pool.py POOL S0 S1
"""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).parent
_lc = open(HERE / "learning_curve.py").read()
exec(_lc[:_lc.index("texts = texts_of(POOL_); Q = encode(texts)")])      # zr_4b and the shared code, unchanged
from sklearn.linear_model import LogisticRegression
CS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1); LAMS = (0.0, 0.25, 0.5, 1.0, 2.0); KAPPA = 10.0


def bll(succ, n, p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(succ * np.log(p) + (n - succ) * np.log(1 - p)).sum() / max(n.sum(), 1))


def platt(lo_c, y_c, lo_t):
    if not 0 < y_c.mean() < 1:
        return sig(lo_t)
    return LogisticRegression(max_iter=2000, C=1e6).fit(lo_c[:, None], y_c).predict_proba(lo_t[:, None])[:, 1]


def project(d, tr):
    sc = StandardScaler().fit(d["X"][tr]); Xs = sc.transform(d["X"][tr])
    _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
    return sc.transform(d["X"]) @ Vt.T, max(1, d["X"].shape[1] // 2560)


def binom(Z, succ, n):
    w = np.r_[succ, n - succ]; kw = w > 0
    return np.vstack([Z, Z])[kw], np.r_[np.ones(len(Z)), np.zeros(len(Z))][kw], w[kw]


def independent(d, Z, scale, tr, ca, crit):
    M = d["M"]; out = np.zeros((len(Z), M))
    for k in range(M):
        best = (CS[3], np.inf)
        for c in CS:
            if crit == "ll":
                Xb, yb, wb = binom(Z[tr], d["succ"][tr, k], d["n"][tr, k]); clf = LogisticRegression(max_iter=2000, C=c / scale).fit(Xb, yb, sample_weight=wb)
            else:
                clf = LogisticRegression(max_iter=2000, C=c / scale).fit(Z[tr], d["y0"][tr, k])
            lo = clf.decision_function(Z[ca])
            v = bll(d["succ"][ca, k], d["n"][ca, k], sig(lo)) if crit == "ll" else -auc(d["y0"][ca, k], lo)
            if v < best[1]:
                best = (c, v)
        clf = LogisticRegression(max_iter=2000, C=best[0] / scale)
        Xb, yb, wb = binom(Z[tr], d["succ"][tr, k], d["n"][tr, k]); clf.fit(Xb, yb, sample_weight=wb)
        lo = clf.decision_function(Z); out[:, k] = platt(lo[ca], d["y0"][ca, k], lo)
    return np.clip(out, 1e-4, 1 - 1e-4)


def shared_design(Z, M, lam):
    N, r = Z.shape; E = np.eye(M)
    blocks = [np.repeat(Z, M, 0)]                                          # (query, route) rows, query-major
    if lam > 0:
        blocks.append(lam * (np.repeat(Z, M, 0)[:, None, :] * np.tile(E, (N, 1))[:, :, None]).reshape(N * M, M * r))
    blocks.append(KAPPA * np.tile(E, (N, 1)))
    return np.hstack(blocks)


def shared(d, Z, scale, tr, ca):
    M = d["M"]; best = None
    st, nt = d["succ"][tr].ravel(), d["n"][tr].ravel()
    for lam in LAMS:
        Dt = shared_design(Z[tr], M, lam); Dc = shared_design(Z[ca], M, lam); Xb, yb, wb = binom(Dt, st, nt)
        for c in CS:
            clf = LogisticRegression(max_iter=3000, C=c / scale).fit(Xb, yb, sample_weight=wb)
            v = bll(d["succ"][ca].ravel(), d["n"][ca].ravel(), sig(clf.decision_function(Dc)))
            if best is None or v < best[0]:
                best = (v, lam, c, clf)
    _, lam, c, clf = best; N = len(Z)
    lo = np.concatenate([clf.decision_function(shared_design(Z[i:i + 2000], M, lam)) for i in range(0, N, 2000)]).reshape(N, M)
    raw = sig(lo); cal = np.stack([platt(lo[ca, k], d["y0"][ca, k], lo[:, k]) for k in range(M)], 1)
    return np.clip(cal, 1e-4, 1 - 1e-4), np.clip(raw, 1e-4, 1 - 1e-4), dict(lam=lam, C=c)


def pair_acc(q, P):
    ok = tot = 0
    for i in range(q.shape[1]):
        for j in range(i + 1, q.shape[1]):
            dq = q[:, i] - q[:, j]; mk = dq != 0; ok += int((np.sign(P[mk, i] - P[mk, j]) == np.sign(dq[mk])).sum()); tot += int(mk.sum())
    return ok / max(tot, 1)


res = {"pool": POOL_, "seeds": {}}; outf = OUT / f"success_pool_{POOL_.lower().replace('-', '')}_{S0}_{S1}{TAG}.json"
for s in range(S0, S1):
    os.environ["DRAW"] = str(s); d = load(POOL_); N = len(d["ids"]); tr, ca, ev = d["tr"], d["ca"], d["ev"]
    S, T = view(d, tr), view(d, np.arange(N)); tk = fit_predict("ours", S, T); C = cost_of(d, tk)
    Z, scale = project(d, tr)
    Ps = {"ours": d["P"], "ours-ll": independent(d, Z, scale, tr, ca, "ll")}
    Ps["shared"], Ps["shared-raw"], cfg = shared(d, Z, scale, tr, ca)
    Ps["zr4b+cost"], _, _ = zr_4b(d, S)
    Cm = cost_of(d, np.repeat(d["med"][None], N, 0)); rb = np.random.default_rng(0); BS = [ev] + [ev[rb.integers(0, len(ev), len(ev))] for _ in range(NBOOT)]
    r_ = {a: dict(vs_median=[], vs_mix=[], ours_vs_it=[]) for a in Ps}
    for ii in BS:
        H0 = curve(d, d["P"], Cm, ii); Hm = hull([(d["paid"][ii, k].mean(), d["q"][ii, k].mean()) for k in range(d["M"])])
        H = {a: curve(d, P, C, ii) for a, P in Ps.items()}
        for a in Ps:
            r_[a]["vs_median"].append(saved(H[a], H0) * 100); r_[a]["vs_mix"].append(saved(H[a], Hm) * 100); r_[a]["ours_vs_it"].append(saved(H["ours"], H[a]) * 100)
    out = {"_cfg": cfg}
    for a, v in r_.items():
        out[a] = {k: float(x[0]) for k, x in v.items()} | {k + "_bs": [float(y) for y in x[1:]] for k, x in v.items()}
        out[a]["auc"] = float(np.mean([auc(d["y0"][ev, m], Ps[a][ev, m]) for m in range(d["M"])])); out[a]["pair_acc"] = pair_acc(d["q"][ev], Ps[a][ev])
    res["seeds"][s] = out
    print(f"  set {s}: {cfg}  " + "  ".join(f"{a}: vs median {o['vs_median']:+5.1f} vs mix {o['vs_mix']:+5.1f} AUC {o['auc']:.3f} pair {o['pair_acc']:.3f}"
                                         for a, o in out.items() if a != "_cfg"), flush=True)
    json.dump(res, open(outf, "w"), indent=1, default=float)
print("DONE", flush=True)

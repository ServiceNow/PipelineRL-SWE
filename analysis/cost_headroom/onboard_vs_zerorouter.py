"""Low-shot onboarding head-to-head: ours vs ZeroRouter-style (arXiv 2601.06220), at matched k.
Hold out route h (the "new model"); every OTHER route keeps its full heads (identical in all arms), so only the new model's
success + cost estimates differ. k labelled training problems for h, 20 draws of them.
  ours        cost = shared level (mean of the other routes' probe log-length) + 1 offset; success = logistic in the shared
              difficulty (mean logit of the other routes' success predictions), 2 parameters
  zr          ZeroRouter-style, sized to a 4-model pool: stage 1 fits a 1-D 2PL IRT, P(u solves i) = sigmoid(a_i (theta_u - b_i)),
              on the OTHER routes' TRAIN outcomes (binomial over draws, Gaussian priors, L-BFGS); stage 2 regresses (log a_i, b_i)
              on the same prefill features (the other routes' success logits + squares) to place every query. New model:
              fit theta_h on the k anchors (anchors' stage-1 a_i, b_i; N(0,1) prior); cost = per-bin mean output of the anchors,
              bins = K quantiles of s = a*b (Eq. 8-10), empty bin -> anchors' overall mean
  zr-dopt     zr with its D-optimal anchors (top-k Fisher information a_i^2 p_i (1 - p_i) at theta = 0) instead of random
  naive       median output of the k + constant success rate
Metric: gain vs the FULL pool's paper rule at matched accuracy (as onboard_full.py) and max reachable test accuracy.
Usage: python onboard_vs_zerorouter.py <pool> <cost file> [routes to hold out, comma]
"""
import json, sys, numpy as np
from pathlib import Path
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R, hull, cost_at

name, CF = sys.argv[1], sys.argv[2]
D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; M = len(S); pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
HOLD = sys.argv[3].split(",") if len(sys.argv) > 3 else S
v = t["valid"].astype(bool); okd = (t["final_outcome"] & t["valid"]).astype(bool); ct = t["completion_tokens"].astype(float)
pt = t["prompt_tokens"].astype(float); n = v.sum(2); avail = n > 0; succ = (okd & v).sum(2)
inp = np.nan_to_num(np.nanmean(np.where(v, pt, np.nan), 2))
pin = np.array([MK[s][0] for s in S]) / 1e6 * 100; pout = np.array([MK[s][1] for s in S]) / 1e6 * 100
outm = np.where(avail, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan); Y = np.log(np.maximum(outm, 1))
Q = np.where(avail, succ / np.maximum(n, 1), 0); Cr = np.where(avail, inp * pin + outm * pout, 1e9)
sp = json.load(open(D / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]]); te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
_lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D / "content_preds.jsonl")}
P = np.clip(np.array([_lp[p] for p in pids]), 1e-4, 1 - 1e-4); LOGIT = np.log(P / (1 - P))
lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(D / CF)}
LC = np.array([lc[p] for p in pids]) * 100; MU = np.log(np.maximum((LC - inp * pin) / pout, 1.0))
VS = np.geomspace(1e-5, 100, 250); med = np.array([np.median(ct[tr, m][v[tr, m]]) for m in range(M)])
sig = lambda z: 1 / (1 + np.exp(-z))


def frontier(Pm, C, ii):
    pts = []
    for V in VS:
        m = np.where(avail[ii], Pm[ii] * V - C[ii], -np.inf).argmax(1); r = np.arange(len(ii))
        pts.append((np.mean(Cr[ii][r, m]), np.mean(Q[ii][r, m])))
    return hull(pts)


H0 = frontier(P, inp * pin + med[None] * pout, te)


def gain(Pm, C):
    H = frontier(Pm, C, te); lo, hi = max(H[0][1], H0[0][1]), min(H[-1][1], H0[-1][1])
    T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    return 1 - float(np.exp(np.nanmean(np.log([cost_at(H, x) / cost_at(H0, x) for x in T])))), H[-1][1]


def fit_irt(others):
    """1-D 2PL on TRAIN outcomes of the given routes: returns log a_i, b_i (train problems), theta_u."""
    ii = tr; K = len(others); N = len(ii)
    s_, n_ = succ[np.ix_(ii, others)].astype(float), n[np.ix_(ii, others)].astype(float)

    def nll(x):
        la, b, th = x[:N], x[N:2 * N], x[2 * N:]
        z = np.exp(la)[:, None] * (th[None] - b[:, None]); p = np.clip(sig(z), 1e-6, 1 - 1e-6)
        ll = (s_ * np.log(p) + (n_ - s_) * np.log(1 - p)).sum()
        pr = 0.5 * (la ** 2).sum() / 0.25 + 0.5 * (b ** 2).sum() / 4 + 0.5 * (th ** 2).sum()
        e = s_ - n_ * p                                              # d ll / dz
        a = np.exp(la)
        g_la = -(e * (th[None] - b[:, None])).sum(1) * a + la / 0.25
        g_b = (e.sum(1) * a) + b / 4
        g_th = -(e * a[:, None]).sum(0) + th
        return -ll + pr, np.r_[g_la, g_b, g_th]
    x0 = np.r_[np.zeros(N), -LOGIT[np.ix_(ii, others)].mean(1) / 2, np.zeros(K)]
    x = minimize(nll, x0, jac=True, method="L-BFGS-B").x
    return x[:N], x[N:2 * N], x[2 * N:]


rng = np.random.default_rng(0); full = gain(P, LC); res = {}
print(f"{name}: full heads {full[0]*100:.1f}% (max acc {full[1]*100:.1f}); rows: held-out route, k -> gain % (max acc %)")
for hname in HOLD:
    h = S.index(hname); others = [m for m in range(M) if m != h]
    # ---- ZeroRouter-style latent from the other routes
    la, b, th = fit_irt(others)
    F = np.c_[np.delete(LOGIT, h, 1), np.delete(LOGIT, h, 1) ** 2]
    fa = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-2, 1e4, 13))).fit(F[tr], la)
    fb = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-2, 1e4, 13))).fit(F[tr], b)
    A_all = np.exp(fa.predict(F)); B_all = fb.predict(F)
    A_all[tr] = np.exp(la); B_all[tr] = b                           # training queries keep their stage-1 positions (anchors)
    s_all = A_all * B_all; edges = np.quantile(s_all[tr], np.linspace(0, 1, 6)[1:-1]); bin_all = np.searchsorted(edges, s_all)
    lvl = np.nanmean(np.delete(MU, h, 1), 1); dbar = np.nanmean(np.delete(LOGIT, h, 1), 1)
    trh = tr[avail[tr, h]]
    p0 = sig(np.exp(la) * (0 - b)); fisher = np.exp(la) ** 2 * p0 * (1 - p0)
    pos = {i: j for j, i in enumerate(tr)}
    dopt_order = trh[np.argsort(-fisher[[pos[i] for i in trh]])]
    without = gain(np.where(np.arange(M)[None] == h, 0.0, P), np.where(np.arange(M)[None] == h, 1e9, LC))
    print(f"  {hname:<9} without it {without[0]*100:5.1f}% ({without[1]*100:.1f})")
    for k in (5, 10, 20, 50, 200):
        if k > len(trh):
            continue
        acc = {a: [] for a in ("ours", "zr", "zr-dopt", "naive")}
        for rep in range(20):
            kk = rng.choice(trh, k, replace=False)
            for arm in acc:
                if arm == "zr-dopt" and rep > 0:
                    continue                                          # deterministic anchors: one draw
                an = dopt_order[:k] if arm == "zr-dopt" else kk
                Pm = P.copy(); C = LC.copy()
                yk = np.concatenate([okd[i, h][v[i, h]] for i in an]).astype(int)
                if arm == "ours":
                    off = np.mean(Y[an, h] - lvl[an]); smear = np.mean(np.exp(Y[an, h] - (lvl[an] + off)))
                    C[:, h] = inp[:, h] * pin[h] + np.exp(lvl + off) * smear * pout[h]
                    Xk = np.repeat(dbar[an], n[an, h])
                    Pm[:, h] = LogisticRegression(C=1.0).fit(Xk[:, None], yk).predict_proba(dbar[:, None])[:, 1] if len(set(yk)) == 2 else np.clip(yk.mean(), .02, .98)
                elif arm in ("zr", "zr-dopt"):
                    ai, bi = A_all[an], B_all[an]; si, ni = succ[an, h].astype(float), n[an, h].astype(float)
                    f_ = lambda x: -(si * np.log(np.clip(sig(ai * (x[0] - bi)), 1e-6, 1)) + (ni - si) * np.log(np.clip(1 - sig(ai * (x[0] - bi)), 1e-6, 1))).sum() + 0.5 * x[0] ** 2
                    thh = minimize(f_, [0.0], method="L-BFGS-B").x[0]
                    Pm[:, h] = sig(A_all * (thh - B_all))
                    bm = np.array([np.nanmean(outm[an, h][bin_all[an] == j]) if (bin_all[an] == j).any() else np.nanmean(outm[an, h]) for j in range(5)])
                    C[:, h] = inp[:, h] * pin[h] + bm[bin_all] * pout[h]
                else:
                    Pm[:, h] = np.clip(yk.mean(), .02, .98); C[:, h] = inp[:, h] * pin[h] + np.nanmedian(outm[an, h]) * pout[h]
                acc[arm].append(gain(Pm, C))
        line = "  ".join(f"{a} {np.mean([g for g, _ in r])*100:6.1f}% ({np.mean([x for _, x in r])*100:.1f})" for a, r in acc.items())
        print(f"     k={k:<4} {line}")
        res[f"{hname}|{k}"] = {a: [float(np.mean([g for g, _ in r])), float(np.mean([x for _, x in r]))] for a, r in acc.items()}
    res[f"{hname}|without"] = list(without)
res["full"] = list(full)
json.dump(res, open(Path(__file__).parent / f"onboard_vs_zr_{name}.json", "w"), indent=1, default=float)

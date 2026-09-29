"""Hardened ZeroRouter reproduction on our 5-model pools (extends zr_dimsweep.py, 4.A.32): seeds, bin counts, paired CIs.
Same stages as zr_dimsweep.py (D-dim 2PL IRT by MAP on the pool's TRAIN outcomes; stage 2 = frozen 4B prefill -> PCA(256) ->
ridge onto (log alpha, b); cost = per-model mean train output in K bins of s = alpha^T b).
For D in {1, 5}, stage-1 seed in {0, 1, 2}, K in {5, 10, 20}:
  routing gain vs the paper rule at matched accuracy for ours / zr / zr success + our cost / our success + zr cost, with a PAIRED
  bootstrap (500 test resamples, identical across arms) of ours - zr and ours - (our success + zr cost)
  onboarding (K=5 bins, as the paper's small anchor sets need): 20 random anchor draws per held-out route at k = 10, 50; paired
  difference ours - zr per draw (same anchors), mean over routes with a 95% interval over draws
Usage: python zr_repro.py
"""
import json, sys, numpy as np, torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R, hull, cost_at
from baseline_cost_heads import rich
from zr_dimsweep import fit_stage1, fit_theta, POOLS

sig = lambda z: 1 / (1 + np.exp(-z))
VS = np.geomspace(1e-5, 100, 250); out = {}
for label, name, cfile, act in POOLS:
    D_ = R / name; t = np.load(D_ / "tensors.npz", allow_pickle=True)
    S = [str(s) for s in t["model_slots"]]; M = len(S); pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
    v = t["valid"].astype(bool); okd = (t["final_outcome"] & t["valid"]).astype(bool); ct = t["completion_tokens"].astype(float)
    pt = t["prompt_tokens"].astype(float); n = v.sum(2); avail = n > 0; succ = (okd & v).sum(2)
    inp = np.nan_to_num(np.nanmean(np.where(v, pt, np.nan), 2))
    pin = np.array([MK[s][0] for s in S]) / 1e6 * 100; pout = np.array([MK[s][1] for s in S]) / 1e6 * 100
    outm = np.where(avail, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan); Y = np.log(np.maximum(outm, 1))
    Q = np.where(avail, succ / np.maximum(n, 1), 0); Cr = np.where(avail, inp * pin + outm * pout, 1e9)
    sp = json.load(open(D_ / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]]); te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
    _lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D_ / "content_preds.jsonl")}
    P = np.clip(np.array([_lp[p] for p in pids]), 1e-4, 1 - 1e-4); LOGIT = np.log(P / (1 - P))
    lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(D_ / cfile)}
    LC = np.array([lc[p] for p in pids]) * 100; MU = np.log(np.maximum((LC - inp * pin) / pout, 1.0))
    med = np.array([np.median(ct[tr, m][v[tr, m]]) for m in range(M)])
    X = rich(R / act, pids); Z = PCA(256, random_state=0).fit(X[tr]).transform(X); Z /= Z[tr].std(0) + 1e-6
    PC = inp * pin + med[None] * pout

    def curve(Pm, C, ii):                          # vectorised frontier over V -> hull
        U = np.where(avail[ii][None], Pm[ii][None] * VS[:, None, None] - C[ii][None], -np.inf); m = U.argmax(2)
        a = np.take_along_axis(np.broadcast_to(Q[ii], U.shape), m[..., None], 2)[..., 0].mean(1)
        c = np.take_along_axis(np.broadcast_to(Cr[ii], U.shape), m[..., None], 2)[..., 0].mean(1)
        return hull(list(zip(c, a)))

    def gain(Pm, C, ii=te):
        H, H0 = curve(Pm, C, ii), curve(P, PC, ii); lo, hi = max(H[0][1], H0[0][1]), min(H[-1][1], H0[-1][1])
        T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
        return 1 - float(np.exp(np.nanmean(np.log([cost_at(H, x) / cost_at(H0, x) for x in T]))))

    def latent(routes, D, seed):
        la, b, th = fit_stage1(succ[np.ix_(tr, routes)], n[np.ix_(tr, routes)], D, seed)
        pr = RidgeCV(alphas=np.geomspace(1, 1e5, 11)).fit(Z[tr], np.c_[la, b]).predict(Z)
        A = np.exp(pr[:, :D]); B = pr[:, D:]; A[tr] = np.exp(la); B[tr] = b
        return A, B, th

    def bins(A, B, K):
        s = (A * B).sum(1); e = np.quantile(s[tr], np.linspace(0, 1, K + 1)[1:-1]); return np.searchsorted(e, s)

    rng = np.random.default_rng(0); BS = [rng.choice(te, len(te)) for _ in range(500)]
    ours = gain(P, LC); res = {"ours": ours}
    print(f"\n===== {label}: ours {ours*100:.1f}% vs paper rule", flush=True)
    for D in (1, 5):
        for seed in (0, 1, 2):
            A, B, th = latent(list(range(M)), D, seed)
            Pz = sig((A[:, None, :] * (th[None] - B[:, None, :])).sum(-1))
            for K in (5, 10, 20):
                bn = bins(A, B, K)
                tab = np.array([[np.nanmean(outm[tr[bn[tr] == j], m]) if (bn[tr] == j).any() else np.nanmean(outm[tr, m]) for j in range(K)] for m in range(M)])
                Cz = inp * pin + tab.T[bn] * pout
                g = {"zr": gain(Pz, Cz), "zr succ + our cost": gain(Pz, LC), "our succ + zr cost": gain(P, Cz)}
                row = f"  D={D} seed={seed} K={K:<2} " + "  ".join(f"{a} {x*100:5.1f}%" for a, x in g.items())
                if K == 10:                           # paired bootstrap at the default bin count
                    d1 = np.array([gain(P, LC, ii) - gain(Pz, Cz, ii) for ii in BS]) * 100
                    d2 = np.array([gain(P, LC, ii) - gain(P, Cz, ii) for ii in BS]) * 100
                    g["ours-zr"] = [float(d1.mean()), *np.percentile(d1, [2.5, 97.5])]
                    g["ours-(our succ+zr cost)"] = [float(d2.mean()), *np.percentile(d2, [2.5, 97.5])]
                    row += f"   ours-zr {d1.mean():+5.1f} [{np.percentile(d1,2.5):+.1f}, {np.percentile(d1,97.5):+.1f}]" \
                           f"   ours-(our succ+zr cost) {d2.mean():+5.1f} [{np.percentile(d2,2.5):+.1f}, {np.percentile(d2,97.5):+.1f}]"
                print(row, flush=True); res[f"D{D}|s{seed}|K{K}"] = g
        # onboarding at this D (seed 0), K=5 bins, 20 anchor draws, paired per draw
        ob = {10: [], 50: []}
        for h in range(M):
            others = [m for m in range(M) if m != h]; A, B, th = latent(others, D, 0); bh = bins(A, B, 5)
            lvl = np.nanmean(np.delete(MU, h, 1), 1); dbar = np.nanmean(np.delete(LOGIT, h, 1), 1); trh = tr[avail[tr, h]]
            for k in (10, 50):
                for _ in range(20):
                    kk = rng.choice(trh, k, replace=False); yk = np.concatenate([okd[i, h][v[i, h]] for i in kk]).astype(int)
                    Pm, C = P.copy(), LC.copy()
                    off = np.mean(Y[kk, h] - lvl[kk]); sm = np.mean(np.exp(Y[kk, h] - (lvl[kk] + off)))
                    C[:, h] = inp[:, h] * pin[h] + np.exp(lvl + off) * sm * pout[h]
                    Xk = np.repeat(dbar[kk], n[kk, h])
                    Pm[:, h] = LogisticRegression(C=1.0).fit(Xk[:, None], yk).predict_proba(dbar[:, None])[:, 1] if len(set(yk)) == 2 else np.clip(yk.mean(), .02, .98)
                    go = gain(Pm, C)
                    Pm, C = P.copy(), LC.copy(); tn = fit_theta(A[kk], B[kk], succ[kk, h], n[kk, h], D)
                    Pm[:, h] = sig((A * (tn[None] - B)).sum(1))
                    bm = np.array([np.nanmean(outm[kk, h][bh[kk] == j]) if (bh[kk] == j).any() else np.nanmean(outm[kk, h]) for j in range(5)])
                    C[:, h] = inp[:, h] * pin[h] + bm[bh] * pout[h]
                    ob[k].append((h, go, gain(Pm, C)))
        for k, r in ob.items():
            r = np.array(r); dm = np.array([np.mean([x[1] - x[2] for x in r if x[0] == h]) for h in range(M)]) * 100
            per = (r[:, 1] - r[:, 2]).reshape(M, 20).mean(0) * 100      # mean over routes, per draw index
            res[f"D{D}|onboard_k{k}"] = {"ours": float(r[:, 1].mean()), "zr": float(r[:, 2].mean()), "diff_by_route": dm.tolist(),
                                         "diff_ci_over_draws": [float(per.mean()), *np.percentile(per, [2.5, 97.5])]}
            print(f"  D={D} onboarding k={k}: ours {r[:,1].mean()*100:.1f}% zr {r[:,2].mean()*100:.1f}%  ours-zr {per.mean():+.1f} "
                  f"[{np.percentile(per,2.5):+.1f}, {np.percentile(per,97.5):+.1f}] (over anchor draws); by route "
                  + " ".join(f"{S[h]} {x:+.1f}" for h, x in enumerate(dm)), flush=True)
    out[label] = res
json.dump(out, open(Path(__file__).parent / "zr_repro.json", "w"), indent=1, default=float)

"""Learning curves (NEW_PATH 4.A.121): how each estimator's saving grows with the number of labelled training problems n, on the 10
training sets. Per training set (DRAW=s) and n in NS (default 50, 100, 200, all): the training problems are subsampled to n (seed
3000 + 1000 s + n; n = all keeps the set); calibration and test are unchanged. Two panels:
  cost      cost estimators fitted on the n problems, all routing with the SAME success probes, fitted on the full training set (the
            tab:allest protocol): ours, median, mean, fromsuccess, zr-bins (difficulty bins), gbm, mixllm (MixLLM-style), knn,
            zr-pricing (ZeroRouter bins, our success), scx (one length per query), rtr (Route-To-Reason)
  system    whole routers fitted on the n problems, each with its own success model: ours (our success probes refitted on n + our cost
            probe), ours-median (our success on n + median pricing), zr_4b (ZeroRouter, its latent read from the 4B prefill), zr_pub
            (ZeroRouter as published, fine-tuned DistilBERT), carrot (CARROT-style kNN success and kNN cost)
Saving at matched accuracy relative to ONE fixed reference for every n and panel: median pricing from the full training set with the
full-set success probes (the paper's median rule), point + NBOOT test resamples (rng 0, the same in every set and n); and ours vs each
arm at the same n and panel (paired). Configurations ((D, K), kNN size, epochs) are chosen on calibration as in the source scripts.
Usage: (GPU) REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned NBOOT=200 OUT_DIR=... python learning_curve.py POOL S0 S1
"""
import json, os, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
POOL_, S0, S1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); sys.argv = [sys.argv[0], "lcurve", POOL_]
HERE = Path(__file__).parent
exec(open(HERE / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
_xe = open(HERE / "extra_estimators_draws.py").read()
exec(_xe[_xe.index("from transformers import AutoModel"):_xe.index("Q = encode(texts_of(POOL_))")])   # encode, RTR, rtr, scx: unchanged
_zf = open(HERE / "zr_faithful_draws.py").read()
exec(_zf[_zf.index("_ze = open"):_zf.index("texts = texts_of(POOL_)")])                     # their encoder, stage2_pub, zr_from_positions
OUT = Path(os.environ.get("OUT_DIR", HERE)); TAG = os.environ.get("RESULT_TAG", "")
NS = [int(x) if x != "all" else 0 for x in os.environ.get("NS", "50,100,200,all").split(",")]


def zr_4b(d, S):
    """tmlr_strengthen.zr_paper with the PCA size capped below n (256 needs >256 training problems)"""
    tr, ca, N = d["tr"], d["ca"], len(d["ids"]); Ltr = d["L"][tr]
    Z = PCA(min(256, len(tr) - 1), random_state=0).fit(d["X"][tr]).transform(d["X"]); Z /= Z[tr].std(0) + 1e-6
    H0 = curve(d, d["P"], cost_of(d, np.repeat(d["med"][None], N, 0)), ca); best = None
    for D in (1, 5):
        la, bb, th = fit_stage1(S["succ"], S["n"], D, 0); pr = RidgeCV(alphas=np.geomspace(1, 1e5, 11)).fit(Z[tr], np.c_[la, bb]).predict(Z)
        A = np.exp(pr[:, :D]); B = pr[:, D:]; A[tr] = np.exp(la); B[tr] = bb; Pz = np.clip(sig((A[:, None, :] * (th[None] - B[:, None, :])).sum(-1)), 1e-4, 1 - 1e-4)
        sc = (A * B).sum(1)
        for K in (5, 10, 20):
            e_ = np.quantile(sc[tr], np.linspace(0, 1, K + 1)[1:-1]); bn = np.searchsorted(e_, sc)
            tab = np.array([[Ltr[bn[tr] == j, k].mean() if (bn[tr] == j).any() else Ltr[:, k].mean() for j in range(K)] for k in range(d["M"])])
            g = saved(curve(d, Pz, cost_of(d, tab.T[bn]), ca), H0)
            if best is None or g > best[0]:
                best = (g, Pz, tab.T[bn], (D, K))
    return best[1], best[2], best[3]


def success_n(d, tr):
    """our success probes (success_linear) refitted on tr; a route with no success or no failure among tr's draws gets its rate"""
    A = dict(X=d["X"], y0=d["y0"], succ=d["succ"], n=d["n"]); sub = lambda ii: {k: x[ii] for k, x in A.items()}
    s_, n_ = d["succ"][tr].sum(0), d["n"][tr].sum(0); bad = (s_ == 0) | (s_ == n_)
    if not bad.any():
        return success_linear("X", sub(tr), sub(d["ca"]), sub(np.arange(len(d["ids"]))))
    keep = np.where(~bad)[0]; P = np.tile(np.clip(s_ / np.maximum(n_, 1), 1e-4, 1 - 1e-4), (len(d["ids"]), 1))
    B = {k: (x[:, keep] if x.ndim == 2 and x.shape[1] == d["M"] else x) for k, x in A.items()}; subB = lambda ii: {k: x[ii] for k, x in B.items()}
    P[:, keep] = success_linear("X", subB(tr), subB(d["ca"]), subB(np.arange(len(d["ids"])))); return P


def bins(d, P, tr):
    """difficulty bins (tmlr_strengthen draws 'zr-bins'): ten bins of the mean success logit, per-route mean training length"""
    dbar = np.log(P / (1 - P)).mean(1); e_ = np.quantile(dbar[tr], np.linspace(0, 1, 11)[1:-1]); b_ = np.searchsorted(e_, dbar)
    zb = np.stack([np.array([d["L"][tr][b_[tr] == j, k].mean() if (b_[tr] == j).any() else d["L"][tr, k].mean() for j in range(10)])[b_] for k in range(d["M"])], 1)
    return zb * (d["L"][tr].mean(0) / zb[tr].mean(0))


def zr_pub(d, S, H0c):
    tr = d["tr"]; Fz = (Fraw - Fraw[tr].mean(0)) / (Fraw[tr].std(0) + 1e-6); cands, eps = [], {}
    for D in (1, 5):
        la, bb, th = fit_stage1(S["succ"], S["n"], D, 0); LA, B, eps[D] = stage2_pub(enc, Fz, tr, la, bb, D)
        cands += zr_from_positions(d, d["ca"], H0c, LA, B, th, D, tr, la, bb)
    best = max(cands, key=lambda c: -np.inf if np.isnan(c[0]) else c[0]); return best[1], best[2], (*best[3], eps)


def carrot(d, S, Ca, T):
    Pk, nn = success_knn(S, Ca, T); dk = dict(d, P=Pk); H0 = curve(dk, Pk, cost_of(dk, np.repeat(d["med"][None], len(d["ids"]), 0)), d["ca"]); sc = {}
    for k_ in (1, 3, 5, 10, 20):
        if k_ <= len(S["X"]):
            t_ = np.zeros((len(d["ids"]), d["M"])); t_[d["ca"]] = fit_predict("knn", S, Ca, k_); sc[k_] = saved(curve(dk, Pk, cost_of(dk, t_), d["ca"]), H0)
    k_ = max(sc, key=lambda c: -np.inf if np.isnan(sc[c]) else sc[c]); return np.clip(Pk, 1e-4, 1 - 1e-4), fit_predict("knn", S, T, k_), (nn, k_)


texts = texts_of(POOL_); Q = encode(texts)
enc = tok(texts, truncation=True, max_length=512, padding="max_length", return_tensors="pt"); Fraw = np.array([feats(x) for x in texts], np.float32)
res = {"pool": POOL_, "ns": NS, "seeds": {}}
outf = OUT / f"learning_curve_{POOL_.lower().replace('-', '')}_{S0}_{S1}{TAG}.json"
for s in range(S0, S1):
    os.environ["DRAW"] = str(s); d = load(POOL_); N = len(d["ids"]); tr_full, ca, ev = d["tr"], d["ca"], d["ev"]; Pfull, medfull = d["P"], d["med"]
    Cm = cost_of(d, np.repeat(medfull[None], N, 0)); rb = np.random.default_rng(0); BS = [ev] + [ev[rb.integers(0, len(ev), len(ev))] for _ in range(NBOOT)]
    H0 = [curve(d, Pfull, Cm, ii) for ii in BS]; res["seeds"][s] = {"_n": dict(train=len(tr_full), calibration=len(ca), test=len(ev))}
    Ho = [curve(d, Pfull, cost_of(d, d["L"]), ii) for ii in BS]; res["seeds"][s]["oracle"] = dict(vs_ref=[saved(h, h0) * 100 for h, h0 in zip(Ho, H0)][0])
    for n in NS:
        tr = tr_full if n == 0 or n >= len(tr_full) else np.sort(np.random.default_rng(3000 + 1000 * s + n).choice(tr_full, n, replace=False))
        key = "all" if tr is tr_full else str(n); CT = d["CT"]
        med = np.array([np.nanmedian(CT[tr, k]) for k in range(d["M"])])
        out = {"n_train": int(len(tr))}
        for panel in ("cost", "system"):
            Pn = Pfull if panel == "cost" else (Pfull if tr is tr_full else success_n(d, tr))
            dn = dict(d, tr=tr, P=Pn, med=med); S, T, Ca = view(dn, tr), view(dn, np.arange(N)), view(dn, ca); tk, Ps, cfg_ = {}, {}, {}
            if panel == "cost":
                cfg = choose(dn, S, Ca); cfg_ = {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}
                for a in ("ours", "median", "mean", "fromsuccess", "gbm", "mixllm", "knn"):
                    tk[a] = fit_predict(a, S, T, cfg.get(a))
                tk["zr-pricing"] = fit_predict("zerorouter", S, T, cfg["zerorouter"]); tk["zr-bins"] = bins(dn, Pn, tr)
                tk["scx"] = scx(dn); tk["rtr"], cfg_["rtr_epochs"] = rtr(dn, Q, False)
            else:
                tk["ours"] = fit_predict("ours", S, T); tk["ours-median"] = np.repeat(med[None], N, 0)
                Ps["zr_4b"], tk["zr_4b"], c4 = zr_4b(dn, S); cfg_["zr_4b"] = list(c4)
                H0c = curve(dn, Pn, cost_of(dn, np.repeat(med[None], N, 0)), ca)
                Ps["zr_pub"], tk["zr_pub"], cp = zr_pub(dn, S, H0c); cfg_["zr_pub"] = [cp[0], cp[1], {str(k): v for k, v in cp[2].items()}]
                Ps["carrot"], tk["carrot"], cfg_["carrot"] = carrot(dn, S, Ca, T)
            C = {a: cost_of(d, x) for a, x in tk.items()}; Pa = lambda a: Ps.get(a, Pn)
            Hs = [{a: curve(d, Pa(a), C[a], ii) for a in C} for ii in BS]; r_ = {"_cfg": cfg_}
            for a in C:
                vm = np.array([saved(H[a], h0) for H, h0 in zip(Hs, H0)]) * 100; vd = np.array([saved(H["ours"], H[a]) for H in Hs]) * 100
                r_[a] = dict(vs_ref=float(vm[0]), vs_ref_bs=vm[1:].tolist(), ours_vs_it=float(vd[0]), ours_vs_it_bs=vd[1:].tolist(),
                             auc=float(np.mean([auc(d["y0"][ev, m], Pa(a)[ev, m]) for m in range(d["M"])])), r2=float(np.mean(r2_routes(d, tk[a], ev))))
            out[panel] = r_
            print(f"  set {s} n={key:>3s} {panel:6s}: vs ref " + "  ".join(f"{a} {r_[a]['vs_ref']:+5.1f}" for a in C)
                  + " | ours vs it " + "  ".join(f"{a} {r_[a]['ours_vs_it']:+5.1f}" for a in C if a != "ours"), flush=True)
        res["seeds"][s][key] = out
        json.dump(res, open(outf, "w"), indent=1, default=float)
print("DONE", flush=True)

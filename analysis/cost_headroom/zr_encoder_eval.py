"""Route with ZeroRouter's OWN encoder (zr_encoder.py outputs) vs the 4B-prefill stand-in used in 4.A.32/4.A.33, and vs ours.
Per pool and D in {1, 5}: stage-2 fit quality on test queries is not observable (their latent exists only for train queries), so
we report (1) the held-out-train R2 of (log alpha, b) for both readers (DistilBERT+features vs 4B->PCA->ridge, same split) and
(2) routing gain vs the paper rule (K = 5 / 10 / 20 bins): zr[DistilBERT], zr[4B reader], their success + our cost, ours.
Usage: python zr_encoder_eval.py
"""
import json, sys, numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R, hull, cost_at
from baseline_cost_heads import rich
from zr_dimsweep import POOLS

sig = lambda z: 1 / (1 + np.exp(-z)); VS = np.geomspace(1e-5, 100, 250); out = {}
for label, name, cfile, act in POOLS:
    D_ = R / name; t = np.load(D_ / "tensors.npz", allow_pickle=True)
    S = [str(s) for s in t["model_slots"]]; M = len(S); pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
    v = t["valid"].astype(bool); okd = (t["final_outcome"] & t["valid"]).astype(bool); ct = t["completion_tokens"].astype(float)
    pt = t["prompt_tokens"].astype(float); n = v.sum(2); avail = n > 0; succ = (okd & v).sum(2)
    inp = np.nan_to_num(np.nanmean(np.where(v, pt, np.nan), 2))
    pin = np.array([MK[s][0] for s in S]) / 1e6 * 100; pout = np.array([MK[s][1] for s in S]) / 1e6 * 100
    outm = np.where(avail, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan)
    Q = np.where(avail, succ / np.maximum(n, 1), 0); Cr = np.where(avail, inp * pin + outm * pout, 1e9)
    sp = json.load(open(D_ / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]]); te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
    _lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D_ / "content_preds.jsonl")}
    P = np.clip(np.array([_lp[p] for p in pids]), 1e-4, 1 - 1e-4)
    lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(D_ / cfile)}
    LC = np.array([lc[p] for p in pids]) * 100; med = np.array([np.median(ct[tr, m][v[tr, m]]) for m in range(M)]); PC = inp * pin + med[None] * pout
    X = rich(R / act, pids); Z = PCA(256, random_state=0).fit(X[tr]).transform(X); Z /= Z[tr].std(0) + 1e-6

    def gain(Pm, C):
        def curve(Pm_, C_):
            U = np.where(avail[te][None], Pm_[te][None] * VS[:, None, None] - C_[te][None], -np.inf); m = U.argmax(2)
            a = np.take_along_axis(np.broadcast_to(Q[te], U.shape), m[..., None], 2)[..., 0].mean(1)
            c = np.take_along_axis(np.broadcast_to(Cr[te], U.shape), m[..., None], 2)[..., 0].mean(1); return hull(list(zip(c, a)))
        H, H0 = curve(Pm, C), curve(P, PC); lo, hi = max(H[0][1], H0[0][1]), min(H[-1][1], H0[-1][1])
        T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
        return 1 - float(np.exp(np.nanmean(np.log([cost_at(H, x) / cost_at(H0, x) for x in T]))))

    res = {"ours": gain(P, LC)}; print(f"\n===== {label}: ours {res['ours']*100:.1f}%")
    for D in (1, 5):
        f = D_ / f"zr_encoder_D{D}.npz"
        if not f.exists():
            print(f"  D={D}: encoder output missing"); continue
        z = np.load(f, allow_pickle=True); la, b, th = z["train_log_alpha"], z["train_b"], z["theta"]
        tgt = np.c_[la, b]; rng = np.random.default_rng(0); perm = rng.permutation(tr); hold, fit_ = perm[: len(tr) // 10], perm[len(tr) // 10:]
        pos = {i: j for j, i in enumerate(tr)}
        r2 = lambda pred, idx: 1 - ((pred - tgt[[pos[i] for i in idx]]) ** 2).sum() / ((tgt[[pos[i] for i in idx]] - tgt.mean(0)) ** 2).sum()
        rd = RidgeCV(alphas=np.geomspace(1, 1e5, 11)).fit(Z[fit_], tgt[[pos[i] for i in fit_]])
        dist_pred = np.c_[z["log_alpha"], z["b"]]
        print(f"  D={D}: held-out-train R2 of the latent: DistilBERT+features {r2(dist_pred[hold], hold):.2f}, 4B reader {r2(rd.predict(Z[hold]), hold):.2f}")
        readers = {"zr[DistilBERT]": (np.exp(z["log_alpha"]), z["b"].copy()),
                   "zr[4B reader]": tuple(np.split(RidgeCV(alphas=np.geomspace(1, 1e5, 11)).fit(Z[tr], tgt).predict(Z), 2, axis=1))}
        readers["zr[4B reader]"] = (np.exp(readers["zr[4B reader]"][0]), readers["zr[4B reader]"][1])
        for rname, (A, B) in readers.items():
            A = A.copy(); B = B.copy(); A[tr] = np.exp(la); B[tr] = b
            Pz = sig((A[:, None, :] * (th[None] - B[:, None, :])).sum(-1)); row = {}
            for K in (5, 10, 20):
                s = (A * B).sum(1); e = np.quantile(s[tr], np.linspace(0, 1, K + 1)[1:-1]); bn = np.searchsorted(e, s)
                tab = np.array([[np.nanmean(outm[tr[bn[tr] == j], m]) if (bn[tr] == j).any() else np.nanmean(outm[tr, m]) for j in range(K)] for m in range(M)])
                row[K] = gain(Pz, inp * pin + tab.T[bn] * pout)
            row["succ+our cost"] = gain(Pz, LC); res[f"D{D}|{rname}"] = row
            print(f"    {rname:<16} K=5 {row[5]*100:5.1f}%  K=10 {row[10]*100:5.1f}%  K=20 {row[20]*100:5.1f}%  | their success + our cost {row['succ+our cost']*100:5.1f}%")
    out[label] = res
json.dump(out, open(Path(__file__).parent / "zr_encoder_eval.json", "w"), indent=1, default=float)

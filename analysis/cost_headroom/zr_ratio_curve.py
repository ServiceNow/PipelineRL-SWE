"""Cost of ZeroRouter relative to ours as a function of accuracy (headline figure, right panel). ZeroRouter's configuration is the
calibration-chosen one (zr_best_ci.json). Test frontiers over V (upper hulls); at 25 accuracies spanning the range both reach
(5-95% trimmed), ratio = cost_ZR(acc) / cost_ours(acc); 95% band from 500 paired test resamples (same resample for both arms;
a resample contributes only at accuracies both hulls reach). Writes zr_ratio_curve.json.
Usage: python zr_ratio_curve.py
"""
import json, sys, numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R, hull, cost_at
from baseline_cost_heads import rich
from zr_dimsweep import fit_stage1, POOLS

sig = lambda z: 1 / (1 + np.exp(-z)); VS = np.geomspace(1e-5, 100, 250)
ci = json.load(open(Path(__file__).parent / "zr_best_ci.json")); out = {}
for label, name, cfile, act in POOLS:
    cfg = ci[label]["calibration-chosen"]["config"]; D, seed, K = (int(x[1:]) for x in cfg.split("|"))
    D_ = R / name; t = np.load(D_ / "tensors.npz", allow_pickle=True)
    S = [str(s) for s in t["model_slots"]]; M = len(S); pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
    v = t["valid"].astype(bool); okd = (t["final_outcome"] & t["valid"]).astype(bool); ct = t["completion_tokens"].astype(float)
    pt = t["prompt_tokens"].astype(float); n = v.sum(2); avail = n > 0; succ = (okd & v).sum(2)
    inp = np.nan_to_num(np.nanmean(np.where(v, pt, np.nan), 2)); pin = np.array([MK[s][0] for s in S]) / 1e6 * 100; pout = np.array([MK[s][1] for s in S]) / 1e6 * 100
    outm = np.where(avail, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan); Q = np.where(avail, succ / np.maximum(n, 1), 0); Cr = np.where(avail, inp * pin + outm * pout, 1e9)
    sp = json.load(open(D_ / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]]); te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
    _lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D_ / "content_preds.jsonl")}; P = np.clip(np.array([_lp[p] for p in pids]), 1e-4, 1 - 1e-4)
    lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(D_ / cfile)}; LC = np.array([lc[p] for p in pids]) * 100
    X = rich(R / act, pids); Z = PCA(256, random_state=0).fit(X[tr]).transform(X); Z /= Z[tr].std(0) + 1e-6
    la, b, th = fit_stage1(succ[tr], n[tr], D, seed); pr = RidgeCV(alphas=np.geomspace(1, 1e5, 11)).fit(Z[tr], np.c_[la, b]).predict(Z)
    A = np.exp(pr[:, :D]); B = pr[:, D:]; A[tr] = np.exp(la); B[tr] = b; Pz = sig((A[:, None, :] * (th[None] - B[:, None, :])).sum(-1))
    s = (A * B).sum(1); e = np.quantile(s[tr], np.linspace(0, 1, K + 1)[1:-1]); bn = np.searchsorted(e, s)
    tab = np.array([[np.nanmean(outm[tr[bn[tr] == j], m]) if (bn[tr] == j).any() else np.nanmean(outm[tr, m]) for j in range(K)] for m in range(M)]); Cz = inp * pin + tab.T[bn] * pout

    def curve(Pm, C, ii):
        U = np.where(avail[ii][None], Pm[ii][None] * VS[:, None, None] - C[ii][None], -np.inf); m = U.argmax(2)
        a = np.take_along_axis(np.broadcast_to(Q[ii], U.shape), m[..., None], 2)[..., 0].mean(1); c = np.take_along_axis(np.broadcast_to(Cr[ii], U.shape), m[..., None], 2)[..., 0].mean(1)
        return hull(list(zip(c, a)))
    Ho, Hz = curve(P, LC, te), curve(Pz, Cz, te); lo, hi = max(Ho[0][1], Hz[0][1]), min(Ho[-1][1], Hz[-1][1])
    acc = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 25)
    ratio = np.array([cost_at(Hz, a) / cost_at(Ho, a) for a in acc])
    rng = np.random.default_rng(0); B_ = []
    for _ in range(500):
        ii = rng.choice(te, len(te)); ho, hz = curve(P, LC, ii), curve(Pz, Cz, ii)
        B_.append([cost_at(hz, a) / cost_at(ho, a) if (ho[0][1] <= a <= ho[-1][1] and hz[0][1] <= a <= hz[-1][1]) else np.nan for a in acc])
    B_ = np.array(B_, float)
    out[label] = {"config": cfg, "acc": acc.tolist(), "ratio": ratio.tolist(), "lo": np.nanpercentile(B_, 2.5, 0).tolist(), "hi": np.nanpercentile(B_, 97.5, 0).tolist()}
    print(f"{label}: ratio ZR/ours from {ratio.min():.2f} to {ratio.max():.2f} over accuracy {acc[0]*100:.0f}-{acc[-1]*100:.0f}%", flush=True)
json.dump(out, open(Path(__file__).parent / "zr_ratio_curve.json", "w"), indent=1)

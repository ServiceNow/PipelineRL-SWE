"""Paired bootstrap of ours - ZeroRouter, with ZeroRouter's configuration (D in {1, 5}, seed in {0, 1, 2}, K in {5, 10, 20},
4B reader) chosen two ways: (a) on CALIBRATION, applied once to test -- the fair protocol, used in the headline figure; (b) the best
on TEST (zr_repro.json) -- an upper bound for ZeroRouter.
Same pipeline as zr_repro.py; 1000 test resamples, identical for both arms. Writes zr_best_ci.json.
Usage: python zr_best_ci.py
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
rep = json.load(open(Path(__file__).parent / "zr_repro.json")); out = {}
for label, name, cfile, act in POOLS:
    cfgs = {k: v["zr"] for k, v in rep[label].items() if "|K" in k}; best = max(cfgs, key=cfgs.get)
    D, seed, K = int(best.split("|")[0][1:]), int(best.split("|")[1][1:]), int(best.split("|")[2][1:])
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

    def gain(Pm, C, ii):
        def curve(Pm_, C_):
            U = np.where(avail[ii][None], Pm_[ii][None] * VS[:, None, None] - C_[ii][None], -np.inf); m = U.argmax(2)
            a = np.take_along_axis(np.broadcast_to(Q[ii], U.shape), m[..., None], 2)[..., 0].mean(1)
            c = np.take_along_axis(np.broadcast_to(Cr[ii], U.shape), m[..., None], 2)[..., 0].mean(1); return hull(list(zip(c, a)))
        H, H0 = curve(Pm, C), curve(P, PC); lo, hi = max(H[0][1], H0[0][1]), min(H[-1][1], H0[-1][1])
        T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
        return 1 - float(np.exp(np.nanmean(np.log([cost_at(H, x) / cost_at(H0, x) for x in T]))))

    cal = np.array([pi[str(p)] for p in sp["calibration_problem_ids"]])

    def zr_arm(D_, seed_, K_):
        la, b, th = fit_stage1(succ[tr], n[tr], D_, seed_)
        pr = RidgeCV(alphas=np.geomspace(1, 1e5, 11)).fit(Z[tr], np.c_[la, b]).predict(Z)
        A = np.exp(pr[:, :D_]); B = pr[:, D_:]; A[tr] = np.exp(la); B[tr] = b
        Pz = sig((A[:, None, :] * (th[None] - B[:, None, :])).sum(-1))
        s_ = (A * B).sum(1); e = np.quantile(s_[tr], np.linspace(0, 1, K_ + 1)[1:-1]); bn = np.searchsorted(e, s_)
        tab = np.array([[np.nanmean(outm[tr[bn[tr] == j], m]) if (bn[tr] == j).any() else np.nanmean(outm[tr, m]) for j in range(K_)] for m in range(M)])
        return Pz, inp * pin + tab.T[bn] * pout
    arms = {}
    for D_ in (1, 5):
        for seed_ in (0, 1, 2):
            for K_ in (5, 10, 20):
                arms[f"D{D_}|s{seed_}|K{K_}"] = zr_arm(D_, seed_, K_)
    cal_gain = {k: gain(*v, cal) for k, v in arms.items()}; chosen = max(cal_gain, key=cal_gain.get)
    rng = np.random.default_rng(0); BS = [rng.choice(te, len(te)) for _ in range(1000)]
    res = {}
    for tag, cfg in (("calibration-chosen", chosen), ("test-best (upper bound for ZR)", best)):
        Pz, Cz = arms[cfg]; d = np.array([(gain(P, LC, ii) - gain(Pz, Cz, ii)) * 100 for ii in BS])
        res[tag] = {"config": cfg, "ours": gain(P, LC, te), "zr": gain(Pz, Cz, te), "diff": float(d.mean()), "ci": [float(x) for x in np.percentile(d, [2.5, 97.5])]}
        print(f"{label} [{tag}] ZeroRouter {cfg}: {res[tag]['zr']*100:.1f}% vs ours {res[tag]['ours']*100:.1f}%; ours - ZR {d.mean():+.1f} "
              f"[{np.percentile(d,2.5):+.1f}, {np.percentile(d,97.5):+.1f}]", flush=True)
    out[label] = res
json.dump(out, open(Path(__file__).parent / "zr_best_ci.json", "w"), indent=1)

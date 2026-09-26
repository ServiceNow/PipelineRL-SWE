#!/usr/bin/env python3
"""Entry-state DISTRIBUTIONAL belief per (problem, route), from the 4B prefill -- for the Bellman replay.

Per route m, a spike-and-slab over the per-draw success rate q:
    pi0 = sigmoid(u_m . z)                   P(route never solves this problem)
    q ~ Beta(a, b) otherwise, a = softplus(v_m . z), b = softplus(w_m . z)
z = PCA of the rich prefill features (all layers x {mean,last}), fit on TRAIN only. Trained by the marginal
likelihood of ALL of the problem's draws on that route (s successes, f failures):
    L = log[ pi0 * 1{s=0} + (1-pi0) * B(a+s, b+f) / B(a, b) ]
The spread matters to the policy: after f failures the posterior mean is
    P(next succeeds | f fails) = (1 - pi0') * a / (a + b + f),  pi0' = pi0 / (pi0 + (1-pi0) B(a,b+f)/B(a,b))
so a vague belief collapses fast (switch / give up) and a sharp one barely moves (resample).

LEVEL FIX (PAPER_OUTLINE 3b-lxxxiii: the old head ranked well but its level was off): per route, three
scalars fit on CALIBRATION by marginal likelihood -- a shift on logit(pi0), a shift on logit(a/(a+b)), and
a scale on the concentration a+b. Reports calibration of the entry mean vs observed pass rates.
"""
from __future__ import annotations
import argparse, json, itertools
from pathlib import Path
import numpy as np, torch
from scipy.special import betaln, expit, logit
from sklearn.decomposition import PCA


def marg_ll(pi0, a, b, s, f):
    """log marginal likelihood, numpy, elementwise."""
    slab = betaln(a + s, b + f) - betaln(a, b)
    spike = np.where(s == 0, np.log(np.clip(pi0, 1e-12, 1)), -np.inf)
    return np.logaddexp(spike, np.log(np.clip(1 - pi0, 1e-12, 1)) + slab)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=600)
    a = ap.parse_args()
    torch.manual_seed(0)
    T = Path(a.tensors_dir)
    t = np.load(T / "tensors.npz", allow_pickle=True)
    v = t["valid"].astype(bool); ok = (t["final_outcome"] & t["valid"]).astype(bool)
    slots = [str(x) for x in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]
    S = (ok & v).sum(2).astype(float); F = (v & ~ok).sum(2).astype(float)
    z = np.load(a.activations, allow_pickle=True)
    feats = np.concatenate([z[k].reshape(len(z[k]), -1) for k in ("mean", "last") if k in z.files], 1)
    fid = {str(p): i for i, p in enumerate(z["problem_ids"])}
    X = feats[[fid[p] for p in pids]].astype(np.float32)
    sp = json.loads((T / "split_manifest.json").read_text())
    pix = {p: i for i, p in enumerate(pids)}
    idx = {k: np.array([pix[p] for p in sp[f"{k}_problem_ids"]]) for k in ("train", "calibration", "test")}
    tr = idx["train"]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = PCA(min(a.pca, len(tr) - 1), random_state=0).fit((X[tr] - mu) / sd).transform((X - mu) / sd).astype(np.float32)
    Z = np.concatenate([Z / np.sqrt(Z.shape[1]), np.ones((len(Z), 1), np.float32)], 1)   # + bias
    params = np.zeros((len(pids), len(slots), 3))
    print(f"{len(pids)} problems, {len(slots)} routes; PCA {Z.shape[1]-1}; train {len(tr)} / cal {len(idx['calibration'])}")
    for m, slot in enumerate(slots):
        keep = tr[(S[tr, m] + F[tr, m]) > 0]
        Zt = torch.tensor(Z[keep]); st = torch.tensor(S[keep, m]); ft = torch.tensor(F[keep, m])
        W = torch.zeros(Z.shape[1], 3, requires_grad=True)
        with torch.no_grad():   # init at the pooled rate, moderate concentration, small spike
            rate = S[keep, m].sum() / max(1.0, (S[keep, m] + F[keep, m]).sum())
            W[-1] = torch.tensor([-2.0, float(np.log(np.expm1(2 * rate + 0.05))), float(np.log(np.expm1(2 * (1 - rate) + 0.05)))])
        opt = torch.optim.Adam([W], lr=0.03)
        for _ in range(a.epochs):
            out = Zt @ W
            pi0 = torch.sigmoid(out[:, 0]); aa = torch.nn.functional.softplus(out[:, 1]) + 1e-3
            bb = torch.nn.functional.softplus(out[:, 2]) + 1e-3
            slab = (torch.lgamma(aa + st) + torch.lgamma(bb + ft) - torch.lgamma(aa + bb + st + ft)
                    - torch.lgamma(aa) - torch.lgamma(bb) + torch.lgamma(aa + bb))
            spike = torch.where(st == 0, torch.log(pi0.clamp_min(1e-12)), torch.full_like(pi0, -1e30))
            ll = torch.logaddexp(spike, torch.log((1 - pi0).clamp_min(1e-12)) + slab)
            loss = -ll.mean() + a.l2 * (W[:-1] ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            out = torch.tensor(Z) @ W
            params[:, m, 0] = torch.sigmoid(out[:, 0]).numpy()
            params[:, m, 1] = (torch.nn.functional.softplus(out[:, 1]) + 1e-3).numpy()
            params[:, m, 2] = (torch.nn.functional.softplus(out[:, 2]) + 1e-3).numpy()
        # level fix on CALIBRATION: shift logit(pi0), shift logit(mean), scale concentration
        cal = idx["calibration"]; cal = cal[(S[cal, m] + F[cal, m]) > 0]
        p0, A, B = params[cal, m, 0], params[cal, m, 1], params[cal, m, 2]
        best = (-np.inf, (0.0, 0.0, 1.0))
        for dp, dm, kc in itertools.product(np.linspace(-3, 3, 13), np.linspace(-2, 2, 17), (0.25, 0.5, 1, 2, 4)):
            mean = expit(logit(np.clip(A / (A + B), 1e-6, 1 - 1e-6)) + dm); conc = (A + B) * kc
            ll = marg_ll(expit(logit(np.clip(p0, 1e-6, 1 - 1e-6)) + dp), mean * conc, (1 - mean) * conc, S[cal, m], F[cal, m]).mean()
            if ll > best[0]:
                best = (ll, (dp, dm, kc))
        dp, dm, kc = best[1]
        mean = expit(logit(np.clip(params[:, m, 1] / params[:, m, 1:].sum(1), 1e-6, 1 - 1e-6)) + dm)
        conc = params[:, m, 1:].sum(1) * kc
        params[:, m, 0] = expit(logit(np.clip(params[:, m, 0], 1e-6, 1 - 1e-6)) + dp)
        params[:, m, 1], params[:, m, 2] = mean * conc, (1 - mean) * conc
        te = idx["test"]; te = te[(S[te, m] + F[te, m]) > 0]
        pm = (1 - params[te, m, 0]) * params[te, m, 1] / params[te, m, 1:].sum(1)
        emp = S[te, m] / (S[te, m] + F[te, m])
        ll_te = marg_ll(params[te, m, 0], params[te, m, 1], params[te, m, 2], S[te, m], F[te, m]).mean()
        base = S[tr, m].sum() / (S[tr, m] + F[tr, m]).sum()
        ll_base = marg_ll(np.zeros(len(te)) + 1e-6, np.full(len(te), 2 * base), np.full(len(te), 2 * (1 - base)), S[te, m], F[te, m]).mean()
        print(f"  {slot:<9} level fix: dpi0 {dp:+.1f} dmean {dm:+.2f} conc x{kc} | test: mean belief {pm.mean():.3f} vs observed "
              f"{emp.mean():.3f}, corr {np.corrcoef(pm, emp)[0,1]:.3f}, marg-LL {ll_te:.3f} (pooled Beta {ll_base:.3f}), "
              f"median pi0 {np.median(params[te, m, 0]):.3f}")
    with open(a.out, "w") as f:
        for i, p in enumerate(pids):
            f.write(json.dumps({"problem_id": p, "params": params[i].tolist(),
                                "p_successes": ((1 - params[i, :, 0]) * params[i, :, 1] / params[i, :, 1:].sum(1)).tolist()}) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

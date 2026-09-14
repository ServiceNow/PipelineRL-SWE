#!/usr/bin/env python3
"""Predict the POSTERIOR OVER SUCCESS COUNT, not a point difficulty -- so the decay is exact.

Draws within a (problem, route) cell are exchangeable, and the replay samples orderings WITHOUT
replacement from K stored draws. So the entire truth about a cell is one integer: k, the number of
draws that succeed. Given k, the probability the next draw succeeds after n observed failures is
exactly k/(K-n) -- no free parameter.

That matters because the measured k distribution is strongly BIMODAL: on this pool 49%/34% of
problems sit at k=0/k=6 for the scout, 9%/64% for gpt-oss-120b. Most problems are always-solved or
never-solved. A point estimate theta-hat ~ 0.5 on such a cell is not "half likely" -- it is a
coin-flip between 0 and 6, and the two have completely different continuation values. The
Beta-Bernoulli decay theta*sigma/(sigma+n) cannot express that: it is a unimodal belief with one
persistence constant, and the right decay rate is precisely what bimodality controls. This is why
the implied sigma measured 0.69 / 0.80 / 6.45 across difficulty terciles -- that 9x spread IS the
bimodality varying, and a learned per-problem sigma is a one-parameter approximation to it.

So predict P(k = j) for j = 0..K and update by Bayes:

    P(k=j | n failures) proportional to  P(k=j) * C(K-j, n) / C(K, n)
    p(next succeeds | n failures)     =  sum_j P(k=j | n) * j / (K-n)

This has no sigma at all, is exact under exchangeability, subsumes the learned-sigma head, and
makes the Bellman lattice exact because every future belief is closed-form from the same posterior.
Only a per-problem representation can supply it: count beliefs have one prior for the whole pool.

Heads are ridge one-vs-all on the class indicators (n=892 << d=20480, so kernel form), clipped and
renormalised. Fitted on the manifest train split only.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--activations", required=True, help="probe npz (the scout's prefill)")
ap.add_argument("--tensors-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--alpha-grid", default="1e2,1e3,1e4,1e5,1e6")
ap.add_argument("--floor", type=float, default=1e-3)
a = ap.parse_args()

z = np.load(a.activations, allow_pickle=True)
X_all = np.concatenate([z[k].reshape(z[k].shape[0], -1) for k in ("mean", "last") if k in z.files],
                       axis=1)
apids = [str(p) for p in z["problem_ids"]]
ai = {p: i for i, p in enumerate(apids)}

T = Path(a.tensors_dir)
t = np.load(T / "tensors.npz", allow_pickle=True)
pk = [str(p) for p in t["problem_ids"]]
slots = [str(s) for s in t["model_slots"]]
Y, V = t["final_outcome"], t["valid"]
K = Y.shape[2]

man = json.loads((T / "split_manifest.json").read_text())
train_ids = {str(x) for x in man["train_problem_ids"]}
cal_ids = {str(x) for x in man["calibration_problem_ids"]}
keep = [i for i, p in enumerate(pk) if p in ai]
X = X_all[[ai[pk[i]] for i in keep]]
pk = [pk[i] for i in keep]
Y, V = Y[keep], V[keep]
tr = np.array([p in train_ids for p in pk])
cal = np.array([p in cal_ids for p in pk])
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
Xs = (X - mu) / sd
G = Xs @ Xs.T
print(f"{len(pk)} problems, {tr.sum()} train / {cal.sum()} calibration, K={K}, d={X.shape[1]}")


def kridge_fit(idx, y, alpha):
    A = G[np.ix_(idx, idx)] + alpha * np.eye(len(idx))
    return np.linalg.solve(A, y - y.mean()), y.mean()


def kridge_pred(idx, coef, mean):
    return G[:, idx] @ coef + mean


tri = np.where(tr)[0]
P = np.zeros((len(pk), len(slots), K + 1))
for m, s in enumerate(slots):
    kcount = (Y[:, m, :] & V[:, m, :]).sum(1)
    nvalid = V[:, m, :].sum(1)
    # Cells with fewer than K valid draws: rescale the count onto the K-draw scale so every
    # problem's label lives on one support. Rounding is the honest choice -- we cannot invent
    # draws that were never run.
    kk = np.where(nvalid > 0, np.rint(kcount * K / np.maximum(nvalid, 1)), 0).astype(int)
    kk = np.clip(kk, 0, K)
    best = (None, np.inf)
    for alpha in [float(v) for v in a.alpha_grid.split(",") if v.strip()]:
        Q = np.zeros((len(pk), K + 1))
        for j in range(K + 1):
            c, mn = kridge_fit(tri, (kk[tri] == j).astype(float), alpha)
            Q[:, j] = kridge_pred(tri, c, mn)
        Q = np.clip(Q, a.floor, None); Q /= Q.sum(1, keepdims=True)
        # Select on calibration log-loss of the TRUE class: this is a distribution, so it must be
        # scored by a proper scoring rule, not by the AUC of any summary of it.
        ll = -np.log(Q[cal, kk[cal]]).mean()
        if ll < best[1]:
            best = (alpha, ll, Q)
    alpha, ll, Q = best
    P[:, m, :] = Q
    ehat = (Q * np.arange(K + 1)).sum(1) / K
    print(f"  {s:8s} alpha={alpha:g}  cal log-loss {ll:.4f}  "
          f"mean E[k]/K {ehat.mean():.3f} (truth {kk.mean()/K:.3f})  "
          f"P(k=0)+P(k=K) mean {(Q[:, 0] + Q[:, K]).mean():.3f}")

with open(a.out, "w") as fh:
    for i, p in enumerate(pk):
        fh.write(json.dumps({"problem_id": p,
                             "p_k": [[float(x) for x in P[i, m]] for m in range(len(slots))]}) + "\n")
print(f"wrote {a.out}")

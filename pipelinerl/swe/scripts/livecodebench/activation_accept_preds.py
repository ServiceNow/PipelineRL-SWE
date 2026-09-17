#!/usr/bin/env python3
"""Per-problem q = P(correct | weak verifier PASS), predicted from the probe.

The optional-accept policy (§3b-lxii) decides whether to trust an acceptance by comparing the
banked value R*q_m against the surplus of buying more. Until now q_m was a POOL-LEVEL constant
per route -- 0.725 / 0.939 / 0.951 on LCB. That is exactly the mistake this project exists to
document: per-problem q is strongly BIMODAL, and a constant averages it away.

  route    problems with >=1 weak PASS    q=0     q=1    0<q<1    sd
  scout                           708   35.7%   56.6%     7.6%  0.472
  oss20                           817    7.2%   79.8%    13.0%  0.283
  oss120                          856    5.3%   82.5%    12.3%  0.257

So for 35.7% of problems EVERY scout acceptance is false, and for 56.6% every one is real. A
policy told "72.5%" cannot tell those apart; a per-problem prediction can. Same probe, same
forward pass, no extra inference cost -- the activations are already computed for the belief head.

Fitted on the manifest train split only, penalty selected per route on calibration. Problems with
no observed weak PASS on a route fall back to that route's train mean.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--activations", required=True)
ap.add_argument("--tensors-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--alpha-grid", default="1e2,1e3,1e4,1e5,1e6")
a = ap.parse_args()

z = np.load(a.activations, allow_pickle=True)
X_all = np.concatenate([z[k].reshape(z[k].shape[0], -1) for k in ("mean", "last") if k in z.files],
                       axis=1)
ai = {str(p): i for i, p in enumerate(z["problem_ids"])}

T = Path(a.tensors_dir)
t = np.load(T / "tensors.npz", allow_pickle=True)
pk = [str(p) for p in t["problem_ids"]]
slots = [str(s) for s in t["model_slots"]]
F, W, V = t["final_outcome"], t["weak_verifier_outcome"], t["valid"]
keep = [i for i, p in enumerate(pk) if p in ai]
X = X_all[[ai[pk[i]] for i in keep]]
pk = [pk[i] for i in keep]
F, W, V = F[keep], W[keep], V[keep]

man = json.loads((T / "split_manifest.json").read_text())
tr = np.array([p in {str(x) for x in man["train_problem_ids"]} for p in pk])
cal = np.array([p in {str(x) for x in man["calibration_problem_ids"]} for p in pk])
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
Xs = (X - mu) / sd
G = Xs @ Xs.T
print(f"{len(pk)} problems, {tr.sum()} train / {cal.sum()} calibration, d={X.shape[1]}")

Q = np.zeros((len(pk), len(slots)))
for m, s in enumerate(slots):
    sel = V[:, m, :] & W[:, m, :]
    n = sel.sum(1)
    has = n > 0
    q = np.where(has, (F[:, m, :] & sel).sum(1) / np.maximum(n, 1), np.nan)
    pool = float(np.nanmean(q[tr & has]))
    y = np.where(has, q, pool)
    fit = np.where(tr & has)[0]
    best = (None, np.inf)
    for al in [float(v_) for v_ in a.alpha_grid.split(",") if v_.strip()]:
        c = np.linalg.solve(G[np.ix_(fit, fit)] + al * np.eye(len(fit)), y[fit] - y[fit].mean())
        pr = np.clip(G[:, fit] @ c + y[fit].mean(), 0.0, 1.0)
        m_ = cal & has
        mse = float(((pr[m_] - y[m_]) ** 2).mean()) if m_.sum() > 5 else np.inf
        if mse < best[1]:
            best = (al, mse, pr)
    al, mse, pr = best
    # Problems never observed to pass on this route keep the pool prior: we have no evidence.
    Q[:, m] = np.where(has, pr, pool)
    te = ~tr & ~cal & has
    print(f"  {s:8s} alpha={al:g}  cal MSE {mse:.4f}  pool q={pool:.3f}  "
          f"test corr(pred,true)={np.corrcoef(pr[te], y[te])[0,1]:.3f}  spread sd={pr.std():.3f}")

with open(a.out, "w") as fh:
    for i, p in enumerate(pk):
        fh.write(json.dumps({"problem_id": p,
                             "q_accept": [float(x) for x in Q[i]]}) + "\n")
print(f"wrote {a.out}")

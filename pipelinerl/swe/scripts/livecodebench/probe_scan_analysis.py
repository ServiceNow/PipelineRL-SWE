#!/usr/bin/env python3
"""Score every encoder / prompt variant in a probe scan under ONE pipeline.

Three axes, all free at inference (3b-lii showed scaling the probe is not the lever):
  * ENCODER  -- family, size, specialisation. Every candidate is one prefill of a small model.
  * PROMPT   -- the probe has always read activations under the SOLVING prompt. Asking the model
                to judge the problem instead costs the identical forward pass.
  * SCALARS  -- prompt NLL, next-token entropy, next-token max log-prob. Already computed by the
                same pass and previously discarded. Reported alone and as an addition, because a
                signal that is free still has to earn its place against one that is also free.

Identical protocol for every candidate -- same manifest split, same kernel-ridge head, penalty
selected per route on CALIBRATION, scored on TEST -- so only the axis under test varies. Absolute
AUCs here are not the deployed head's (binarised label, different target); the RANKING is the claim.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--scan-dir", required=True)
ap.add_argument("--tensors-dir", required=True)
ap.add_argument("--alpha-grid", default="1e3,1e4,1e5,1e6")
a = ap.parse_args()

T = Path(a.tensors_dir)
t = np.load(T / "tensors.npz", allow_pickle=True)
pk = [str(p) for p in t["problem_ids"]]
slots = [str(s) for s in t["model_slots"]]
Y, V = t["final_outcome"], t["valid"]
rate = np.array([[(Y[i, m, :] & V[i, m, :]).sum() / max(V[i, m, :].sum(), 1)
                  for m in range(len(slots))] for i in range(len(pk))])
man = json.loads((T / "split_manifest.json").read_text())
tr = np.array([p in {str(x) for x in man["train_problem_ids"]} for p in pk])
cal = np.array([p in {str(x) for x in man["calibration_problem_ids"]} for p in pk])
te = ~tr & ~cal


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if y.sum() in (0, len(y)): return np.nan
    r = np.argsort(np.argsort(s)) + 1
    return (r[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (len(y) - y.sum()))


def score(X):
    """Mean test AUC over routes, penalty chosen per route on calibration."""
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xs = (X - mu) / sd
    G = Xs @ Xs.T
    tri = np.where(tr)[0]
    out = []
    for m in range(len(slots)):
        yb = (rate[:, m] > 0.5).astype(float)
        best = (-1, None)
        for al in [float(v) for v in a.alpha_grid.split(",") if v.strip()]:
            c = np.linalg.solve(G[np.ix_(tri, tri)] + al * np.eye(len(tri)), yb[tri] - yb[tri].mean())
            pr = G[:, tri] @ c + yb[tri].mean()
            v = auc(yb[cal], pr[cal])
            if v == v and v > best[0]: best = (v, pr)
        out.append(auc(yb[te], best[1][te]))
    return out


def load_npz(f):
    z = np.load(f, allow_pickle=True)
    ai = {str(p): i for i, p in enumerate(z["problem_ids"])}
    idx = [ai[p] for p in pk]
    X = np.concatenate([z[q].reshape(z[q].shape[0], -1) for q in ("mean", "last")], axis=1)[idx]
    sc = z["scalars"][idx] if "scalars" in z.files else None
    return X, sc


files = sorted(Path(a.scan_dir).glob("*.npz"))
print(f"{len(files)} candidates in {a.scan_dir}\n")
print(f"{'candidate':24s}{'features':>10s}" + "".join(f"{('AUC '+s):>12s}" for s in slots) + f"{'mean':>9s}")
print("-" * (34 + 12 * len(slots) + 9))
rowsout = []
for f in files:
    X, sc = load_npz(f)
    aucs = score(X)
    rowsout.append((f.stem, "activations", np.mean(aucs)))
    print(f"{f.stem:24s}{X.shape[1]:10d}" + "".join(f"{v:12.4f}" for v in aucs) + f"{np.mean(aucs):9.4f}")
    if sc is not None and np.isfinite(sc).all():
        s_only = score(sc)
        both = score(np.c_[X, sc * X.std() / (sc.std(0) + 1e-8)])
        print(f"{'  + free scalars only':24s}{sc.shape[1]:10d}"
              + "".join(f"{v:12.4f}" for v in s_only) + f"{np.mean(s_only):9.4f}")
        print(f"{'  + activations+scalars':24s}{X.shape[1]+sc.shape[1]:10d}"
              + "".join(f"{v:12.4f}" for v in both) + f"{np.mean(both):9.4f}")
        rowsout += [(f.stem, "scalars only", np.mean(s_only)), (f.stem, "both", np.mean(both))]

if rowsout:
    print("\nranked by mean test AUC:")
    for n, k, v in sorted(rowsout, key=lambda r: -r[2])[:12]:
        print(f"  {v:.4f}  {n} ({k})")

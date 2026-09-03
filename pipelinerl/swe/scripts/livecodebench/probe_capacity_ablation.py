#!/usr/bin/env python3
"""Is a LINEAR probe on frozen activations leaving anything on the table?

The method uses logistic/ridge regression on the scout's pre-generation state. That choice
needs justifying rather than assuming, and the cheap way to test it is more capacity on the
SAME features: if a nonlinear head does not beat a linear one, "linear is sufficient" becomes a
measured claim.

This is deliberately not the 8B LoRA comparison. The LoRA asks a different and less central
question -- whether a richer *representation* helps -- and costs a reachable-dataset rebuild
plus ~83 GPU-minutes. Capacity on fixed features costs seconds and answers the methodological
objection directly.

Protocol: layer and hyperparameters chosen on the 170-problem calibration split, reported on
the held-out 171. Both targets are covered, since the paper conditions both halves of the
utility rule:
  SUCCESS  per-route solve-at-any-depth and pool solvability, scored by AUC
  COST     log mean completion tokens, scored by R2 against the per-route constant
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipelinerl.swe.scripts.livecodebench.mdp_utils import load_split_manifest, split_indices


def auc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler

    d = np.load(args.activations, allow_pickle=True)
    PRE = d["pre"]                      # load once: npz members decompress on every access
    apids = [str(p) for p in d["problem_ids"]]
    layers = list(d["layers"])

    T = Path(args.tensors_dir)
    t = np.load(T / "tensors.npz", allow_pickle=True)
    tp = [str(p) for p in t["problem_ids"]]
    slots = [str(s) for s in t["model_slots"]]
    ok = t["final_outcome"] & t["valid"]
    ct = t["completion_tokens"].astype(float)
    valid = t["valid"]

    man = load_split_manifest(T / "split_manifest.json", tp)
    itr, ical, ite = split_indices(man, tp)
    name = {i: "train" for i in itr}
    name.update({i: "cal" for i in ical})
    name.update({i: "test" for i in ite})
    pos = {p: i for i, p in enumerate(tp)}

    common = [p for p in apids if p in pos]
    rows = np.array([apids.index(p) for p in common])
    g = np.array([name[pos[p]] for p in common])
    tr, cal, te = g == "train", g == "cal", g == "test"
    print(f"{len(common)} problems | train {tr.sum()} cal {cal.sum()} test {te.sum()}")
    print(f"layers available: {layers}\n")

    targets = {s: np.array([bool(ok[pos[p], mi, :].any()) for p in common])
               for mi, s in enumerate(slots)}
    targets["pool"] = np.array([bool(ok[pos[p]].any()) for p in common])

    print("=== SUCCESS (AUC on the held-out test split) ===")
    print(f"{'target':<8} {'linear':>18} {'MLP-256':>18} {'MLP-64':>18}")
    for name_t, y in targets.items():
        cells = []
        for kind in ("linear", "mlp256", "mlp64"):
            best = (-1, None)
            for j, L in enumerate(layers):
                X = PRE[rows, j, :]
                sc = StandardScaler().fit(X[tr])
                A, B, C = sc.transform(X[tr]), sc.transform(X[cal]), sc.transform(X[te])
                if kind == "linear":
                    scores_cal, scores_te = [], []
                    for Cval in (0.01, 0.05, 0.2):
                        m = LogisticRegression(max_iter=400, C=Cval).fit(A, y[tr])
                        scores_cal.append((auc(m.predict_proba(B)[:, 1], y[cal]),
                                           m.predict_proba(C)[:, 1]))
                    v, pred = max(scores_cal, key=lambda z: z[0])
                else:
                    h = (256,) if kind == "mlp256" else (64,)
                    accs = []
                    for seed in range(args.seeds):
                        m = MLPClassifier(hidden_layer_sizes=h, alpha=1.0, max_iter=400,
                                          random_state=seed, early_stopping=True)
                        m.fit(A, y[tr])
                        accs.append((auc(m.predict_proba(B)[:, 1], y[cal]),
                                     m.predict_proba(C)[:, 1]))
                    v = float(np.mean([a[0] for a in accs]))
                    pred = np.mean([a[1] for a in accs], axis=0)
                if v > best[0]:
                    best = (v, pred)
            cells.append(auc(best[1], y[te]))
        print(f"{name_t:<8} {cells[0]:18.4f} {cells[1]:18.4f} {cells[2]:18.4f}")

    print("\n=== COST (R2 vs the per-route constant, held-out test split) ===")
    print(f"{'target':<8} {'ridge':>18} {'MLP-256':>18} {'MLP-64':>18}")
    for mi, s in enumerate(slots):
        y = np.array([np.log(max(ct[pos[p], mi, :][valid[pos[p], mi, :]].mean(), 1.0))
                      for p in common])
        const = y[tr].mean()
        cells = []
        for kind in ("ridge", "mlp256", "mlp64"):
            best = (-1e9, None)
            for j, L in enumerate(layers):
                X = PRE[rows, j, :]
                sc = StandardScaler().fit(X[tr])
                A, B, C = sc.transform(X[tr]), sc.transform(X[cal]), sc.transform(X[te])
                if kind == "ridge":
                    m = Ridge(alpha=100.0).fit(A, y[tr])
                    v, pred = -np.abs(m.predict(B) - y[cal]).mean(), m.predict(C)
                else:
                    h = (256,) if kind == "mlp256" else (64,)
                    ps = []
                    for seed in range(args.seeds):
                        m = MLPRegressor(hidden_layer_sizes=h, alpha=1.0, max_iter=400,
                                         random_state=seed, early_stopping=True).fit(A, y[tr])
                        ps.append((-np.abs(m.predict(B) - y[cal]).mean(), m.predict(C)))
                    v = float(np.mean([p[0] for p in ps]))
                    pred = np.mean([p[1] for p in ps], axis=0)
                if v > best[0]:
                    best = (v, pred)
            cells.append(1 - ((y[te] - best[1]) ** 2).sum() / ((y[te] - const) ** 2).sum())
        print(f"{s:<8} {cells[0]:18.4f} {cells[1]:18.4f} {cells[2]:18.4f}")

    print("\nIf the MLP columns do not beat the linear one, 'a linear probe suffices' is")
    print("a measured claim rather than a convenience, and capacity is not the bottleneck.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Does capacity help? Same question, three times the evaluation data.

A single temporal split leaves 171 test problems, which cannot resolve the 0.02 AUC gap
between a linear probe and an MLP on the same frozen activations: the interval came back
[-0.0661, +0.1120]. That is underpowered, not negative, and it is the reason "why a linear
probe?" is still an open objection.

The rolling-origin folds already on disk give three contiguous date-block splits over the same
892 problems, pooling to 535 test problems at no collection cost. The folds file carries ONLY
problem ids and date boundaries -- no outcomes -- so although it was written alongside the
corrupted collection it is label-independent and is applied here to the CLEAN tensors.

Statistics: each fold fits on its own train, selects layer and hyperparameters on its own
calibration, and predicts its own test. Scores from different folds are not mutually
calibrated, so AUCs are computed WITHIN fold and pooled as a size-weighted mean. The bootstrap
resamples problems within each fold and recomputes that weighted mean, which keeps the
resampling unit the problem while never comparing scores across folds.

Decides one thing: if capacity does not help across folds, the linear choice is measured rather
than assumed and the 8B LoRA stays optional. If it does help, the LoRA is worth its
reachable-dataset rebuild plus ~83 GPU-minutes, and the parity claim needs restating.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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
    ap.add_argument("--tensors-dir", required=True, help="CLEAN tensors")
    ap.add_argument("--folds", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler

    d = np.load(args.activations, allow_pickle=True)
    PRE = d["pre"]                     # load once; npz members decompress on every access
    apids = [str(p) for p in d["problem_ids"]]
    layers = list(d["layers"])

    T = Path(args.tensors_dir)
    t = np.load(T / "tensors.npz", allow_pickle=True)
    tp = [str(p) for p in t["problem_ids"]]
    slots = [str(s) for s in t["model_slots"]]
    ok = t["final_outcome"] & t["valid"]
    ct = t["completion_tokens"].astype(float)
    valid = t["valid"]
    pos = {p: i for i, p in enumerate(tp)}
    apos = {p: i for i, p in enumerate(apids)}

    folds = json.loads(Path(args.folds).read_text())["folds"]
    universe = set(tp) & set(apids)
    print(f"{len(folds)} folds | clean tensors {len(tp)} problems | activations {len(apids)}")

    targets = {s: (lambda mi: (lambda p: bool(ok[pos[p], mi, :].any())))(mi)
               for mi, s in enumerate(slots)}
    targets["pool"] = lambda p: bool(ok[pos[p]].any())

    def features(ids, j):
        return PRE[np.array([apos[p] for p in ids]), j, :]

    def run_success(tgt_fn):
        """Per fold: (test ids, linear scores, mlp scores, labels)."""
        out = []
        for fo in folds:
            tr = [p for p in map(str, fo["train_problem_ids"]) if p in universe]
            ca = [p for p in map(str, fo["calibration_problem_ids"]) if p in universe]
            teI = [p for p in map(str, fo["test_problem_ids"]) if p in universe]
            ytr = np.array([tgt_fn(p) for p in tr])
            yca = np.array([tgt_fn(p) for p in ca])
            yte = np.array([tgt_fn(p) for p in teI])
            preds = {}
            for kind in ("linear", "mlp"):
                best = (-1, None)
                for j, _ in enumerate(layers):
                    Xtr, Xca, Xte = features(tr, j), features(ca, j), features(teI, j)
                    sc = StandardScaler().fit(Xtr)
                    A, B, C = sc.transform(Xtr), sc.transform(Xca), sc.transform(Xte)
                    if kind == "linear":
                        for Cv in (0.01, 0.05, 0.2):
                            m = LogisticRegression(max_iter=400, C=Cv).fit(A, ytr)
                            v = auc(m.predict_proba(B)[:, 1], yca)
                            if v == v and v > best[0]:
                                best = (v, m.predict_proba(C)[:, 1])
                    else:
                        ps = []
                        for s_ in range(args.seeds):
                            m = MLPClassifier(hidden_layer_sizes=(64,), alpha=1.0, max_iter=400,
                                              random_state=s_, early_stopping=True).fit(A, ytr)
                            ps.append((auc(m.predict_proba(B)[:, 1], yca),
                                       m.predict_proba(C)[:, 1]))
                        v = float(np.nanmean([p[0] for p in ps]))
                        if v == v and v > best[0]:
                            best = (v, np.mean([p[1] for p in ps], axis=0))
                preds[kind] = best[1]
            out.append((preds["linear"], preds["mlp"], yte))
        return out

    rng = np.random.default_rng(0)

    def pooled(per_fold):
        w = np.array([len(y) for _, _, y in per_fold], float)
        w /= w.sum()
        a_l = np.nansum([w[i] * auc(f[0], f[2]) for i, f in enumerate(per_fold)])
        a_m = np.nansum([w[i] * auc(f[1], f[2]) for i, f in enumerate(per_fold)])
        diffs = []
        for _ in range(args.boot):
            acc = 0.0
            okd = True
            for i, (pl, pm, y) in enumerate(per_fold):
                b = rng.integers(0, len(y), len(y))
                if y[b].all() or not y[b].any():
                    okd = False
                    break
                acc += w[i] * (auc(pm[b], y[b]) - auc(pl[b], y[b]))
            if okd:
                diffs.append(acc)
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return a_l, a_m, a_m - a_l, lo, hi, (diffs > 0).mean(), int(w.size)

    print(f"\n=== SUCCESS: linear vs MLP-64, pooled over folds "
          f"({sum(len([p for p in map(str, f['test_problem_ids']) if p in universe]) for f in folds)} test problems) ===")
    print(f"{'target':<8} {'linear':>8} {'MLP-64':>8} {'diff':>9} {'95% CI':>22} {'P(MLP better)':>14}")
    for name, fn in targets.items():
        pf = run_success(fn)
        a_l, a_m, dd, lo, hi, p, _ = pooled(pf)
        print(f"{name:<8} {a_l:8.4f} {a_m:8.4f} {dd:+9.4f} [{lo:+8.4f},{hi:+8.4f}] {p:14.3f}")

    print("\n=== COST: ridge vs MLP-64, pooled R2 vs each fold's own constant ===")
    print(f"{'route':<8} {'ridge R2':>10} {'MLP R2':>10}")
    for mi, s in enumerate(slots):
        def ylen(p):
            x = ct[pos[p], mi, :][valid[pos[p], mi, :]]
            return np.log(max(x.mean(), 1.0)) if len(x) else 0.0
        num_r = num_m = den = 0.0
        for fo in folds:
            tr = [p for p in map(str, fo["train_problem_ids"]) if p in universe]
            ca = [p for p in map(str, fo["calibration_problem_ids"]) if p in universe]
            teI = [p for p in map(str, fo["test_problem_ids"]) if p in universe]
            ytr = np.array([ylen(p) for p in tr]); yca = np.array([ylen(p) for p in ca])
            yte = np.array([ylen(p) for p in teI]); const = ytr.mean()
            got = {}
            for kind in ("ridge", "mlp"):
                best = (-1e9, None)
                for j, _ in enumerate(layers):
                    Xtr, Xca, Xte = features(tr, j), features(ca, j), features(teI, j)
                    sc = StandardScaler().fit(Xtr)
                    A, B, C = sc.transform(Xtr), sc.transform(Xca), sc.transform(Xte)
                    if kind == "ridge":
                        m = Ridge(alpha=100.0).fit(A, ytr)
                        v, pr = -np.abs(m.predict(B) - yca).mean(), m.predict(C)
                    else:
                        ps = []
                        for s_ in range(args.seeds):
                            m = MLPRegressor(hidden_layer_sizes=(64,), alpha=1.0, max_iter=400,
                                             random_state=s_, early_stopping=True).fit(A, ytr)
                            ps.append((-np.abs(m.predict(B) - yca).mean(), m.predict(C)))
                        v = float(np.mean([q[0] for q in ps]))
                        pr = np.mean([q[1] for q in ps], axis=0)
                    if v > best[0]:
                        best = (v, pr)
                got[kind] = best[1]
            num_r += ((yte - got["ridge"]) ** 2).sum()
            num_m += ((yte - got["mlp"]) ** 2).sum()
            den += ((yte - const) ** 2).sum()
        print(f"{s:<8} {1 - num_r / den:10.4f} {1 - num_m / den:10.4f}")

    print("\nIf the success CIs straddle zero across 535 problems, capacity is not the")
    print("bottleneck and the linear choice is justified rather than merely convenient.")


if __name__ == "__main__":
    main()

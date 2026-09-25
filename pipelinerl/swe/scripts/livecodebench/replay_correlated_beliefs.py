#!/usr/bin/env python3
"""Idea A, second gate: does letting a failure on route X lower beliefs about route Y win at matched spend?

Verifier regime: every draw's pass/fail is observed and a pass ends the episode. Two beliefs, both
logistic regressions fit on TRAIN-split histories (random route orders, stopping at first success):
  independent  P(next draw of m passes) ~ logit prior_m + own failures on m          (RoR-style)
  correlated   ... + failures observed on the OTHER routes                            (Idea A)
The policy is identical for both: draw the route maximising P*V - c_m (learned cost head) and give up
when that is <= 0. Budgets are matched by a calibration-split value grid mixed to the target spend;
test is scored once with problem-level paired bootstrap CIs.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def feats(lp, own, other, correlated):
    return [lp, np.log1p(own)] + ([np.log1p(other)] if correlated else [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--orderings", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--budgets", default="0.02,0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-route", action="store_true", help="fit one belief per target route")
    a = ap.parse_args()
    T = Path(a.tensors_dir)
    t = np.load(T / "tensors.npz", allow_pickle=True)
    ok = (t["final_outcome"] & t["valid"]).astype(bool); valid = t["valid"].astype(bool)
    slots = [str(s) for s in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]
    price = dict(kv.split("=") for kv in a.prices.split(","))
    real = np.stack([(t["prompt_tokens"][:, m] + t["completion_tokens"][:, m]) * float(price[s]) / 1e6
                     for m, s in enumerate(slots)], 1)
    cp = {json.loads(l)["problem_id"]: json.loads(l) for l in open(T / "content_preds.jsonl")}
    cc = {json.loads(l)["problem_id"]: json.loads(l) for l in open(T / "cost_preds.jsonl")}
    LP = np.array([[logit(cp[p]["p_successes"][m]) for m in range(len(slots))] for p in pids])
    C = np.array([cc[p]["expected_costs"][:len(slots)] for p in pids])
    sp = json.loads((T / "split_manifest.json").read_text())
    idx = {k: np.array([pids.index(p) for p in sp[f"{k}_problem_ids"]]) for k in ("train", "calibration", "test")}
    rng = np.random.default_rng(a.seed)
    nP, nM, nD = valid.shape
    order = np.array([[[rng.permutation(nD) for _ in range(nM)] for _ in range(a.orderings)] for _ in range(nP)])

    def draws(i, o, m):
        return [d for d in order[i, o, m] if valid[i, m, d]]

    # ---- fit both beliefs on train histories (random route choice, stop at first success)
    models = {}
    for name, corr in (("independent", False), ("correlated", True)):
        X, y, tgt = [], [], []
        for i in idx["train"]:
            for o in range(a.orderings):
                q = {m: draws(i, o, m) for m in range(nM)}
                f = np.zeros(nM)
                for _ in range(a.max_steps):
                    av = [m for m in range(nM) if len(q[m]) > f[m]]
                    if not av:
                        break
                    m = int(rng.choice(av)); d = q[m][int(f[m])]
                    X.append(feats(LP[i, m], f[m], f.sum() - f[m], corr)); y.append(ok[i, m, d]); tgt.append(m)
                    if ok[i, m, d]:
                        break
                    f[m] += 1
        X, y, tgt = np.array(X), np.array(y), np.array(tgt)
        if a.per_route:
            # one belief per TARGET route: a cheap route's own failures can be far less informative
            # about its next draw than a strong route's, which one shared coefficient cannot express
            models[name] = [LogisticRegression(C=1.0, max_iter=2000).fit(X[tgt == m], y[tgt == m]) for m in range(nM)]
        else:
            models[name] = [LogisticRegression(C=1.0, max_iter=2000).fit(X, y)] * nM
        for m in range(nM):
            print(f"{name} {slots[m]:<9} coef {np.round(models[name][m].coef_[0], 2)}", flush=True)

    def run(ids, name, V):
        corr = name == "correlated"; clf = models[name]
        acc = np.zeros(len(ids)); cost = np.zeros(len(ids))
        for k, i in enumerate(ids):
            for o in range(a.orderings):
                q = {m: draws(i, o, m) for m in range(nM)}; f = np.zeros(nM); spent = 0.0; solved = False
                for _ in range(a.max_steps):
                    av = [m for m in range(nM) if len(q[m]) > f[m]]
                    if not av:
                        break
                    P = np.array([clf[m].predict_proba([feats(LP[i, m], f[m], f.sum() - f[m], corr)])[0, 1] for m in av])
                    net = P * V - C[i, av]
                    if net.max() <= 0:
                        break
                    m = av[int(np.argmax(net))]; d = q[m][int(f[m])]; spent += real[i, m, d]
                    if ok[i, m, d]:
                        solved = True; break
                    f[m] += 1
                acc[k] += solved / a.orderings; cost[k] += spent * 100 / a.orderings
        return acc, cost

    Vs = np.geomspace(1e-4, 1.0, 26)
    out = {"slots": slots, "per_route": a.per_route,
           "coef": {k: [c.coef_[0].tolist() for c in v] for k, v in models.items()}, "results": {}}
    test_res = {}
    for name in models:
        cal = [(0.0, 0.0, None)] + [(*[x.mean() for x in run(idx["calibration"], name, V)][::-1], V) for V in Vs]
        cal = sorted(cal)                      # (cal_cost, cal_acc, V)
        cache = {}
        def test_at(V):
            if V not in cache:
                cache[V] = run(idx["test"], name, V) if V is not None else (np.zeros(len(idx["test"])),) * 2
            return cache[V]
        for b in [float(x) for x in a.budgets.split(",")]:
            best = None
            for lo in cal:
                for hi in cal:
                    if lo[0] <= b <= hi[0] and hi[0] > lo[0]:
                        w = (b - lo[0]) / (hi[0] - lo[0]); acc = (1 - w) * lo[1] + w * hi[1]
                    elif lo is hi and lo[0] <= b:
                        w = 0.0; acc = lo[1]
                    else:
                        continue
                    if best is None or acc > best[0]:
                        best = (acc, lo[2], hi[2], w)
            _, vlo, vhi, w = best
            (al, cl), (ah, ch) = test_at(vlo), test_at(vhi)
            test_res[(name, b)] = ((1 - w) * al + w * ah, (1 - w) * cl + w * ch)
        print(f"{name}: calibration grid done", flush=True)
    print(f"\n{T.name} test ({len(idx['test'])} problems); correlated - independent, 95% paired CI")
    for b in [float(x) for x in a.budgets.split(",")]:
        (A1, C1), (A2, C2) = test_res[("independent", b)], test_res[("correlated", b)]
        d = A2 - A1; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
        line = (f"  {b:>5}c  independent {A1.mean()*100:5.1f}%@{C1.mean():.3f}  correlated {A2.mean()*100:5.1f}%@{C2.mean():.3f}"
                f"  delta {d.mean()*100:+.1f} [{np.percentile(bs,2.5)*100:+.1f},{np.percentile(bs,97.5)*100:+.1f}]")
        print(line, flush=True)
        out["results"][str(b)] = {"independent": [A1.mean(), C1.mean()], "correlated": [A2.mean(), C2.mean()],
                                  "delta": d.mean(), "ci": [np.percentile(bs, 2.5), np.percentile(bs, 97.5)]}
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

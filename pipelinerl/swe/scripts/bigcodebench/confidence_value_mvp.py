#!/usr/bin/env python3
"""Is a model's own confidence in its sample worth collecting? Minimal test.

Same draws, two ways to pick which one to submit:
  prior only   P(correct) = the probe's per-problem prior for that attempt's model
  + confidence P(correct) = per-model logistic regression on [logit prior, confidence]
Confidence is fixed in advance as the answer's mean token logprob -- no search over scores.
Each model's regression is fit on 4/5 of the problems and applied to the held-out 1/5.

Two held sets:
  within   4 draws from one model (the prior is identical, so "prior only" is a random pick)
  across   1 draw from each model in the pilot (the prior picks the model it trusts most)
Spend is identical between the two pickers by construction, so any gain is free accuracy.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot-dir", required=True)
    ap.add_argument("--tensors-dir", required=True, help="for the probe prior (content_preds.jsonl)")
    ap.add_argument("--routes", required=True)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    routes = a.routes.split(",")
    T = Path(a.tensors_dir)
    slots = [str(s) for s in np.load(T / "tensors.npz", allow_pickle=True)["model_slots"]]
    prior = {}
    for line in open(T / "content_preds.jsonl"):
        r = json.loads(line)
        prior[str(r["problem_id"])] = {s: r["p_successes"][slots.index(s)] for s in routes}

    # draws[route][pid] = list of (correct, confidence)
    draws = {m: defaultdict(list) for m in routes}
    for m in routes:
        for f in sorted(Path(a.pilot_dir).glob(f"{m}_*_d*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                lp = [t[1] for t in (r.get("logprobs") or []) if t[1] is not None]
                if r.get("finish_reason") == "error" or len(lp) < 5 or r["problem_id"] not in prior:
                    continue
                draws[m][r["problem_id"]].append((bool(r["resolved"]), float(np.mean(lp))))
    pids = sorted(set.intersection(*[{p for p, v in draws[m].items() if len(v) >= a.draws}
                                     for m in routes]))
    fold = {p: i % 5 for i, p in enumerate(rng.permutation(pids))}
    print(f"{len(pids)} problems with >= {a.draws} logprob draws on every model in {routes}")

    # cross-fitted P(correct | prior, confidence) for every draw
    post = {m: {} for m in routes}
    for m in routes:
        for k in range(5):
            tr = [(logit(prior[p][m]), c, y) for p in pids if fold[p] != k for y, c in draws[m][p]]
            X = np.array([[x, c] for x, c, _ in tr]); y = np.array([t[2] for t in tr])
            mu, sd = X.mean(0), X.std(0) + 1e-9
            clf = LogisticRegression(C=1.0).fit((X - mu) / sd, y)
            for p in pids:
                if fold[p] == k:
                    Z = (np.array([[logit(prior[p][m]), c] for _, c in draws[m][p]]) - mu) / sd
                    post[m][p] = clf.predict_proba(Z)[:, 1]
        w = clf.coef_[0]
        print(f"  {m:<9} last-fold coefficients (standardised): prior {w[0]:+.2f}  confidence {w[1]:+.2f}")

    def report(name, rows):
        rows = np.array(rows)            # columns: prior-only, +confidence, perfect
        d = rows[:, 1] - rows[:, 0]
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f"  {name:<22} prior only {rows[:,0].mean()*100:5.1f}%   + confidence {rows[:,1].mean()*100:5.1f}%"
              f"   perfect {rows[:,2].mean()*100:5.1f}%   gain {d.mean()*100:+.1f} "
              f"[{np.percentile(bs,2.5)*100:+.1f}, {np.percentile(bs,97.5)*100:+.1f}]")

    print(f"\nPick 1 to submit (resampled 20x per problem; 95% CI by problem bootstrap):")
    for m in routes:
        rows = []
        for p in pids:
            v = draws[m][p]; q = post[m][p]; acc = np.zeros(3)
            for _ in range(20):
                S = rng.choice(len(v), a.draws, replace=False)
                y = np.array([v[i][0] for i in S])
                acc += [y[0], y[np.argmax(q[S])], y.max()]  # prior ties -> first = random pick
            rows.append(acc / 20)
        report(f"within {m} ({a.draws})", rows)
    rows = []
    for p in pids:
        acc = np.zeros(3)
        for _ in range(20):
            picks = [(m, rng.integers(len(draws[m][p]))) for m in routes]
            y = np.array([draws[m][p][i][0] for m, i in picks])
            pr = np.array([prior[p][m] for m, _ in picks])
            q = np.array([post[m][p][i] for m, i in picks])
            acc += [y[np.argmax(pr)], y[np.argmax(q)], y.max()]
        rows.append(acc / 20)
    report(f"across ({len(routes)} models)", rows)


if __name__ == "__main__":
    main()

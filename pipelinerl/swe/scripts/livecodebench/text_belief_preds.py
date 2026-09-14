#!/usr/bin/env python3
"""Per-problem beliefs from CHEAP text signals, as the control for the representation claim.

The belief-head result says activation beliefs beat count beliefs. Count beliefs have NO
per-problem prior at all, so that comparison cannot distinguish two very different claims:

  (a) per-problem beliefs help, and any cheap per-problem signal would do;
  (b) the ACTIVATIONS specifically carry something text does not.

This emits the same `p_successes` format from signals that need no forward pass of any model:

  --signal length   problem length alone (2 features: chars, words). The most trivial
                    per-problem signal that exists -- if this recovers the gain, the claim is
                    about the policy having ANY per-problem prior, not about representations.
  --signal tfidf    TF-IDF of the problem statement + logistic regression per route.

Matched to the activation pipeline on everything that is not the feature set: same manifest
train/calibration split, penalty selected per route on calibration by the same criterion, and the
same clipping. Anything else would repeat the rigged-comparison error of 2026-09-xx, where TF-IDF
got RidgeCV over 13 alphas and the activations got a hand-picked constant.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--tensors-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--signal", choices=("length", "tfidf"), required=True)
ap.add_argument("--C-grid", default="0.01,0.03,0.1,0.3,1,3,10,30")
a = ap.parse_args()

T = Path(a.tensors_dir)
t = np.load(T / "tensors.npz", allow_pickle=True)
pk = [str(p) for p in t["problem_ids"]]
slots = [str(s) for s in t["model_slots"]]
Y, V = t["final_outcome"], t["valid"]
text = {}
for line in open(T / "problems.jsonl"):
    r = json.loads(line)
    text[str(r["problem_id"])] = str(r.get("problem_statement", ""))
docs = [text.get(p, "") for p in pk]

man = json.loads((T / "split_manifest.json").read_text())
tr = np.array([p in {str(x) for x in man["train_problem_ids"]} for p in pk])
cal = np.array([p in {str(x) for x in man["calibration_problem_ids"]} for p in pk])

if a.signal == "length":
    X = np.c_[[len(d) for d in docs], [len(d.split()) for d in docs]].astype(float)
    X = StandardScaler().fit(X[tr]).transform(X)
else:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000)
    vec.fit([d for d, k in zip(docs, tr) if k])
    X = vec.transform(docs)
print(f"{a.signal}: {len(pk)} problems, features {X.shape[1]}, "
      f"{tr.sum()} train / {cal.sum()} calibration")

P = np.zeros((len(pk), len(slots)))
for m, s in enumerate(slots):
    y = ((Y[:, m, :] & V[:, m, :]).sum(1) > 0).astype(int)   # "any draw solves it"
    best = (None, np.inf)
    for C in [float(v) for v in a.C_grid.split(",") if v.strip()]:
        lr = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
        q = np.clip(lr.predict_proba(X[cal])[:, 1], 1e-6, 1 - 1e-6)
        ll = -(y[cal] * np.log(q) + (1 - y[cal]) * np.log(1 - q)).mean()
        if ll < best[1]:
            best = (C, ll)
    C, ll = best
    lr = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
    # Scale "solves at all" down to a per-draw rate, matching what p_successes means for the
    # activation head: the train pass@1 of the route, times the relative ranking this gives.
    raw = np.clip(lr.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    rate = float((Y[tr, m, :] & V[tr, m, :]).sum() / max(V[tr, m, :].sum(), 1))
    P[:, m] = np.clip(raw * rate / max(raw[tr].mean(), 1e-9), 1e-4, 1 - 1e-4)
    print(f"  {s:8s} C={C:g}  cal log-loss {ll:.4f}  mean p {P[:, m].mean():.3f} (train rate {rate:.3f})")

with open(a.out, "w") as fh:
    for i, p in enumerate(pk):
        fh.write(json.dumps({"problem_id": p,
                             "p_successes": [float(x) for x in P[i]]}) + "\n")
print(f"wrote {a.out}")

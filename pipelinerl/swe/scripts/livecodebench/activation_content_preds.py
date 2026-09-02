#!/usr/bin/env python3
"""Turn scout activations into the replay's `--content-preds` file.

The sequential policy's beliefs currently come from a Qwen3-Embedding-8B LoRA scoring
rendered state text, at 6.42 encoder calls per episode. A linear probe on the scout's own
last-token hidden state matches or beats that encoder on every head -- oss120 0.8306 vs
0.7529, pool solvability 0.7947 vs 0.7335 -- and costs nothing extra, because the scout
forward pass is already paid under the mandatory-scout protocol.

This fits one probe per route on the TRAIN split and emits static per-problem priors
theta_m(x) in the shape the replay expects. Depth is left to the analytic decay, which is
the factorized structure the project already uses: the plan's own mechanism analysis found
per-problem decay worthless and theta on the expensive route to be where the money is.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--activations", required=True)
ap.add_argument("--tensors-dir", required=True)
ap.add_argument("--layer", type=int, default=18)
ap.add_argument("--C", type=float, default=0.05)
ap.add_argument("--out", required=True)
a = ap.parse_args()

d = np.load(a.activations, allow_pickle=True)
pids = [str(p) for p in d["problem_ids"]]; layers = list(d["layers"])
t = np.load(Path(a.tensors_dir) / "tensors.npz", allow_pickle=True)
tp = [str(p) for p in t["problem_ids"]]; ti = {p: i for i, p in enumerate(tp)}
probs = {str(json.loads(l)["problem_id"]): json.loads(l)
         for l in open(Path(a.tensors_dir) / "problems.jsonl") if l.strip()}
ok = t["final_outcome"] & t["valid"]; slots = [str(s) for s in t["model_slots"]]

keep = [i for i, p in enumerate(pids) if p in ti]
pk = [pids[i] for i in keep]
X = d["pre"][keep][:, layers.index(a.layer)]
tr = np.array([probs[p].get("source_temporal_split") == "train" for p in pk])
print(f"{len(pk)} problems, fitting on {tr.sum()} train, layer {a.layer}")

P = np.zeros((len(pk), len(slots)))
for j, s in enumerate(slots):
    # pass@1 is the next-draw prior the replay decays; it is the theta the policy acts on.
    y = ok[[ti[p] for p in pk], j, 0]
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=a.C).fit(sc.transform(X[tr]), y[tr])
    P[:, j] = clf.predict_proba(sc.transform(X))[:, 1]
    print(f"  {s:8s} train base rate {y[tr].mean():.3f}  mean predicted {P[:, j].mean():.3f}")

with open(a.out, "w") as f:
    for i, p in enumerate(pk):
        f.write(json.dumps({"problem_id": p, "p_successes": [float(x) for x in P[i]]}) + "\n")
print(f"wrote {a.out}")

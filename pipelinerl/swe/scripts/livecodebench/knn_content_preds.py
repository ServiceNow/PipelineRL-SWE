#!/usr/bin/env python3
"""kNN belief head: theta_m(x) from the outcomes of the nearest TRAINING problems.

"Rethinking Predictive Modeling for LLM Routing: When Simple kNN Beats Complex Learned Routers"
(2505.12601) argues nearest-neighbour retrieval beats learned routers. That is a claim about the
BELIEF, not the policy, so the fair test is to emit theta in the same format the learned probe
uses and run the identical replay -- same abstention rule (max_m Q <= 0), same decay, same costs.
kNN does not "do" abstention or resampling; the policy does, and it is unchanged.

Distance is cosine on the same standardised activations the learned head sees, so the comparison
isolates the estimator rather than the representation.
"""
import argparse, json
import numpy as np
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--activations", required=True)
ap.add_argument("--tensors-dir", required=True)
ap.add_argument("--k", type=int, default=25)
ap.add_argument("--out", required=True)
a = ap.parse_args()

z = np.load(a.activations, allow_pickle=True)
apids = [str(p) for p in z["problem_ids"]]
X = np.concatenate([z["content_mean"].reshape(len(apids), -1),
                    z["content_last"].reshape(len(apids), -1)], axis=1).astype(np.float32)

t = np.load(Path(a.tensors_dir) / "tensors.npz", allow_pickle=True)
tp = [str(p) for p in t["problem_ids"]]
ti = {p: i for i, p in enumerate(tp)}
ok = t["final_outcome"] & t["valid"]
slots = [str(s) for s in t["model_slots"]]

man = json.loads((Path(a.tensors_dir) / "split_manifest.json").read_text())
train_ids = {str(x) for x in man["train_problem_ids"]}

pk = [p for p in apids if p in ti]
Xk = X[[apids.index(p) for p in pk]]
tr = np.array([p in train_ids for p in pk])
# pass@1 on draw 0 -- the same target the learned head predicts
Y = np.stack([ok[[ti[p] for p in pk], j, 0].astype(float) for j in range(len(slots))], axis=1)

mu, sd = Xk[tr].mean(0), Xk[tr].std(0) + 1e-8
Z = (Xk - mu) / sd
Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
S = Z @ Z[tr].T                                   # cosine similarity to every training problem
idx_tr = np.where(tr)[0]
P = np.zeros((len(pk), len(slots)))
for i in range(len(pk)):
    sim = S[i].copy()
    if tr[i]:                                     # never retrieve itself
        sim[np.where(idx_tr == i)[0]] = -np.inf
    nb = np.argpartition(-sim, a.k)[: a.k]
    P[i] = Y[idx_tr[nb]].mean(0)
P = np.clip(P, 0.001, 0.999)
print(f"{len(pk)} problems, k={a.k}, neighbours drawn from {tr.sum()} train")
for j, s in enumerate(slots):
    print(f"  {s:8s} train base {Y[tr, j].mean():.3f}  knn mean {P[:, j].mean():.3f}")
with open(a.out, "w") as f:
    for i, p in enumerate(pk):
        f.write(json.dumps({"problem_id": p, "p_successes": [float(x) for x in P[i]]}) + "\n")
print(f"wrote {a.out}")

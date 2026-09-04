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
ap.add_argument("--activations", help="one file used for every route (the cheap scout probe)")
ap.add_argument("--activations-file", action="append", default=[], metavar="ROUTE=PATH",
                help=("Per-candidate mode: each route's head is fit on its OWN activations, "
                      "i.e. arXiv 2602.09924's method. Requires weights for every pool member, "
                      "so it is not deployable on an API-served pool -- run for completeness."))
ap.add_argument("--rich", action="store_true", help="Concatenate ALL layers x {mean,last} instead of one layer of one readout. The single-layer probe captured only 11-40% of the between-problem variance ceiling on TACO; concatenating roughly doubles cost R2 there (oss20 0.093->0.216) and lifts LCB too (oss120 belief AUC 0.792->0.864, cost R2 0.700->0.792). The gain is FEATURES, not capacity: an MLP on the single layer collapses to negative R2.")
ap.add_argument("--readout", default="pre",
                help="which stored readout to use; 'mean' is the template-neutral one")
ap.add_argument("--tensors-dir", required=True)
ap.add_argument("--layer", type=int, default=18,
                help="absolute layer for the single-file mode")
ap.add_argument("--layer-frac", type=float, default=0.0,
                help=("pick each model's layer by RELATIVE depth instead. Required in "
                      "per-candidate mode: gpt-oss-20b has 24 layers and no layer 18, so "
                      "absolute indices do not transfer across model families."))
ap.add_argument("--C", type=float, default=0.05)
ap.add_argument("--out", required=True)
a = ap.parse_args()

def _load(path):
    z = np.load(path, allow_pickle=True)
    if a.rich:
        parts = [z[k].reshape(z[k].shape[0], -1) for k in ("mean", "last") if k in z.files]
        return [str(p) for p in z["problem_ids"]], np.concatenate(parts, axis=1)[:, None, :], [1]
    key = a.readout if a.readout in z.files else "pre"
    return [str(p) for p in z["problem_ids"]], z[key], list(z["layers"])

per_route = {}
for spec in a.activations_file:
    r, _, path = spec.partition("=")
    per_route[r] = _load(path)
if not per_route and not a.activations:
    raise SystemExit("need --activations or --activations-file")
pids, Xall, layers = _load(a.activations) if a.activations else next(iter(per_route.values()))
t = np.load(Path(a.tensors_dir) / "tensors.npz", allow_pickle=True)
tp = [str(p) for p in t["problem_ids"]]; ti = {p: i for i, p in enumerate(tp)}
probs = {str(json.loads(l)["problem_id"]): json.loads(l)
         for l in open(Path(a.tensors_dir) / "problems.jsonl") if l.strip()}
ok = t["final_outcome"] & t["valid"]; slots = [str(s) for s in t["model_slots"]]

keep = [i for i, p in enumerate(pids) if p in ti]
pk = [pids[i] for i in keep]
def _pick(ls):
    if a.rich:
        return 0
    if a.layer_frac:
        return min(range(len(ls)), key=lambda i: abs(ls[i] / max(ls) - a.layer_frac))
    return ls.index(a.layer)

X = Xall[keep][:, _pick(layers)]
# Use the canonical split manifest, NOT source_temporal_split. They coincide on LCB but
# not on TACO, whose dates are 79.8% Unix-epoch sentinels and whose manifest is therefore a
# random split -- fitting on the temporal "train" there would train on manifest test.
_man = json.loads((Path(a.tensors_dir) / "split_manifest.json").read_text())
_train_ids = {str(x) for x in _man["train_problem_ids"]}
tr = np.array([p in _train_ids for p in pk])
print(f"{len(pk)} problems, fitting on {tr.sum()} train, layer {a.layer}")

P = np.zeros((len(pk), len(slots)))
for j, s in enumerate(slots):
    # pass@1 is the next-draw prior the replay decays; it is the theta the policy acts on.
    y = ok[[ti[p] for p in pk], j, 0]
    if s in per_route:
        rp, rX, rl = per_route[s]
        ridx = {q: i for i, q in enumerate(rp)}
        Xr = rX[[ridx[q] for q in pk]][:, _pick(rl)]
    else:
        Xr = X
    sc = StandardScaler().fit(Xr[tr])
    clf = LogisticRegression(max_iter=2000,
                             C=a.C / max(1, Xr.shape[1] // 2560)).fit(sc.transform(Xr[tr]), y[tr])
    P[:, j] = clf.predict_proba(sc.transform(Xr))[:, 1]
    print(f"  {s:8s} train base rate {y[tr].mean():.3f}  mean predicted {P[:, j].mean():.3f}")

with open(a.out, "w") as f:
    for i, p in enumerate(pk):
        f.write(json.dumps({"problem_id": p, "p_successes": [float(x) for x in P[i]]}) + "\n")
print(f"wrote {a.out}")

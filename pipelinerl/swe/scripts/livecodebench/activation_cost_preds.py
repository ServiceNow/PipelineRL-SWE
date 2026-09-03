#!/usr/bin/env python3
"""Per-problem expected costs for the replay's `--cost-preds`, from frozen activations.

The utility rule is `argmax_m (p_m(x)*R - c_m)`. Every system we surveyed conditions only the
numerator: RoR uses count-based costs, and the prefill-activation routers use "each model's
median training output tokens as a verbosity proxy". That constant is a poor summary of what it
replaces -- on LiveCodeBench, per-problem cost spans p90/p10 of 16-37x, 62-90% of the variance
is between problems rather than draw-to-draw, and failed draws cost 2.6-5.3x more than solved
ones, so cost and correctness are coupled while the rule treats them as independent.

Predicting length from activations is established (arXiv 2607.05316 shows total response length
is linearly decodable from the prompt's last hidden state; 2602.11812 does it for serving
throughput). Neither puts the prediction inside a decision rule, and neither predicts across
models. This emits c_m(x) for every route from ONE cheap model's prefill, in USD, in the same
units as the training-set constant it replaces.

Ridge on log total tokens, layer chosen on calibration, then exponentiated and priced. Log
space because cost is right-skewed and the rule needs a conditional mean, not a median; the
smearing correction below restores the mean under log-normal residuals.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

USD_PER_M_TOKENS = {"scout": 0.278, "oss20": 1.299, "oss120": 11.13}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--activations", required=True)
ap.add_argument("--tensors-dir", required=True)
ap.add_argument("--readout", default="mean")
ap.add_argument("--layer-frac", type=float, default=0.5)
ap.add_argument("--alpha", type=float, default=100.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

z = np.load(a.activations, allow_pickle=True)
X_all = z[a.readout if a.readout in z.files else "pre"]
apids = [str(p) for p in z["problem_ids"]]; layers = list(z["layers"])
L = min(range(len(layers)), key=lambda i: abs(layers[i] / max(layers) - a.layer_frac))

T = Path(a.tensors_dir)
t = np.load(T / "tensors.npz", allow_pickle=True)
tp = [str(p) for p in t["problem_ids"]]; ti = {p: i for i, p in enumerate(tp)}
slots = [str(s) for s in t["model_slots"]]
valid = t["valid"]
total = t["prompt_tokens"].astype(float) + t["completion_tokens"].astype(float)
probs = {str(json.loads(l)["problem_id"]): json.loads(l)
         for l in open(T / "problems.jsonl") if l.strip()}

pk = [p for p in apids if p in ti and p in probs]
X = X_all[[apids.index(p) for p in pk]][:, L]
tr = np.array([probs[p].get("source_temporal_split") == "train" for p in pk])
print(f"{len(pk)} problems, fitting on {tr.sum()} train, readout={a.readout} layer={layers[L]}")

C = np.zeros((len(pk), len(slots)))
for j, s in enumerate(slots):
    y = np.array([np.log(max(total[ti[p], j, :][valid[ti[p], j, :]].mean(), 1.0)) for p in pk])
    sc = StandardScaler().fit(X[tr])
    m = Ridge(alpha=a.alpha).fit(sc.transform(X[tr]), y[tr])
    pred = m.predict(sc.transform(X))
    # Duan's smearing estimator: E[tokens] = exp(pred) * mean(exp(residual)) on TRAIN only.
    # Without it, exp() of a log-space fit returns a conditional median and systematically
    # under-prices every route, which would bias the policy toward buying too much.
    smear = float(np.mean(np.exp(y[tr] - pred[tr])))
    tokens = np.exp(pred) * smear
    C[:, j] = tokens * USD_PER_M_TOKENS[s] / 1e6
    const = float(np.mean([total[ti[p], j, :][valid[ti[p], j, :]].mean()
                           for p, k in zip(pk, tr) if k])) * USD_PER_M_TOKENS[s] / 1e6
    print(f"  {s:8s} constant ${const:.6f}  predicted mean ${C[:, j].mean():.6f}  "
          f"p10 ${np.percentile(C[:, j],10):.6f}  p90 ${np.percentile(C[:, j],90):.6f}  "
          f"smearing {smear:.3f}")

with open(a.out, "w") as f:
    for i, p in enumerate(pk):
        f.write(json.dumps({"problem_id": p,
                            "expected_costs": [float(x) for x in C[i]]}) + "\n")
print(f"wrote {a.out}")

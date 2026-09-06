#!/usr/bin/env python3
"""Diagnostic cost heads that interpolate between the per-route constant and the oracle.

DIAGNOSTIC ONLY -- the oracle end reads test-split outcomes and is not a deployable policy.
Its purpose is to answer a question the real head cannot: is query-conditioned cost failing on
TACO because the *idea* does not pay there, or because our estimate is too weak? Blending

    c_lambda(x) = (1 - lambda) * c_const + lambda * c_true(x)

sweeps exactly that axis. lambda=0 is what RoR spends (the per-route training mean), lambda=1 is
perfect foresight, and the frontier as a function of lambda says how good a cost head would have
to be before conditioning pays -- a requirement stated in units a future method can be measured
against, rather than a post-hoc excuse.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

USD_PER_M_TOKENS = {"scout": 0.278, "oss20": 1.299, "oss120": 11.13}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--tensors-dir", required=True)
ap.add_argument("--lam", type=float, required=True, help="0 = per-route constant, 1 = oracle")
ap.add_argument("--out", required=True)
a = ap.parse_args()

z = np.load(Path(a.tensors_dir) / "tensors.npz")
ids = [str(p) for p in z["problem_ids"]]
ct, valid = z["completion_tokens"], z["valid"]
slots = [str(s) for s in z["model_slots"]]
man = json.loads((Path(a.tensors_dir) / "split_manifest.json").read_text())
train = {str(x) for x in man["train_problem_ids"]}

rows = []
for j, s in enumerate(slots):
    per = np.array([
        ct[i, j][valid[i, j]].mean() if valid[i, j].any() else np.nan for i in range(len(ids))
    ])
    const = float(np.nanmean([per[i] for i, p in enumerate(ids) if p in train]))
    per = np.where(np.isnan(per), const, per)
    rows.append(((1.0 - a.lam) * const + a.lam * per) * USD_PER_M_TOKENS[s] / 1e6)

C = np.stack(rows, axis=1)
with open(a.out, "w") as f:
    for i, p in enumerate(ids):
        f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in C[i]]}) + "\n")
print(f"lambda={a.lam}: " + "  ".join(
    f"{s} mean ${C[:, j].mean():.5f} p10 ${np.percentile(C[:, j], 10):.5f} "
    f"p90 ${np.percentile(C[:, j], 90):.5f}" for j, s in enumerate(slots)))
print(f"wrote {a.out}")

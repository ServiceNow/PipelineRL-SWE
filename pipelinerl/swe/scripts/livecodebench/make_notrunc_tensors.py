#!/usr/bin/env python3
"""Build the no-truncation counterfactual tensors: mark every at-cap draw invalid.

A draw that hits the token cap is a runaway generation -- it costs the full cap and, on both
LiveCodeBench and TACO, essentially never solves the problem (LCB scout 0.0%, oss20 3.6%; TACO
0.0% on both). Cheap routes do this far more than the 120B does, so part of any cost advantage
over a count-based baseline could be "the probe learned which prompts make small models ramble"
rather than difficulty prediction. This builds the world where that channel does not exist, so
the two explanations can be separated (paper section 6.9).

It is a counterfactual, not a fix: deleting the runaway draws deletes a route entirely on the
problems where every draw ran away (4 on LCB oss20, 14 on TACO oss20), so the ablated world is
slightly easier than a world with a genuinely larger cap. Refit the cost head against the output
directory before replaying, or the policy is charged for spending it can no longer incur.
"""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--tensors-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cap", type=float, default=0.0,
                help="Completion length counted as truncated. Default 0 infers it from the "
                     "maximum observed completion, which is the cap the collection ran under.")
a = ap.parse_args()

src, dst = Path(a.tensors_dir), Path(a.out)
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "tensors.npz":
        shutil.copy2(f, dst / f.name)

z = np.load(src / "tensors.npz")
d = {k: z[k] for k in z.files}
cap = a.cap or float(d["completion_tokens"].max())
drop = (d["completion_tokens"] >= cap - 1) & d["valid"]
d["valid"] = d["valid"] & ~drop
np.savez_compressed(dst / "tensors.npz", **d)

print(f"cap {cap:.0f}: dropped {int(drop.sum())} of {int(z['valid'].sum())} valid draws")
for i, s in enumerate(list(z["model_slots"])):
    empty = int((d["valid"][:, i, :].sum(axis=1) == 0).sum())
    print(f"  {s:8s} {int(z['valid'][:, i, :].sum()):5d} -> {int(d['valid'][:, i, :].sum()):5d} valid"
          f"   ({int(drop[:, i, :].sum()):4d} dropped, {empty} problems left with no draw at all)")
print(f"wrote {dst}/tensors.npz")

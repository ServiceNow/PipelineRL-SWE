"""Compare regenerated activations with the original ones (internal check, needs the original files).

Usage: python tools/compare_activations.py <regenerated.npz> <original.npz>
Prints, per readout and layer, the median cosine similarity and the median relative L2 error over the shared problems.
"""
import sys
import numpy as np

a, b = (np.load(f, allow_pickle=True) for f in sys.argv[1:3])
ib = {str(p): i for i, p in enumerate(b["problem_ids"])}
ids = [str(p) for p in a["problem_ids"] if str(p) in ib]; ia = {str(p): i for i, p in enumerate(a["problem_ids"])}
print(len(ids), "shared problems; layers", list(a["layers"]), "vs", list(b["layers"]))
for k in ("mean", "last"):
    X = a[k][[ia[p] for p in ids]].astype(np.float64); Y = b[k][[ib[p] for p in ids]].astype(np.float64)
    cos = (X * Y).sum(-1) / (np.linalg.norm(X, axis=-1) * np.linalg.norm(Y, axis=-1))
    rel = np.linalg.norm(X - Y, axis=-1) / np.linalg.norm(Y, axis=-1)
    print(k, "cos median by layer", np.round(np.median(cos, 0), 5), "min", round(float(cos.min()), 5),
          "| rel err median", np.round(np.median(rel, 0), 4), "max", round(float(rel.max()), 4))

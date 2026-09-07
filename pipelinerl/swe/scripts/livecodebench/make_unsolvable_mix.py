#!/usr/bin/env python3
"""Resample a pool to a target fraction of pool-unsolvable problems.

The value of the give-up action should scale with how many problems nothing in the pool can
solve: with none, abstention is worthless; with many, it is most of the available saving. Our
two datasets sit at 8.3% (LiveCodeBench) and 40.0% (TACO) and differ in a dozen other ways, so
they cannot separate that variable from everything else. This holds one pool fixed and varies
only the mix, turning the scope condition into a dose-response curve.

Problems are dropped, never duplicated, so every retained cell is real data; the split
proportions are preserved so the manifest stays honest.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--tensors-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--unsolvable-frac", type=float, required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

src, dst = Path(a.tensors_dir), Path(a.out)
dst.mkdir(parents=True, exist_ok=True)
z = np.load(src / "tensors.npz")
ids = [str(p) for p in z["problem_ids"]]
solved = (z["final_outcome"] & z["valid"]).any(axis=(1, 2))
man = json.loads((src / "split_manifest.json").read_text())
rng = np.random.default_rng(a.seed)

keep_idx: list[int] = []
new_man = {k: v for k, v in man.items() if not k.endswith("_problem_ids")}
for key in ("train_problem_ids", "calibration_problem_ids", "test_problem_ids"):
    sel = {str(x) for x in man[key]}
    s = [i for i, p in enumerate(ids) if p in sel and solved[i]]
    u = [i for i, p in enumerate(ids) if p in sel and not solved[i]]
    # keep as many problems as the requested mix allows without duplicating any
    n_s, n_u = len(s), len(u)
    f = a.unsolvable_frac
    total = min(int(n_s / (1 - f)) if f < 1 else n_u, int(n_u / f) if f > 0 else n_s)
    want_u = int(round(total * f)); want_s = total - want_u
    take = list(rng.permutation(s)[:want_s]) + list(rng.permutation(u)[:want_u])
    keep_idx += [int(i) for i in take]
    new_man[key] = [ids[int(i)] for i in take]

keep_idx = sorted(keep_idx)
d = {k: (z[k][keep_idx] if z[k].shape[:1] == (len(ids),) else z[k]) for k in z.files}
np.savez_compressed(dst / "tensors.npz", **d)
(dst / "split_manifest.json").write_text(json.dumps(new_man, indent=1))
for f_ in src.iterdir():
    if f_.name in ("tensors.npz", "split_manifest.json"):
        continue
    if f_.name == "problems.jsonl":
        kept = {ids[i] for i in keep_idx}
        with open(dst / f_.name, "w") as o:
            for line in open(f_):
                if line.strip() and str(json.loads(line)["problem_id"]) in kept:
                    o.write(line)
    else:
        shutil.copy2(f_, dst / f_.name)

sub = solved[keep_idx]
print(f"target unsolvable {a.unsolvable_frac:.0%} -> actual {1 - sub.mean():.1%}, "
      f"{len(keep_idx)} problems "
      f"(train {len(new_man['train_problem_ids'])}, cal {len(new_man['calibration_problem_ids'])}, "
      f"test {len(new_man['test_problem_ids'])})")

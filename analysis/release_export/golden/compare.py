"""Compare two output trees file by file (internal). Exact by default: JSON/JSONL values and npz arrays must be equal
(NaN == NaN); reports each differing file with its largest numeric difference. Logs are skipped.
Usage: python compare.py <baseline dir> <candidate dir> [--tol 0] [--rename old=new ...]
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            if k not in a or k not in b:
                out.append((f"{path}/{k}", "missing in " + ("candidate" if k in a else "baseline"))); continue
            walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"length {len(a)} vs {len(b)}")); return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (a == b or (math.isnan(a) and math.isnan(b))):
            out.append((path, abs(a - b)))
    elif a != b:
        out.append((path, f"{str(a)[:40]!r} vs {str(b)[:40]!r}"))


def load(f):
    if f.suffix == ".json":
        return json.loads(f.read_text())
    if f.suffix == ".jsonl":
        return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if f.suffix == ".npz":
        z = np.load(f, allow_pickle=False); return {k: z[k].tolist() for k in z.files}
    return f.read_bytes()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("cand"); ap.add_argument("--tol", type=float, default=0)
    ap.add_argument("--rename", nargs="*", default=[]); a = ap.parse_args()
    ren = dict(r.split("=", 1) for r in a.rename)
    B, C = Path(a.base), Path(a.cand)
    files = sorted(p.relative_to(B) for p in B.rglob("*") if p.is_file() and "logs" not in p.parts and "cmds" not in p.parts)
    bad = 0
    for rel in files:
        c = C / ren.get(str(rel), str(rel))
        if not c.exists():
            print("MISSING", rel); bad += 1; continue
        d = []; walk(load(B / rel), load(c), "", d)
        num = [x for _, x in d if isinstance(x, float)]; other = [(p, x) for p, x in d if not isinstance(x, float)]
        if other or (num and max(num) > a.tol):
            bad += 1
            print(f"DIFF {rel}: {len(d)} values, max |diff| {max(num) if num else 0:.3g}", *[f"\n   {p}: {x}" for p, x in d[:5]])
    extra = sorted(p.relative_to(C) for p in C.rglob("*") if p.is_file() and "logs" not in p.parts and "cmds" not in p.parts)
    extra = [p for p in extra if str(p) not in {ren.get(str(r), str(r)) for r in files}]
    for p in extra:
        print("EXTRA", p)
    print(f"{len(files)} files, {bad} differ, {len(extra)} extra")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

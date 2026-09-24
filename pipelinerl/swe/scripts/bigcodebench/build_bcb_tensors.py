#!/usr/bin/env python3
"""Tensor bundle for the BigCodeBench pool, mirroring build_pool_v2_tensors.py.

Separate from the LCB builder only because BCB carries its own split manifest (written by the
collector) and has no source scout collection to inherit a split from. Same schema, so every
downstream analysis -- oracle headroom, best-fixed-cascade, the replay -- runs unchanged.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rungs", required=True, help="label:draws,...")
    ap.add_argument("--calibration-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0)
    a = ap.parse_args()
    pool, out = Path(a.pool_dir), Path(a.output_dir)
    spec = [(s.split(":")[0], int(s.split(":")[1])) for s in a.rungs.split(",") if s.strip()]
    slots = [s for s, _ in spec]; K = max(k for _, k in spec)
    man = json.loads((pool / "bcb_split_manifest.json").read_text())
    pids = sorted(set(man["train"]) | set(man["eval"]))
    pidx = {p: i for i, p in enumerate(pids)}
    P, M = len(pids), len(slots)
    final = np.zeros((P, M, K), bool); valid = np.zeros((P, M, K), bool)
    ptok = np.zeros((P, M, K), np.float32); ctok = np.zeros((P, M, K), np.float32)
    infra = 0
    for mi, (label, nd) in enumerate(spec):
        got = {}
        for f in pool.glob(f"{label}_*_d*.jsonl"):
            m = re.match(rf"{re.escape(label)}_(train|eval)_d(\d+)\.jsonl$", f.name)
            if not m:
                continue
            for line in open(f):
                if line.strip():
                    r = json.loads(line); got[(str(r["problem_id"]), int(m.group(2)))] = r
        for pid, pi in pidx.items():
            for k in range(nd):
                r = got.get((pid, k))
                if r is None:
                    continue
                if r.get("finish_reason") == "error" or not str(r.get("full_output") or "").strip():
                    infra += 1; continue        # transport / empty: not evidence about the model
                valid[pi, mi, k] = True
                final[pi, mi, k] = bool(r.get("resolved"))
                ptok[pi, mi, k] = float(r.get("prompt_tokens", 0))
                ctok[pi, mi, k] = float(r.get("completion_tokens", 0))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "tensors.npz", final_outcome=final, execution_outcome=final,
                        weak_verifier_outcome=final, valid=valid, prompt_tokens=ptok,
                        completion_tokens=ctok, problem_ids=np.array(pids),
                        model_slots=np.array(slots), schema_version=np.array(3))
    rng = np.random.default_rng(a.split_seed)
    rest = list(man["train"]); rng.shuffle(rest)
    ncal = int(round(len(rest) * a.calibration_fraction))
    (out / "split_manifest.json").write_text(json.dumps(
        {"train_problem_ids": sorted(rest[ncal:]), "calibration_problem_ids": sorted(rest[:ncal]),
         "test_problem_ids": sorted(man["eval"]), "split_mode": "bcb_random"}, indent=1))
    print(f"{P} tasks x {M} rungs x up to {K} draws; {infra} infra failures masked")
    for mi, (label, nd) in enumerate(spec):
        v = valid[:, mi, :nd]
        if v.sum():
            print(f"  {label:<10}{v.sum():>6}/{P*nd:<6} valid ({v.sum()/(P*nd)*100:>5.1f}%)  "
                  f"pass@1 {final[:, mi, :nd][v].mean()*100:>5.1f}%")
    print("wrote", out)


if __name__ == "__main__":
    main()

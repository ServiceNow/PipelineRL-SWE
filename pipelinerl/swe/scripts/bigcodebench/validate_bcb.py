#!/usr/bin/env python3
"""Decide which BigCodeBench tasks are safe to measure a model on, in THIS environment.

Runs every task's own reference solution REPEATS times and classifies:
  keep       -- passed every repeat. Only these enter the pool.
  flaky      -- passed some repeats and not others: unseeded randomness, wall-clock or network.
  missing_dep-- failed every repeat with ImportError/ModuleNotFoundError: our sandbox, not the task.
  broken     -- failed every repeat for another reason: the reference does not satisfy its own tests.
  timeout    -- exceeded the per-run limit.
A model measured on anything but `keep` is being scored on our plumbing (PAPER_OUTLINE 3b-lxxxix).

Writes bcb_validation.json with per-task status, and bcb_keep.json with the surviving task_ids.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from pipelinerl.swe.scripts.bigcodebench.bcb_grader import grade, reference_code   # noqa: E402

_DS = None
def _task(i: int, repeats: int, timeout: int):
    global _DS
    if _DS is None:
        from datasets import load_dataset
        _DS = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    r = _DS[i]
    runs = [grade(reference_code(r), r["test"], timeout=timeout) for _ in range(repeats)]
    n_ok = sum(g.ok for g in runs)
    det = " | ".join(g.detail for g in runs if not g.ok)[:500]
    if n_ok == repeats:                       status = "keep"
    elif n_ok > 0:                            status = "flaky"
    elif any(g.status == "timeout" for g in runs): status = "timeout"
    elif "ModuleNotFoundError" in det or "ImportError" in det: status = "missing_dep"
    else:                                     status = "broken"
    return {"idx": i, "task_id": r["task_id"], "status": status, "n_ok": n_ok,
            "repeats": repeats, "libs": r["libs"], "detail": det[:300]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    from datasets import load_dataset
    n = len(load_dataset("bigcode/bigcodebench", split="v0.1.4"))
    idx = list(range(n if not a.limit else min(n, a.limit)))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_task, i, a.repeats, a.timeout) for i in idx]
        for k, f in enumerate(futs):
            rows.append(f.result())
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(idx)}  {Counter(r['status'] for r in rows)}", flush=True)
    (out / "bcb_validation.json").write_text(json.dumps(rows, indent=1))
    keep = [r["task_id"] for r in rows if r["status"] == "keep"]
    (out / "bcb_keep.json").write_text(json.dumps(keep, indent=1))
    c = Counter(r["status"] for r in rows)
    print(f"\n{len(idx)} tasks: " + "  ".join(f"{k}={v}" for k, v in c.most_common()))
    print(f"kept {len(keep)} ({len(keep)/len(idx)*100:.1f}%) -> {out/'bcb_keep.json'}")
    for st in ["flaky", "missing_dep", "broken", "timeout"]:
        ex_ = [r for r in rows if r["status"] == st][:3]
        for r in ex_:
            print(f"  [{st}] {r['task_id']} libs={r['libs']} :: {r['detail'][:150]}")


if __name__ == "__main__":
    main()

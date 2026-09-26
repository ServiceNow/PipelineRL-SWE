#!/usr/bin/env python3
"""Build the CodeContests task file for the routing pool (cost-head replication, NEW_PATH / PRIOR_ART 7).

Keeps Codeforces stdin/stdout problems with a Python 3 reference solution; selects up to --max-tests
tests (public, then private, then generated); and keeps a problem only if some Python 3 reference
passes ALL selected tests under our grader and time limit. That validation drops problems with several
acceptable outputs (special judges) and ones too slow in Python, which would otherwise score correct
model answers as wrong.
"""
from __future__ import annotations
import argparse, glob, json, re
from multiprocessing import Pool
import pandas as pd
from pipelinerl.swe.scripts.codecontests.cc_grader import grade

PY3 = 3   # CodeContests language code for PYTHON3


def slug(name: str) -> str:
    return "cc_" + re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:60]


def validate(task):
    for ref in task.pop("_refs"):
        g = grade(ref, [tuple(t) for t in task["tests"]], task["timeout_s"])
        if g.ok:
            return task, True
    return task, False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/mnt/llmd/data/code_contests/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tests", type=int, default=20)
    ap.add_argument("--max-refs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    df = pd.concat([pd.read_parquet(f).assign(cc_split=f.split("/")[-1].split("-")[0])
                    for f in sorted(glob.glob(f"{a.data_dir}/*.parquet"))])
    tasks = []
    for _, r in df.iterrows():
        if r["source"] != 2 or r["input_file"] or r["output_file"]:
            continue
        refs = [s for l, s in zip(r["solutions"]["language"], r["solutions"]["solution"]) if l == PY3][: a.max_refs]
        if not refs:
            continue
        tests = []
        for key in ("public_tests", "private_tests", "generated_tests"):
            tests += list(zip(r[key]["input"], r[key]["output"]))
        tests = tests[: a.max_tests]
        if len(tests) < 3:
            continue
        tl = (r["time_limit"] or {}).get("seconds") or 2
        tasks.append({"problem_id": slug(r["name"]), "name": r["name"], "prompt": r["description"],
                      "tests": [[i, o] for i, o in tests], "timeout_s": float(min(max(3 * tl, 4), 20)),
                      "cf_rating": int(r["cf_rating"] or 0), "cc_split": r["cc_split"],
                      "cf_tags": list(r["cf_tags"]), "_refs": refs})
    seen = set(); uniq = []
    for t in tasks:
        if t["problem_id"] not in seen:
            seen.add(t["problem_id"]); uniq.append(t)
    print(f"{len(df)} rows -> {len(uniq)} Codeforces stdin/stdout problems with a Python 3 reference", flush=True)
    kept = 0
    with Pool(a.workers) as pool, open(a.out, "w") as f:
        for t, ok in pool.imap_unordered(validate, uniq, chunksize=2):
            if ok:
                f.write(json.dumps(t) + "\n"); kept += 1
    print(f"kept {kept} problems whose reference passes all selected tests -> {a.out}")


if __name__ == "__main__":
    main()

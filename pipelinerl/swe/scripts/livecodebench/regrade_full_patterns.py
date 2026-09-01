#!/usr/bin/env python3
"""Re-grade saved generations one test at a time, so failure PATTERNS exist.

The pinned LiveCodeBench runner returns at the first failing test. Every `result_codes`
array in the collection is therefore a prefix, T...TF, and the tests after the first
failure were never executed. Two consequences that this script removes:

  1. "Fraction of tests passed" is the position of the first failure divided by suite size,
     not a measure of how much of the problem was solved. The project measured its signal
     content as nil (AUC 0.53) -- on that degenerate encoding.
  2. Cross-model comparison can only ask "did they stop at the same index", not "did they
     fail the same tests". 59.1% of failed draws stop at index 0, so most of the population
     collapses into one indistinguishable bucket. A cross-model agreement statistic computed
     on prefixes measured 0.50-0.60 AUC, which is not evidence about failure-set agreement.

Fix: call the same pinned grader once per test, with a single-test evaluation sample. There
is no short-circuit to suppress when the suite has one test, so the pinned semantics are
preserved exactly and the full pattern falls out. No generation is re-run; every program is
read from the saved draw files, so this costs CPU and no API spend.

The runner forks a worker per call and concurrent forks corrupt labels, so grading is
serialized. Output is one row per (problem, model, draw) carrying the complete pattern.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from pipelinerl.swe.scripts.livecodebench.collect_lcb_trajectories import (
    LCB_DATASET_REVISION,
    LCB_EVALUATOR_COMMIT,
    evaluate_code,
    load_lcb,
    problem_id,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def single_test_sample(sample: dict[str, Any], i: int) -> dict[str, Any]:
    """One test from an evaluation sample, preserving call-based vs stdin grading.

    `fn_name` is written only when it is actually present: the LCB evaluator selects
    call-based grading on the KEY existing, so emitting `"fn_name": null` silently grades a
    stdin problem as if it needed a Solution class. That defect mis-graded 14% of the TACO
    build and is not being reintroduced here.
    """
    io = json.loads(sample["input_output"])
    out: dict[str, Any] = {"inputs": [io["inputs"][i]], "outputs": [io["outputs"][i]]}
    if io.get("fn_name"):
        out["fn_name"] = io["fn_name"]
    return {"input_output": json.dumps(out)}


def suite_size(sample: dict[str, Any]) -> int:
    return len(json.loads(sample["input_output"]).get("inputs", []))


def read_draw(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[str(r["problem_id"])] = r
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draw-file", action="append", required=True, metavar="LABEL=PATH",
                   help="Repeatable. e.g. oss120=/.../oss120_eval_d0.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--only-all-fail", action="store_true",
                   help="Restrict to problems where EVERY supplied draw failed. That is the "
                        "population where an abstention decision is live and where the "
                        "signal currently degenerates to a single bit.")
    p.add_argument("--max-tests", type=int, default=50,
                   help="Cap tests graded per problem (median suite is 16). Recorded per row.")
    p.add_argument("--eval-timeout", type=int, default=10)
    p.add_argument("--min-date", default="2023-09-01")
    p.add_argument("--release-version", default="release_v6")
    p.add_argument("--dataset-revision", default=LCB_DATASET_REVISION)
    p.add_argument("--limit", type=int, default=0, help="Smoke-test on this many problems.")
    args = p.parse_args()

    draws: dict[str, dict[str, dict]] = {}
    for spec in args.draw_file:
        label, _, path = spec.partition("=")
        draws.setdefault(label, {}).update(read_draw(Path(path)))
    labels = sorted(draws)

    rows = load_lcb(min_date=args.min_date, release_version=args.release_version,
                    dataset_revision=args.dataset_revision)
    by_pid = {problem_id(r): r for r in rows}

    shared = sorted(set.intersection(*[set(d) for d in draws.values()]) & set(by_pid))
    if args.only_all_fail:
        shared = [pid for pid in shared
                  if all(not draws[l][pid].get("resolved") for l in labels)]
    if args.limit:
        shared = shared[: args.limit]
    logger.info("%d problems x %d draws to regrade", len(shared), len(labels))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                done.add((r["problem_id"], r["label"]))
        logger.info("resuming: %d rows already graded", len(done))

    with open(out_path, "a") as out_f:
        for n, pid in enumerate(shared, 1):
            problem = by_pid[pid]
            total = suite_size(problem["_evaluation_sample"])
            n_tests = min(total, args.max_tests)
            for label in labels:
                if (pid, label) in done:
                    continue
                code = draws[label][pid].get("code") or ""
                pattern: list[bool] = []
                codes: list[Any] = []
                if code.strip():
                    for i in range(n_tests):
                        probe = dict(problem)
                        probe["_single_test"] = single_test_sample(problem["_evaluation_sample"], i)
                        rep = evaluate_code(code, probe, args.eval_timeout, "_single_test")
                        rc = rep["result_codes"]
                        pattern.append(bool(rc) and rc[0] is True)
                        codes.append(rc[0] if rc else None)
                out_f.write(json.dumps({
                    "problem_id": pid,
                    "label": label,
                    "resolved_original": bool(draws[label][pid].get("resolved")),
                    "suite_size": total,
                    "n_tests_graded": n_tests,
                    "empty_code": not code.strip(),
                    "pattern": pattern,
                    "per_test_codes": codes,
                    "n_passed": sum(pattern),
                    "_lcb_evaluator_commit": LCB_EVALUATOR_COMMIT,
                    "_lcb_dataset_revision": args.dataset_revision,
                    "_grading": "per_test_no_short_circuit",
                }) + "\n")
                out_f.flush()
            if n % 10 == 0:
                logger.info("regraded %d/%d problems", n, len(shared))


if __name__ == "__main__":
    main()

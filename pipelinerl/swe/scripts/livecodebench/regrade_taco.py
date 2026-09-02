#!/usr/bin/env python3
"""Re-grade saved TACO generations, fixing two defects that cost real solves.

Neither fix needs a new generation. Every program is read from the stored `full_output`.

DEFECT 1 -- the null `fn_name` key. `build_taco_problems.py` wrote `"fn_name": null` into the
evaluation sample whenever TACO had no function name. The LiveCodeBench evaluator selects
call-based grading on the KEY BEING PRESENT, so stdin problems were graded as if they needed
a `Solution` class. 1,774 of 2,000 problems carry the null key, and 7.2% of medium/hard draws
show the resulting `module 'tmp_sol' has no attribute 'Solution'` failure. The builder was
fixed, but the problems file on disk predates the fix, so the key is stripped here.

DEFECT 2 -- code extraction takes the FIRST fenced block. `extract_code` returns the first
```python block in the output. Models that explain before they solve put an illustrative
fragment first, so the grader receives prose. A real example, recorded as a syntax error:

    full_output starts  '**Solution Explanation**\\n\\nFor every integer `i (0 <= i <= N)` ...'
    extracted code      'dp[i] = dp[i-1] + dp[i-3] + dp[i-4]   (terms with negative index are 0)'

That is 14.0% of medium/hard draws, the largest single defect signature, and it is not
truncation. This selects the last fenced block that actually parses as Python instead.

The same extractor is used for LiveCodeBench, where the identical signature accounts for
roughly 11% of failures, so that collection is affected too. This script deliberately does
not touch it -- LCB labels are re-derived by their own pipeline -- but the defect is shared.

Truncation is the third defect and the only one needing re-collection: 29-30% of TACO draws
sit at the 4096-token cap. This script reports it but cannot fix it.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
from pathlib import Path
from typing import Any

from pipelinerl.swe.scripts.livecodebench.collect_lcb_trajectories import (
    LCB_EVALUATOR_COMMIT,
    evaluate_code,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FENCE_RE = re.compile(r"```(?:[Pp]ython)?\s*\n(.*?)```", re.DOTALL)
CAP_TOLERANCE = 50


def extract_code_robust(output: str) -> tuple[str, str]:
    """Return (code, how). Prefer the LAST fenced block that parses as Python.

    Rationale in the module docstring: taking the first block hands the grader an
    illustrative fragment whenever the model explains before it solves.
    """
    blocks = [b.strip() for b in FENCE_RE.findall(output or "") if b.strip()]
    for b in reversed(blocks):
        try:
            ast.parse(b)
            return b, "last_parsing_fence"
        except SyntaxError:
            continue
    if blocks:
        return max(blocks, key=len), "longest_fence_none_parse"
    stripped = (output or "").strip()
    try:
        ast.parse(stripped)
        return stripped, "whole_output"
    except SyntaxError:
        return stripped, "whole_output_unparsed"


def strip_null_fn_name(sample: dict[str, Any]) -> dict[str, Any]:
    io = json.loads(sample["input_output"])
    if "fn_name" in io and not io["fn_name"]:
        io.pop("fn_name")
    return {"input_output": json.dumps(io)}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", required=True, help="taco_problems.jsonl")
    ap.add_argument("--draw-file", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--output", required=True)
    ap.add_argument("--eval-timeout", type=int, default=10)
    ap.add_argument("--old-cap", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    problems: dict[str, dict] = {}
    n_stripped = 0
    for line in open(args.problems):
        if not line.strip():
            continue
        p = json.loads(line)
        pid = str(p.get("question_id") or p.get("problem_id"))
        before = json.loads(p["_evaluation_sample"]["input_output"])
        p["_evaluation_sample"] = strip_null_fn_name(p["_evaluation_sample"])
        if "fn_name" in before and not before["fn_name"]:
            n_stripped += 1
        problems[pid] = p
        # collect_lcb_trajectories.problem_id() builds "<platform>_<question_id>"; the draw
        # files carry that composite, so index both ways.
        problems[f"{p.get('platform')}_{pid}"] = p
    logger.info("%d problems, stripped a null fn_name from %d", len(problems) // 2, n_stripped)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                done.add((r["problem_id"], r["label"]))
        logger.info("resuming: %d rows already regraded", len(done))

    stats: dict[str, dict[str, int]] = {}
    with open(out_path, "a") as out_f:
        for spec in args.draw_file:
            label, _, path = spec.partition("=")
            rows = read_rows(Path(path))
            if args.limit:
                rows = rows[: args.limit]
            st = stats.setdefault(label, {"n": 0, "was": 0, "now": 0, "rescued": 0,
                                          "lost": 0, "no_problem": 0, "at_cap": 0})
            for i, r in enumerate(rows, 1):
                pid = str(r["problem_id"])
                if (pid, label) in done:
                    continue
                prob = problems.get(pid)
                if prob is None:
                    st["no_problem"] += 1
                    continue
                was = bool(r.get("resolved") if "resolved" in r else r.get("scout_correct"))
                code, how = extract_code_robust(r.get("full_output") or "")
                rep = evaluate_code(code, prob, args.eval_timeout, "_evaluation_sample")
                now = bool(rep["resolved"])
                st["n"] += 1
                st["was"] += was
                st["now"] += now
                st["rescued"] += (now and not was)
                st["lost"] += (was and not now)
                st["at_cap"] += (r.get("completion_tokens") or 0) >= args.old_cap - CAP_TOLERANCE
                out_f.write(json.dumps({
                    "problem_id": pid, "label": label,
                    "resolved_before": was, "resolved": now,
                    "extraction": how, "code_changed": code != (r.get("code") or r.get("patch_text") or ""),
                    "result_codes": rep["result_codes"],
                    "eval_metadata": rep["metadata"],
                    "completion_tokens": r.get("completion_tokens"),
                    "difficulty": prob.get("difficulty"),
                    "_lcb_evaluator_commit": LCB_EVALUATOR_COMMIT,
                    "_regrade": "fn_name_stripped+robust_extraction",
                }) + "\n")
                if i % 100 == 0:
                    out_f.flush()
                    logger.info("%s: %d/%d  solve %.1f%% -> %.1f%%", label, i, len(rows),
                                100 * st["was"] / max(st["n"], 1), 100 * st["now"] / max(st["n"], 1))

    print(f"\n{'route':<10} {'n':>6} {'before':>8} {'after':>8} {'delta':>8} {'rescued':>8} {'lost':>6} {'at cap':>7}")
    for label, s in stats.items():
        if not s["n"]:
            continue
        b, a = 100 * s["was"] / s["n"], 100 * s["now"] / s["n"]
        print(f"{label:<10} {s['n']:6d} {b:7.1f}% {a:7.1f}% {a-b:+7.1f}pt {s['rescued']:8d} "
              f"{s['lost']:6d} {100*s['at_cap']/s['n']:6.1f}%")
    print("\n'at cap' is the remaining defect and needs re-collection, not re-grading.")


if __name__ == "__main__":
    main()

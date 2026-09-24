#!/usr/bin/env python3
"""Grade a BigCodeBench candidate solution against the task's unittest suite.

Unlike LiveCodeBench there is no official runner to pin, so this file IS the grading contract and
has to be conservative: a task only counts once its own reference solution passes it repeatedly in
THIS environment. Everything else -- a missing library, a network-dependent test, a test that
depends on wall-clock time or an unseeded shuffle -- is a property of our sandbox, not of the
model, and must be excluded before any model is measured (PAPER_OUTLINE 3b-lxxxv/lxxxix: every
collection artifact this project has hit was mistaken for model behaviour first).

A candidate is graded by running `<candidate code>\n\n<task test>` in a fresh subprocess and
reading unittest's own result, so a candidate that redefines helpers or imports differently is
still graded on behaviour. Timeouts, crashes and import errors are failures for a candidate but
are reported separately for a reference, which is what lets the validator tell the three cases
apart.
"""
from __future__ import annotations
import json, subprocess, sys, tempfile, textwrap
from dataclasses import dataclass
from pathlib import Path

RUNNER = '''
import json, sys, unittest, io, os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")          # never open a window
_out = io.StringIO()
try:
    suite = unittest.defaultTestLoader.loadTestsFromName("TestCases", sys.modules["__main__"])
    res = unittest.TextTestRunner(stream=_out, verbosity=0).run(suite)
    print("__BCB__" + json.dumps({
        "ran": res.testsRun,
        "failures": len(res.failures), "errors": len(res.errors),
        "skipped": len(res.skipped),
        "ok": res.wasSuccessful() and res.testsRun > 0,
        "detail": [str(t[1])[:400] for t in (res.failures + res.errors)][:3],
    }))
except Exception as e:
    print("__BCB__" + json.dumps({"ran": 0, "ok": False, "harness_error": f"{type(e).__name__}: {e}"[:400]}))
'''


@dataclass
class Result:
    ok: bool
    ran: int = 0
    failures: int = 0
    errors: int = 0
    status: str = ""          # pass | fail | timeout | crash | harness_error
    detail: str = ""


def grade(candidate_code: str, test_code: str, timeout: int = 60) -> Result:
    program = candidate_code + "\n\n" + test_code + "\n" + RUNNER
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "prog.py"
        p.write_text(program)
        try:
            cp = subprocess.run([sys.executable, str(p)], cwd=td, capture_output=True,
                                text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return Result(False, status="timeout")
    for line in (cp.stdout or "").splitlines():
        if line.startswith("__BCB__"):
            d = json.loads(line[len("__BCB__"):])
            if "harness_error" in d:
                return Result(False, status="harness_error", detail=d["harness_error"])
            return Result(bool(d["ok"]), d.get("ran", 0), d.get("failures", 0), d.get("errors", 0),
                          "pass" if d["ok"] else "fail", " | ".join(d.get("detail", []))[:400])
    return Result(False, status="crash", detail=((cp.stderr or "")[-400:]))


def reference_code(task: dict) -> str:
    """The reference solution as a runnable module: signature block + canonical body."""
    return task["code_prompt"] + task["canonical_solution"]

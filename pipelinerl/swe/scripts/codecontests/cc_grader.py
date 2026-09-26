"""Grade a Python stdin/stdout solution on CodeContests tests.

Comparison is token-wise (whitespace-insensitive), with a 1e-6 relative/absolute tolerance when both
tokens parse as floats. Problems that accept several correct outputs cannot be graded this way; the
prepare step drops them by requiring a reference solution to pass every selected test, the same
"validate the grader on known-correct code first" rule as BigCodeBench.
"""
from __future__ import annotations
import os, subprocess, sys, tempfile
from dataclasses import dataclass


@dataclass
class Grade:
    ok: bool
    status: str          # pass | wrong | timeout | error | empty
    n_passed: int
    n_tests: int


def _same(a: str, b: str) -> bool:
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if abs(fx - fy) > 1e-6 * max(1.0, abs(fy)):
            return False
    return True


def grade(code: str, tests: list[tuple[str, str]], timeout_s: float, stop_on_fail: bool = True) -> Grade:
    if not code.strip():
        return Grade(False, "empty", 0, len(tests))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sol.py")
        with open(path, "w") as f:
            f.write(code)
        passed = 0
        for inp, out in tests:
            try:
                r = subprocess.run([sys.executable, path], input=inp, capture_output=True, text=True,
                                   timeout=timeout_s, cwd=d)
            except subprocess.TimeoutExpired:
                if stop_on_fail:
                    return Grade(False, "timeout", passed, len(tests))
                continue
            if r.returncode != 0:
                if stop_on_fail:
                    return Grade(False, "error", passed, len(tests))
                continue
            if _same(r.stdout, out):
                passed += 1
            elif stop_on_fail:
                return Grade(False, "wrong", passed, len(tests))
        return Grade(passed == len(tests), "pass" if passed == len(tests) else "wrong", passed, len(tests))

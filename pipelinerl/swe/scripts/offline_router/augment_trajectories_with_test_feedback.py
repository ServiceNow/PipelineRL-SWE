#!/usr/bin/env python3
"""
Augment trajectories_{train,eval}.jsonl with scout test-execution feedback
from a Daytona eval run, adding a 'test_feedback' field to each row.

The field is a plain-text summary of what the scout's patch did to the test
suite — which tests failed, which passed — formatted for use as predictor
input alongside the patch and (optionally) the CoT trace.

For SWE-Smith Daytona evals (run_swesmith_eval_daytona.py output):
  - report.json has: resolved, patch_exists
  - test_output.txt has: raw pytest stdout, parsed to extract test names

Writes augmented files to --output-dir as trajectories_{train,eval}.jsonl.
Instances with no corresponding Daytona report get test_feedback="".

Usage:
  python augment_trajectories_with_test_feedback.py \\
    --trajectories-dir  /mnt/.../instruct_patches_trajectories_XYZ \\
    --daytona-log-dir   logs/run_evaluation/scout_eval_TIMESTAMP \\
    --output-dir        /mnt/.../instruct_patches_trajectories_XYZ_with_testfb \\
    [--split train]  # omit to process both splits
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_test_output(test_output: str) -> dict:
    """Extract test failure signal from raw pytest stdout."""
    failing, passing = [], []

    for line in test_output.splitlines():
        # Lines like "tests/foo.py::test_bar FAILED" or "PASSED"
        m = re.match(r'\s*(\S+::[\w\[\]@,. -]+)\s+(PASSED|FAILED|ERROR)\s*', line)
        if m:
            test_name, status = m.group(1), m.group(2)
            if status in ("FAILED", "ERROR"):
                failing.append(test_name)
            elif status == "PASSED":
                passing.append(test_name)

    # Extract the short test summary lines (one per failure, with reason)
    summary_lines = []
    in_summary = False
    for line in test_output.splitlines():
        if "short test summary info" in line:
            in_summary = True
            continue
        if in_summary:
            if line.startswith("=") or not line.strip():
                in_summary = False
            else:
                summary_lines.append(line.strip())

    # Fallback: look for "FAILED xxx - reason" pattern
    if not summary_lines:
        for line in test_output.splitlines():
            if re.match(r'FAILED\s+\S+', line):
                summary_lines.append(line.strip())

    return {
        "failing": failing,
        "passing": passing,
        "summary_lines": summary_lines[:5],  # cap at 5 to avoid huge inputs
    }


def format_test_feedback(resolved: bool, patch_exists: bool, parsed: dict | None) -> str:
    """Render test execution info as a plain-text block for predictor input."""
    if not patch_exists:
        return "Scout test execution: NO PATCH (patch generation failed or empty)"

    if resolved:
        n_pass = len(parsed["passing"]) if parsed else 0
        return f"Scout test execution: PASSED ({n_pass} tests passing)"

    if parsed is None:
        return "Scout test execution: FAILED (no test output available)"

    failing = parsed["failing"]
    passing = parsed["passing"]
    summary = parsed["summary_lines"]

    parts = [f"Scout test execution: FAILED ({len(failing)} test(s) still failing)"]

    if failing:
        names = ", ".join(f.split("::")[-1] for f in failing[:8])
        if len(failing) > 8:
            names += f", ... (+{len(failing)-8} more)"
        parts.append(f"Failing tests: {names}")

    if passing:
        parts.append(f"Passing tests: {len(passing)}")

    if summary:
        parts.append("Failure summary:")
        for line in summary:
            # Truncate very long lines
            parts.append(f"  {line[:200]}")

    return "\n".join(parts)


def load_daytona_feedback(log_dir: Path) -> dict[str, str]:
    """
    Walk a Daytona eval log directory, return {instance_id: test_feedback_text}.
    Expects structure: log_dir/{instance_id}/report.json + test_output.txt
    """
    feedback: dict[str, str] = {}
    if not log_dir.exists():
        print(f"  [warn] Daytona log dir not found: {log_dir}")
        return feedback

    for instance_dir in log_dir.iterdir():
        if not instance_dir.is_dir():
            continue
        iid = instance_dir.name
        report_path = instance_dir / "report.json"
        test_output_path = instance_dir / "test_output.txt"

        if not report_path.exists():
            continue

        try:
            report = json.loads(report_path.read_text())
        except Exception:
            continue

        resolved = bool(report.get("resolved", False))
        patch_exists = bool(report.get("patch_exists", True))

        # If the report has tests_status (SWE-bench Verified style), use it directly
        tests_status = report.get("tests_status") or {}
        # Handle nested: sometimes keyed by instance_id
        if iid in tests_status:
            tests_status = tests_status[iid].get("tests_status", {})

        if tests_status and "FAIL_TO_PASS" in tests_status:
            f2p = tests_status["FAIL_TO_PASS"]
            failing = f2p.get("failure", [])
            fixed = f2p.get("success", [])
            parsed = {"failing": failing, "passing": fixed, "summary_lines": []}
        elif test_output_path.exists():
            raw = test_output_path.read_text(errors="replace")
            parsed = parse_test_output(raw)
        else:
            parsed = None

        feedback[iid] = format_test_feedback(resolved, patch_exists, parsed)

    return feedback


def augment_split(traj_path: Path, feedback: dict[str, str], output_path: Path) -> None:
    if not traj_path.exists():
        print(f"  [skip] {traj_path} not found")
        return

    n_total = n_matched = n_missing = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(traj_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            iid = str(row.get("problem_id") or row.get("instance_id") or "").strip()
            n_total += 1
            if iid in feedback:
                row["test_feedback"] = feedback[iid]
                n_matched += 1
            else:
                row["test_feedback"] = ""
                n_missing += 1
            fout.write(json.dumps(row) + "\n")

    print(f"  {output_path.name}: {n_total} rows, {n_matched} matched, {n_missing} missing feedback")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-dir", required=True,
                        help="Dir containing trajectories_{train,eval}.jsonl")
    parser.add_argument("--daytona-log-dir-train", default=None,
                        help="Daytona log dir for train split (logs/run_evaluation/RUN_ID_train)")
    parser.add_argument("--daytona-log-dir-eval", default=None,
                        help="Daytona log dir for eval split (logs/run_evaluation/RUN_ID_eval)")
    parser.add_argument("--daytona-log-dir", default=None,
                        help="Single log dir used for both splits (if train/eval not separate)")
    parser.add_argument("--output-dir", required=True,
                        help="Output dir for augmented trajectories_{train,eval}.jsonl")
    parser.add_argument("--split", choices=["train", "eval", "both"], default="both")
    args = parser.parse_args()

    traj_dir = Path(args.trajectories_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "eval"] if args.split == "both" else [args.split]

    for split in splits:
        log_dir_arg = getattr(args, f"daytona_log_dir_{split}", None) or args.daytona_log_dir
        if not log_dir_arg:
            print(f"[{split}] No daytona log dir specified — writing empty test_feedback")
            feedback = {}
        else:
            log_dir = Path(log_dir_arg)
            print(f"[{split}] Loading Daytona feedback from {log_dir}")
            feedback = load_daytona_feedback(log_dir)
            print(f"  → {len(feedback)} instances with feedback")

        traj_path = traj_dir / f"trajectories_{split}.jsonl"
        output_path = out_dir / f"trajectories_{split}.jsonl"
        print(f"[{split}] Augmenting {traj_path} → {output_path}")
        augment_split(traj_path, feedback, output_path)

    print("\n[done]")


if __name__ == "__main__":
    main()

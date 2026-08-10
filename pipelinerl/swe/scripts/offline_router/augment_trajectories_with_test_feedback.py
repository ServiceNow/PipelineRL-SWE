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


MAX_TEST_NAMES_DEFAULT = 200


def format_test_feedback(
    resolved: bool,
    patch_exists: bool,
    parsed: dict | None,
    max_test_names: int = MAX_TEST_NAMES_DEFAULT,
) -> str:
    """Render test execution info as a labeled list for predictor input.

    Format (all ablation modes are derived in the training script from the
    raw _tf_* fields; this function produces the canonical 'full' format):
      Scout test execution: FAILED (8/576 tests failing)
      FAILED: test_foo
      FAILED: test_bar
      PASSED: test_baz
      ...
      (+N more PASSED, +M more FAILED — see totals above)
      Total: 568 passed, 8 failed
    """
    if not patch_exists:
        return "Scout test execution: NO PATCH (patch generation failed or empty)"

    failing = parsed["failing"] if parsed else []
    passing = parsed["passing"] if parsed else []
    n_fail = len(failing)
    n_pass = len(passing)
    total = n_fail + n_pass

    if resolved:
        header = f"Scout test execution: PASSED ({n_pass} tests passing)"
    elif total == 0:
        return "Scout test execution: FAILED (no test output available)"
    else:
        header = f"Scout test execution: FAILED ({n_fail}/{total} tests failing)"

    parts = [header]

    # Prioritise showing all failing names, fill remainder with passing
    fail_slots = min(n_fail, max_test_names)
    pass_slots = min(n_pass, max(0, max_test_names - fail_slots))

    for name in failing[:fail_slots]:
        parts.append(f"FAILED: {name}")
    for name in passing[:pass_slots]:
        parts.append(f"PASSED: {name}")

    overflow_fail = n_fail - fail_slots
    overflow_pass = n_pass - pass_slots
    overflow_parts = []
    if overflow_fail > 0:
        overflow_parts.append(f"+{overflow_fail} more FAILED")
    if overflow_pass > 0:
        overflow_parts.append(f"+{overflow_pass} more PASSED")
    if overflow_parts:
        parts.append(f"({', '.join(overflow_parts)} — see totals above)")

    parts.append(f"Total: {n_pass} passed, {n_fail} failed")
    return "\n".join(parts)


def load_daytona_feedback(log_dir: Path, max_test_names: int = MAX_TEST_NAMES_DEFAULT) -> dict[str, str]:
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

        # SWE-bench Verified wraps all fields under the instance_id key;
        # SWE-Smith puts them at the top level.
        report_data = report.get(iid, report)

        resolved = bool(report_data.get("resolved", False))
        patch_exists = bool(report_data.get("patch_exists", True))
        tests_status = report_data.get("tests_status") or {}

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

        feedback[iid] = {
            "text": format_test_feedback(resolved, patch_exists, parsed, max_test_names=max_test_names),
            "failing": (parsed["failing"] if parsed else []),
            "passing": (parsed["passing"] if parsed else []),
            "resolved": resolved,
            "patch_exists": patch_exists,
        }

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
                fb = feedback[iid]
                row["test_feedback"] = fb["text"]
                row["_tf_failing"] = fb["failing"]
                row["_tf_passing"] = fb["passing"]
                row["_tf_resolved"] = fb["resolved"]
                row["_tf_patch_exists"] = fb["patch_exists"]
                n_matched += 1
            else:
                row["test_feedback"] = ""
                row["_tf_failing"] = []
                row["_tf_passing"] = []
                row["_tf_resolved"] = False
                row["_tf_patch_exists"] = True
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
    parser.add_argument("--max-test-names", type=int, default=MAX_TEST_NAMES_DEFAULT,
                        help="Max total test names in the full-format block (default 200). "
                             "Failing names are shown first; remaining slots go to passing.")
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
            feedback = load_daytona_feedback(log_dir, max_test_names=args.max_test_names)
            print(f"  → {len(feedback)} instances with feedback")

        traj_path = traj_dir / f"trajectories_{split}.jsonl"
        output_path = out_dir / f"trajectories_{split}.jsonl"
        print(f"[{split}] Augmenting {traj_path} → {output_path}")
        augment_split(traj_path, feedback, output_path)

    print("\n[done]")


if __name__ == "__main__":
    main()

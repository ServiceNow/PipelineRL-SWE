#!/usr/bin/env python
"""Evaluate SWE-smith predictions with one eai job per instance (instead of a Daytona sandbox).

Same per-instance steps as run_swesmith_eval_daytona.py (checkout the instance branch, HEAD~1, apply the patch, revert the test
files, run the profile's test command in the testbed env), run as an eai job on the instance's image; grading is local with
swesmith's get_eval_report (F2P only by default). --gold N evaluates the gold fix, i.e. the instance's bug patch reversed, for N
instances spread across as many repository images as possible; --gold N --empty is the negative control (no patch: must fail).
Jobs run as root (eai alphabit run-as-root=true), as in Daytona. With --no-root they run as uid 13011, and since SWE-smith images
keep /testbed root-owned (755) the job then copies /testbed to /tmp/testbed and runs there (test command paths rewritten,
PYTHONPATH pointed at the copy; tools under /root, e.g. pydantic's uv, stay unreadable); the --empty control checks that the
tests really import the copy and not the image's install in /testbed.

Usage:
  python run_swesmith_eval_eai.py --gold 300 --run-id smith_gold --wait 0     # submit, then rerun the same command to grade
  python run_swesmith_eval_eai.py --predictions-path preds.jsonl --run-id smith_x
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
from swesmith.harness.grading import get_eval_report
from swesmith.profiles import registry
from swesmith.profiles.python import PythonProfile

sys.path.insert(0, str(Path(__file__).parent))
from run_swebench_eval_daytona import GIT_APPLY_CMDS  # noqa: F401  (same fallbacks, spelled out in RUN_SH)
from run_swebench_eval_eai import ROOT, grade_wait, submit_all
from run_swesmith_eval_daytona import ACTIVATE, normalize_prediction, repair_patch

RUN_SH = """#!/bin/bash
D={d}
export HOME=/tmp; git config --global --add safe.directory '*'
echo "uid=$(id -u) user=$(whoami) host=$(hostname)" > $D/env.txt; ls -ld /testbed >> $D/env.txt
date +%s > $D/started
{copy}
cd {tb}
(timeout 60 git fetch || true) > $D/checkout.txt 2>&1
if ! (git checkout {iid} && git checkout HEAD~1) >> $D/checkout.txt 2>&1; then echo checkout_failed > $D/patch_failed; date +%s > $D/done; exit 0; fi
applied=0
for cmd in "git apply {rev} --verbose" "git apply {rev} --verbose --reject" "patch {rev} --batch --fuzz=5 -p1 -i"; do
  if [ ! -s $D/patch.diff ] || $cmd $D/patch.diff >> $D/apply.txt 2>&1; then applied=1; break; fi
done
if [ $applied = 1 ]; then
  {revert}
  ({activate} 2>/dev/null; /bin/bash $D/eval.sh) > $D/test_output.txt 2>&1
else echo patch_failed > $D/patch_failed; fi
date +%s > $D/done
"""


def main() -> None:
    p = argparse.ArgumentParser("Evaluate SWE-smith predictions with one eai job per instance.")
    p.add_argument("--predictions-path")
    p.add_argument("--gold", type=int, default=0)
    p.add_argument("--instances", default="", help="comma-separated subset of instance ids")
    p.add_argument("--empty", action="store_true", help="with --gold: empty patches (negative control)")
    p.add_argument("--hf-dataset", default="SWE-bench/SWE-smith")
    p.add_argument("--run-id", required=True)
    p.add_argument("--account", default="snow.research.rlar")
    p.add_argument("--results-obj", default="snow.research.tapes.results")
    p.add_argument("--cpu", type=int, default=2)
    p.add_argument("--mem", type=int, default=8)
    p.add_argument("--max-run-time", type=int, default=0, help="floor on the job time limit (default: profile timeout + 300 s)")
    p.add_argument("--wait", type=int, default=1800)
    p.add_argument("--submit-threads", type=int, default=16)
    p.add_argument("--no-root", dest="root", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    from datasets import load_dataset
    rows = load_dataset(args.hf_dataset, split="train")
    if args.gold:                                                  # round-robin over images, first instance of each first
        by_img: dict[str, list[int]] = {}
        for j, (im, f2p) in enumerate(zip(rows["image_name"], rows["FAIL_TO_PASS"])):  # Python only (the Go test command is malformed)
            if f2p:
                by_img.setdefault(im, []).append(j)
        py = {im for im, v in by_img.items() if isinstance(registry.get_from_inst(dict(rows[v[0]])), PythonProfile)}
        by_img = {im: v for im, v in by_img.items() if im in py}
        pick, k = [], 0
        while len(pick) < args.gold and any(k < len(v) for v in by_img.values()):
            pick += [v[k] for v in by_img.values() if k < len(v)][:args.gold - len(pick)]; k += 1
        inst = {rows[j]["instance_id"]: dict(rows[j]) for j in pick}
        preds = {i: normalize_prediction({KEY_INSTANCE_ID: i, KEY_PREDICTION: v["patch"], KEY_MODEL: "gold"}) for i, v in inst.items()}
    else:
        preds = {r[KEY_INSTANCE_ID]: normalize_prediction(r) for r in map(json.loads, open(args.predictions_path)) if r}
        inst = {r["instance_id"]: dict(r) for r in rows if r["instance_id"] in preds}
        preds = {i: v for i, v in preds.items() if i in inst}
    if args.instances:
        keep = set(args.instances.split(",")); preds = {i: v for i, v in preds.items() if i in keep}
    print(f"{len(preds)} instances on {len({inst[i]['image_name'] for i in preds})} images", flush=True)

    run_dir = ROOT / args.run_id; t0 = time.time(); todo = {}; mrt = {}
    for iid, pred in preds.items():
        d = run_dir / iid; d.mkdir(parents=True, exist_ok=True)
        if (d / "submitted").exists():
            continue
        rp = registry.get_from_inst(inst[iid]); test_cmd, _ = rp.get_test_cmd(inst[iid], f2p_only=True)
        f2p, p2p = rp.get_test_files(inst[iid]); tf = " ".join(f2p + p2p)
        patch = "" if args.empty else pred[KEY_PREDICTION] if args.gold else repair_patch(pred[KEY_PREDICTION], inst[iid])
        (d / "patch.diff").write_text(patch if not patch.strip() or patch.endswith("\n") else patch + "\n")
        tb = "/testbed" if args.root else "/tmp/testbed"
        (d / "eval.sh").write_text("\n".join(["#!/bin/bash", "set -uxo pipefail", f"cd {tb}"]
                                             + ([] if args.root else ["export PYTHONPATH=/tmp/testbed:/tmp/testbed/src${PYTHONPATH:+:$PYTHONPATH}"])
                                             + [": '>>>>> Start Test Output'", test_cmd.replace("/testbed", tb), ": '>>>>> End Test Output'", ""]))
        copy = ":" if args.root else "cp -r /testbed /tmp/testbed 2> $D/copy.txt; du -sh /tmp/testbed >> $D/env.txt"
        (d / "run.sh").write_text(RUN_SH.format(d=d, iid=iid, copy=copy, tb=tb, rev="-R" if args.gold else "", activate=ACTIVATE,
                                                revert=f"git checkout -- {tf} >> $D/apply.txt 2>&1" if tf else ":"))
        todo[iid] = d; mrt[iid] = rp.timeout + 300
    args.max_run_time = max([args.max_run_time, *mrt.values()]) if mrt else 1200
    submit_all(todo, args, lambda i: inst[i]["image_name"], prefix="swes")
    if args.dry_run:
        return
    grade_wait(run_dir, preds, args, t0, lambda iid, pred, log: nonvacuous(get_eval_report(pred, inst[iid], Path(log), f2p_only=True)))


def nonvacuous(rep: dict) -> dict:
    """the grader reports resolved when no FAIL_TO_PASS test appears in the log at all; count that as not resolved"""
    if rep.get("resolved") and not rep.get("tests_status", {}).get("FAIL_TO_PASS", {}).get("success"):
        rep["resolved"], rep["vacuous"] = False, True
    return rep


if __name__ == "__main__":
    main()

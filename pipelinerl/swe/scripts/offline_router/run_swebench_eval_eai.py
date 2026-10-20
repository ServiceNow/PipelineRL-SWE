#!/usr/bin/env python3
"""Evaluate SWE-bench Verified predictions with one eai job per instance (instead of a Daytona sandbox).

Same per-instance logic as run_swebench_eval_daytona.py, but each instance runs as its own eai job on the SWE-bench eval image:
the orchestrator stages patch.diff + eval.sh + run.sh on /mnt/llmd/results, submits the job, waits for the job to write
test_output.txt and a done marker to the same directory, and grades the log locally with swebench's get_eval_report.

Usage:
  python run_swebench_eval_eai.py --gold 3 --run-id eai_smoke          # gold patches for the first 3 Verified instances
  python run_swebench_eval_eai.py --predictions-path preds.jsonl --run-id verified_x --instances a,b,c
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec

ROOT = Path("/mnt/llmd/results/exps/aristides/swebench/eai_eval")
ARCH = "x86_64"
RUN_SH = """#!/bin/bash
D={d}
echo "uid=$(id -u) user=$(whoami) host=$(hostname)" > $D/env.txt
date +%s > $D/started
cd /testbed
applied=0
for cmd in "git apply --verbose" "git apply --verbose --reject" "patch --batch --fuzz=5 -p1 -i"; do
  if [ ! -s $D/patch.diff ] || $cmd $D/patch.diff >> $D/apply.txt 2>&1; then applied=1; break; fi
done
if [ $applied = 1 ]; then /bin/bash $D/eval.sh > $D/test_output.txt 2>&1; else echo patch_failed > $D/patch_failed; fi
date +%s > $D/done
"""


def _image_name(instance_id: str) -> str:
    return f"ghcr.io/epoch-research/swe-bench.eval.{ARCH}.{instance_id.lower()}:latest"


def submit(iid: str, d: Path, args) -> str:
    name = re.sub(r"[^a-z0-9_]", "_", f"swev_{args.run_id}_{iid}".lower())[:60]
    cmd = ["eai", "job", "new", "--restartable", "--name", name, "--account", args.account, "--image", _image_name(iid),
           "--cpu", str(args.cpu), "--mem", str(args.mem), "--gpu", "0", "--max-run-time", str(args.max_run_time),
           "--data", f"{args.results_obj}:/mnt/llmd/results", "--field", "id", "--", "/bin/bash", str(d / "run.sh")]
    if args.dry_run:
        print(" ".join(shlex.quote(c) for c in cmd)); return "dry"
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    p = argparse.ArgumentParser("Evaluate SWE-bench Verified predictions with one eai job per instance.")
    p.add_argument("--predictions-path", help="JSONL with {instance_id, model_patch, model} per line")
    p.add_argument("--gold", type=int, default=0, help="instead of predictions: the gold patch of the first N instances")
    p.add_argument("--instances", default="", help="comma-separated subset of instance ids")
    p.add_argument("--run-id", required=True)
    p.add_argument("--account", default="snow.research.rlar")
    p.add_argument("--results-obj", default="snow.research.tapes.results")
    p.add_argument("--cpu", type=int, default=2)
    p.add_argument("--mem", type=int, default=8)
    p.add_argument("--max-run-time", type=int, default=1200)
    p.add_argument("--wait", type=int, default=1800, help="seconds to wait for all jobs before grading what finished")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    from datasets import load_dataset
    inst = {r["instance_id"]: dict(r) for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
    if args.gold:
        preds = {i: {KEY_INSTANCE_ID: i, KEY_PREDICTION: inst[i]["patch"], KEY_MODEL: "gold"} for i in list(inst)[:args.gold]}
    else:
        preds = {}
        for line in open(args.predictions_path):
            if line.strip():
                r = json.loads(line); i = r.get("instance_id")
                preds[i] = {KEY_INSTANCE_ID: i, KEY_PREDICTION: r.get("model_patch") or r.get(KEY_PREDICTION, ""), KEY_MODEL: r.get("model", "unknown")}
    if args.instances:
        keep = set(args.instances.split(",")); preds = {i: v for i, v in preds.items() if i in keep}
    preds = {i: v for i, v in preds.items() if i in inst}

    run_dir = ROOT / args.run_id; t0 = time.time(); jobs, specs = {}, {}
    for iid, pred in preds.items():
        d = run_dir / iid; d.mkdir(parents=True, exist_ok=True)
        if (d / "submitted").exists():                                    # already submitted: rerun only waits and grades
            continue
        specs[iid] = make_test_spec(inst[iid]); patch = pred[KEY_PREDICTION]
        (d / "patch.diff").write_text(patch if not patch.strip() or patch.endswith("\n") else patch + "\n")
        (d / "eval.sh").write_text(specs[iid].eval_script); (d / "run.sh").write_text(RUN_SH.format(d=d))
        jobs[iid] = submit(iid, d, args); (d / "submitted").write_text(f"{time.time()}\n{jobs[iid]}\n")
        print(f"submitted {iid}: {jobs[iid]}", flush=True)
    if args.dry_run:
        return

    while time.time() - t0 < args.wait and not all((run_dir / i / "done").exists() for i in preds):
        time.sleep(10)
    out = []
    for iid, pred in preds.items():
        d = run_dir / iid; r = {"instance_id": iid, "job_id": jobs.get(iid)}
        if (d / "started").exists():
            r["queue_to_start_s"] = float((d / "started").read_text()) - float((d / "submitted").read_text().split()[0])
        if not (d / "done").exists():
            r |= {"status": "unfinished", "resolved": False}
        elif (d / "patch_failed").exists():
            r |= {"status": "patch_failed", "resolved": False}
        else:
            rep = get_eval_report(test_spec=specs.get(iid) or make_test_spec(inst[iid]), prediction=pred,
                                  test_log_path=str(d / "test_output.txt"), include_tests_status=True)
            (d / "report.json").write_text(json.dumps(rep, indent=2))
            r |= {"status": "completed", "resolved": bool(rep.get(iid, rep).get("resolved", False)),
                  "run_s": float((d / "done").read_text()) - float((d / "started").read_text())}
        out.append(r); print(json.dumps(r), flush=True)
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    n = sum(r["resolved"] for r in out)
    print(f"\n{n}/{len(out)} resolved, wall {time.time() - t0:.0f}s; results in {run_dir / 'results.jsonl'}")


if __name__ == "__main__":
    main()

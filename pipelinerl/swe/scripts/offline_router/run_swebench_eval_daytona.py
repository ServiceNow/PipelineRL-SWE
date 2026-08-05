#!/usr/bin/env python3
"""Evaluate SWE-bench Verified predictions via Daytona sandboxes.

Unlike run_swesmith_eval_daytona.py, the Docker image already contains the
BUGGY state (base commit) — we just apply the fix patch and run the test suite.
No git checkout of a bug commit is needed.

Usage:
  python run_swebench_eval_daytona.py \
    --predictions-path /mnt/.../predictions_oss120b.jsonl \
    --run-id verified_oss120b_real \
    --concurrency 32
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import traceback
from pathlib import Path
from typing import Any

from daytona import AsyncDaytona, CreateSandboxFromImageParams
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec
from tqdm.auto import tqdm

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
]

ACTIVATE = "source /opt/miniconda3/bin/activate testbed 2>/dev/null || source /opt/miniconda3/bin/activate && conda activate testbed 2>/dev/null || true"
TTL_MINUTES = int(os.environ.get("DAYTONA_TTL", "5"))
ARCH = "x86_64"


def _image_name(instance_id: str) -> str:
    return f"swebench/sweb.eval.{ARCH}.{instance_id.lower()}:latest"


async def _eval_one(
    daytona: AsyncDaytona,
    pred: dict[str, Any],
    test_spec: Any,
    run_id: str,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    instance_id = pred[KEY_INSTANCE_ID]
    log_dir = Path(RUN_EVALUATION_LOG_DIR) / run_id / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)

    async with sem:
        sandbox = None
        try:
            sandbox = await daytona.create(
                CreateSandboxFromImageParams(
                    image=_image_name(instance_id),
                    auto_delete_interval=TTL_MINUTES,
                ),
                timeout=120,
            )

            patch = pred.get(KEY_PREDICTION, "").strip()
            if patch:
                await sandbox.fs.upload_file(patch.encode(), "/tmp/patch.diff")
                applied = False
                for cmd in GIT_APPLY_CMDS:
                    r = await sandbox.process.exec(
                        f"cd /testbed && {cmd} /tmp/patch.diff", timeout=30
                    )
                    if r.exit_code == 0:
                        applied = True
                        break
                if not applied:
                    (log_dir / LOG_REPORT).write_text(
                        json.dumps({"resolved": False, "patch_failed": True}, indent=2)
                    )
                    return {"status": "patch_failed", "resolved": False, "instance_id": instance_id}

                # Revert test files in case the patch accidentally modified them
                test_files = list(test_spec.fail_to_pass) + list(test_spec.pass_to_pass)
                if test_files:
                    # test files are test function names, not paths — skip revert
                    pass

            eval_sh = test_spec.eval_script
            await sandbox.fs.upload_file(eval_sh.encode(), "/tmp/eval.sh")
            r = await sandbox.process.exec(
                "/bin/bash /tmp/eval.sh", timeout=test_spec.timeout + 60
            )
            test_output = r.result or ""
            test_log_path = log_dir / LOG_TEST_OUTPUT
            test_log_path.write_text(test_output)

            report = get_eval_report(
                test_spec=test_spec,
                prediction=pred,
                log_path=test_log_path,
                include_tests_status=True,
            )
            report[KEY_MODEL] = pred.get(KEY_MODEL, "unknown")
            (log_dir / LOG_REPORT).write_text(json.dumps(report, indent=2))
            resolved = bool(report.get("resolved", False))
            return {"status": "completed", "resolved": resolved, "instance_id": instance_id}

        except asyncio.TimeoutError:
            (log_dir / LOG_REPORT).write_text(
                json.dumps({"timed_out": True, "resolved": False}, indent=2)
            )
            return {"status": "timeout", "resolved": False, "instance_id": instance_id}
        except Exception as exc:
            print(f"\n[error] {instance_id}: {exc}")
            traceback.print_exc()
            (log_dir / LOG_REPORT).write_text(
                json.dumps({"error": str(exc), "resolved": False}, indent=2)
            )
            return {"status": "error", "resolved": False, "instance_id": instance_id}
        finally:
            if sandbox is not None:
                try:
                    await sandbox.delete()
                except Exception:
                    pass


async def run_all(
    predictions: dict[str, dict[str, Any]],
    test_specs: dict[str, Any],
    run_id: str,
    concurrency: int,
    redo_existing: bool,
) -> list[dict[str, Any]]:
    log_dir_parent = Path(RUN_EVALUATION_LOG_DIR) / run_id
    remaining = dict(predictions)
    if not redo_existing and log_dir_parent.exists():
        skip = sum(
            1 for iid in list(remaining)
            if (log_dir_parent / iid / LOG_REPORT).exists()
            and not remaining.pop(iid, None)
        )
        if skip:
            print(f"Skipping {skip} already-completed instances. Remaining: {len(remaining)}")

    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []

    async with AsyncDaytona() as daytona:
        tasks = [
            asyncio.create_task(
                _eval_one(daytona, pred, test_specs[iid], run_id, sem)
            )
            for iid, pred in remaining.items()
            if iid in test_specs
        ]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
            result = await coro
            results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate SWE-bench Verified predictions via Daytona.")
    parser.add_argument("--predictions-path", required=True,
                        help="JSONL with {instance_id, model_patch, model} per line")
    parser.add_argument("--run-id", required=True, help="Unique run identifier for log dir")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--redo-existing", action="store_true")
    parser.add_argument("--output-path", default=None,
                        help="Write summary JSONL to this path (default: <predictions-path>.results.jsonl)")
    args = parser.parse_args()

    preds_path = Path(args.predictions_path)
    raw_preds = {}
    with preds_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            iid = row.get("instance_id") or row.get(KEY_INSTANCE_ID)
            if iid:
                raw_preds[iid] = {
                    KEY_INSTANCE_ID: iid,
                    KEY_PREDICTION: row.get("model_patch") or row.get(KEY_PREDICTION, ""),
                    KEY_MODEL: row.get("model", "unknown"),
                }
    print(f"Loaded {len(raw_preds)} predictions from {preds_path}")

    print("Loading SWE-bench Verified dataset from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test", trust_remote_code=True)
    instances = {row["instance_id"]: dict(row) for row in ds}

    missing = [iid for iid in raw_preds if iid not in instances]
    if missing:
        print(f"Warning: {len(missing)} predictions not found in dataset, skipping")
        for iid in missing:
            del raw_preds[iid]

    print("Building test specs...")
    test_specs = {}
    for iid, inst in instances.items():
        if iid not in raw_preds:
            continue
        try:
            test_specs[iid] = make_test_spec(inst)
        except Exception as e:
            print(f"  Warning: could not build test spec for {iid}: {e}")
    print(f"Built {len(test_specs)} test specs")

    results = asyncio.run(run_all(
        predictions=raw_preds,
        test_specs=test_specs,
        run_id=args.run_id,
        concurrency=args.concurrency,
        redo_existing=args.redo_existing,
    ))

    n_resolved = sum(1 for r in results if r.get("resolved"))
    n_total = len(results)
    print(f"\nResults: {n_resolved}/{n_total} resolved ({100*n_resolved/n_total:.1f}%)" if n_total else "No results")

    out_path = Path(args.output_path) if args.output_path else preds_path.with_suffix(".results.jsonl")
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()

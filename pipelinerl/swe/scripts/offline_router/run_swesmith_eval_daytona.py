#!/usr/bin/env python
"""Evaluate SWE-Smith predictions via Daytona sandboxes instead of local Docker.

Drop-in replacement for run_swesmith_eval_with_prune.py.
Each instance gets its own ephemeral sandbox (auto_delete_interval=TTL_MINUTES).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import tempfile
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
from swesmith.harness.grading import get_eval_report
from swesmith.profiles import registry
from tqdm.auto import tqdm

FAKE_INDEX_RE = re.compile(r"^index 0{7,40}\.\.1{7,40}(?: \d+)?$")
DIFF_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
]

ACTIVATE = "source /opt/miniconda3/bin/activate && conda activate testbed"

TTL_MINUTES = int(os.environ.get("DAYTONA_TTL", "3"))


# --- patch hash repair (mirrors repair_swe_smith_eval_patch_hashes.py) ---

def git_blob_hash(text: str) -> str:
    data = text.encode("utf-8", errors="surrogateescape")
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()[:7]


def paths_from_patch(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        m = DIFF_RE.match(line)
        if m:
            path = m.group(2)
            if path not in paths:
                paths.append(path)
    return paths


def strip_fake_index_lines(patch: str) -> str:
    lines = [line for line in patch.splitlines() if not FAKE_INDEX_RE.match(line.strip())]
    text = "\n".join(lines)
    if text and patch.endswith("\n"):
        text += "\n"
    return text


def apply_patch_to_files(file_contents: dict[str, str], patch: str, touched_paths: list[str]) -> dict[str, str] | None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for path in touched_paths:
            if path not in file_contents:
                return None
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file_contents[path], encoding="utf-8", errors="surrogateescape")
        patch_path = root / "prediction.patch"
        patch_path.write_text(strip_fake_index_lines(patch), encoding="utf-8", errors="surrogateescape")
        result = subprocess.run(
            ["patch", "-p1", "--batch", "--forward", "--reject-file=-", "-i", str(patch_path)],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            return None
        return {path: (root / path).read_text(encoding="utf-8", errors="surrogateescape") for path in touched_paths}


def replace_fake_index_lines(patch: str, old_contents: dict[str, str], new_contents: dict[str, str]) -> str:
    current_path: str | None = None
    out: list[str] = []
    for line in patch.splitlines():
        m = DIFF_RE.match(line)
        if m:
            current_path = m.group(2)
            out.append(line)
            continue
        if FAKE_INDEX_RE.match(line.strip()):
            if current_path and current_path in old_contents and current_path in new_contents:
                out.append(f"index {git_blob_hash(old_contents[current_path])}..{git_blob_hash(new_contents[current_path])} 100644")
            else:
                out.append(line)
            continue
        out.append(line)
    text = "\n".join(out)
    if text and patch.endswith("\n"):
        text += "\n"
    return text


def parse_file_contents(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    return {}


def repair_patch(patch: str, instance: dict[str, Any]) -> str:
    """Replace fake index lines with real git blob hashes; fall back to stripping if patch won't apply."""
    if not patch:
        return patch
    has_fake = any(FAKE_INDEX_RE.match(line.strip()) for line in patch.splitlines())
    if not has_fake:
        return patch
    file_contents = parse_file_contents(
        instance.get("repair_file_contents") or instance.get("gold_file_contents") or {}
    )
    if not file_contents:
        return strip_fake_index_lines(patch)
    touched = paths_from_patch(patch)
    new_contents = apply_patch_to_files(file_contents, patch, touched)
    if new_contents is None:
        return strip_fake_index_lines(patch)
    return replace_fake_index_lines(patch, file_contents, new_contents)


# --- prediction loading ---

def normalize_prediction(pred: dict[str, Any]) -> dict[str, Any]:
    out = dict(pred)
    patch = out.get(KEY_PREDICTION) or out.get("model_patch") or out.get("patch") or ""
    out[KEY_PREDICTION] = patch
    out["patch"] = patch
    out["model_patch"] = patch
    out.setdefault(KEY_MODEL, out.get("model_name_or_path") or out.get("model") or "unknown")
    return out


def load_rows(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if path.endswith(".json"):
        return json.loads(p.read_text())
    raise ValueError(f"Expected .json or .jsonl, got {path}")


ALL_LANGUAGE_DATASETS = [
    "SWE-bench/SWE-smith-py",
    "SWE-bench/SWE-smith-rs",
    "SWE-bench/SWE-smith-java",
    "SWE-bench/SWE-smith-go",
]


def load_dataset_hf(dataset_name: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    return list(ds)


def load_datasets_hf(dataset_names: list[str], split: str) -> dict[str, Any]:
    from datasets import load_dataset
    merged: dict[str, Any] = {}
    for name in dataset_names:
        print(f"Loading HuggingFace dataset: {name} / {split}")
        try:
            ds = load_dataset(name, split=split)
            for row in ds:
                iid = row.get(KEY_INSTANCE_ID)
                if iid and iid not in merged:
                    merged[iid] = row
            print(f"  → {len(ds)} instances (total merged so far: {len(merged)})")
        except Exception as exc:
            print(f"  [warn] failed to load {name}: {exc}")
    return merged


def enrich_with_file_contents(dataset: dict[str, Any], bugged_context_path: str) -> None:
    """Merge repair_file_contents from local bugged_context dataset (keyed by 'id')."""
    from datasets import load_from_disk
    print(f"Loading file contents from {bugged_context_path}")
    bc = load_from_disk(bugged_context_path)
    enriched = 0
    for row in bc:
        iid = row.get("id") or row.get("instance_id")
        if iid and iid in dataset:
            dataset[iid]["repair_file_contents"] = row.get("repair_file_contents") or row.get("gold_file_contents")
            enriched += 1
    print(f"Enriched {enriched}/{len(dataset)} instances with file contents")


# --- eval ---

async def run_one(
    daytona: AsyncDaytona,
    pred: dict[str, Any],
    instance: dict[str, Any],
    run_id: str,
    f2p_only: bool,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    instance_id = pred[KEY_INSTANCE_ID]
    rp = registry.get_from_inst(instance)
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)

    async with sem:
        sandbox = None
        try:
            params = CreateSandboxFromImageParams(
                image=instance["image_name"],
                auto_delete_interval=TTL_MINUTES,
            )
            sandbox = await daytona.create(params, timeout=120)

            # The Docker image contains CLEAN code. The instance_id is a branch with 2 commits:
            #   HEAD:   bug commit + F2P test files removed (so tests don't trivially pass)
            #   HEAD~1: bug commit with test files present ← eval state we need
            # git fetch failure is non-fatal (branches are local); both checkouts must succeed.
            r = await sandbox.process.exec(
                f"cd /testbed && (git fetch || true) && git checkout {instance_id} && git checkout HEAD~1",
                timeout=60,
            )
            if r.exit_code != 0:
                (log_dir / LOG_TEST_OUTPUT).write_text(r.result or "")
                (log_dir / LOG_REPORT).write_text(json.dumps({"resolved": False, "checkout_failed": True}, indent=4))
                return {"status": "checkout_failed", "resolved": False, "instance_id": instance_id}

            patch = repair_patch(pred[KEY_PREDICTION], instance)
            if patch:
                await sandbox.fs.upload_file(patch.encode(), "/tmp/patch.diff")
                applied = False
                for cmd_template in GIT_APPLY_CMDS:
                    if "/tmp/patch.diff" not in cmd_template:
                        full_cmd = f"cd /testbed && {cmd_template} /tmp/patch.diff"
                    else:
                        full_cmd = f"cd /testbed && {cmd_template}"
                    r = await sandbox.process.exec(full_cmd, timeout=30)
                    if r.exit_code == 0:
                        applied = True
                        break
                if not applied:
                    (log_dir / LOG_TEST_OUTPUT).write_text("")
                    (log_dir / LOG_REPORT).write_text(json.dumps({"resolved": False, "patch_failed": True}, indent=4))
                    return {"status": "patch_failed", "resolved": False, "instance_id": instance_id}

                # Revert test files in case the prediction patch accidentally modified them
                f2p_files, p2p_files = rp.get_test_files(instance)
                test_files = f2p_files + p2p_files
                if test_files:
                    await sandbox.process.exec(
                        f"cd /testbed && git checkout -- {' '.join(test_files)}",
                        timeout=30,
                    )

            test_cmd, _ = rp.get_test_cmd(instance, f2p_only=f2p_only)
            eval_sh = "\n".join([
                "#!/bin/bash",
                "set -uxo pipefail",
                "cd /testbed",
                ": '>>>>> Start Test Output'",
                test_cmd,
                ": '>>>>> End Test Output'",
                "",
            ])
            await sandbox.fs.upload_file(eval_sh.encode(), "/tmp/eval.sh")
            r = await sandbox.process.exec(
                f"{ACTIVATE} && /bin/bash /tmp/eval.sh",
                timeout=rp.timeout + 30,
            )
            test_output = r.result or ""
            test_log_path = log_dir / LOG_TEST_OUTPUT
            test_log_path.write_text(test_output)

            report = get_eval_report(pred, instance, test_log_path, f2p_only=f2p_only)
            report[KEY_MODEL] = pred[KEY_MODEL]
            (log_dir / LOG_REPORT).write_text(json.dumps(report, indent=4))
            resolved = bool(report.get("resolved", False))
            return {"status": "completed", "resolved": resolved, "instance_id": instance_id}

        except asyncio.TimeoutError:
            (log_dir / LOG_REPORT).write_text(json.dumps({"timed_out": True, "resolved": False}, indent=4))
            return {"status": "timeout", "resolved": False, "instance_id": instance_id}
        except Exception as exc:
            print(f"\n[error] {instance_id}: {exc}")
            traceback.print_exc()
            (log_dir / LOG_REPORT).write_text(json.dumps({"error": str(exc), "resolved": False}, indent=4))
            return {"status": "error", "resolved": False, "instance_id": instance_id}
        finally:
            if sandbox is not None:
                try:
                    await sandbox.delete()
                except Exception:
                    pass


async def run_all(
    predictions: dict[str, Any],
    dataset: dict[str, Any],
    run_id: str,
    f2p_only: bool,
    concurrency: int,
    redo_existing: bool,
) -> dict[str, Any]:
    log_dir_parent = RUN_EVALUATION_LOG_DIR / run_id
    remaining = dict(predictions)
    if not redo_existing and log_dir_parent.exists():
        completed_count = 0
        for iid in list(remaining):
            if (log_dir_parent / iid / LOG_REPORT).exists():
                del remaining[iid]
                completed_count += 1
        if completed_count:
            print(f"Skipping {completed_count} already-completed instances. Remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to evaluate.")
        return {}

    payloads = []
    for iid, pred in remaining.items():
        inst = dataset.get(iid)
        if inst is None:
            print(f"[warn] instance {iid} not in dataset, skipping")
            continue
        payloads.append((pred, inst))

    sem = asyncio.Semaphore(concurrency)
    stats: dict[str, int] = {"resolved": 0, "unresolved": 0, "timeout": 0, "error": 0, "patch_failed": 0, "checkout_failed": 0}

    async with AsyncDaytona() as daytona:
        with tqdm(total=len(payloads), desc="Daytona eval") as pbar:
            tasks = [
                asyncio.create_task(run_one(daytona, pred, inst, run_id, f2p_only, sem))
                for pred, inst in payloads
            ]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                s = result["status"]
                if s == "completed":
                    if result["resolved"]:
                        stats["resolved"] += 1
                    else:
                        stats["unresolved"] += 1
                else:
                    stats[s] = stats.get(s, 0) + 1
                pbar.set_postfix(stats)
                pbar.update()

    return stats


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate SWE-Smith predictions via Daytona sandboxes.")
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--dataset_path", default=None, help="Local .json/.jsonl or omit to fetch from HF")
    parser.add_argument("--hf_dataset", default=None, help="Single HuggingFace dataset (legacy; prefer --hf_datasets)")
    parser.add_argument("--hf_datasets", nargs="+", default=None,
                        help="HuggingFace dataset names to load in sequence and merge (default: all language splits)")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--f2p_only", action="store_true", default=True)
    parser.add_argument("--no_f2p_only", dest="f2p_only", action="store_false")
    parser.add_argument("--redo_existing", action="store_true")
    parser.add_argument("--instance_ids", nargs="+")
    parser.add_argument("--bugged_context_path", default=None, help="Local HF dataset dir with repair_file_contents (for patch hash repair)")
    args = parser.parse_args()

    predictions_raw = load_rows(args.predictions_path)
    predictions = {row[KEY_INSTANCE_ID]: normalize_prediction(row) for row in predictions_raw}
    if args.instance_ids:
        predictions = {k: v for k, v in predictions.items() if k in set(args.instance_ids)}

    if args.dataset_path:
        dataset_rows = load_rows(args.dataset_path)
        dataset = {row[KEY_INSTANCE_ID]: row for row in dataset_rows}
    elif args.hf_dataset:
        print(f"Fetching dataset from HuggingFace: {args.hf_dataset} / {args.hf_split}")
        dataset_rows = load_dataset_hf(args.hf_dataset, args.hf_split)
        dataset = {row[KEY_INSTANCE_ID]: row for row in dataset_rows}
    else:
        names = args.hf_datasets or ALL_LANGUAGE_DATASETS
        dataset = load_datasets_hf(names, args.hf_split)

    missing = [iid for iid in predictions if iid not in dataset]
    if missing:
        print(f"[warn] {len(missing)} prediction IDs not found in dataset: {missing[:5]}...")

    if args.bugged_context_path:
        enrich_with_file_contents(dataset, args.bugged_context_path)

    stats = asyncio.run(run_all(
        predictions=predictions,
        dataset=dataset,
        run_id=args.run_id,
        f2p_only=args.f2p_only,
        concurrency=args.concurrency,
        redo_existing=args.redo_existing,
    ))

    log_dir_parent = RUN_EVALUATION_LOG_DIR / args.run_id
    ids_resolved, ids_unresolved = [], []
    for pred in predictions.values():
        iid = pred[KEY_INSTANCE_ID]
        report_path = log_dir_parent / iid / LOG_REPORT
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        if report.get("resolved", False):
            ids_resolved.append(iid)
        else:
            ids_unresolved.append(iid)

    summary = {
        "resolved": len(ids_resolved),
        "unresolved": len(ids_unresolved),
        "total": len(predictions),
        "ids_resolved": ids_resolved,
        "ids_unresolved": ids_unresolved,
    }
    (log_dir_parent / LOG_REPORT).write_text(json.dumps(summary, indent=4))
    print(f"\nResolved {len(ids_resolved)}/{len(predictions)} instances.")
    print(f"Report: {log_dir_parent / LOG_REPORT}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

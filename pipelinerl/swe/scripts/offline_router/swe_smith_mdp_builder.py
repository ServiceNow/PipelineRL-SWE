"""Build full-sandbox-execution SWE-Smith tensors from saved multi-rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pipelinerl.swe.load_datasets import load_local_swe_dataset
from pipelinerl.swe.scripts.livecodebench.mdp_utils import (
    SPLIT_SCHEMA_VERSION,
    TENSOR_SCHEMA_VERSION,
    write_split_manifest,
)
from pipelinerl.swe.scripts.offline_router.common import problem_id_from_item


MODEL_SPECS = [
    ("scout", "qwen3_4b_instruct_2507", "Qwen_Qwen3-4B-Instruct-2507"),
    ("oss20", "gpt_oss_20b", "openai_gpt-oss-20b"),
    ("oss120", "gpt_oss_120b", "openai_gpt-oss-120b"),
]


def read_outputs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pid = str(row.get("instance_id") or row.get("problem_id") or "").strip()
        if not pid or pid in rows:
            raise ValueError(f"Missing or duplicate instance ID in {path}: {pid!r}")
        rows[pid] = row
    return rows


def read_reports(run_dir: Path) -> dict[str, dict[str, Any]]:
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    rows = {}
    for path in sorted(run_dir.glob("*/report.json")):
        report = json.loads(path.read_text())
        if "resolved" in report:
            rows[path.parent.name] = report
    return rows


def execution_feedback(report: dict[str, Any]) -> str:
    status = "PASSED" if bool(report["resolved"]) else "FAILED"
    parts = [f"Sandbox execution: {status}"]
    for group in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        result = (report.get("tests_status") or {}).get(group) or {}
        passed = [str(value) for value in result.get("success") or []]
        failed = [str(value) for value in result.get("failure") or []]
        parts.append(f"{group}: passed={len(passed)}/{len(passed) + len(failed)}")
        if failed:
            suffix = " ..." if len(failed) > 8 else ""
            parts.append(f"{group} failures: {', '.join(failed[:8])}{suffix}")
    if not bool(report.get("patch_exists", True)):
        parts.append("Patch was empty or could not be applied")
    return "; ".join(parts)


def load_problems(dataset_path: str, dataset_name: str) -> dict[str, dict[str, Any]]:
    rows = load_local_swe_dataset(
        dataset_names=[dataset_name] if dataset_name else [], dataset_path=dataset_path,
        shuffle=False, seed=0, dataset_label=dataset_name or None, max_samples=None,
    )
    result = {}
    for row in rows:
        pid = problem_id_from_item(row)
        statement = str(row.get("problem_statement") or "").strip()
        if statement:
            result[pid] = {
                "problem_id": pid, "problem_statement": statement,
                "repo": str(row.get("repo") or ""),
                "base_commit": str(row.get("base_commit") or ""),
            }
    return result


def write_external_holdout(
    path: Path, problem_ids: list[str], ids_path: Path, seed: int, cal_fraction: float
) -> dict[str, Any]:
    heldout = {line.strip() for line in ids_path.read_text().splitlines() if line.strip()}
    all_ids = set(problem_ids)
    if heldout - all_ids:
        raise ValueError(f"{len(heldout - all_ids)} held-out IDs lack complete tensors")
    development = sorted(all_ids - heldout)
    if not heldout or len(development) < 2:
        raise ValueError("External holdout needs non-empty test and at least two development IDs")
    rng = np.random.default_rng(seed)
    shuffled = [development[int(i)] for i in rng.permutation(len(development))]
    n_cal = min(max(int(round(len(shuffled) * cal_fraction)), 1), len(shuffled) - 1)
    manifest = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "method": "external_test_ids_plus_development_permutation", "seed": seed,
        "development_calibration_fraction": cal_fraction,
        "train_problem_ids": sorted(shuffled[n_cal:]),
        "calibration_problem_ids": sorted(shuffled[:n_cal]),
        "test_problem_ids": sorted(heldout), "heldout_ids_path": str(ids_path),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", action="append", required=True)
    parser.add_argument("--report-root", action="append", required=True)
    parser.add_argument("--report-split", action="append", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-name", default="swe_smith_test_bugged_context")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-draws", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--heldout-ids")
    parser.add_argument("--development-calibration-fraction", type=float, default=1 / 3)
    parser.add_argument("--require-complete", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (len(args.trace_root) == len(args.report_root) == len(args.report_split)):
        raise ValueError("Trace-root/report-root/report-split counts must match")
    if args.num_draws < 2:
        raise ValueError("Sequential resampling requires at least two draws")
    problems = load_problems(args.dataset_path, args.dataset_name)
    keys = [(slot, draw) for slot, _, _ in MODEL_SPECS for draw in range(args.num_draws)]
    outputs: dict[tuple[str, int], dict[str, dict[str, Any]]] = {key: {} for key in keys}
    reports: dict[tuple[str, int], dict[str, dict[str, Any]]] = {key: {} for key in keys}
    sources = []
    for trace_value, report_value, split in zip(
        args.trace_root, args.report_root, args.report_split, strict=True
    ):
        trace_root, report_root = Path(trace_value), Path(report_value)
        source = {"trace_root": trace_value, "report_root": report_value, "report_split": split, "routes": {}}
        for slot, route, model_slug in MODEL_SPECS:
            for draw in range(args.num_draws):
                output_path = trace_root / route / f"rollout_{draw}" / "collect" / "models" / model_slug / "outputs.jsonl"
                run_dir = report_root / f"swe_smith_{split}_{route}_r{draw}"
                new_outputs, new_reports = read_outputs(output_path), read_reports(run_dir)
                if set(outputs[(slot, draw)]) & set(new_outputs):
                    raise ValueError(f"Duplicate IDs across collections for {slot} draw {draw}")
                outputs[(slot, draw)].update(new_outputs)
                reports[(slot, draw)].update(new_reports)
                source["routes"][f"{slot}_d{draw}"] = {"outputs": len(new_outputs), "real_reports": len(new_reports)}
        sources.append(source)

    pids = sorted(set(problems).intersection(*(set(table) for table in outputs.values())))
    shape = (len(pids), len(MODEL_SPECS), args.num_draws)
    final = np.zeros(shape, bool)
    valid = np.zeros(shape, bool)
    proxy = np.zeros(shape, np.float32)
    prompt_tokens = np.zeros(shape, np.float32)
    completion_tokens = np.zeros(shape, np.float32)
    records = []
    for pi, pid in enumerate(pids):
        for mi, (slot, _, _) in enumerate(MODEL_SPECS):
            for draw in range(args.num_draws):
                output, report = outputs[(slot, draw)][pid], reports[(slot, draw)].get(pid)
                if report is None:
                    continue
                valid[pi, mi, draw], final[pi, mi, draw] = True, bool(report["resolved"])
                proxy[pi, mi, draw] = float(output.get("proxy_reward") or 0)
                prompt_tokens[pi, mi, draw] = float(output.get("prompt_tokens") or 0)
                completion_tokens[pi, mi, draw] = float(output.get("output_tokens") or 0)
                patch = str(output.get("model_patch") or "").strip()
                records.append({
                    "problem_id": pid, "model_slot": slot, "draw_index": draw,
                    "code": patch or str(output.get("output_text") or ""), "model_patch": patch,
                    "full_execution_feedback": execution_feedback(report),
                    "final_outcome": bool(report["resolved"]),
                    "proxy_verifier_score": float(output.get("proxy_reward") or 0),
                    "proxy_verifier_outcome": bool(output.get("proxy_success", False)),
                    "prompt_tokens": float(output.get("prompt_tokens") or 0),
                    "completion_tokens": float(output.get("output_tokens") or 0),
                })

    complete = valid.all(axis=(1, 2))
    keep = complete if args.require_complete else valid.any(axis=(1, 2))
    if not keep.any():
        raise ValueError("No problems satisfy the validity policy")
    kept_ids, kept_set = [pids[i] for i in np.flatnonzero(keep)], set()
    kept_set.update(kept_ids)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "tensors.npz", final_outcome=final[keep], execution_outcome=final[keep],
        proxy_verifier_score=proxy[keep], valid=valid[keep],
        prompt_tokens=prompt_tokens[keep], completion_tokens=completion_tokens[keep],
        problem_ids=np.array(kept_ids), model_slots=np.array([x[0] for x in MODEL_SPECS]),
        schema_version=np.array(TENSOR_SCHEMA_VERSION),
    )
    (out / "problems.jsonl").write_text("".join(json.dumps(problems[pid]) + "\n" for pid in kept_ids))
    (out / "draw_records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records if row["problem_id"] in kept_set))
    if args.heldout_ids:
        manifest = write_external_holdout(
            out / "split_manifest.json", kept_ids, Path(args.heldout_ids),
            args.split_seed, args.development_calibration_fraction,
        )
    else:
        manifest = write_split_manifest(
            out / "split_manifest.json", kept_ids, seed=args.split_seed,
            train_fraction=args.train_fraction, calibration_fraction=args.calibration_fraction,
        )
    summary = {
        "schema_version": TENSOR_SCHEMA_VERSION,
        "protocol": "swe_smith_scout_first_full_sandbox_execution",
        "model_slots": [x[0] for x in MODEL_SPECS], "num_draws": args.num_draws,
        "candidate_problems": len(pids), "complete_problems": int(complete.sum()),
        "kept_problems": len(kept_ids), "missing_reports_are_failures": False,
        "split_counts": {name: len(manifest[f"{name}_problem_ids"]) for name in ("train", "calibration", "test")},
        "solve_rates": {slot: float(final[keep, mi].mean()) for mi, (slot, _, _) in enumerate(MODEL_SPECS)},
        "collections": sources,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

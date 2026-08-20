#!/usr/bin/env python3
"""Serially regrade a corrected LCB collection and validate its infrastructure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pipelinerl.swe.scripts.livecodebench.collect_lcb_trajectories import (
    LCB_DATASET_REVISION,
    LCB_EVALUATOR_COMMIT,
    build_labels_parquet,
    evaluate_code,
    extract_code,
    format_test_feedback,
    is_evaluator_infrastructure_error,
    load_lcb,
    problem_id,
)


def _read_latest(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if path.exists():
        with open(path) as in_f:
            for line in in_f:
                row = json.loads(line)
                latest[row["problem_id"]] = row
    return latest


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as out_f:
        for row in rows:
            out_f.write(json.dumps(row) + "\n")
    temporary.replace(path)


def regrade_scout(
    dataset_rows: list[dict[str, Any]],
    path: Path,
    timeout: int,
    feedback_tests: str,
    dataset_revision: str,
    errors_only: bool = False,
) -> list[dict[str, Any]]:
    existing = _read_latest(path)
    missing = [problem_id(row) for row in dataset_rows if problem_id(row) not in existing]
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} scout rows")

    repaired = []
    for index, dataset_row in enumerate(dataset_rows, start=1):
        pid = problem_id(dataset_row)
        row = existing[pid]
        has_runner_error = is_evaluator_infrastructure_error(
            row.get("_lcb_feedback_result_codes"),
            row.get("_lcb_feedback_eval_metadata"),
            row.get("_tf_failing"),
        ) or is_evaluator_infrastructure_error(
            row.get("_lcb_result_codes"), row.get("_lcb_eval_metadata")
        )
        if errors_only and not has_runner_error:
            repaired.append(row)
            continue
        code = row.get("patch_text") or extract_code(row.get("full_output", ""))
        full_suite_size = (
            dataset_row["_n_public_tests"] + dataset_row["_n_private_tests"]
        )
        feedback_suite_size = (
            dataset_row["_n_public_tests"]
            if feedback_tests == "public"
            else full_suite_size
        )
        feedback_key = (
            "_public_evaluation_sample"
            if feedback_tests == "public"
            else "_evaluation_sample"
        )
        feedback_report = evaluate_code(
            code, dataset_row, timeout=timeout, sample_key=feedback_key
        )
        full_report = (
            evaluate_code(code, dataset_row, timeout=timeout)
            if feedback_tests == "public"
            else feedback_report
        )
        row.update(
            {
                "patch_text": code,
                "scout_correct": full_report["resolved"],
                "test_feedback": format_test_feedback(
                    feedback_report, feedback_suite_size
                ),
                "_tf_failing": feedback_report["failing"],
                "_tf_passing": feedback_report["passing"],
                "_tf_resolved": feedback_report["resolved"],
                "_tf_patch_exists": bool(code.strip()),
                "_tf_total": feedback_suite_size,
                "_lcb_feedback_result_codes": feedback_report["result_codes"],
                "_lcb_feedback_eval_metadata": feedback_report["metadata"],
                "_lcb_result_codes": full_report["result_codes"],
                "_lcb_eval_metadata": full_report["metadata"],
                "_lcb_evaluator_commit": LCB_EVALUATOR_COMMIT,
                "_lcb_dataset_revision": dataset_revision,
                "_lcb_feedback_tests": feedback_tests,
                "_lcb_test_count": feedback_suite_size,
                "_lcb_full_test_count": full_suite_size,
            }
        )
        repaired.append(row)
        if index % 50 == 0:
            print(f"Regraded scout {index}/{len(dataset_rows)}", flush=True)

    _write_jsonl_atomic(path, repaired)
    return repaired


def regrade_oracle(
    dataset_rows: list[dict[str, Any]],
    path: Path,
    timeout: int,
    dataset_revision: str,
    errors_only: bool = False,
) -> list[dict[str, Any]]:
    existing = _read_latest(path)
    missing = [problem_id(row) for row in dataset_rows if problem_id(row) not in existing]
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} oracle rows")

    repaired = []
    for index, dataset_row in enumerate(dataset_rows, start=1):
        pid = problem_id(dataset_row)
        row = existing[pid]
        has_runner_error = is_evaluator_infrastructure_error(
            row.get("result_codes"), row.get("eval_metadata")
        )
        if errors_only and not has_runner_error:
            repaired.append(row)
            continue
        full_output = row.get("full_output", "")
        if full_output.strip():
            code = row.get("code") or extract_code(full_output)
            report = evaluate_code(code, dataset_row, timeout=timeout)
            row.update(
                {
                    "resolved": report["resolved"],
                    "code": code,
                    "result_codes": report["result_codes"],
                    "eval_metadata": report["metadata"],
                    "_lcb_evaluator_commit": LCB_EVALUATOR_COMMIT,
                    "_lcb_dataset_revision": dataset_revision,
                }
            )
        repaired.append(row)
        if index % 50 == 0:
            print(f"Regraded oracle {index}/{len(dataset_rows)}", flush=True)

    _write_jsonl_atomic(path, repaired)
    return repaired


def validate_split(
    split: str,
    dataset_rows: list[dict[str, Any]],
    output_dir: Path,
    dataset_revision: str,
) -> None:
    expected_ids = {problem_id(row) for row in dataset_rows}
    scout = _read_latest(output_dir / f"scout_{split}.jsonl")
    oracle = _read_latest(output_dir / f"oracle_{split}.jsonl")

    errors = []
    if set(scout) != expected_ids:
        errors.append(f"scout IDs {len(scout)}/{len(expected_ids)}")
    if set(oracle) != expected_ids:
        errors.append(f"oracle IDs {len(oracle)}/{len(expected_ids)}")

    scout_runner_errors = [
        pid
        for pid, row in scout.items()
        if is_evaluator_infrastructure_error(
            row.get("_lcb_feedback_result_codes"),
            row.get("_lcb_feedback_eval_metadata"),
            row.get("_tf_failing"),
        )
        or is_evaluator_infrastructure_error(
            row.get("_lcb_result_codes"), row.get("_lcb_eval_metadata")
        )
    ]
    oracle_runner_errors = [
        pid
        for pid, row in oracle.items()
        if is_evaluator_infrastructure_error(
            row.get("result_codes"), row.get("eval_metadata")
        )
    ]
    oracle_call_errors = [
        pid
        for pid, row in oracle.items()
        if not row.get("full_output", "").strip()
        and str((row.get("eval_metadata") or {}).get("error_message", ""))
        != "EmptyGeneration"
    ]
    bad_provenance = [
        pid
        for pid, row in [*scout.items(), *oracle.items()]
        if row.get("_lcb_evaluator_commit") != LCB_EVALUATOR_COMMIT
        or row.get("_lcb_dataset_revision") != dataset_revision
    ]

    if scout_runner_errors:
        errors.append(f"scout runner errors={len(scout_runner_errors)}")
    if oracle_runner_errors:
        errors.append(f"oracle runner errors={len(oracle_runner_errors)}")
    if oracle_call_errors:
        errors.append(f"oracle call errors={len(oracle_call_errors)}")
    if bad_provenance:
        errors.append(f"bad provenance={len(bad_provenance)}")

    labels_path = output_dir / split / "labels.parquet"
    trajectory_path = output_dir / f"trajectories_{split}.jsonl"
    label_rows = len(pd.read_parquet(labels_path)) if labels_path.exists() else 0
    trajectory_rows = (
        sum(1 for _ in open(trajectory_path)) if trajectory_path.exists() else 0
    )
    if label_rows != len(expected_ids):
        errors.append(f"label rows={label_rows}/{len(expected_ids)}")
    if trajectory_rows != len(expected_ids):
        errors.append(f"trajectory rows={trajectory_rows}/{len(expected_ids)}")

    if errors:
        raise ValueError(f"{split} validation failed: " + ", ".join(errors))
    print(
        f"{split} valid: {len(expected_ids)} rows, "
        f"scout_solved={sum(bool(row.get('scout_correct')) for row in scout.values())}, "
        f"oracle_solved={sum(bool(row.get('resolved')) for row in oracle.values())}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--release-version", default="release_v6")
    parser.add_argument("--dataset-revision", default=LCB_DATASET_REVISION)
    parser.add_argument("--min-date", default="2023-09-01")
    parser.add_argument("--temporal-cutoff", default="2024-10-01")
    parser.add_argument("--eval-timeout", type=int, default=10)
    parser.add_argument(
        "--scout-feedback-tests", choices=["public", "all"], default="public"
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Regrade only rows marked with evaluator infrastructure failures",
    )
    args = parser.parse_args()

    rows = load_lcb(
        min_date=args.min_date,
        release_version=args.release_version,
        dataset_revision=args.dataset_revision,
    )
    splits = {
        "train": [row for row in rows if row["contest_date"] < args.temporal_cutoff],
        "eval": [row for row in rows if row["contest_date"] >= args.temporal_cutoff],
    }
    output_dir = Path(args.output_dir)

    if not args.validate_only:
        for split, split_rows in splits.items():
            scout_rows = regrade_scout(
                split_rows,
                output_dir / f"scout_{split}.jsonl",
                timeout=args.eval_timeout,
                feedback_tests=args.scout_feedback_tests,
                dataset_revision=args.dataset_revision,
                errors_only=args.errors_only,
            )
            oracle_rows = regrade_oracle(
                split_rows,
                output_dir / f"oracle_{split}.jsonl",
                timeout=args.eval_timeout,
                dataset_revision=args.dataset_revision,
                errors_only=args.errors_only,
            )
            build_labels_parquet(
                oracle_rows, output_dir / split / "labels.parquet"
            )
            _write_jsonl_atomic(
                output_dir / f"trajectories_{split}.jsonl", scout_rows
            )
        return

    for split, split_rows in splits.items():
        validate_split(split, split_rows, output_dir, args.dataset_revision)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import argparse
import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any

from pipelinerl.swe.scripts.new.export_traces_to_sb_cli import (
    _build_git_patch,
    _index_dataset_rows,
    _suggest_submit_command,
)

logger = logging.getLogger(__name__)


def _load_instance_filter(path: str | None) -> set[str] | None:
    if not path:
        return None
    values = {line.strip() for line in Path(path).open() if line.strip()}
    logger.info("Loaded %d explicit instance ids from %s", len(values), path)
    return values


def _record_keys(record: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key_name in ("instance_id", "problem_id", "id"):
        value = record.get(key_name)
        if value is None:
            continue
        value_s = str(value).strip()
        if value_s and value_s.lower() != "none" and value_s not in keys:
            keys.append(value_s)
    return keys


def _maybe_run_submit(args: argparse.Namespace, predictions_path: Path) -> None:
    submit_cmd = _suggest_submit_command(args, predictions_path)
    if not submit_cmd:
        return
    if not args.run_sb_cli:
        logger.info("sb-cli command: %s", submit_cmd)
        return
    logger.info("Running: %s", submit_cmd)
    subprocess.run(shlex.split(submit_cmd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export expert_eval.jsonl records into an sb-cli compatible predictions JSON."
    )
    parser.add_argument("--input-jsonl", action="append", required=True, help="expert_eval.jsonl path(s). Repeatable.")
    parser.add_argument("--dataset-path", action="append", required=True, help="Preprocessed local SWE dataset path(s).")
    parser.add_argument("--output-json", required=True, help="Output predictions JSON path for sb-cli.")
    parser.add_argument("--output-jsonl", default=None, help="Optional output JSONL predictions path.")
    parser.add_argument("--output-summary", default=None, help="Optional output summary JSON path.")
    parser.add_argument("--instance-ids-path", default=None, help="Optional newline-delimited instance ids to keep.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--skip-empty-patches", action="store_true", help="Skip rows whose reconstructed patch is empty.")
    parser.add_argument("--run-sb-cli", action="store_true", help="Invoke `sb-cli submit ...` after writing predictions.")
    parser.add_argument("--sb-subset", default=None, help="sb-cli subset, e.g. swe-bench_lite or swe-bench_verified.")
    parser.add_argument("--sb-split", default=None, help="sb-cli split, e.g. dev or test.")
    parser.add_argument("--sb-run-id", default=None)
    parser.add_argument("--sb-output-dir", default=None)
    parser.add_argument("--sb-instance-ids", default=None)
    parser.add_argument("--sb-overwrite", action="store_true")
    parser.add_argument("--sb-no-gen-report", action="store_true")
    parser.add_argument("--sb-no-wait", action="store_true")
    parser.add_argument("--sb-no-verify", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    instance_filter = _load_instance_filter(args.instance_ids_path)
    dataset_rows_by_id, _ = _index_dataset_rows(args.dataset_path)

    stats = {
        "n_records_loaded": 0,
        "n_filtered_out_by_instance_filter": 0,
        "n_missing_dataset_row": 0,
        "n_missing_repair_output": 0,
        "n_patch_reconstruction_errors": 0,
        "n_empty_patches": 0,
        "n_predictions_written": 0,
    }
    predictions_by_instance: dict[str, dict[str, str]] = {}
    predictions_jsonl: list[dict[str, str]] = []

    for input_jsonl in args.input_jsonl:
        with Path(input_jsonl).open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                stats["n_records_loaded"] += 1
                record_in = json.loads(line)
                instance_id = None
                dataset_row = None
                for key in _record_keys(record_in):
                    if instance_filter is not None and key not in instance_filter:
                        continue
                    dataset_row = dataset_rows_by_id.get(key)
                    if dataset_row is not None:
                        instance_id = key
                        break
                if instance_filter is not None and instance_id is None:
                    record_key_set = set(_record_keys(record_in))
                    if record_key_set and record_key_set.isdisjoint(instance_filter):
                        stats["n_filtered_out_by_instance_filter"] += 1
                        continue
                if dataset_row is None or instance_id is None:
                    stats["n_missing_dataset_row"] += 1
                    continue

                repair_output = record_in.get("repair_output")
                if not isinstance(repair_output, str):
                    logger.warning("Missing repair_output for instance %s", instance_id)
                    stats["n_missing_repair_output"] += 1
                    continue

                try:
                    model_patch, _n_edits, _patch_status = _build_git_patch(
                        dataset_row.get("file_contents") or {},
                        repair_output,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Patch reconstruction failed for %s: %s", instance_id, exc)
                    model_patch = ""
                    stats["n_patch_reconstruction_errors"] += 1

                if not model_patch:
                    stats["n_empty_patches"] += 1
                    if args.skip_empty_patches:
                        continue

                model_name_or_path = args.model_name_or_path or str(record_in.get("source") or "expert_eval")
                out_record = {
                    "instance_id": instance_id,
                    "model_patch": model_patch,
                    "model_name_or_path": model_name_or_path,
                }
                predictions_by_instance[instance_id] = {
                    "model_patch": model_patch,
                    "model_name_or_path": model_name_or_path,
                }
                predictions_jsonl.append(out_record)
                stats["n_predictions_written"] += 1

                if args.limit is not None and stats["n_predictions_written"] >= args.limit:
                    break
        if args.limit is not None and stats["n_predictions_written"] >= args.limit:
            break

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump(predictions_by_instance, handle, indent=2, sort_keys=True)

    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else output_json.with_suffix(".jsonl")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as handle:
        for record in predictions_jsonl:
            handle.write(json.dumps(record) + "\n")

    summary_path = Path(args.output_summary) if args.output_summary else output_json.with_name(
        output_json.stem + "_summary.json"
    )
    summary = {
        **stats,
        "input_jsonl": args.input_jsonl,
        "output_json": str(output_json),
        "output_jsonl": str(output_jsonl),
        "suggested_sb_cli_command": _suggest_submit_command(args, output_json),
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    logger.info("Wrote %d predictions to %s", stats["n_predictions_written"], output_json)
    logger.info("Wrote JSONL companion file to %s", output_jsonl)
    logger.info("Wrote summary to %s", summary_path)
    if summary["suggested_sb_cli_command"]:
        logger.info("Suggested sb-cli command: %s", summary["suggested_sb_cli_command"])

    _maybe_run_submit(args, output_json)


if __name__ == "__main__":
    main()

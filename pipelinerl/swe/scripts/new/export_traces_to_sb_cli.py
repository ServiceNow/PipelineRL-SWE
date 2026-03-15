#!/usr/bin/env python
import argparse
import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any

from pipelinerl.swe.load_datasets import load_local_swe_dataset
from pipelinerl.swe.scripts.new.router_trace_utils import load_router_traces
from pipelinerl.swe.scripts.repair_eval_utils import extract_search_replace_edits
from pipelinerl.swe.utils.repair_utils import apply_edits_to_files, get_normalized_patch

logger = logging.getLogger(__name__)


def _trace_problem_keys(trace: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key_name in ("instance_id", "problem_id", "id"):
        value = trace.get(key_name)
        if value is None:
            continue
        value_s = str(value).strip()
        if value_s and value_s.lower() != "none" and value_s not in keys:
            keys.append(value_s)
    return keys


def _dataset_row_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key_name in ("id", "problem_id", "instance_id"):
        value = row.get(key_name)
        if value is None:
            continue
        value_s = str(value).strip()
        if value_s and value_s.lower() != "none" and value_s not in keys:
            keys.append(value_s)
    return keys


def _repo_commit_key(repo: Any, base_commit: Any) -> str | None:
    repo_s = str(repo or "").strip()
    base_commit_s = str(base_commit or "").strip()
    if not repo_s or not base_commit_s or repo_s.lower() == "none" or base_commit_s.lower() == "none":
        return None
    return f"{repo_s}::{base_commit_s}"


def _index_dataset_rows(dataset_paths: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    rows_by_repo_commit: dict[str, dict[str, Any]] = {}
    duplicates = 0
    repo_commit_duplicates = 0
    total_rows = 0
    for dataset_path in dataset_paths:
        rows = load_local_swe_dataset(
            dataset_names=[],
            dataset_path=dataset_path,
            shuffle=False,
            seed=42,
        )
        total_rows += len(rows)
        for row in rows:
            for key in _dataset_row_keys(row):
                if key in rows_by_key:
                    duplicates += 1
                    continue
                rows_by_key[key] = row
            repo_commit = _repo_commit_key(row.get("repo"), row.get("base_commit"))
            if repo_commit:
                if repo_commit in rows_by_repo_commit:
                    repo_commit_duplicates += 1
                else:
                    rows_by_repo_commit[repo_commit] = row
    logger.info(
        "Indexed %d dataset rows from %d path(s) into %d unique ids (%d duplicate ids ignored) and %d repo/base_commit pairs (%d duplicates ignored)",
        total_rows,
        len(dataset_paths),
        len(rows_by_key),
        duplicates,
        len(rows_by_repo_commit),
        repo_commit_duplicates,
    )
    return rows_by_key, rows_by_repo_commit


def _load_instance_filter(path: str | None) -> set[str] | None:
    if not path:
        return None
    values = {line.strip() for line in Path(path).open() if line.strip()}
    logger.info("Loaded %d explicit instance ids from %s", len(values), path)
    return values


def _select_route(trace: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    if args.route == "policy":
        route = trace.get("policy")
        return (route if isinstance(route, dict) else None), "policy"

    experts = trace.get("experts") or []
    if not isinstance(experts, list) or not experts:
        return None, "expert"

    if args.expert_rank is not None:
        for idx, expert in enumerate(experts):
            if not isinstance(expert, dict):
                continue
            expert_rank = expert.get("expert_rank")
            try:
                if expert_rank is not None and int(expert_rank) == args.expert_rank:
                    return expert, f"expert_{args.expert_rank}"
            except (TypeError, ValueError):
                pass
            if idx == args.expert_rank:
                return expert, f"expert_{args.expert_rank}"
        return None, f"expert_{args.expert_rank}"

    if args.expert_model_substring:
        needle = args.expert_model_substring.lower()
        for expert in experts:
            if not isinstance(expert, dict):
                continue
            model_name = str(expert.get("model_name") or "")
            if needle in model_name.lower():
                expert_rank = expert.get("expert_rank")
                label = f"expert_{expert_rank}" if expert_rank is not None else model_name
                return expert, label
        return None, f"expert_{args.expert_model_substring}"

    raise ValueError("Expert route requested but neither --expert-rank nor --expert-model-substring was provided.")


def _build_git_patch(file_contents: dict[str, str], repair_output: str) -> tuple[str, int, str | None]:
    edits = extract_search_replace_edits(repair_output)
    if not edits:
        return "", 0, "no_edits"

    modified_contents = apply_edits_to_files(file_contents, edits, silent=False)
    patch_dict = get_normalized_patch(file_contents, modified_contents)
    if not patch_dict:
        return "", len(edits), "empty_patch"

    diff_parts: list[str] = []
    for file_path, patch in patch_dict.items():
        diff_parts.append(f"diff --git a/{file_path} b/{file_path}")
        diff_parts.append("index 0000000..1111111 100644")
        diff_parts.append(f"--- a/{file_path}")
        diff_parts.append(f"+++ b/{file_path}")
        diff_parts.append(patch)

    text = "\n".join(diff_parts)
    if text and not text.endswith("\n"):
        text += "\n"
    return text, len(edits), None


def _suggest_submit_command(args: argparse.Namespace, predictions_path: Path) -> str | None:
    if not args.sb_subset or not args.sb_split:
        return None
    cmd = [
        "sb-cli",
        "submit",
        args.sb_subset,
        args.sb_split,
        "--predictions_path",
        str(predictions_path),
    ]
    if args.sb_run_id:
        cmd += ["--run_id", args.sb_run_id]
    if args.sb_output_dir:
        cmd += ["--output_dir", args.sb_output_dir]
    if args.sb_instance_ids:
        cmd += ["--instance_ids", args.sb_instance_ids]
    if args.sb_overwrite:
        cmd += ["--overwrite", "1"]
    if args.sb_no_gen_report:
        cmd += ["--gen_report", "0"]
    if args.sb_no_wait:
        cmd += ["--wait_for_evaluation", "0"]
    if args.sb_no_verify:
        cmd += ["--verify_submission", "0"]
    return shlex.join(cmd)


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
        description="Export policy/expert predictions from router traces into an sb-cli compatible JSON file."
    )
    parser.add_argument("--input-glob", action="append", required=True, help="Router trace JSONL glob(s). Repeatable.")
    parser.add_argument("--dataset-path", action="append", required=True, help="Preprocessed local SWE dataset path(s).")
    parser.add_argument("--output-json", required=True, help="Output predictions JSON path for sb-cli.")
    parser.add_argument("--output-jsonl", default=None, help="Optional output JSONL predictions path.")
    parser.add_argument("--output-summary", default=None, help="Optional output summary JSON path.")
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--all-model-versions", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--instance-ids-path", default=None, help="Optional newline-delimited instance ids to keep.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--route", choices=["policy", "expert"], default="expert")
    parser.add_argument("--expert-rank", type=int, default=None)
    parser.add_argument("--expert-model-substring", default=None)
    parser.add_argument("--model-name-or-path", default=None, help="Override model_name_or_path in the exported predictions.")
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

    split = None if args.split == "all" else args.split
    traces = load_router_traces(
        input_globs=args.input_glob,
        split=split,
        latest_model_only=not args.all_model_versions,
        dedupe_by_problem=not args.keep_duplicates,
    )
    if not traces:
        raise ValueError("No traces found after filtering.")

    instance_filter = _load_instance_filter(args.instance_ids_path)
    dataset_rows_by_key = _index_dataset_rows(args.dataset_path)
    dataset_rows_by_id, dataset_rows_by_repo_commit = dataset_rows_by_key

    predictions_by_instance: dict[str, dict[str, str]] = {}
    predictions_jsonl: list[dict[str, str]] = []
    stats = {
        "n_traces_loaded": len(traces),
        "n_filtered_out_by_instance_filter": 0,
        "n_missing_dataset_row": 0,
        "n_missing_route": 0,
        "n_missing_repair_output": 0,
        "n_patch_reconstruction_errors": 0,
        "n_empty_patches": 0,
        "n_predictions_written": 0,
        "n_matched_by_repo_commit": 0,
        "route": args.route,
        "expert_rank": args.expert_rank,
        "expert_model_substring": args.expert_model_substring,
    }

    for trace in traces:
        trace_keys = _trace_problem_keys(trace)

        dataset_row = None
        instance_id = None
        for key in trace_keys:
            if instance_filter is not None and key not in instance_filter:
                continue
            dataset_row = dataset_rows_by_id.get(key)
            if dataset_row is not None:
                instance_id = key
                break
        if dataset_row is None:
            repo_commit = _repo_commit_key(trace.get("repo"), trace.get("base_commit"))
            if repo_commit:
                candidate_row = dataset_rows_by_repo_commit.get(repo_commit)
                candidate_instance_id = str(candidate_row.get("id") or "").strip() if candidate_row is not None else ""
                if candidate_row is not None:
                    if instance_filter is not None and candidate_instance_id not in instance_filter:
                        stats["n_filtered_out_by_instance_filter"] += 1
                        continue
                    dataset_row = candidate_row
                    instance_id = candidate_instance_id
                    stats["n_matched_by_repo_commit"] += 1
        if instance_filter is not None and dataset_row is None and trace_keys:
            if not any(key in instance_filter for key in trace_keys):
                stats["n_filtered_out_by_instance_filter"] += 1
                continue
        if dataset_row is None or instance_id is None:
            stats["n_missing_dataset_row"] += 1
            continue

        route_obj, route_label = _select_route(trace, args)
        if route_obj is None:
            logger.warning("Missing route %s for instance %s", route_label, instance_id)
            stats["n_missing_route"] += 1
            continue

        repair_output = route_obj.get("repair_output")
        if not isinstance(repair_output, str):
            repair_output = route_obj.get("output_text")
        if not isinstance(repair_output, str):
            logger.warning("Missing repair_output for instance %s", instance_id)
            stats["n_missing_repair_output"] += 1
            continue

        try:
            model_patch, n_edits, patch_status = _build_git_patch(dataset_row.get("file_contents") or {}, repair_output)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Patch reconstruction failed for %s: %s", instance_id, exc)
            model_patch = ""
            n_edits = 0
            patch_status = "reconstruction_error"
            stats["n_patch_reconstruction_errors"] += 1

        if not model_patch:
            stats["n_empty_patches"] += 1
            if args.skip_empty_patches:
                continue

        model_name_or_path = args.model_name_or_path or str(route_obj.get("model_name") or route_label)
        record = {
            "instance_id": instance_id,
            "model_patch": model_patch,
            "model_name_or_path": model_name_or_path,
        }
        predictions_by_instance[instance_id] = {
            "model_patch": model_patch,
            "model_name_or_path": model_name_or_path,
        }
        predictions_jsonl.append(record)
        stats["n_predictions_written"] += 1

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

    summary_path = Path(args.output_summary) if args.output_summary else output_json.with_name(output_json.stem + "_summary.json")
    summary = {
        **stats,
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

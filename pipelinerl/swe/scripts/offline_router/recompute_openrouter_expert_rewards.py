#!/usr/bin/env python
"""Recompute expert rewards in an offline-router collection from saved outputs."""

import argparse
import difflib
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_from_disk
from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

logger = logging.getLogger(__name__)

EXT_LANGUAGE_MAP = {
    ".py": "Python",
    ".pyi": "Python",
    ".ipynb": "Python",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".kt": "Kotlin",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".cpp": "C++",
    ".cc": "C++",
    ".c": "C",
    ".cs": "C#",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".scala": "Scala",
}


class FormatError(Exception):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-collection-dir", required=True)
    parser.add_argument("--output-collection-dir", required=True)
    parser.add_argument("--expert-route-idx", type=int, default=1)
    parser.add_argument("--train-dataset-names", default="swe_bench_train")
    parser.add_argument("--train-dataset-path", default="/mnt/llmd/data/swebench/ds_train")
    parser.add_argument("--train-dataset-label", default=None)
    parser.add_argument("--train-max-samples", type=int, default=4096)
    parser.add_argument("--eval-dataset-names", default="swebench_lite")
    parser.add_argument("--eval-dataset-path", default="/mnt/llmd/data/swebench_lite/ds")
    parser.add_argument("--eval-dataset-label", default=None)
    parser.add_argument("--eval-max-samples", type=int, default=500)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _split_names(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_file_contents(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        if isinstance(parsed, dict):
            return {str(key): str(value) for key, value in parsed.items()}
    return {}


def _resolve_dataset_label(item: dict[str, Any], dataset_names: list[str], dataset_label: str | None) -> str:
    if dataset_label:
        return dataset_label
    row_dataset = item.get("dataset")
    if isinstance(row_dataset, str) and row_dataset:
        return row_dataset
    if dataset_names:
        return str(dataset_names[0])
    return "swe"


def _load_local_swe_dataset(
    dataset_names: list[str],
    dataset_path: str,
    shuffle: bool,
    seed: int,
    dataset_label: str | None,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    dataset = load_from_disk(dataset_path)
    samples: list[dict[str, Any]] = []
    for item in dataset:
        file_contents = _parse_file_contents(item.get("gold_file_contents", "{}"))
        if not file_contents:
            continue
        item_id = item.get("issue_id", "") or item.get("instance_id", "") or item.get("id", "")
        samples.append(
            {
                "id": item_id,
                "dataset": _resolve_dataset_label(item, dataset_names, dataset_label),
                "repo": item.get("repo", ""),
                "base_commit": item.get("base_commit", ""),
                "problem_statement": item.get("problem_statement"),
                "patch": item.get("patch"),
                "file_contents": file_contents,
            }
        )
    if shuffle:
        random.seed(seed)
        random.shuffle(samples)
    if max_samples is not None and max_samples > 0 and len(samples) > max_samples:
        samples = samples[:max_samples]
    return samples


def _load_split_dataset(args: argparse.Namespace, split_name: str) -> dict[str, dict[str, Any]]:
    if split_name == "train":
        dataset = _load_local_swe_dataset(
            dataset_names=_split_names(args.train_dataset_names),
            dataset_path=args.train_dataset_path,
            shuffle=bool(args.shuffle),
            seed=int(args.seed),
            dataset_label=args.train_dataset_label,
            max_samples=args.train_max_samples,
        )
    elif split_name == "eval":
        dataset = _load_local_swe_dataset(
            dataset_names=_split_names(args.eval_dataset_names),
            dataset_path=args.eval_dataset_path,
            shuffle=bool(args.shuffle),
            seed=int(args.seed),
            dataset_label=args.eval_dataset_label,
            max_samples=args.eval_max_samples,
        )
    else:
        raise ValueError(f"Unsupported split: {split_name}")
    return {problem_id_from_item(item): item for item in dataset}


def problem_id_from_item(item: dict[str, Any]) -> str:
    for key in ("problem_id", "issue_id", "instance_id", "id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    repo = str(item.get("repo") or "").strip()
    base_commit = str(item.get("base_commit") or "").strip()
    if repo and base_commit:
        return f"{repo}@{base_commit}"
    raise ValueError("Missing problem identifier (problem_id/issue_id/instance_id/id)")


def extract_search_replace_edits(solution_text: str) -> list[dict[str, str]]:
    edits: list[dict[str, str]] = []

    def _extract_from_block(block: str) -> None:
        lines = block.split("\n")
        file_path = None
        start_index = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("###"):
                file_path = line.strip()[3:].strip()
                start_index = i + 1
                break
        if not file_path:
            return
        search_start = search_end = replace_start = replace_end = None
        for i, line in enumerate(lines[start_index:], start=start_index):
            if "<<<<<<< SEARCH" in line:
                search_start = i + 1
            elif "=======" in line and search_start is not None:
                search_end = i
                replace_start = i + 1
            elif ">>>>>>> REPLACE" in line and replace_start is not None:
                replace_end = i
                break
        if None in (search_start, search_end, replace_start, replace_end):
            return
        edits.append(
            {
                "file_path": file_path,
                "search": "\n".join(lines[search_start:search_end]),
                "replace": "\n".join(lines[replace_start:replace_end]),
            }
        )

    code_blocks: list[str] = []
    in_block = False
    current: list[str] = []
    for line in solution_text.split("\n"):
        if line.strip().startswith("```"):
            if in_block:
                code_blocks.append("\n".join(current))
                current = []
            in_block = not in_block
        elif in_block:
            current.append(line)
    for block in code_blocks:
        _extract_from_block(block)
    if not edits and "<<<<<<< SEARCH" in solution_text and ">>>>>>> REPLACE" in solution_text:
        _extract_from_block(solution_text)
    return edits


def _generate_unified_diff(old_code: str, new_code: str, n_context: int = 3) -> str:
    diff = difflib.unified_diff(
        old_code.splitlines(),
        new_code.splitlines(),
        fromfile="old",
        tofile="new",
        lineterm="",
        n=n_context,
    )
    try:
        next(diff)
        next(diff)
        return "\n".join(diff)
    except StopIteration:
        return ""


def _apply_edits_to_files(file_contents: dict[str, str], edits: list[dict[str, str]]) -> dict[str, str]:
    new_content_dict = dict(file_contents)
    for edit in edits:
        file_path = edit.get("file_path", "")
        search_text = edit.get("search", "")
        replace_text = edit.get("replace", "")
        if search_text == replace_text:
            raise FormatError("Search and replace blocks are identical")
        if file_path not in new_content_dict:
            raise FormatError(f"File {file_path} not found in file_contents")
        current_content = new_content_dict[file_path]
        if search_text not in current_content:
            raise FormatError(f"Search text not found in {file_path}: {search_text}")
        new_content_dict[file_path] = current_content.replace(search_text, replace_text, 1)
    return new_content_dict


def _get_filelevel_diff(patch_text: str) -> dict[str, str]:
    try:
        patch = PatchSet(patch_text)
    except (UnidiffParseError, Exception):
        return {}
    return {patchfile.path: "\n".join(str(hunk).strip() for hunk in patchfile).strip() for patchfile in patch}


def _get_normalized_patch(code_context: dict[str, str], new_content_dict: dict[str, str]) -> dict[str, str]:
    patch_dict: dict[str, str] = {}
    for path, new_content in new_content_dict.items():
        patch = _generate_unified_diff(code_context.get(path, ""), new_content)
        if patch:
            patch_dict[path] = patch
    return patch_dict


def _compute_change_similarities(
    pred_patch: dict[str, str],
    oracle_patch: dict[str, str],
) -> list[dict[str, Any]]:
    similarities: list[dict[str, Any]] = []
    for path in set(oracle_patch.keys()).union(set(pred_patch.keys())):
        pred_change = pred_patch.get(path, "")
        oracle_change = oracle_patch.get(path, "")
        if oracle_change == "" or pred_change == "":
            similarity = 0.0
        else:
            similarity = difflib.SequenceMatcher(None, pred_change, oracle_change, autojunk=False).ratio()
        similarities.append(
            {
                "path": path,
                "pred_change": pred_change,
                "oracle_change": oracle_change,
                "similarity": similarity,
            }
        )
    return similarities


def calculate_precise_reward(
    file_contents: dict[str, str],
    oracle_patch_text: str,
    predicted_edits: list[dict[str, str]],
) -> tuple[float, dict[str, Any]]:
    try:
        if len(predicted_edits) == 0:
            raise FormatError("No valid search blocks found")
        oracle_patch = _get_filelevel_diff(oracle_patch_text)
        pred_new_content = _apply_edits_to_files(file_contents, predicted_edits)
        pred_patch = _get_normalized_patch(file_contents, pred_new_content)
        similarities = _compute_change_similarities(pred_patch, oracle_patch)
        if len(similarities) == 0:
            return 1.0, {"similarities": []}
        reward = sum(sim["similarity"] for sim in similarities) / len(similarities)
        return reward, {
            "similarities": similarities,
            "num_files_changed": len(similarities),
            "oracle_files": list(oracle_patch.keys()),
            "predicted_files": list(pred_patch.keys()),
        }
    except FormatError as exc:
        return 0.0, {"format_error": True, "error_message": str(exc)}
    except Exception as exc:  # pylint: disable=broad-except
        return 0.0, {"error": str(exc)}


def _failure_type(success: bool, reward_metadata: dict[str, Any]) -> str:
    if bool(reward_metadata.get("format_error")):
        return "format"
    if success:
        return "none"
    return "semantic"


def _set_list_value(row: dict[str, Any], key: str, idx: int, value: Any) -> None:
    values = list(row.get(key) or [])
    while len(values) <= idx:
        values.append(None)
    values[idx] = value
    row[key] = values


def _copy_metadata(input_dir: Path, output_dir: Path) -> None:
    for name in ("collection_config.json", "metadata.json"):
        src = input_dir / name
        if src.exists():
            dst = output_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _recompute_split(args: argparse.Namespace, split_name: str) -> dict[str, Any]:
    input_dir = Path(args.input_collection_dir) / split_name
    output_dir = Path(args.output_collection_dir) / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    problems = _load_split_dataset(args, split_name)

    n_rows = 0
    n_missing_problem = 0
    old_failure_counts: dict[str, int] = {}
    new_failure_counts: dict[str, int] = {}
    old_reward_sum = 0.0
    new_reward_sum = 0.0
    changed_reward_count = 0
    parse_recovered_count = 0

    for shard in sorted(input_dir.glob("*.parquet")):
        rows = pq.read_table(shard).to_pylist()
        new_rows: list[dict[str, Any]] = []
        for row in rows:
            n_rows += 1
            problem_id = str(row.get("problem_id") or "")
            problem = problems.get(problem_id)
            if problem is None:
                n_missing_problem += 1
                new_rows.append(row)
                continue
            idx = int(args.expert_route_idx)
            outputs = row.get("route_outputs") or []
            output_text = str(outputs[idx] if len(outputs) > idx and outputs[idx] is not None else "")
            old_rewards = row.get("route_rewards") or []
            old_reward = float(old_rewards[idx] if len(old_rewards) > idx and old_rewards[idx] is not None else 0.0)
            old_failure = "unknown"
            old_failures = row.get("route_failure_types") or []
            if len(old_failures) > idx and old_failures[idx]:
                old_failure = str(old_failures[idx])
            old_failure_counts[old_failure] = old_failure_counts.get(old_failure, 0) + 1
            old_reward_sum += old_reward

            edits = extract_search_replace_edits(output_text)
            reward, metadata = calculate_precise_reward(
                file_contents=problem.get("file_contents") or {},
                oracle_patch_text=str(problem.get("patch") or ""),
                predicted_edits=edits,
            )
            success = bool(reward and reward > float(args.success_threshold))
            new_failure = _failure_type(success, metadata)
            new_failure_counts[new_failure] = new_failure_counts.get(new_failure, 0) + 1
            new_reward_sum += float(reward)
            if abs(float(reward) - old_reward) > 1e-12:
                changed_reward_count += 1
            if old_failure == "format" and new_failure != "format":
                parse_recovered_count += 1

            _set_list_value(row, "route_rewards", idx, float(reward))
            _set_list_value(row, "performance_targets", idx, float(reward))
            _set_list_value(row, "route_successes", idx, success)
            _set_list_value(row, "route_failure_types", idx, new_failure)
            new_rows.append(row)

        pq.write_table(pa.Table.from_pylist(new_rows), output_dir / shard.name)

    return {
        "split": split_name,
        "n_rows": n_rows,
        "n_missing_problem": n_missing_problem,
        "old_mean_reward": old_reward_sum / n_rows if n_rows else 0.0,
        "new_mean_reward": new_reward_sum / n_rows if n_rows else 0.0,
        "changed_reward_count": changed_reward_count,
        "parse_recovered_count": parse_recovered_count,
        "old_failure_counts": old_failure_counts,
        "new_failure_counts": new_failure_counts,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    input_dir = Path(args.input_collection_dir)
    output_dir = Path(args.output_collection_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise ValueError(f"Output directory exists and is non-empty: {output_dir}. Pass --overwrite to replace files.")
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_metadata(input_dir, output_dir)

    summaries = []
    for split_name in ("train", "eval"):
        if (input_dir / split_name).exists():
            summary = _recompute_split(args, split_name)
            summaries.append(summary)
            logger.info("Recomputed %s: %s", split_name, summary)
    with (output_dir / "reward_recompute_summary.json").open("w") as handle:
        json.dump({"input_collection_dir": str(input_dir), "split_summaries": summaries}, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

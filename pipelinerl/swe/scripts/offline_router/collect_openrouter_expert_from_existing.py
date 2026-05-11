#!/usr/bin/env python
"""Collect an OpenRouter expert route while reusing an existing primary route."""

import argparse
import asyncio
import difflib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_from_disk
from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

logger = logging.getLogger(__name__)

REPAIR_SYSTEM_PROMPT = (
    "You are a helpful coding assistant. You will see a bug report and the relevant files. "
    "Produce SEARCH/REPLACE patches using the exact format requested."
)

REPAIR_TEMPLATE = (
    "Analyze the following code to find and fix bugs. Use this format:\n\n"
    "<think>\n"
    "[Your analysis process - be as detailed as you want until you're confident in your solution]\n"
    "</think>\n\n"
    "<solution>\n"
    "[Your SEARCH/REPLACE edits using this format:]\n\n"
    "### filename.py\n"
    "<<<<<<< SEARCH\n"
    "[exact code to find]\n"
    "=======\n"
    "[replacement code]\n"
    ">>>>>>> REPLACE\n"
    "</solution>\n\n"
    "IMPORTANT REQUIREMENTS:\n"
    "- Every SEARCH/REPLACE edit must use the exact format above\n"
    "- The SEARCH block must contain a contiguous chunk of lines that exist in the source code\n"
    "- PROPER INDENTATION IS CRITICAL - if you want to add '    print(x)', you must include all those spaces\n"
    "- Wrap each SEARCH/REPLACE edit in a code block\n"
    "- Use separate code blocks for multiple edits\n\n"
    "Example:\n"
    "```python\n"
    "### mathweb/flask/app.py\n"
    "<<<<<<< SEARCH\n"
    "from flask import Flask\n"
    "=======\n"
    "import math\n"
    "from flask import Flask\n"
    ">>>>>>> REPLACE\n"
    "```\n\n"
    "Here is the issue:\n"
    "--- BEGIN ISSUE ---\n"
    "{problem_statement}\n"
    "--- END ISSUE ---\n\n"
    "Below are the code files that may contain bugs:\n"
    "{file_contents}"
)

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
    parser.add_argument("--source-collection-dir", required=True)
    parser.add_argument("--train-source-collection-dir", default=None)
    parser.add_argument("--eval-source-collection-dir", default=None)
    parser.add_argument("--reuse-expert-collection-dir", default=None)
    parser.add_argument("--reuse-train-expert-collection-dir", default=None)
    parser.add_argument("--reuse-eval-expert-collection-dir", default=None)
    parser.add_argument("--reuse-expert-route-idx", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="google/gemini-3-flash-preview")
    parser.add_argument("--expert-label", default=None)
    parser.add_argument("--base-url", default="https://openrouter.ai/api")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--train-dataset-names", default="swe_bench_train")
    parser.add_argument("--train-dataset-path", default="/mnt/llmd/data/swebench/ds_train")
    parser.add_argument("--train-dataset-label", default=None)
    parser.add_argument("--train-max-samples", type=int, default=4096)
    parser.add_argument("--eval-dataset-names", default="swebench_lite")
    parser.add_argument("--eval-dataset-path", default="/mnt/llmd/data/swebench_lite/ds")
    parser.add_argument("--eval-dataset-label", default=None)
    parser.add_argument("--eval-max-samples", type=int, default=500)
    parser.add_argument("--collect-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--collect-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=15000)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--connector-limit", type=int, default=64)
    parser.add_argument("--max-concurrent-problems", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--progress-log-every", type=int, default=25)
    parser.add_argument("--openrouter-title", default="PipelineRL-SWE offline router")
    parser.add_argument("--openrouter-referer", default=None)
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


def load_local_swe_dataset(
    dataset_names: list[str],
    dataset_path: str,
    shuffle: bool = True,
    seed: int = 42,
    dataset_label: str | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    dataset = load_from_disk(dataset_path)
    samples: list[dict[str, Any]] = []
    for item in dataset:
        file_contents = _parse_file_contents(item.get("gold_file_contents", "{}"))
        if not file_contents:
            continue
        item_id = item.get("issue_id", "") or item.get("instance_id", "") or item.get("id", "")
        all_file_stats = item.get("all_file_stats", "{}")
        if isinstance(all_file_stats, dict):
            all_file_stats = json.dumps(all_file_stats)
        elif not isinstance(all_file_stats, str):
            all_file_stats = "{}"
        samples.append(
            {
                "id": item_id,
                "dataset": _resolve_dataset_label(item, dataset_names, dataset_label),
                "repo": item.get("repo", ""),
                "base_commit": item.get("base_commit", ""),
                "problem_statement": item.get("problem_statement"),
                "patch": item.get("patch"),
                "file_contents": file_contents,
                "all_file_stats": all_file_stats,
            }
        )
    if shuffle:
        random.seed(seed)
        random.shuffle(samples)
    if max_samples is not None and max_samples > 0 and len(samples) > max_samples:
        samples = samples[:max_samples]
    return samples


def _load_dataset_for_split(args: argparse.Namespace, split_name: str) -> list[dict[str, Any]]:
    if split_name == "train":
        dataset_names = _split_names(args.train_dataset_names)
        dataset_path = args.train_dataset_path
        dataset_label = args.train_dataset_label
        max_samples = args.train_max_samples
    elif split_name == "eval":
        dataset_names = _split_names(args.eval_dataset_names)
        dataset_path = args.eval_dataset_path
        dataset_label = args.eval_dataset_label
        max_samples = args.eval_max_samples
    else:
        raise ValueError(f"Unsupported split: {split_name}")

    dataset = load_local_swe_dataset(
        dataset_names=dataset_names,
        dataset_path=dataset_path,
        shuffle=bool(args.shuffle),
        seed=int(args.seed),
        dataset_label=dataset_label,
        max_samples=max_samples,
    )
    return sorted(dataset, key=problem_id_from_item)


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


def infer_language_from_problem(problem: dict[str, Any]) -> str:
    for key in ("language", "lang", "repo_language"):
        value = problem.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    dataset = str(problem.get("dataset") or "").lower()
    if "smith_py" in dataset or dataset.endswith("_py"):
        return "Python"
    if "smith_go" in dataset or dataset.endswith("_go"):
        return "Go"
    if "smith_rs" in dataset or dataset.endswith("_rs"):
        return "Rust"
    if "smith_java" in dataset or dataset.endswith("_java"):
        return "Java"

    file_contents = problem.get("file_contents") or {}
    ext_counts: dict[str, int] = {}
    if isinstance(file_contents, dict):
        for path in file_contents:
            label = EXT_LANGUAGE_MAP.get(Path(str(path)).suffix.lower())
            if label:
                ext_counts[label] = ext_counts.get(label, 0) + 1
    if ext_counts:
        return max(ext_counts.items(), key=lambda item: item[1])[0]
    return "Unknown"


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, subvalue in value.items():
            if "api_key" in str(key).lower():
                sanitized[key] = "***"
            else:
                sanitized[key] = sanitize_for_json(subvalue)
        return sanitized
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _format_file_context(file_contents: dict[str, str]) -> str:
    return "\n".join(f"### {path}\n```\n{content}\n```\n" for path, content in file_contents.items())


def build_repair_messages(problem_statement: str, file_contents: dict[str, str]) -> tuple[list[dict[str, str]], str]:
    file_context = _format_file_context(file_contents)
    user_content = REPAIR_TEMPLATE.format(
        problem_statement=problem_statement,
        file_contents=file_context,
    )
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ], file_context


def _read_api_key(args: argparse.Namespace) -> str:
    key = os.environ.get(args.api_key_env)
    if key:
        return key.strip()
    if args.api_key_file:
        path = Path(args.api_key_file)
        if path.exists():
            key = path.read_text().strip()
            if key:
                return key
            raise ValueError(f"OpenRouter API key file is empty: {path}")
    raise ValueError(
        f"Missing OpenRouter API key. Set {args.api_key_env} or pass --api-key-file pointing to a readable file."
    )


def _load_source_rows(split_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not split_dir.exists():
        return rows
    for shard in sorted(split_dir.glob("*.parquet")):
        table = pq.read_table(shard)
        for row in table.to_pylist():
            problem_id = row.get("problem_id")
            if isinstance(problem_id, str) and problem_id:
                rows[problem_id] = row
    return rows


def _load_existing_problem_ids(split_dir: Path) -> tuple[set[str], int]:
    existing: set[str] = set()
    max_index = -1
    if not split_dir.exists():
        return existing, 0
    for shard in sorted(split_dir.glob("*.parquet")):
        try:
            max_index = max(max_index, int(shard.stem.split("-")[-1]))
        except ValueError:
            pass
        try:
            table = pq.read_table(shard, columns=["problem_id"])
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to read existing shard %s: %s", shard, exc)
            continue
        for value in table.column("problem_id").to_pylist():
            if isinstance(value, str) and value:
                existing.add(value)
    return existing, max_index + 1


def _row_count_in_shard(path: Path) -> int:
    try:
        return int(pq.read_metadata(path).num_rows)
    except Exception:  # pylint: disable=broad-except
        return 0


def _write_parquet_shard(split_dir: Path, shard_index: int, rows: list[dict[str, Any]]) -> Path:
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"{split_dir.name}-{shard_index:05d}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _list_value(row: dict[str, Any], key: str, index: int, default: Any) -> Any:
    values = row.get(key)
    if isinstance(values, list) and len(values) > index:
        return values[index]
    return default


def _source_collection_dir_for_split(args: argparse.Namespace, split_name: str) -> Path:
    if split_name == "train" and args.train_source_collection_dir:
        return Path(args.train_source_collection_dir)
    if split_name == "eval" and args.eval_source_collection_dir:
        return Path(args.eval_source_collection_dir)
    return Path(args.source_collection_dir)


def _reuse_expert_collection_dir_for_split(args: argparse.Namespace, split_name: str) -> Path | None:
    raw: str | None
    if split_name == "train":
        raw = args.reuse_train_expert_collection_dir or args.reuse_expert_collection_dir
    elif split_name == "eval":
        raw = args.reuse_eval_expert_collection_dir or args.reuse_expert_collection_dir
    else:
        raw = args.reuse_expert_collection_dir
    return Path(raw) if raw else None


def _derive_failure_type(success: bool, reward_metadata: dict[str, Any], request_error: str | None) -> str:
    if request_error:
        return "request_error"
    if bool(reward_metadata.get("format_error")):
        return "format"
    if success:
        return "none"
    return "semantic"


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


async def chat_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    parameters: dict[str, Any],
    api_key: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[str, dict[str, Any], float]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)
    payload = {"model": model_name, "messages": messages} | (parameters or {})
    start = time.time()
    async with session.post(url, json=payload, headers=headers) as response:
        response.raise_for_status()
        data = await response.json()
    latency = time.time() - start
    text = data["choices"][0]["message"]["content"] or ""
    usage = data.get("usage", {})
    return text, usage, latency


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


def _get_normalized_patch(code_context: dict[str, str], new_content_dict: dict[str, str]) -> dict[str, str]:
    patch_dict: dict[str, str] = {}
    for path, new_content in new_content_dict.items():
        patch = _generate_unified_diff(code_context.get(path, ""), new_content)
        if patch:
            patch_dict[path] = patch
    return patch_dict


def _get_filelevel_diff(patch_text: str) -> dict[str, str]:
    try:
        patch = PatchSet(patch_text)
    except (UnidiffParseError, Exception):
        return {}
    return {patchfile.path: "\n".join(str(hunk).strip() for hunk in patchfile).strip() for patchfile in patch}


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
        logger.warning("Format error calculating precise reward: %s", exc)
        return 0.0, {"format_error": True, "error_message": str(exc)}
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Unexpected error calculating precise reward: %s", exc)
        return 0.0, {"error": str(exc)}


async def _run_openrouter_expert(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    api_key: str,
    repair_messages: list[dict[str, str]],
    file_contents: dict[str, str],
    oracle_patch: str,
) -> dict[str, Any]:
    request_error: str | None = None
    usage: dict[str, Any] = {}
    repair_text = ""
    start = time.time()
    parameters: dict[str, Any] = {
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
    }
    extra_headers: dict[str, str] = {}
    if args.openrouter_title:
        extra_headers["X-Title"] = str(args.openrouter_title)
    if args.openrouter_referer:
        extra_headers["HTTP-Referer"] = str(args.openrouter_referer)

    try:
        # chat_completion handles the OpenAI-compatible response shape and usage accounting.
        repair_text, usage, latency = await chat_completion(
            session=session,
            base_url=args.base_url,
            model_name=args.model,
            messages=repair_messages,
            parameters=parameters,
            api_key=api_key,
            extra_headers=extra_headers,
        )
    except Exception as exc:  # pylint: disable=broad-except
        latency = time.time() - start
        request_error = str(exc)

    if request_error:
        reward = 0.0
        reward_metadata: dict[str, Any] = {"request_error": True, "error": request_error}
    else:
        edits = extract_search_replace_edits(repair_text)
        reward, reward_metadata = calculate_precise_reward(file_contents, oracle_patch, edits)

    success = bool(reward and reward > float(args.success_threshold))
    return {
        "output_text": repair_text,
        "reward": float(reward or 0.0),
        "success": success,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
        "latency_s": float(latency),
        "failure_type": _derive_failure_type(success, reward_metadata, request_error),
        "reward_metadata": reward_metadata,
    }


def _expert_from_reuse_row(
    row: dict[str, Any],
    route_idx: int,
    success_threshold: float,
) -> dict[str, Any]:
    reward = float(
        _list_value(
            row,
            "route_rewards",
            route_idx,
            _list_value(row, "performance_targets", route_idx, 0.0),
        )
        or 0.0
    )
    return {
        "output_text": str(_list_value(row, "route_outputs", route_idx, "") or ""),
        "reward": reward,
        "success": bool(_list_value(row, "route_successes", route_idx, reward > success_threshold)),
        "prompt_tokens": int(_list_value(row, "route_prompt_tokens", route_idx, 0) or 0),
        "output_tokens": int(_list_value(row, "route_output_tokens", route_idx, 0) or 0),
        "latency_s": float(_list_value(row, "route_latencies_s", route_idx, 0.0) or 0.0),
        "failure_type": str(_list_value(row, "route_failure_types", route_idx, "unknown") or "unknown"),
    }


def _build_collected_row(
    problem: dict[str, Any],
    source_row: dict[str, Any],
    expert: dict[str, Any],
    args: argparse.Namespace,
    split_name: str,
) -> dict[str, Any]:
    problem_id = problem_id_from_item(problem)
    problem_statement = problem.get("problem_statement")
    if not isinstance(problem_statement, str) or not problem_statement.strip():
        raise ValueError(f"Problem {problem_id} missing problem_statement")

    primary_output = _list_value(
        source_row,
        "route_outputs",
        0,
        source_row.get("primary_output_text") or "",
    )
    primary_reward = float(_list_value(source_row, "route_rewards", 0, 0.0) or 0.0)
    primary_success = bool(_list_value(source_row, "route_successes", 0, primary_reward > float(args.success_threshold)))

    return {
        "problem_id": problem_id,
        "dataset": str(problem.get("dataset") or source_row.get("dataset") or ""),
        "split": split_name,
        "repo": str(problem.get("repo") or source_row.get("repo") or ""),
        "base_commit": str(problem.get("base_commit") or source_row.get("base_commit") or ""),
        "language": str(source_row.get("language") or infer_language_from_problem(problem)),
        "problem_statement": problem_statement,
        "prompt_text": str(source_row.get("prompt_text") or ""),
        "primary_output_text": str(primary_output or ""),
        "performance_targets": [primary_reward, float(expert["reward"])],
        "route_rewards": [primary_reward, float(expert["reward"])],
        "route_successes": [primary_success, bool(expert["success"])],
        "route_prompt_tokens": [
            int(_list_value(source_row, "route_prompt_tokens", 0, 0) or 0),
            int(expert["prompt_tokens"]),
        ],
        "route_output_tokens": [
            int(_list_value(source_row, "route_output_tokens", 0, 0) or 0),
            int(expert["output_tokens"]),
        ],
        "route_latencies_s": [
            float(_list_value(source_row, "route_latencies_s", 0, 0.0) or 0.0),
            float(expert["latency_s"]),
        ],
        "route_outputs": [str(primary_output or ""), str(expert["output_text"] or "")],
        "route_failure_types": [
            str(_list_value(source_row, "route_failure_types", 0, "unknown") or "unknown"),
            str(expert["failure_type"]),
        ],
    }


async def _collect_problem(
    problem: dict[str, Any],
    source_row: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
    session: aiohttp.ClientSession,
    split_name: str,
) -> dict[str, Any]:
    problem_id = problem_id_from_item(problem)
    file_contents = problem.get("file_contents") or {}
    if not isinstance(file_contents, dict) or not file_contents:
        raise ValueError(f"Problem {problem_id} missing file_contents")
    problem_statement = problem.get("problem_statement")
    if not isinstance(problem_statement, str) or not problem_statement.strip():
        raise ValueError(f"Problem {problem_id} missing problem_statement")

    repair_messages, _stage_input = build_repair_messages(problem_statement, file_contents)
    expert = await _run_openrouter_expert(
        session=session,
        args=args,
        api_key=api_key,
        repair_messages=repair_messages,
        file_contents=file_contents,
        oracle_patch=str(problem.get("patch") or ""),
    )
    return _build_collected_row(problem, source_row, expert, args, split_name)


def _preseed_reused_expert_rows(
    args: argparse.Namespace,
    split_name: str,
    split_dir: Path,
    dataset: list[dict[str, Any]],
    source_rows: dict[str, dict[str, Any]],
    existing_ids: set[str],
    next_shard_index: int,
) -> tuple[int, int, int]:
    reuse_dir = _reuse_expert_collection_dir_for_split(args, split_name)
    if reuse_dir is None:
        return next_shard_index, 0, 0

    reuse_rows = _load_source_rows(reuse_dir / split_name)
    if not reuse_rows:
        logger.warning("No reusable expert rows found for split=%s in %s", split_name, reuse_dir / split_name)
        return next_shard_index, 0, 0

    buffered_rows: list[dict[str, Any]] = []
    reused = 0
    failed = 0
    success_threshold = float(args.success_threshold)
    route_idx = int(args.reuse_expert_route_idx)
    for problem in dataset:
        problem_id = problem_id_from_item(problem)
        if problem_id in existing_ids:
            continue
        source_row = source_rows.get(problem_id)
        reuse_row = reuse_rows.get(problem_id)
        if source_row is None or reuse_row is None:
            continue
        try:
            expert = _expert_from_reuse_row(reuse_row, route_idx, success_threshold)
            buffered_rows.append(_build_collected_row(problem, source_row, expert, args, split_name))
            existing_ids.add(problem_id)
            reused += 1
        except Exception as exc:  # pylint: disable=broad-except
            failed += 1
            logger.exception("Failed to reuse %s problem %s: %s", split_name, problem_id, exc)

        if len(buffered_rows) >= int(args.shard_size):
            shard_path = _write_parquet_shard(split_dir, next_shard_index, buffered_rows)
            logger.info(
                "%s reused expert rows: wrote=%d shard=%s source=%s",
                split_name,
                len(buffered_rows),
                shard_path.name,
                reuse_dir,
            )
            next_shard_index += 1
            buffered_rows = []

    if buffered_rows:
        shard_path = _write_parquet_shard(split_dir, next_shard_index, buffered_rows)
        logger.info(
            "%s reused expert rows: wrote=%d shard=%s source=%s",
            split_name,
            len(buffered_rows),
            shard_path.name,
            reuse_dir,
        )
        next_shard_index += 1

    logger.info(
        "%s reusable expert preseed complete: reuse_source_rows=%d reused=%d failed=%d",
        split_name,
        len(reuse_rows),
        reused,
        failed,
    )
    return next_shard_index, reused, failed


async def _collect_split(args: argparse.Namespace, split_name: str, api_key: str) -> dict[str, Any]:
    source_collection_dir = _source_collection_dir_for_split(args, split_name)
    source_rows = _load_source_rows(source_collection_dir / split_name)
    dataset = _load_dataset_for_split(args, split_name)
    split_dir = Path(args.output_dir) / split_name
    existing_ids, next_shard_index = _load_existing_problem_ids(split_dir)
    next_shard_index, reused_expert_rows, reuse_failed = _preseed_reused_expert_rows(
        args=args,
        split_name=split_name,
        split_dir=split_dir,
        dataset=dataset,
        source_rows=source_rows,
        existing_ids=existing_ids,
        next_shard_index=next_shard_index,
    )

    pending: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    skipped_existing = 0
    skipped_missing_source = 0
    for problem in dataset:
        problem_id = problem_id_from_item(problem)
        if problem_id in existing_ids:
            skipped_existing += 1
            continue
        source_row = source_rows.get(problem_id)
        if source_row is None:
            skipped_missing_source += 1
            continue
        pending.append((len(pending), problem, source_row))

    logger.info(
        "Starting %s OpenRouter expert collection: dataset_rows=%d source_rows=%d "
        "reused_expert=%d skipped_existing=%d skipped_missing_source=%d pending=%d shard_size=%d source_dir=%s",
        split_name,
        len(dataset),
        len(source_rows),
        reused_expert_rows,
        skipped_existing,
        skipped_missing_source,
        len(pending),
        int(args.shard_size),
        source_collection_dir,
    )

    connector = aiohttp.TCPConnector(limit=int(args.connector_limit))
    timeout = aiohttp.ClientTimeout(total=float(args.request_timeout))
    semaphore = asyncio.Semaphore(int(args.max_concurrent_problems))
    max_in_flight = max(1, int(args.max_concurrent_problems) * 4)

    written = 0
    failed = 0
    buffered_rows: list[dict[str, Any]] = []
    ready_rows: dict[int, dict[str, Any] | None] = {}
    next_ready_idx = 0
    shard_row_counts: list[int] = []

    async def _task_wrapper(
        order_idx: int,
        problem: dict[str, Any],
        source_row: dict[str, Any],
        session: aiohttp.ClientSession,
    ) -> tuple[int, dict[str, Any] | None]:
        async with semaphore:
            try:
                row = await _collect_problem(
                    problem=problem,
                    source_row=source_row,
                    args=args,
                    api_key=api_key,
                    session=session,
                    split_name=split_name,
                )
                return order_idx, row
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Failed to collect %s problem %s: %s", split_name, problem_id_from_item(problem), exc)
                return order_idx, None

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        pending_iter = iter(pending)
        in_flight: set[asyncio.Task] = set()
        while True:
            while len(in_flight) < max_in_flight:
                try:
                    order_idx, problem, source_row = next(pending_iter)
                except StopIteration:
                    break
                in_flight.add(asyncio.create_task(_task_wrapper(order_idx, problem, source_row, session)))
            if not in_flight:
                break

            done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                order_idx, row = await future
                ready_rows[order_idx] = row
                while next_ready_idx in ready_rows:
                    ready = ready_rows.pop(next_ready_idx)
                    next_ready_idx += 1
                    if ready is None:
                        failed += 1
                        continue
                    buffered_rows.append(ready)
                    written += 1
                    if len(buffered_rows) >= int(args.shard_size):
                        shard_path = _write_parquet_shard(split_dir, next_shard_index, buffered_rows)
                        shard_row_counts.append(_row_count_in_shard(shard_path))
                        logger.info(
                            "%s progress: processed=%d/%d written=%d failed=%d shards=%d last_shard=%s",
                            split_name,
                            written + failed,
                            len(pending),
                            written,
                            failed,
                            len(shard_row_counts),
                            shard_path.name,
                        )
                        next_shard_index += 1
                        buffered_rows = []
                    elif (written + failed) % max(1, int(args.progress_log_every)) == 0:
                        logger.info(
                            "%s progress: processed=%d/%d written=%d failed=%d buffered=%d in_flight=%d",
                            split_name,
                            written + failed,
                            len(pending),
                            written,
                            failed,
                            len(buffered_rows),
                            len(in_flight),
                        )

    if buffered_rows:
        shard_path = _write_parquet_shard(split_dir, next_shard_index, buffered_rows)
        shard_row_counts.append(_row_count_in_shard(shard_path))

    logger.info(
        "Finished %s collection: pending=%d written=%d failed=%d shards=%d",
        split_name,
        len(pending),
        written,
        failed,
        len(shard_row_counts),
    )
    return {
        "split": split_name,
        "n_dataset_rows": len(dataset),
        "n_source_rows": len(source_rows),
        "n_skipped_existing": skipped_existing,
        "n_skipped_missing_source": skipped_missing_source,
        "n_pending": len(pending),
        "n_written": written,
        "n_failed": failed,
        "n_reused_expert_rows": reused_expert_rows,
        "n_reuse_failed": reuse_failed,
        "shard_row_counts": shard_row_counts,
    }


def _safe_config(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    payload["api_key_env"] = str(args.api_key_env)
    if payload.get("api_key_file"):
        payload["api_key_file"] = str(payload["api_key_file"])
    return sanitize_for_json(payload)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = _read_api_key(args)

    write_json(output_dir / "collection_config.json", _safe_config(args))
    started_at = time.time()
    split_summaries: list[dict[str, Any]] = []
    if args.collect_train:
        split_summaries.append(asyncio.run(_collect_split(args, "train", api_key)))
    if args.collect_eval:
        split_summaries.append(asyncio.run(_collect_split(args, "eval", api_key)))

    metadata = {
        "schema_version": 1,
        "collection_started_at_unix": started_at,
        "collection_completed_at_unix": time.time(),
        "source_collection_dir": str(args.source_collection_dir),
        "route_labels": ["primary_model", args.expert_label or f"expert_0:{args.model}"],
        "route_model_names": ["primary_model", args.model],
        "prompt_builder": "pipelinerl.swe.scripts.repair_eval_utils.build_repair_messages",
        "reward_function": "pipelinerl.swe.utils.repair_utils.calculate_precise_reward",
        "primary_route_source": "copied_from_source_collection_route_0",
        "expert_route_source": "openrouter_chat_completions",
        "split_summaries": split_summaries,
    }
    write_json(output_dir / "metadata.json", metadata)
    logger.info("OpenRouter expert collection complete: output_dir=%s", output_dir)


if __name__ == "__main__":
    main()

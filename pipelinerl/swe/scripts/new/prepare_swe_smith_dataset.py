#!/usr/bin/env python
import argparse
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
import git
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _first_non_empty(item: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = item.get(key)
        if value is not None and value != "":
            return value
    return default


def _as_dict_maybe_json(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, list):
        file_map: dict[str, str] = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            path = item.get("path") or item.get("file_path") or item.get("filename") or item.get("name")
            content = item.get("content") or item.get("text")
            if path and content is not None:
                file_map[str(path)] = str(content)
        return file_map
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict) or isinstance(parsed, list):
                return _as_dict_maybe_json(parsed)
        except json.JSONDecodeError:
            return {}
    return {}


def _source_slug(dataset_name: str) -> str:
    return dataset_name.replace("/", "__").replace("-", "_").lower()


def _extract_file_contents(item: dict[str, Any]) -> dict[str, str]:
    for key in [
        "gold_file_contents",
        "file_contents",
        "gold_files",
        "gold_files_content",
        "files",
    ]:
        parsed = _as_dict_maybe_json(item.get(key))
        if parsed:
            return parsed
    return {}


def _parse_repo_with_commit_suffix(repo_value: str) -> tuple[str | None, str | None]:
    # Expected SWE-smith format example:
    #   swesmith/Automattic__mongoose.5f57a5bb
    text = (repo_value or "").strip()
    if not text:
        return None, None
    payload = text.split("/", 1)[1] if "/" in text else text
    if "." not in payload:
        repo_name = payload.replace("__", "/")
        return (repo_name or None), None
    prefix, maybe_sha = payload.rsplit(".", 1)
    repo_name = prefix.replace("__", "/")
    if re.fullmatch(r"[0-9a-fA-F]{6,40}", maybe_sha):
        return (repo_name or None), maybe_sha.lower()
    return (repo_name or None), None


def _extract_repo_and_commit(item: dict[str, Any]) -> tuple[str | None, str | None]:
    repo_raw = _first_non_empty(item, ["repo", "repo_name", "repository"], default="")
    repo_name, commit_prefix = _parse_repo_with_commit_suffix(str(repo_raw))

    if not repo_name:
        # Fallback: derive from image name if present.
        image_name = str(_first_non_empty(item, ["image", "image_name"], default=""))
        # Example:
        # swebench/swesmith.x86_64.automattic_1776_mongoose.5f57a5bb
        if "." in image_name:
            parts = image_name.split(".")
            if len(parts) >= 2:
                maybe_sha = parts[-1]
                if re.fullmatch(r"[0-9a-fA-F]{6,40}", maybe_sha):
                    commit_prefix = commit_prefix or maybe_sha.lower()
                # heuristic for owner/repo from last non-sha segment
                if len(parts) >= 3:
                    token = parts[-2]
                    if "_" in token:
                        tok_parts = token.split("_")
                        if len(tok_parts) >= 2:
                            owner = tok_parts[0]
                            repo = tok_parts[-1]
                            repo_name = f"{owner}/{repo}"

    explicit_commit = _first_non_empty(item, ["base_commit", "commit", "base_sha", "sha"])
    if explicit_commit:
        commit_prefix = str(explicit_commit)
    return repo_name, commit_prefix


def _parse_patch_files(patch: str) -> list[str]:
    if not patch:
        return []
    files = re.findall(r"^--- a/(.+)$", patch, re.MULTILINE)
    if not files:
        files = re.findall(r"^diff --git a/(.+?) b/", patch, re.MULTILINE)
    # Preserve order, dedupe.
    seen = set()
    ordered: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered


def _ensure_repo(local_dir: Path, repo_name: str) -> Path | None:
    repo_path = local_dir / repo_name.replace("/", "_")
    url = f"https://github.com/{repo_name}.git"
    try:
        if repo_path.exists() and (repo_path / ".git").exists():
            repo = git.Repo(repo_path)
            repo.remotes.origin.fetch()
            return repo_path
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        git.Repo.clone_from(url, repo_path)
        return repo_path
    except Exception as exc:
        logger.warning("Failed to prepare repo %s (%s): %s", repo_name, repo_path, exc)
        return None


def _resolve_commit(repo_path: Path, commit_prefix: str) -> str | None:
    try:
        repo = git.Repo(repo_path)
        resolved = repo.git.rev_parse(f"{commit_prefix}^{{commit}}").strip()
        return resolved
    except Exception as exc:
        logger.warning("Could not resolve commit prefix %s in %s: %s", commit_prefix, repo_path, exc)
        return None


def _fetch_gold_file_contents_from_repo(repo_path: Path, commit_hash: str, patch: str) -> dict[str, str]:
    files = _parse_patch_files(patch)
    if not files:
        return {}
    repo = git.Repo(repo_path)
    contents: dict[str, str] = {}
    for file_path in files:
        try:
            contents[file_path] = repo.git.show(f"{commit_hash}:{file_path}")
        except Exception:
            continue
    return contents


def _normalize_item(
    item: dict[str, Any],
    source_name: str,
    require_file_contents: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    repo, base_commit = _extract_repo_and_commit(item)
    patch = _first_non_empty(item, ["patch", "gold_patch", "target_patch", "diff", "model_patch"])
    problem_statement = _first_non_empty(
        item,
        ["problem_statement", "problem", "issue_description", "statement", "prompt"],
    )
    item_id = _first_non_empty(item, ["id", "issue_id", "instance_id", "problem_id"], default="")

    file_contents = _extract_file_contents(item)
    if not repo:
        return None, "missing_repo"
    if not base_commit:
        return None, "missing_base_commit"
    if not patch:
        return None, "missing_patch"
    if not problem_statement:
        return None, "missing_problem_statement"
    if require_file_contents and not file_contents:
        return None, "missing_gold_file_contents"

    split = str(item.get("split") or "").lower()
    split = split if split in {"train", "test"} else ""

    return {
        "id": str(item_id),
        "dataset": str(item.get("dataset") or _source_slug(source_name)),
        "dataset_source": source_name,
        "split": split,
        "repo": str(repo),
        "base_commit": str(base_commit),
        "problem_statement": str(problem_statement),
        "patch": str(patch),
        "gold_file_contents": json.dumps(file_contents),
        "all_file_stats": "{}",
    }, None


def _split_within_repo(rows: list[dict[str, Any]], train_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_fraction <= 0.0 or train_fraction >= 1.0:
        raise ValueError(f"train_fraction must be in (0,1), got {train_fraction}")
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_repo[row["repo"]].append(row)

    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for _, repo_rows in by_repo.items():
        local_rows = list(repo_rows)
        rng.shuffle(local_rows)
        n = len(local_rows)
        if n <= 1:
            train_rows.extend(local_rows)
            continue
        n_train = int(round(n * train_fraction))
        n_train = max(1, min(n - 1, n_train))
        train_rows.extend(local_rows[:n_train])
        test_rows.extend(local_rows[n_train:])
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def _split_disjoint_by_repo(
    rows: list[dict[str, Any]],
    train_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_fraction <= 0.0 or train_fraction >= 1.0:
        raise ValueError(f"train_fraction must be in (0,1), got {train_fraction}")

    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_repo[row["repo"]].append(row)

    rng = random.Random(seed)
    repos = list(by_repo.keys())
    rng.shuffle(repos)

    target_train = int(round(len(rows) * train_fraction))
    target_train = max(1, min(len(rows) - 1, target_train))

    # Greedy assignment by descending repo size gives stable volume control.
    repos = sorted(repos, key=lambda repo: len(by_repo[repo]), reverse=True)
    train_repo_set: set[str] = set()
    train_count = 0
    for repo in repos:
        repo_size = len(by_repo[repo])
        if train_count < target_train:
            train_repo_set.add(repo)
            train_count += repo_size
        else:
            break

    # Ensure both sides are non-empty at repo level.
    if len(train_repo_set) == 0 and repos:
        train_repo_set.add(repos[0])
    if len(train_repo_set) == len(repos) and len(repos) > 1:
        train_repo_set.remove(repos[-1])

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for repo, repo_rows in by_repo.items():
        if repo in train_repo_set:
            train_rows.extend(repo_rows)
        else:
            test_rows.extend(repo_rows)

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def _token_count_for_row(row: dict[str, Any], tokenizer: Any) -> int:
    file_contents = _as_dict_maybe_json(row.get("gold_file_contents", "{}"))
    joined_files = "\n\n".join(file_contents[k] for k in sorted(file_contents.keys()))
    patch = str(row.get("patch") or "")
    # Explicitly match repair input + answer signal.
    text = f"{joined_files}\n\n<PATCH>\n{patch}"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return int(len(tokens))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SWE-smith datasets into PipelineRL local SWE format.")
    parser.add_argument(
        "--hf-dataset",
        action="append",
        required=True,
        help="Hugging Face dataset name. Repeat to combine multiple datasets.",
    )
    parser.add_argument(
        "--hf-split",
        default="train",
        help="HF split name used for all datasets (default: train).",
    )
    parser.add_argument("--train-output-path", required=True, help="Output path for processed train dataset (save_to_disk).")
    parser.add_argument("--test-output-path", required=True, help="Output path for processed test dataset (save_to_disk).")
    parser.add_argument(
        "--split-strategy",
        choices=["field", "within-repo", "disjoint-repo"],
        default="field",
        help=(
            "field: use existing split field if present, fallback to within-repo if absent; "
            "within-repo: split inside each repo (repos overlap across train/test); "
            "disjoint-repo: split at repo level so train/test repos are disjoint."
        ),
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--tokenizer-model",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Tokenizer used for token-length filtering.",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=0,
        help="Filter out rows where tokens(gold_file_contents + patch) exceeds this value. <=0 disables.",
    )
    parser.add_argument(
        "--repos-base-dir",
        default="",
        help="Local cache dir for cloning repos when reconstructing gold_file_contents from patch+commit.",
    )
    parser.add_argument(
        "--reconstruct-missing-gold-files",
        action="store_true",
        help="If gold_file_contents is missing, clone/fetch repo and read patch files at base commit.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    loaded: list[tuple[str, Any]] = []
    total_loaded_rows = 0
    for dataset_name in args.hf_dataset:
        logger.info("Loading HF dataset %s split=%s", dataset_name, args.hf_split)
        ds = load_dataset(dataset_name, split=args.hf_split, trust_remote_code=True)
        loaded.append((dataset_name, ds))
        total_loaded_rows += len(ds)
    if not loaded:
        raise ValueError("No datasets loaded")
    logger.info("Loaded %d total rows before normalization", total_loaded_rows)

    normalized_rows: list[dict[str, Any]] = []
    skipped = 0
    drop_reasons = Counter()
    for source_name, ds in loaded:
        for item in tqdm(ds, desc=f"Normalize {source_name}", unit="row"):
            normalized, drop_reason = _normalize_item(
                dict(item),
                source_name=source_name,
                require_file_contents=not args.reconstruct_missing_gold_files,
            )
            if normalized is None:
                skipped += 1
                if drop_reason:
                    drop_reasons[drop_reason] += 1
                continue
            normalized_rows.append(normalized)
    if not normalized_rows:
        raise ValueError(
            "No valid normalized rows. "
            f"Top drop reasons: {dict(drop_reasons)}. "
            "Schema likely mismatched expected keys for repo/base_commit/patch/problem_statement/gold_file_contents."
        )
    logger.info("Normalized rows=%d skipped=%d", len(normalized_rows), skipped)

    if args.reconstruct_missing_gold_files:
        if not args.repos_base_dir:
            raise ValueError("--reconstruct-missing-gold-files requires --repos-base-dir")
        repos_base = Path(args.repos_base_dir)
        repos_base.mkdir(parents=True, exist_ok=True)
        reconstructed = 0
        reconstruction_failed = 0
        for row in tqdm(normalized_rows, desc="Reconstruct gold files", unit="row"):
            current_files = _as_dict_maybe_json(row.get("gold_file_contents", "{}"))
            if current_files:
                continue
            repo_name = str(row.get("repo") or "")
            commit_prefix = str(row.get("base_commit") or "")
            patch = str(row.get("patch") or "")
            if not repo_name or not commit_prefix or not patch:
                reconstruction_failed += 1
                continue
            repo_path = _ensure_repo(repos_base, repo_name)
            if repo_path is None:
                reconstruction_failed += 1
                continue
            full_commit = _resolve_commit(repo_path, commit_prefix)
            if not full_commit:
                reconstruction_failed += 1
                continue
            contents = _fetch_gold_file_contents_from_repo(repo_path, full_commit, patch)
            if not contents:
                reconstruction_failed += 1
                continue
            row["base_commit"] = full_commit
            row["gold_file_contents"] = json.dumps(contents)
            reconstructed += 1
        logger.info(
            "Gold file reconstruction done: reconstructed=%d failed=%d",
            reconstructed,
            reconstruction_failed,
        )
        # Drop rows still missing file contents.
        kept_after_reconstruct = []
        for row in normalized_rows:
            if _as_dict_maybe_json(row.get("gold_file_contents", "{}")):
                kept_after_reconstruct.append(row)
            else:
                drop_reasons["missing_gold_file_contents_after_reconstruct"] += 1
        normalized_rows = kept_after_reconstruct
        if not normalized_rows:
            raise ValueError("No rows left after reconstruction; check repo/commit parsing.")

    token_filtered_out = 0
    token_counts: list[int] = []
    if args.max_total_tokens > 0:
        logger.info(
            "Applying token filter max_total_tokens=%d with tokenizer=%s",
            args.max_total_tokens,
            args.tokenizer_model,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=True)
        kept_rows: list[dict[str, Any]] = []
        for row in tqdm(normalized_rows, desc="Token filter", unit="row"):
            token_count = _token_count_for_row(row, tokenizer)
            row["token_count"] = token_count
            token_counts.append(token_count)
            if token_count > args.max_total_tokens:
                token_filtered_out += 1
                continue
            kept_rows.append(row)
        normalized_rows = kept_rows
        logger.info(
            "Token filtering kept=%d filtered_out=%d",
            len(normalized_rows),
            token_filtered_out,
        )
        if not normalized_rows:
            raise ValueError("Token filtering removed all rows; raise max_total_tokens or inspect dataset.")

    if args.split_strategy == "field":
        train_rows = [row for row in normalized_rows if row["split"] == "train"]
        test_rows = [row for row in normalized_rows if row["split"] == "test"]
        if not train_rows or not test_rows:
            logger.warning(
                "split field not sufficient (train=%d test=%d), falling back to within-repo split",
                len(train_rows),
                len(test_rows),
            )
            train_rows, test_rows = _split_within_repo(normalized_rows, args.train_fraction, args.split_seed)
    elif args.split_strategy == "disjoint-repo":
        train_rows, test_rows = _split_disjoint_by_repo(normalized_rows, args.train_fraction, args.split_seed)
    else:
        train_rows, test_rows = _split_within_repo(normalized_rows, args.train_fraction, args.split_seed)

    for row in train_rows:
        row["split"] = "train"
    for row in test_rows:
        row["split"] = "test"

    train_ds = Dataset.from_list(train_rows)
    test_ds = Dataset.from_list(test_rows)

    train_output = Path(args.train_output_path)
    test_output = Path(args.test_output_path)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving train dataset: %s rows=%d", train_output, len(train_ds))
    train_ds.save_to_disk(str(train_output))
    logger.info("Saving test dataset: %s rows=%d", test_output, len(test_ds))
    test_ds.save_to_disk(str(test_output))

    train_repos = {row["repo"] for row in train_rows}
    test_repos = {row["repo"] for row in test_rows}
    overlap = train_repos & test_repos
    summary = {
        "hf_datasets": args.hf_dataset,
        "hf_split": args.hf_split,
        "split_strategy": args.split_strategy,
        "train_fraction": args.train_fraction,
        "split_seed": args.split_seed,
        "n_total": len(normalized_rows),
        "n_skipped_schema_mismatch": skipped,
        "schema_drop_reasons": dict(drop_reasons),
        "reconstruct_missing_gold_files": bool(args.reconstruct_missing_gold_files),
        "repos_base_dir": args.repos_base_dir,
        "n_filtered_by_token_limit": token_filtered_out,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "n_train_repos": len(train_repos),
        "n_test_repos": len(test_repos),
        "n_repo_overlap": len(overlap),
        "repo_disjoint": len(overlap) == 0,
        "dataset_distribution": dict(Counter(row["dataset_source"] for row in normalized_rows)),
    }
    if token_counts:
        token_counts_sorted = sorted(token_counts)
        summary["token_stats_pre_filter"] = {
            "min": token_counts_sorted[0],
            "p50": token_counts_sorted[len(token_counts_sorted) // 2],
            "p90": token_counts_sorted[int(0.9 * (len(token_counts_sorted) - 1))],
            "p99": token_counts_sorted[int(0.99 * (len(token_counts_sorted) - 1))],
            "max": token_counts_sorted[-1],
        }
        summary["max_total_tokens"] = args.max_total_tokens
        summary["tokenizer_model"] = args.tokenizer_model
    summary_path = train_output.parent / "prepare_swe_smith_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()

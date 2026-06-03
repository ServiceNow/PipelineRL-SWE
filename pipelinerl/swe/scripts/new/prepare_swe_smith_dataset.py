#!/usr/bin/env python
import argparse
import difflib
import hashlib
import json
import logging
import random
import re
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
import git
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _git_blob_hash(text: str) -> str:
    data = text.encode("utf-8", errors="surrogateescape")
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()[:7]


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
    # Official SWE-bench rows use ordinary GitHub repo names such as
    # DataDog/integrations-core. Do not treat the owner as a dataset namespace.
    if "/" in text:
        payload_candidate = text.split("/", 1)[1]
        has_swesmith_owner_repo = "__" in payload_candidate
        has_commit_suffix = bool(re.search(r"\.[0-9a-fA-F]{6,40}$", payload_candidate))
        if not has_swesmith_owner_repo and not has_commit_suffix:
            return text.removesuffix(".git"), None
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


def _run_git_command(
    args: list[str],
    timeout_seconds: int,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds if timeout_seconds > 0 else None,
    )


def _ensure_repo(
    local_dir: Path,
    repo_name: str,
    timeout_seconds: int,
    partial_clone: bool,
    fetch_existing: bool,
) -> Path | None:
    repo_path = local_dir / repo_name.replace("/", "_")
    url = f"https://github.com/{repo_name}.git"
    try:
        if repo_path.exists() and (repo_path / ".git").exists():
            _run_git_command(["rev-parse", "--is-inside-work-tree"], timeout_seconds, cwd=repo_path)
            if fetch_existing:
                fetch_args = ["fetch", "origin"]
                if partial_clone:
                    fetch_args.insert(1, "--filter=blob:none")
                _run_git_command(fetch_args, timeout_seconds, cwd=repo_path)
            return repo_path
        if repo_path.exists():
            logger.warning("Repo path exists but is not a git repo, skipping %s (%s)", repo_name, repo_path)
            return None
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        clone_args = ["clone"]
        if partial_clone:
            clone_args.extend(["--filter=blob:none", "--no-checkout"])
        clone_args.extend([url, str(repo_path)])
        _run_git_command(clone_args, timeout_seconds)
        return repo_path
    except subprocess.TimeoutExpired:
        logger.warning("Timed out preparing repo %s after %ds (%s)", repo_name, timeout_seconds, repo_path)
        return None
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        logger.warning("Failed to prepare repo %s (%s): %s", repo_name, repo_path, stderr)
        return None
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


def _fetch_gold_file_contents_from_raw_url(
    repo_name: str,
    commit_hash: str,
    patch: str,
    timeout_seconds: int,
) -> dict[str, str]:
    files = _parse_patch_files(patch)
    if not files:
        return {}
    contents: dict[str, str] = {}
    timeout = timeout_seconds if timeout_seconds > 0 else None
    for file_path in files:
        quoted_path = urllib.parse.quote(file_path, safe="/")
        url = f"https://raw.githubusercontent.com/{repo_name}/{commit_hash}/{quoted_path}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                if int(response.status) < 200 or int(response.status) >= 300:
                    continue
                contents[file_path] = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            logger.debug("Failed raw file fetch repo=%s file=%s url=%s: %s", repo_name, file_path, url, exc)
            continue
    return contents



def _strip_patch_index_lines(patch: str) -> str:
    lines = []
    for line in patch.splitlines():
        if re.fullmatch(r"index [0-9a-fA-F]{6,40}\.\.[0-9a-fA-F]{6,40}(?: \d+)?", line.strip()):
            continue
        lines.append(line)
    text = "\n".join(lines)
    if text and patch.endswith("\n"):
        text += "\n"
    return text


def _apply_unified_patch_to_contents(file_contents: dict[str, str], patch: str) -> dict[str, str] | None:
    touched = _parse_patch_files(patch)
    if not touched:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for file_path in touched:
            if file_path not in file_contents:
                return None
            target = root / file_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file_contents[file_path], encoding="utf-8", errors="surrogateescape")
        patch_path = root / "bug.patch"
        patch_path.write_text(_strip_patch_index_lines(patch), encoding="utf-8", errors="surrogateescape")
        result = subprocess.run(
            ["patch", "-p1", "--batch", "--forward", "--reject-file=-", "-i", str(patch_path)],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            return None
        out: dict[str, str] = {}
        for file_path in touched:
            target = root / file_path
            if not target.exists():
                return None
            out[file_path] = target.read_text(encoding="utf-8", errors="surrogateescape")
        return out


def _unified_diff_body(old: str, new: str) -> str:
    lines = list(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile="old",
            tofile="new",
            lineterm="",
            n=3,
        )
    )
    if len(lines) <= 2:
        return ""
    return "\n".join(lines[2:])


def _build_fix_patch_from_bugged_to_clean(
    clean_contents: dict[str, str],
    bugged_contents: dict[str, str],
) -> str:
    parts: list[str] = []
    for file_path in sorted(set(clean_contents) | set(bugged_contents)):
        clean = clean_contents.get(file_path)
        bugged = bugged_contents.get(file_path)
        if clean is None or bugged is None or clean == bugged:
            continue
        body = _unified_diff_body(bugged, clean)
        if not body:
            continue
        parts.extend(
            [
                f"diff --git a/{file_path} b/{file_path}",
                f"index {_git_blob_hash(bugged)}..{_git_blob_hash(clean)} 100644",
                f"--- a/{file_path}",
                f"+++ b/{file_path}",
                body,
            ]
        )
    text = "\n".join(parts)
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def _convert_swesmith_rows_to_bugged_context(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert SWE-Smith rows from clean/base context to bugged repair context.

    Canonical output fields:
    - repair_file_contents: the bugged code shown to the solver.
    - reference_file_contents: the clean target code.
    - bug_introducing_patch: SWE-Smith's original clean -> bug patch.
    - repair_target_patch: the bugged -> clean patch used for proxy reward.

    Backward-compatible aliases are also written: gold_file_contents points to
    repair_file_contents and patch points to repair_target_patch.
    """
    kept: list[dict[str, Any]] = []
    stats = {
        "input_rows": len(rows),
        "converted": 0,
        "missing_clean_contents": 0,
        "bug_patch_apply_failed": 0,
        "empty_fix_patch": 0,
    }
    for row in rows:
        clean_contents = _as_dict_maybe_json(row.get("gold_file_contents", "{}"))
        if not clean_contents:
            stats["missing_clean_contents"] += 1
            continue
        bug_patch = str(row.get("patch") or "")
        bugged_contents = _apply_unified_patch_to_contents(clean_contents, bug_patch)
        if bugged_contents is None:
            stats["bug_patch_apply_failed"] += 1
            continue
        full_bugged = dict(clean_contents)
        full_bugged.update(bugged_contents)
        fix_patch = _build_fix_patch_from_bugged_to_clean(clean_contents, full_bugged)
        if not fix_patch:
            stats["empty_fix_patch"] += 1
            continue
        new_row = dict(row)
        reference_json = json.dumps(clean_contents)
        repair_json = json.dumps(full_bugged)
        new_row["reference_file_contents"] = reference_json
        new_row["repair_file_contents"] = repair_json
        new_row["bug_introducing_patch"] = bug_patch
        new_row["repair_target_patch"] = fix_patch
        # Backward-compatible aliases for existing loader/proxy code.
        new_row["clean_file_contents"] = reference_json
        new_row["bug_patch"] = bug_patch
        new_row["fix_patch"] = fix_patch
        new_row["gold_file_contents"] = repair_json
        new_row["patch"] = fix_patch
        new_row["swesmith_bugged_context"] = True
        kept.append(new_row)
        stats["converted"] += 1
    return kept, stats


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
    parser.add_argument("--train-output-path", default="", help="Output path for processed train dataset (save_to_disk).")
    parser.add_argument("--test-output-path", default="", help="Output path for processed test dataset (save_to_disk).")
    parser.add_argument(
        "--single-output-path",
        default="",
        help="If set, save all normalized rows to this dataset path and skip train/test splitting.",
    )
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
    parser.add_argument(
        "--gold-file-source",
        choices=["git", "raw-url"],
        default="git",
        help=(
            "How to reconstruct missing gold_file_contents. "
            "git clones/fetches repos; raw-url fetches touched files directly from raw.githubusercontent.com."
        ),
    )
    parser.add_argument(
        "--max-normalized-rows",
        type=int,
        default=0,
        help="Randomly sample this many normalized rows before reconstructing file contents. <=0 keeps all rows.",
    )
    parser.add_argument("--row-sample-seed", type=int, default=42)
    parser.add_argument(
        "--git-timeout-seconds",
        type=int,
        default=900,
        help="Timeout for each git clone/fetch command. <=0 disables the timeout.",
    )
    parser.add_argument(
        "--no-partial-clone",
        action="store_true",
        help="Use full git clones instead of blobless partial clones.",
    )
    parser.add_argument(
        "--fetch-existing-repos",
        action="store_true",
        help="Run git fetch on repo caches that already exist. Disabled by default to avoid long resumes.",
    )
    parser.add_argument(
        "--swesmith-bugged-context",
        action="store_true",
        help=(
            "For SWE-Smith rows, apply the bug-introducing patch to reconstructed file contents, "
            "use those bugged contents as repair context, and replace patch with the inverse fix patch. "
            "Do not use this for SWE-Bench datasets, whose patch is already the fix direction."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.single_output_path:
        if args.train_output_path or args.test_output_path:
            raise ValueError("--single-output-path cannot be combined with --train-output-path/--test-output-path")
    elif not args.train_output_path or not args.test_output_path:
        raise ValueError("Provide either --single-output-path or both --train-output-path and --test-output-path")

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

    sampled_before_reconstruct = 0
    if args.max_normalized_rows and args.max_normalized_rows > 0 and len(normalized_rows) > args.max_normalized_rows:
        rng = random.Random(int(args.row_sample_seed))
        rng.shuffle(normalized_rows)
        sampled_before_reconstruct = len(normalized_rows) - int(args.max_normalized_rows)
        normalized_rows = normalized_rows[: int(args.max_normalized_rows)]
        logger.info(
            "Sampled normalized rows before reconstruction: kept=%d dropped=%d seed=%d",
            len(normalized_rows),
            sampled_before_reconstruct,
            int(args.row_sample_seed),
        )

    if args.reconstruct_missing_gold_files:
        if args.gold_file_source == "git" and not args.repos_base_dir:
            raise ValueError("--reconstruct-missing-gold-files requires --repos-base-dir")
        repos_base = Path(args.repos_base_dir)
        if args.gold_file_source == "git":
            repos_base.mkdir(parents=True, exist_ok=True)
        repo_cache: dict[str, Path | None] = {}
        commit_cache: dict[tuple[str, str], str | None] = {}
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
            if args.gold_file_source == "raw-url":
                contents = _fetch_gold_file_contents_from_raw_url(
                    repo_name=repo_name,
                    commit_hash=commit_prefix,
                    patch=patch,
                    timeout_seconds=int(args.git_timeout_seconds),
                )
                full_commit = commit_prefix
            else:
                if repo_name not in repo_cache:
                    logger.info("Preparing repo cache for %s", repo_name)
                    repo_cache[repo_name] = _ensure_repo(
                        repos_base,
                        repo_name,
                        timeout_seconds=int(args.git_timeout_seconds),
                        partial_clone=not bool(args.no_partial_clone),
                        fetch_existing=bool(args.fetch_existing_repos),
                    )
                repo_path = repo_cache[repo_name]
                if repo_path is None:
                    reconstruction_failed += 1
                    continue
                commit_key = (str(repo_path), commit_prefix)
                if commit_key not in commit_cache:
                    commit_cache[commit_key] = _resolve_commit(repo_path, commit_prefix)
                full_commit = commit_cache[commit_key]
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

    swesmith_bugged_context_stats: dict[str, int] | None = None
    if args.swesmith_bugged_context:
        logger.info("Converting SWE-Smith rows to bugged repair context")
        normalized_rows, swesmith_bugged_context_stats = _convert_swesmith_rows_to_bugged_context(normalized_rows)
        logger.info("SWE-Smith bugged-context conversion stats: %s", swesmith_bugged_context_stats)
        if not normalized_rows:
            raise ValueError("No rows left after SWE-Smith bugged-context conversion")

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

    if args.single_output_path:
        single_ds = Dataset.from_list(normalized_rows)
        single_output = Path(args.single_output_path)
        single_output.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving single dataset: %s rows=%d", single_output, len(single_ds))
        single_ds.save_to_disk(str(single_output))

        repos = {row["repo"] for row in normalized_rows}
        summary = {
            "hf_datasets": args.hf_dataset,
            "hf_split": args.hf_split,
            "single_output_path": str(single_output),
            "n_total": len(normalized_rows),
            "n_sampled_before_reconstruct": sampled_before_reconstruct,
            "max_normalized_rows": int(args.max_normalized_rows),
            "row_sample_seed": int(args.row_sample_seed),
            "n_skipped_schema_mismatch": skipped,
            "schema_drop_reasons": dict(drop_reasons),
            "reconstruct_missing_gold_files": bool(args.reconstruct_missing_gold_files),
            "repos_base_dir": args.repos_base_dir,
            "gold_file_source": args.gold_file_source,
            "git_timeout_seconds": int(args.git_timeout_seconds),
            "partial_clone": not bool(args.no_partial_clone),
            "fetch_existing_repos": bool(args.fetch_existing_repos),
            "swesmith_bugged_context": bool(args.swesmith_bugged_context),
            "swesmith_bugged_context_stats": swesmith_bugged_context_stats,
            "n_filtered_by_token_limit": token_filtered_out,
            "n_repos": len(repos),
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
        summary_path = single_output.parent / "prepare_swe_bench_single_summary.json"
        with summary_path.open("w") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Summary written to %s", summary_path)
        return

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
        "n_sampled_before_reconstruct": sampled_before_reconstruct,
        "max_normalized_rows": int(args.max_normalized_rows),
        "row_sample_seed": int(args.row_sample_seed),
        "n_skipped_schema_mismatch": skipped,
        "schema_drop_reasons": dict(drop_reasons),
        "reconstruct_missing_gold_files": bool(args.reconstruct_missing_gold_files),
        "repos_base_dir": args.repos_base_dir,
        "gold_file_source": args.gold_file_source,
        "git_timeout_seconds": int(args.git_timeout_seconds),
        "partial_clone": not bool(args.no_partial_clone),
        "fetch_existing_repos": bool(args.fetch_existing_repos),
        "swesmith_bugged_context": bool(args.swesmith_bugged_context),
        "swesmith_bugged_context_stats": swesmith_bugged_context_stats,
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

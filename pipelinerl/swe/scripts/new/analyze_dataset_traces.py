#!/usr/bin/env python
import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipelinerl.swe.scripts.new.router_trace_utils import load_router_traces

logger = logging.getLogger(__name__)


EXT_LANGUAGE_MAP = {
    ".py": "Python",
    ".pyi": "Python",
    ".ipynb": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".go": "Go",
    ".rs": "Rust",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".c": "C",
    ".h": "C/C++",
    ".hpp": "C/C++",
    ".cs": "C#",
    ".php": "PHP",
    ".rb": "Ruby",
    ".swift": "Swift",
    ".scala": "Scala",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
    ".sql": "SQL",
    ".md": "Markdown",
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _ext_from_path(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    return suffix or "<no_ext>"


def _language_from_ext(ext: str) -> str:
    return EXT_LANGUAGE_MAP.get(ext.lower(), "Other/Unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze router trace dataset composition (repos, frequency, languages, token stats).")
    parser.add_argument("--input-glob", action="append", required=True, help="Glob pattern for trace JSONL files. Can be repeated.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="all", choices=["train", "test", "all"])
    parser.add_argument("--all-model-versions", action="store_true", help="Keep all model versions (default: latest only).")
    parser.add_argument("--keep-duplicates", action="store_true", help="Do not dedupe by problem id.")
    parser.add_argument("--top-k", type=int, default=50)
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
        raise ValueError("No traces found for provided filters.")

    logger.info("Loaded %d traces", len(traces))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_counts = Counter()
    dataset_counts = Counter()
    model_version_counts = Counter()
    repo_counts = Counter()
    ext_counts = Counter()
    language_counts = Counter()
    split_repo_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_dataset_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_model_version_counts: dict[str, Counter[str]] = defaultdict(Counter)

    repo_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "policy_reward_sum": 0.0,
            "policy_success_sum": 0.0,
            "policy_prompt_tokens_sum": 0.0,
            "policy_output_tokens_sum": 0.0,
            "num_files_for_repair_sum": 0.0,
        }
    )

    for trace in traces:
        split_value = str(trace.get("split") or "unknown")
        dataset_value = str(trace.get("dataset") or "unknown")
        model_version = str(trace.get("model_version") if trace.get("model_version") is not None else "unknown")
        repo = str(trace.get("repo") or "unknown")

        split_counts[split_value] += 1
        dataset_counts[dataset_value] += 1
        model_version_counts[model_version] += 1
        repo_counts[repo] += 1
        split_repo_counts[split_value][repo] += 1
        split_dataset_counts[split_value][dataset_value] += 1
        split_model_version_counts[split_value][model_version] += 1

        files_for_repair = trace.get("files_for_repair")
        if isinstance(files_for_repair, list):
            repo_stats[repo]["num_files_for_repair_sum"] += float(len(files_for_repair))
            for item in files_for_repair:
                if not isinstance(item, str):
                    continue
                ext = _ext_from_path(item)
                ext_counts[ext] += 1
                language_counts[_language_from_ext(ext)] += 1

        policy = trace.get("policy") or {}
        repo_stats[repo]["count"] += 1.0
        repo_stats[repo]["policy_reward_sum"] += _safe_float(policy.get("reward"), 0.0)
        repo_stats[repo]["policy_success_sum"] += _safe_float(policy.get("success"), 0.0)
        repo_stats[repo]["policy_prompt_tokens_sum"] += _safe_int(policy.get("prompt_tokens"), 0)
        repo_stats[repo]["policy_output_tokens_sum"] += _safe_int(policy.get("output_tokens"), 0)

    top_k = max(1, args.top_k)

    repo_rows: list[dict[str, Any]] = []
    for repo, count in repo_counts.most_common(top_k):
        stats = repo_stats[repo]
        denom = stats["count"] if stats["count"] > 0 else 1.0
        repo_rows.append(
            {
                "repo": repo,
                "count": int(count),
                "share": float(count) / len(traces),
                "avg_policy_reward": stats["policy_reward_sum"] / denom,
                "avg_policy_success": stats["policy_success_sum"] / denom,
                "avg_policy_prompt_tokens": stats["policy_prompt_tokens_sum"] / denom,
                "avg_policy_output_tokens": stats["policy_output_tokens_sum"] / denom,
                "avg_num_files_for_repair": stats["num_files_for_repair_sum"] / denom,
            }
        )

    extension_rows = [
        {"extension": ext, "count": count, "share": float(count) / max(sum(ext_counts.values()), 1)}
        for ext, count in ext_counts.most_common(top_k)
    ]
    language_rows = [
        {"language": language, "count": count, "share": float(count) / max(sum(language_counts.values()), 1)}
        for language, count in language_counts.most_common(top_k)
    ]

    _write_csv(
        output_dir / "repo_frequency.csv",
        repo_rows,
        [
            "repo",
            "count",
            "share",
            "avg_policy_reward",
            "avg_policy_success",
            "avg_policy_prompt_tokens",
            "avg_policy_output_tokens",
            "avg_num_files_for_repair",
        ],
    )
    _write_csv(
        output_dir / "language_frequency.csv",
        language_rows,
        ["language", "count", "share"],
    )
    _write_csv(
        output_dir / "extension_frequency.csv",
        extension_rows,
        ["extension", "count", "share"],
    )
    _write_csv(
        output_dir / "split_frequency.csv",
        [{"split": k, "count": v, "share": float(v) / len(traces)} for k, v in split_counts.items()],
        ["split", "count", "share"],
    )
    _write_csv(
        output_dir / "dataset_frequency.csv",
        [{"dataset": k, "count": v, "share": float(v) / len(traces)} for k, v in dataset_counts.items()],
        ["dataset", "count", "share"],
    )
    _write_csv(
        output_dir / "model_version_frequency.csv",
        [{"model_version": k, "count": v, "share": float(v) / len(traces)} for k, v in model_version_counts.items()],
        ["model_version", "count", "share"],
    )

    per_split_rows: list[dict[str, Any]] = []
    per_split_dataset_rows: list[dict[str, Any]] = []
    per_split_model_version_rows: list[dict[str, Any]] = []
    for split_name, split_total in split_counts.items():
        split_repo_counter = split_repo_counts[split_name]
        for repo, count in split_repo_counter.most_common(top_k):
            per_split_rows.append(
                {
                    "split": split_name,
                    "repo": repo,
                    "count": count,
                    "share_within_split": float(count) / split_total if split_total else 0.0,
                }
            )
        split_dataset_counter = split_dataset_counts[split_name]
        for dataset_name, count in split_dataset_counter.most_common(top_k):
            per_split_dataset_rows.append(
                {
                    "split": split_name,
                    "dataset": dataset_name,
                    "count": count,
                    "share_within_split": float(count) / split_total if split_total else 0.0,
                }
            )
        split_model_counter = split_model_version_counts[split_name]
        for model_ver, count in split_model_counter.items():
            per_split_model_version_rows.append(
                {
                    "split": split_name,
                    "model_version": model_ver,
                    "count": count,
                    "share_within_split": float(count) / split_total if split_total else 0.0,
                }
            )

    _write_csv(
        output_dir / "repo_frequency_by_split.csv",
        per_split_rows,
        ["split", "repo", "count", "share_within_split"],
    )
    _write_csv(
        output_dir / "dataset_frequency_by_split.csv",
        per_split_dataset_rows,
        ["split", "dataset", "count", "share_within_split"],
    )
    _write_csv(
        output_dir / "model_version_frequency_by_split.csv",
        per_split_model_version_rows,
        ["split", "model_version", "count", "share_within_split"],
    )

    train_repos = set(split_repo_counts.get("train", {}).keys())
    test_repos = set(split_repo_counts.get("test", {}).keys())
    repo_overlap_rows = [
        {
            "train_repo_count": len(train_repos),
            "test_repo_count": len(test_repos),
            "intersection_count": len(train_repos & test_repos),
            "train_only_count": len(train_repos - test_repos),
            "test_only_count": len(test_repos - train_repos),
            "jaccard": (len(train_repos & test_repos) / len(train_repos | test_repos)) if (train_repos or test_repos) else 0.0,
            "train_covers_test": (len(test_repos - train_repos) == 0),
            "test_covers_train": (len(train_repos - test_repos) == 0),
            "are_disjoint": len(train_repos & test_repos) == 0,
        }
    ]
    _write_csv(
        output_dir / "repo_overlap_train_vs_test.csv",
        repo_overlap_rows,
        [
            "train_repo_count",
            "test_repo_count",
            "intersection_count",
            "train_only_count",
            "test_only_count",
            "jaccard",
            "train_covers_test",
            "test_covers_train",
            "are_disjoint",
        ],
    )

    _write_csv(
        output_dir / "repo_membership_train_vs_test.csv",
        [
            {
                "repo": repo,
                "in_train": repo in train_repos,
                "in_test": repo in test_repos,
                "train_count": split_repo_counts.get("train", {}).get(repo, 0),
                "test_count": split_repo_counts.get("test", {}).get(repo, 0),
            }
            for repo in sorted(train_repos | test_repos)
        ],
        ["repo", "in_train", "in_test", "train_count", "test_count"],
    )

    summary = {
        "n_traces": len(traces),
        "split_filter": args.split,
        "latest_model_only": not args.all_model_versions,
        "dedupe_by_problem": not args.keep_duplicates,
        "n_unique_repos": len(repo_counts),
        "n_unique_datasets": len(dataset_counts),
        "n_unique_model_versions": len(model_version_counts),
        "n_unique_extensions": len(ext_counts),
        "n_unique_languages": len(language_counts),
        "top_repo": repo_rows[0] if repo_rows else None,
        "top_language": language_rows[0] if language_rows else None,
        "top_extension": extension_rows[0] if extension_rows else None,
        "files": {
            "repo_frequency": "repo_frequency.csv",
            "repo_frequency_by_split": "repo_frequency_by_split.csv",
            "repo_overlap_train_vs_test": "repo_overlap_train_vs_test.csv",
            "repo_membership_train_vs_test": "repo_membership_train_vs_test.csv",
            "language_frequency": "language_frequency.csv",
            "extension_frequency": "extension_frequency.csv",
            "split_frequency": "split_frequency.csv",
            "dataset_frequency": "dataset_frequency.csv",
            "dataset_frequency_by_split": "dataset_frequency_by_split.csv",
            "model_version_frequency": "model_version_frequency.csv",
            "model_version_frequency_by_split": "model_version_frequency_by_split.csv",
        },
        "repo_overlap_train_vs_test": repo_overlap_rows[0],
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Wrote dataset analysis to %s", output_dir)


if __name__ == "__main__":
    main()

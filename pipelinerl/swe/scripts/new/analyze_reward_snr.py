#!/usr/bin/env python
import argparse
import csv
import importlib.util
import json
import logging
from pathlib import Path
from collections import Counter

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - CLI guard
    raise SystemExit("numpy is required for analyze_reward_snr.py") from exc

try:
    import matplotlib.pyplot as plt  # type: ignore

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    from datasets import load_dataset
except ModuleNotFoundError:
    load_dataset = None  # type: ignore[assignment]

try:
    from pipelinerl.swe.scripts.new.router_trace_utils import (
        extract_reward_vector,
        extract_route_labels,
        load_router_traces,
    )
except ModuleNotFoundError:
    module_path = Path(__file__).with_name("router_trace_utils.py")
    spec = importlib.util.spec_from_file_location("router_trace_utils_local", module_path)
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extract_reward_vector = module.extract_reward_vector
    extract_route_labels = module.extract_route_labels
    load_router_traces = module.load_router_traces

logger = logging.getLogger(__name__)

EXT_LANGUAGE_MAP = {
    ".py": "Python",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
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


def _problem_id(trace: dict) -> str:
    return str(trace.get("problem_id") or trace.get("instance_id") or trace.get("id") or "")


def _dataset_name(trace: dict) -> str:
    return str(trace.get("dataset") or "")


def _load_offline_dataset_rows(
    dataset_dir: Path,
    dataset_split: str,
) -> tuple[list[dict], list[str]]:
    if load_dataset is None:
        raise SystemExit("datasets is required for --dataset-dir mode")

    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Offline router metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    route_labels = list(metadata.get("route_labels") or [])
    if not route_labels:
        raise ValueError(f"metadata.json is missing route_labels: {metadata_path}")

    split_names = ["train", "eval"] if dataset_split == "all" else [dataset_split]
    data_files: dict[str, list[str]] = {}
    for split_name in split_names:
        split_dir = dataset_dir / split_name
        files = sorted(str(path) for path in split_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet shards found for split={split_name} in {split_dir}")
        data_files[split_name] = files

    dataset_dict = load_dataset("parquet", data_files=data_files)
    rows: list[dict] = []
    for split_name in split_names:
        split_dataset = dataset_dict[split_name]
        for row in split_dataset:
            rewards = row.get("performance_targets")
            if not isinstance(rewards, list) or len(rewards) < 2:
                continue
            rows.append(
                {
                    "dataset": row.get("dataset"),
                    "problem_id": row.get("problem_id") or row.get("instance_id") or row.get("id"),
                    "language": row.get("language"),
                    "repo": row.get("repo"),
                    "performance_targets": rewards,
                    "split": split_name,
                }
            )
    if not rows:
        raise ValueError(f"No valid rows with performance_targets found in {dataset_dir}")
    return rows, route_labels


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _infer_language(trace: dict) -> str:
    for key in ("language", "lang", "repo_language"):
        value = trace.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    files = trace.get("files_for_repair")
    if not isinstance(files, list) or not files:
        return "Unknown"
    ext_counts = Counter()
    for item in files:
        if not isinstance(item, str):
            continue
        suffix = Path(item).suffix.lower()
        if suffix:
            ext_counts[EXT_LANGUAGE_MAP.get(suffix, "Other/Unknown")] += 1
    if not ext_counts:
        return "Unknown"
    return ext_counts.most_common(1)[0][0]


def _write_task_csv(path: Path, traces: list[dict], rewards: np.ndarray, route_labels: list[str]) -> None:
    fieldnames = [
        "dataset",
        "problem_id",
        "between_expert_variance",
        "between_expert_std",
        "reward_min",
        "reward_max",
        "reward_range",
        "reward_mean",
        "best_route_idx",
        "best_route_label",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        per_task_var = rewards.var(axis=1)
        per_task_std = rewards.std(axis=1)
        per_task_min = rewards.min(axis=1)
        per_task_max = rewards.max(axis=1)
        per_task_mean = rewards.mean(axis=1)
        per_task_best_idx = rewards.argmax(axis=1)
        for idx, trace in enumerate(traces):
            best_idx = int(per_task_best_idx[idx])
            writer.writerow(
                {
                    "dataset": _dataset_name(trace),
                    "problem_id": _problem_id(trace),
                    "between_expert_variance": float(per_task_var[idx]),
                    "between_expert_std": float(per_task_std[idx]),
                    "reward_min": float(per_task_min[idx]),
                    "reward_max": float(per_task_max[idx]),
                    "reward_range": float(per_task_max[idx] - per_task_min[idx]),
                    "reward_mean": float(per_task_mean[idx]),
                    "best_route_idx": best_idx,
                    "best_route_label": route_labels[best_idx],
                }
            )


def _write_route_csv(path: Path, rewards: np.ndarray, route_labels: list[str]) -> None:
    fieldnames = [
        "route_idx",
        "route_label",
        "mean_reward",
        "std_across_tasks",
        "variance_across_tasks",
        "min_reward",
        "max_reward",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for route_idx, label in enumerate(route_labels):
            column = rewards[:, route_idx]
            writer.writerow(
                {
                    "route_idx": route_idx,
                    "route_label": label,
                    "mean_reward": float(column.mean()),
                    "std_across_tasks": float(column.std()),
                    "variance_across_tasks": float(column.var()),
                    "min_reward": float(column.min()),
                    "max_reward": float(column.max()),
                }
            )


def _maybe_plot(output_dir: Path, rewards: np.ndarray, route_labels: list[str]) -> None:
    if not MATPLOTLIB_AVAILABLE:
        logger.info("matplotlib not available; skipping plots")
        return

    between_var = rewards.var(axis=1)
    within_var = rewards.var(axis=0)

    plt.figure(figsize=(10, 4.5))
    plt.hist(between_var, bins=30, color="#4c78a8", edgecolor="black", alpha=0.85)
    plt.axvline(float(between_var.mean()), color="#f58518", linestyle="--", label=f"mean={between_var.mean():.4f}")
    plt.xlabel("Between-expert variance per task")
    plt.ylabel("Count")
    plt.title("Per-task Between-Expert Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "between_expert_variance_hist.png", dpi=200)
    plt.close()

    plt.figure(figsize=(max(7, len(route_labels) * 1.6), 4.8))
    x = np.arange(len(route_labels))
    plt.bar(x, within_var, color="#54a24b")
    plt.xticks(x, route_labels, rotation=20, ha="right")
    plt.ylabel("Variance across tasks")
    plt.title("Within-Expert Variance Across Tasks")
    plt.tight_layout()
    plt.savefig(output_dir / "within_expert_variance_bar.png", dpi=200)
    plt.close()


def _write_language_csv(
    path: Path,
    traces: list[dict],
    rewards: np.ndarray,
    route_labels: list[str],
    ddof: int,
) -> list[dict]:
    grouped: dict[str, list[int]] = {}
    for idx, trace in enumerate(traces):
        grouped.setdefault(_infer_language(trace), []).append(idx)

    rows: list[dict] = []
    fieldnames = [
        "language",
        "n_traces",
        "between_expert_variance_mean",
        "between_expert_variance_median",
        "between_expert_std_mean",
        "within_expert_variance_mean",
        "within_expert_std_mean",
        "snr_variance_ratio_between_over_within",
        "snr_std_ratio_between_over_within",
    ]
    for route_label in route_labels:
        slug = "".join(ch if ch.isalnum() else "_" for ch in route_label.lower()).strip("_")
        fieldnames.extend(
            [
                f"{slug}_mean_reward",
                f"{slug}_within_variance",
                f"{slug}_within_std",
            ]
        )

    for language, idxs in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        sub = rewards[idxs, :]
        between_var = sub.var(axis=1, ddof=ddof)
        between_std = sub.std(axis=1, ddof=ddof)
        within_var = sub.var(axis=0, ddof=ddof)
        within_std = sub.std(axis=0, ddof=ddof)
        row = {
            "language": language,
            "n_traces": int(sub.shape[0]),
            "between_expert_variance_mean": float(between_var.mean()),
            "between_expert_variance_median": float(np.median(between_var)),
            "between_expert_std_mean": float(between_std.mean()),
            "within_expert_variance_mean": float(within_var.mean()),
            "within_expert_std_mean": float(within_std.mean()),
            "snr_variance_ratio_between_over_within": (
                float(between_var.mean() / within_var.mean()) if float(within_var.mean()) > 0 else None
            ),
            "snr_std_ratio_between_over_within": (
                float(between_std.mean() / within_std.mean()) if float(within_std.mean()) > 0 else None
            ),
        }
        for route_idx, route_label in enumerate(route_labels):
            slug = "".join(ch if ch.isalnum() else "_" for ch in route_label.lower()).strip("_")
            row[f"{slug}_mean_reward"] = float(sub[:, route_idx].mean())
            row[f"{slug}_within_variance"] = float(within_var[route_idx])
            row[f"{slug}_within_std"] = float(within_std[route_idx])
        rows.append(row)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze reward-vector signal-vs-noise from router traces.")
    parser.add_argument("--input-glob", action="append", default=[], help="Input JSONL glob. Repeatable.")
    parser.add_argument("--dataset-dir", default=None, help="Offline router dataset directory with train/eval parquet shards.")
    parser.add_argument(
        "--dataset-split",
        default="all",
        choices=["train", "eval", "all"],
        help="Which split(s) to use with --dataset-dir.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for summary outputs.")
    parser.add_argument(
        "--split",
        default=None,
        choices=[None, "train", "test", "all"],
        help="Optional split filter. Use 'all' to disable split filtering.",
    )
    parser.add_argument(
        "--latest-model-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only the latest complete model_version.",
    )
    parser.add_argument(
        "--dedupe-by-problem",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only the latest trace per problem.",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        default=0,
        choices=[0, 1],
        help="Variance degrees of freedom. Default 0 for population variance.",
    )
    parser.add_argument(
        "--group-by-language",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also compute the same SNR breakdown separately per inferred language.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    has_input_globs = bool(args.input_glob)
    has_dataset_dir = bool(args.dataset_dir)
    if has_input_globs == has_dataset_dir:
        raise ValueError("Specify exactly one of --input-glob or --dataset-dir.")

    route_labels: list[str]
    summary_inputs: dict[str, object]
    if has_dataset_dir:
        rows, route_labels = _load_offline_dataset_rows(Path(args.dataset_dir), str(args.dataset_split))
        reward_vectors = [row["performance_targets"] for row in rows]
        traces = rows
        summary_inputs = {
            "input_mode": "offline_dataset",
            "dataset_dir": str(Path(args.dataset_dir)),
            "dataset_split": str(args.dataset_split),
        }
    else:
        split = None if args.split in (None, "all") else args.split
        traces = load_router_traces(
            input_globs=args.input_glob,
            split=split,
            latest_model_only=args.latest_model_only,
            dedupe_by_problem=args.dedupe_by_problem,
        )
        if not traces:
            raise ValueError("No traces matched the requested filters.")
        reward_vectors = [extract_reward_vector(trace) for trace in traces]
        summary_inputs = {
            "input_mode": "router_traces",
            "input_globs": list(args.input_glob),
            "split": split or "all",
            "latest_model_only": bool(args.latest_model_only),
            "dedupe_by_problem": bool(args.dedupe_by_problem),
        }

    dim = min(len(vec) for vec in reward_vectors)
    if dim <= 0:
        raise ValueError("Reward vectors are empty.")
    if dim < 2:
        raise ValueError("Need at least two routes to compute between-expert variance.")

    rewards = np.asarray([vec[:dim] for vec in reward_vectors], dtype=np.float64)
    if has_dataset_dir:
        if len(route_labels) < dim:
            raise ValueError(
                f"metadata route_labels has length {len(route_labels)} but rewards have dimension {dim}"
            )
        route_labels = route_labels[:dim]
    else:
        n_experts = dim - 1
        route_labels = extract_route_labels(traces, n_experts=n_experts)[:dim]

    per_task_between_var = rewards.var(axis=1, ddof=args.ddof)
    per_task_between_std = rewards.std(axis=1, ddof=args.ddof)
    per_route_within_var = rewards.var(axis=0, ddof=args.ddof)
    per_route_within_std = rewards.std(axis=0, ddof=args.ddof)

    mean_between_var = float(per_task_between_var.mean())
    mean_within_var = float(per_route_within_var.mean())
    mean_between_std = float(per_task_between_std.mean())
    mean_within_std = float(per_route_within_std.mean())

    summary = {
        **summary_inputs,
        "n_traces": int(rewards.shape[0]),
        "n_routes": int(rewards.shape[1]),
        "route_labels": route_labels,
        "ddof": int(args.ddof),
        "between_expert_variance_per_task": {
            "mean": mean_between_var,
            "median": float(np.median(per_task_between_var)),
            "std": float(per_task_between_var.std(ddof=args.ddof)),
            "min": float(per_task_between_var.min()),
            "max": float(per_task_between_var.max()),
        },
        "between_expert_std_per_task": {
            "mean": mean_between_std,
            "median": float(np.median(per_task_between_std)),
            "std": float(per_task_between_std.std(ddof=args.ddof)),
            "min": float(per_task_between_std.min()),
            "max": float(per_task_between_std.max()),
        },
        "within_expert_variance_across_tasks": {
            "mean": mean_within_var,
            "median": float(np.median(per_route_within_var)),
            "min": float(per_route_within_var.min()),
            "max": float(per_route_within_var.max()),
            "by_route": {
                route_labels[idx]: {
                    "variance": float(per_route_within_var[idx]),
                    "std": float(per_route_within_std[idx]),
                    "mean_reward": float(rewards[:, idx].mean()),
                }
                for idx in range(len(route_labels))
            },
        },
        "snr": {
            "variance_ratio_between_over_within_mean": (
                mean_between_var / mean_within_var if mean_within_var > 0 else None
            ),
            "std_ratio_between_over_within_mean": (
                mean_between_std / mean_within_std if mean_within_std > 0 else None
            ),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "snr_summary.json", summary)
    _write_task_csv(output_dir / "per_task_between_expert_variance.csv", traces, rewards, route_labels)
    _write_route_csv(output_dir / "per_route_within_expert_variance.csv", rewards, route_labels)
    if args.group_by_language:
        language_rows = _write_language_csv(
            output_dir / "language_snr_summary.csv",
            traces,
            rewards,
            route_labels,
            args.ddof,
        )
        summary["language_breakdown_path"] = str(output_dir / "language_snr_summary.csv")
        summary["languages"] = [row["language"] for row in language_rows]
        _write_json(output_dir / "snr_summary.json", summary)
    _maybe_plot(output_dir, rewards, route_labels)

    logger.info("Loaded %d traces with %d routes", rewards.shape[0], rewards.shape[1])
    logger.info(
        "Between-expert variance per task: mean=%.6f median=%.6f",
        summary["between_expert_variance_per_task"]["mean"],
        summary["between_expert_variance_per_task"]["median"],
    )
    logger.info(
        "Within-expert variance across tasks: mean=%.6f median=%.6f",
        summary["within_expert_variance_across_tasks"]["mean"],
        summary["within_expert_variance_across_tasks"]["median"],
    )
    logger.info(
        "SNR variance ratio (between/within)=%.6f",
        summary["snr"]["variance_ratio_between_over_within_mean"]
        if summary["snr"]["variance_ratio_between_over_within_mean"] is not None
        else float("nan"),
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import argparse
import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt  # type: ignore

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

from pipelinerl.swe.scripts.new.router_trace_utils import (
    extract_reward_vector,
    extract_route_labels,
    load_router_traces,
)

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


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _infer_language(trace: dict[str, Any]) -> str:
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


def _extract_success_vector(trace: dict[str, Any], expected_dim: int) -> list[int]:
    values: list[int] = []
    policy = trace.get("policy") or {}
    policy_success = policy.get("success")
    if policy_success is None:
        policy_reward = _safe_float(policy.get("reward"), 0.0)
        values.append(1 if policy_reward > 0 else 0)
    else:
        values.append(1 if bool(policy_success) else 0)
    experts = trace.get("experts") or []
    for expert in experts[: max(0, expected_dim - 1)]:
        expert = expert or {}
        success = expert.get("success")
        if success is None:
            reward = _safe_float(expert.get("reward"), 0.0)
            values.append(1 if reward > 0 else 0)
        else:
            values.append(1 if bool(success) else 0)
    if len(values) < expected_dim:
        values.extend([0] * (expected_dim - len(values)))
    return values[:expected_dim]


def _update_running_stats(stats: dict[str, float], value: float) -> None:
    n = stats["n"] + 1.0
    delta = value - stats["mean"]
    mean = stats["mean"] + delta / n
    m2 = stats["m2"] + delta * (value - mean)
    stats["n"] = n
    stats["mean"] = mean
    stats["m2"] = m2


def _plot_heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    if not matrix or not row_labels or not col_labels:
        return
    plt.figure(figsize=(max(8, 1.2 * len(col_labels)), max(6, 0.45 * len(row_labels))))
    image = plt.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(image)
    plt.xticks(range(len(col_labels)), col_labels, rotation=30, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_winner_shares(
    winner_rows: list[dict[str, Any]],
    route_labels: list[str],
    out_path: Path,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    if not winner_rows:
        return
    languages = [row["language"] for row in winner_rows]
    plt.figure(figsize=(max(10, 0.45 * len(languages)), 5.5))
    bottom = [0.0 for _ in languages]
    for route_idx, route_label in enumerate(route_labels):
        values = [row.get(f"winner_share_route_{route_idx}", 0.0) for row in winner_rows]
        plt.bar(languages, values, bottom=bottom, label=route_label)
        bottom = [bottom[i] + values[i] for i in range(len(values))]
    plt.ylabel("Winner share")
    plt.title("Winning-route share by language")
    plt.xticks(rotation=30, ha="right")
    if len(route_labels) <= 10:
        plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-language expert/policy performance analysis from router traces.")
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--all-model-versions", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--top-languages", type=int, default=20, help="Keep top-N languages by trace count for plots/tables.")
    parser.add_argument(
        "--experts-only",
        action="store_true",
        help="Exclude policy route and report/plot experts only.",
    )
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

    n_experts = max((len(trace.get("experts") or []) for trace in traces), default=0)
    route_labels = extract_route_labels(traces, n_experts)
    dim = len(route_labels)
    logger.info("Loaded %d traces, routes=%d", len(traces), dim)

    route_indices = list(range(dim))
    if args.experts_only and dim > 1:
        route_indices = list(range(1, dim))
    selected_route_labels = [route_labels[idx] for idx in route_indices]
    if not selected_route_labels:
        raise ValueError("No routes selected for analysis.")
    selected_pairs = [(route_indices[i], route_indices[j]) for i in range(len(route_indices)) for j in range(i + 1, len(route_indices))]

    lang_count = Counter()
    reward_sum = defaultdict(lambda: [0.0 for _ in range(dim)])
    success_sum = defaultdict(lambda: [0 for _ in range(dim)])
    pair_delta_stats = defaultdict(
        lambda: {
            pair: {"n": 0.0, "mean": 0.0, "m2": 0.0}
            for pair in selected_pairs
        }
    )

    for trace in traces:
        rewards = extract_reward_vector(trace)
        if len(rewards) < dim:
            continue
        rewards = rewards[:dim]
        successes = _extract_success_vector(trace, expected_dim=dim)
        language = _infer_language(trace)
        lang_count[language] += 1
        for idx in range(dim):
            reward_sum[language][idx] += _safe_float(rewards[idx], 0.0)
            success_sum[language][idx] += successes[idx]
        for left_idx, right_idx in selected_pairs:
            delta = _safe_float(rewards[left_idx], 0.0) - _safe_float(rewards[right_idx], 0.0)
            _update_running_stats(pair_delta_stats[language][(left_idx, right_idx)], delta)

    top_languages = [name for name, _ in lang_count.most_common(max(1, args.top_languages))]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        {"language": language, "n_traces": lang_count[language], "share": lang_count[language] / len(traces)}
        for language in top_languages
    ]
    _write_csv(output_dir / "language_counts.csv", summary_rows, ["language", "n_traces", "share"])

    metric_rows: list[dict[str, Any]] = []
    for language in top_languages:
        n = max(1, lang_count[language])
        for route_idx, route_label in enumerate(route_labels):
            if route_idx not in route_indices:
                continue
            metric_rows.append(
                {
                    "language": language,
                    "n_traces": lang_count[language],
                    "route_idx": route_idx,
                    "route_label": route_label,
                    "avg_reward": reward_sum[language][route_idx] / n,
                    "success_rate": success_sum[language][route_idx] / n,
                }
            )
    _write_csv(
        output_dir / "language_route_metrics.csv",
        metric_rows,
        ["language", "n_traces", "route_idx", "route_label", "avg_reward", "success_rate"],
    )

    pair_rows: list[dict[str, Any]] = []
    for language in top_languages:
        for left_idx, right_idx in selected_pairs:
            stats = pair_delta_stats[language][(left_idx, right_idx)]
            n_tasks = int(stats["n"])
            if n_tasks <= 0:
                continue
            if n_tasks > 1:
                variance = stats["m2"] / (n_tasks - 1)
                std_delta = variance**0.5 if variance > 0.0 else 0.0
            else:
                std_delta = None
            mean_delta = stats["mean"]
            effect_size = None
            if std_delta is not None and std_delta > 1e-12:
                effect_size = mean_delta / std_delta
            pair_rows.append(
                {
                    "language": language,
                    "left_idx": left_idx,
                    "right_idx": right_idx,
                    "left_label": route_labels[left_idx],
                    "right_label": route_labels[right_idx],
                    "pair_label": f"{route_labels[left_idx]} vs {route_labels[right_idx]}",
                    "mean_delta": mean_delta,
                    "std_delta": std_delta,
                    "effect_size_mean_over_std": effect_size,
                    "n_tasks": n_tasks,
                }
            )
    _write_csv(
        output_dir / "language_pairwise_delta_metrics.csv",
        pair_rows,
        [
            "language",
            "left_idx",
            "right_idx",
            "left_label",
            "right_label",
            "pair_label",
            "mean_delta",
            "std_delta",
            "effect_size_mean_over_std",
            "n_tasks",
        ],
    )

    reward_matrix = []
    success_matrix = []
    for language in top_languages:
        n = max(1, lang_count[language])
        reward_matrix.append([reward_sum[language][idx] / n for idx in route_indices])
        success_matrix.append([success_sum[language][idx] / n for idx in route_indices])

    _plot_heatmap(
        reward_matrix,
        row_labels=top_languages,
        col_labels=selected_route_labels,
        title="Average reward by language and route",
        out_path=output_dir / "avg_reward_heatmap.png",
    )
    _plot_heatmap(
        success_matrix,
        row_labels=top_languages,
        col_labels=selected_route_labels,
        title="Success rate by language and route",
        out_path=output_dir / "success_rate_heatmap.png",
    )

    if MATPLOTLIB_AVAILABLE:
        for idx in route_indices:
            route_label = route_labels[idx]
            langs = top_languages
            values = [reward_sum[language][idx] / max(1, lang_count[language]) for language in langs]
            plt.figure(figsize=(max(10, 0.45 * len(langs)), 4.6))
            plt.bar(langs, values, color="tab:blue")
            plt.title(f"Avg reward by language | {route_label}")
            plt.ylabel("Average reward")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            slug = route_label.replace("/", "_").replace(":", "_")
            plt.savefig(output_dir / f"avg_reward_by_language__{slug}.png")
            plt.close()

            sr_values = [success_sum[language][idx] / max(1, lang_count[language]) for language in langs]
            plt.figure(figsize=(max(10, 0.45 * len(langs)), 4.6))
            plt.bar(langs, sr_values, color="tab:green")
            plt.title(f"Success rate by language | {route_label}")
            plt.ylabel("Success rate")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / f"success_rate_by_language__{slug}.png")
            plt.close()

    logger.info("Wrote per-language analysis to %s", output_dir)


if __name__ == "__main__":
    main()

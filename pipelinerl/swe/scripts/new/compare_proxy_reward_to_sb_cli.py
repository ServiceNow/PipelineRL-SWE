#!/usr/bin/env python
import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt  # type: ignore

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

from pipelinerl.swe.load_datasets import load_local_swe_dataset
from pipelinerl.swe.scripts.new.router_trace_utils import load_router_traces

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


def _repo_commit_key(repo: Any, base_commit: Any) -> str | None:
    repo_s = str(repo or "").strip()
    base_commit_s = str(base_commit or "").strip()
    if not repo_s or not base_commit_s or repo_s.lower() == "none" or base_commit_s.lower() == "none":
        return None
    return f"{repo_s}::{base_commit_s}"


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


def _index_dataset_rows(dataset_paths: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    rows_by_repo_commit: dict[str, dict[str, Any]] = {}
    for dataset_path in dataset_paths:
        rows = load_local_swe_dataset(
            dataset_names=[],
            dataset_path=dataset_path,
            shuffle=False,
            seed=42,
        )
        for row in rows:
            for key in _dataset_row_keys(row):
                rows_by_id.setdefault(key, row)
            repo_commit = _repo_commit_key(row.get("repo"), row.get("base_commit"))
            if repo_commit:
                rows_by_repo_commit.setdefault(repo_commit, row)
    return rows_by_id, rows_by_repo_commit


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
                return expert, model_name
        return None, f"expert_{args.expert_model_substring}"

    raise ValueError("Expert route requested but neither --expert-rank nor --expert-model-substring was provided.")


def _resolve_instance_id(
    trace: dict[str, Any],
    rows_by_id: dict[str, dict[str, Any]],
    rows_by_repo_commit: dict[str, dict[str, Any]],
) -> tuple[str | None, str]:
    for key in _trace_problem_keys(trace):
        if key in rows_by_id:
            return key, "id"
    repo_commit = _repo_commit_key(trace.get("repo"), trace.get("base_commit"))
    if repo_commit and repo_commit in rows_by_repo_commit:
        row = rows_by_repo_commit[repo_commit]
        return str(row.get("id") or "").strip() or None, "repo_commit"
    return None, "missing"


def _load_report(report_json: str) -> tuple[set[str], set[str]]:
    with Path(report_json).open() as handle:
        report = json.load(handle)
    resolved_ids = {str(item) for item in report.get("resolved_ids", [])}
    unresolved_ids = {str(item) for item in report.get("unresolved_ids", [])}
    return resolved_ids, unresolved_ids


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / (var_x * var_y) ** 0.5


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _roc_auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [score for score, label in zip(scores, labels) if label == 1]
    neg = [score for score, label in zip(scores, labels) if label == 0]
    if not pos or not neg:
        return None
    total = 0.0
    denom = len(pos) * len(neg)
    for p in pos:
        for n in neg:
            if p > n:
                total += 1.0
            elif p == n:
                total += 0.5
    return total / denom


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "instance_id",
        "repo",
        "base_commit",
        "route_label",
        "proxy_reward",
        "trace_success",
        "resolved",
        "match_source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _plot(rows: list[dict[str, Any]], route_label: str, out_path: Path) -> None:
    if not MATPLOTLIB_AVAILABLE or not rows:
        return
    xs_resolved = [row["proxy_reward"] for row in rows if row["resolved"] == 1]
    ys_resolved = [1.0 + ((idx % 7) - 3) * 0.015 for idx, row in enumerate(rows) if row["resolved"] == 1]
    xs_unresolved = [row["proxy_reward"] for row in rows if row["resolved"] == 0]
    ys_unresolved = [0.0 + ((idx % 7) - 3) * 0.015 for idx, row in enumerate(rows) if row["resolved"] == 0]

    plt.figure(figsize=(10.5, 5.8))
    if xs_unresolved:
        plt.scatter(xs_unresolved, ys_unresolved, alpha=0.55, s=18, label="Unresolved", color="tab:red")
    if xs_resolved:
        plt.scatter(xs_resolved, ys_resolved, alpha=0.65, s=18, label="Resolved", color="tab:green")
    plt.yticks([0, 1], ["unresolved", "resolved"])
    plt.xlabel("Proxy reward")
    plt.ylabel("Real eval outcome")
    plt.title(f"Proxy reward vs sb-cli outcome | {route_label}")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare per-instance proxy rewards from traces against sb-cli real eval outcomes.")
    parser.add_argument("--input-glob", action="append", required=True, help="Router trace JSONL glob(s).")
    parser.add_argument("--report-json", required=True, help="sb-cli report JSON path.")
    parser.add_argument("--dataset-path", action="append", default=[], help="Local dataset path(s) for old trace fallback matching.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--all-model-versions", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--route", choices=["policy", "expert"], default="expert")
    parser.add_argument("--expert-rank", type=int, default=None)
    parser.add_argument("--expert-model-substring", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    resolved_ids, unresolved_ids = _load_report(args.report_json)
    report_ids = resolved_ids | unresolved_ids
    logger.info(
        "Loaded sb-cli report: resolved=%d unresolved=%d total_completed=%d",
        len(resolved_ids),
        len(unresolved_ids),
        len(report_ids),
    )

    split = None if args.split == "all" else args.split
    traces = load_router_traces(
        input_globs=args.input_glob,
        split=split,
        latest_model_only=not args.all_model_versions,
        dedupe_by_problem=not args.keep_duplicates,
    )
    if not traces:
        raise ValueError("No traces found after filtering.")

    rows_by_id: dict[str, dict[str, Any]] = {}
    rows_by_repo_commit: dict[str, dict[str, Any]] = {}
    if args.dataset_path:
        rows_by_id, rows_by_repo_commit = _index_dataset_rows(args.dataset_path)

    joined_rows: list[dict[str, Any]] = []
    route_label = "policy" if args.route == "policy" else f"expert_{args.expert_rank}"
    missing_route = 0
    missing_instance_id = 0
    not_in_report = 0

    for trace in traces:
        instance_id, match_source = _resolve_instance_id(trace, rows_by_id, rows_by_repo_commit)
        if not instance_id:
            missing_instance_id += 1
            continue
        if instance_id not in report_ids:
            not_in_report += 1
            continue

        route_obj, route_label = _select_route(trace, args)
        if route_obj is None:
            missing_route += 1
            continue

        resolved = 1 if instance_id in resolved_ids else 0
        joined_rows.append(
            {
                "instance_id": instance_id,
                "repo": trace.get("repo"),
                "base_commit": trace.get("base_commit"),
                "route_label": route_label,
                "proxy_reward": _safe_float(route_obj.get("reward"), 0.0),
                "trace_success": bool(route_obj.get("success")),
                "resolved": resolved,
                "match_source": match_source,
            }
        )

    if not joined_rows:
        raise ValueError("No trace/report rows matched. Check --dataset-path for older traces or route selection.")

    proxy_rewards = [float(row["proxy_reward"]) for row in joined_rows]
    labels = [int(row["resolved"]) for row in joined_rows]
    resolved_rewards = [float(row["proxy_reward"]) for row in joined_rows if row["resolved"] == 1]
    unresolved_rewards = [float(row["proxy_reward"]) for row in joined_rows if row["resolved"] == 0]

    summary = {
        "report_json": args.report_json,
        "n_joined": len(joined_rows),
        "n_resolved": sum(labels),
        "n_unresolved": len(labels) - sum(labels),
        "missing_route": missing_route,
        "missing_instance_id": missing_instance_id,
        "not_in_report": not_in_report,
        "mean_proxy_reward_all": sum(proxy_rewards) / len(proxy_rewards),
        "mean_proxy_reward_resolved": (sum(resolved_rewards) / len(resolved_rewards)) if resolved_rewards else None,
        "mean_proxy_reward_unresolved": (sum(unresolved_rewards) / len(unresolved_rewards)) if unresolved_rewards else None,
        "pearson_proxy_vs_resolved": _pearson(proxy_rewards, [float(label) for label in labels]),
        "spearman_proxy_vs_resolved": _spearman(proxy_rewards, [float(label) for label in labels]),
        "roc_auc_proxy_vs_resolved": _roc_auc(proxy_rewards, labels),
        "route_label": route_label,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_instance_proxy_vs_real.csv", joined_rows)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    _plot(joined_rows, route_label, output_dir / "proxy_reward_vs_sb_cli_scatter.png")

    logger.info("Wrote %s", output_dir / "per_instance_proxy_vs_real.csv")
    logger.info("Wrote %s", output_dir / "summary.json")
    if MATPLOTLIB_AVAILABLE:
        logger.info("Wrote %s", output_dir / "proxy_reward_vs_sb_cli_scatter.png")
    else:
        logger.info("matplotlib not available; skipped scatter plot")
    logger.info(
        "Joined=%d resolved=%d unresolved=%d pearson=%s spearman=%s roc_auc=%s",
        summary["n_joined"],
        summary["n_resolved"],
        summary["n_unresolved"],
        summary["pearson_proxy_vs_resolved"],
        summary["spearman_proxy_vs_resolved"],
        summary["roc_auc_proxy_vs_resolved"],
    )


if __name__ == "__main__":
    main()

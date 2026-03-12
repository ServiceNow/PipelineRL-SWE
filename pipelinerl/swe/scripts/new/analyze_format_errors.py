#!/usr/bin/env python
import argparse
import csv
import logging
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _route_obj(trace: dict[str, Any], route_idx: int) -> dict[str, Any] | None:
    if route_idx == 0:
        policy = trace.get("policy")
        return policy if isinstance(policy, dict) else {}
    experts = trace.get("experts") or []
    if not isinstance(experts, list):
        return None
    expert_idx = route_idx - 1
    if expert_idx < 0 or expert_idx >= len(experts):
        return None
    expert = experts[expert_idx]
    return expert if isinstance(expert, dict) else {}


def _trace_problem_keys(trace: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key_name in ("problem_id", "id", "instance_id"):
        value = trace.get(key_name)
        if value is None:
            continue
        value_s = str(value)
        if value_s and value_s not in keys:
            keys.append(value_s)
    return keys


def _format_error_from_route(route: dict[str, Any]) -> tuple[bool, bool]:
    if "format_error" in route:
        return True, _as_bool(route.get("format_error"))
    failure_type = route.get("failure_type")
    if failure_type is not None:
        return True, str(failure_type).strip().lower() == "format"
    return False, False


def _load_dataset_rows(dataset_paths: list[str]) -> dict[str, dict[str, Any]]:
    # Lazy imports so the script still works in environments without HF datasets
    # when --recalculate-with-repair-util is not used.
    from pipelinerl.swe.load_datasets import load_local_swe_dataset

    by_key: dict[str, dict[str, Any]] = {}
    for dataset_path in dataset_paths:
        rows = load_local_swe_dataset(
            dataset_names=[],
            dataset_path=dataset_path,
            shuffle=False,
            seed=42,
        )
        for row in rows:
            for key_name in ("id", "problem_id", "instance_id"):
                value = row.get(key_name)
                if value is None:
                    continue
                value_s = str(value)
                if value_s:
                    by_key[value_s] = row
    return by_key


def _recalculate_format_error(route: dict[str, Any], row: dict[str, Any]) -> tuple[bool, bool]:
    from pipelinerl.swe.agents.repair_agent import RepairNode
    from pipelinerl.swe.utils.repair_utils import calculate_precise_reward

    file_contents = row.get("file_contents")
    patch = row.get("patch")
    if not isinstance(file_contents, dict) or not isinstance(patch, str):
        return False, False

    raw_output = route.get("repair_output")
    if not isinstance(raw_output, str):
        raw_output = route.get("output_text")
    if not isinstance(raw_output, str):
        return False, False

    edits = RepairNode._extract_search_replace_edits(None, raw_output)
    _, metadata = calculate_precise_reward(file_contents, patch, edits)
    return True, bool((metadata or {}).get("format_error", False))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "route_idx",
        "route_label",
        "n_present",
        "n_reward_present",
        "avg_reward",
        "n_success",
        "success_rate",
        "n_format_errors",
        "format_error_rate_all",
        "n_format_field_present",
        "format_error_rate_observed_only",
        "n_recalc_present",
        "n_recalc_format_errors",
        "recalc_format_error_rate_all",
        "recalc_format_error_rate_observed_only",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not MATPLOTLIB_AVAILABLE or not rows:
        return
    labels = [str(row["route_label"]) for row in rows]
    all_rates = [float(row["format_error_rate_all"]) for row in rows]
    observed_rates = [float(row["format_error_rate_observed_only"]) for row in rows]

    fig_w = max(10, 0.65 * len(labels))
    plt.figure(figsize=(fig_w, 5))
    x = list(range(len(labels)))
    width = 0.42
    plt.bar([i - width / 2 for i in x], all_rates, width=width, label="rate over all route calls")
    plt.bar([i + width / 2 for i in x], observed_rates, width=width, label="rate where format field exists")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Format error rate")
    plt.title("Format error rate by route")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "format_error_rates.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize format-error rates per route from router traces.")
    parser.add_argument("--input-glob", action="append", required=True, help="JSONL glob(s), repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--all-model-versions", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--experts-only", action="store_true")
    parser.add_argument(
        "--recalculate-with-repair-util",
        action="store_true",
        help="Recompute format_error from repair_output using calculate_precise_reward + dataset file contents.",
    )
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        help="Dataset path(s) for recalculation (e.g. /mnt/llmd/data/swe_smith/ds_test). Repeatable.",
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

    dataset_rows_by_key: dict[str, dict[str, Any]] = {}
    if args.recalculate_with_repair_util:
        if not args.dataset_path:
            raise ValueError("--dataset-path is required when --recalculate-with-repair-util is enabled.")
        dataset_rows_by_key = _load_dataset_rows(args.dataset_path)
        logger.info(
            "Loaded %d dataset rows for format-error recalculation from %d path(s)",
            len(dataset_rows_by_key),
            len(args.dataset_path),
        )

    n_experts = max((len(trace.get("experts") or []) for trace in traces), default=0)
    route_labels = extract_route_labels(traces, n_experts)
    route_indices = list(range(len(route_labels)))
    if args.experts_only and len(route_indices) > 1:
        route_indices = route_indices[1:]

    rows: list[dict[str, Any]] = []
    for route_idx in route_indices:
        n_present = 0
        n_reward_present = 0
        reward_sum = 0.0
        n_success = 0
        n_format = 0
        n_format_present = 0
        n_recalc = 0
        n_recalc_format = 0
        for trace in traces:
            route = _route_obj(trace, route_idx)
            if route is None:
                continue
            n_present += 1
            if _as_bool(route.get("success")):
                n_success += 1
            rewards = extract_reward_vector(trace)
            if route_idx < len(rewards):
                reward_sum += float(rewards[route_idx])
                n_reward_present += 1
            has_fmt, is_fmt = _format_error_from_route(route)
            if has_fmt:
                n_format_present += 1
                if is_fmt:
                    n_format += 1
            if args.recalculate_with_repair_util:
                dataset_row = None
                for key in _trace_problem_keys(trace):
                    dataset_row = dataset_rows_by_key.get(key)
                    if dataset_row is not None:
                        break
                if dataset_row is not None:
                    recalc_present, recalc_is_fmt = _recalculate_format_error(route, dataset_row)
                    if recalc_present:
                        n_recalc += 1
                        if recalc_is_fmt:
                            n_recalc_format += 1

        if n_present == 0:
            continue
        rows.append(
            {
                "route_idx": route_idx,
                "route_label": route_labels[route_idx],
                "n_present": n_present,
                "n_reward_present": n_reward_present,
                "avg_reward": (reward_sum / n_reward_present) if n_reward_present else 0.0,
                "n_success": n_success,
                "success_rate": n_success / n_present,
                "n_format_errors": n_format,
                "format_error_rate_all": n_format / n_present,
                "n_format_field_present": n_format_present,
                "format_error_rate_observed_only": (n_format / n_format_present) if n_format_present else 0.0,
                "n_recalc_present": n_recalc,
                "n_recalc_format_errors": n_recalc_format,
                "recalc_format_error_rate_all": n_recalc_format / n_present,
                "recalc_format_error_rate_observed_only": (n_recalc_format / n_recalc) if n_recalc else 0.0,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "format_error_summary.csv", rows)
    _plot(rows, output_dir)

    logger.info("Loaded %d traces", len(traces))
    logger.info("Wrote summary CSV: %s", output_dir / "format_error_summary.csv")
    if MATPLOTLIB_AVAILABLE:
        logger.info("Wrote plot: %s", output_dir / "format_error_rates.png")
    else:
        logger.info("matplotlib not available; skipping plot generation")

    for row in rows:
        logger.info(
            "[route=%s] present=%d avg_reward=%.4f success=%.3f format_err_all=%.3f format_err_observed=%.3f recalc_fmt_all=%.3f recalc_fmt_observed=%.3f format_field_coverage=%.3f",
            row["route_label"],
            row["n_present"],
            row["avg_reward"],
            row["success_rate"],
            row["format_error_rate_all"],
            row["format_error_rate_observed_only"],
            row["recalc_format_error_rate_all"],
            row["recalc_format_error_rate_observed_only"],
            (row["n_format_field_present"] / row["n_present"]) if row["n_present"] else 0.0,
        )


if __name__ == "__main__":
    main()

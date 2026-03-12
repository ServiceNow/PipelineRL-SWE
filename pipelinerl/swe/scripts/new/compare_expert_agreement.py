#!/usr/bin/env python
import argparse
import csv
import logging
from pathlib import Path
from typing import Any

from pipelinerl.swe.scripts.new.router_trace_utils import (
    extract_reward_vector,
    extract_route_labels,
    load_router_traces,
)

logger = logging.getLogger(__name__)


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare win/fail agreement rates across all route pairs.")
    parser.add_argument("--input-glob", action="append", required=True, help="Glob pattern for router trace JSONL files.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path for pairwise agreement stats.")
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--all-model-versions", action="store_true", help="Do not filter to latest model version.")
    parser.add_argument("--keep-duplicates", action="store_true", help="Do not dedupe by problem id.")
    parser.add_argument(
        "--win-threshold",
        type=float,
        default=0.0,
        help="Reward threshold for defining a win (reward > threshold).",
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
        raise ValueError("No traces found for provided filters.")

    n_experts = max((len(trace.get("experts") or []) for trace in traces), default=0)
    route_labels = extract_route_labels(traces, n_experts)
    logger.info("Loaded %d traces and %d routes", len(traces), len(route_labels))

    rows: list[dict[str, Any]] = []
    threshold = float(args.win_threshold)
    for left in range(len(route_labels)):
        for right in range(left + 1, len(route_labels)):
            total = 0
            agree = 0
            left_wins_right_fails = 0
            right_wins_left_fails = 0
            both_win = 0
            both_fail = 0

            for trace in traces:
                rewards = extract_reward_vector(trace)
                if len(rewards) <= right:
                    continue
                r_left = float(rewards[left])
                r_right = float(rewards[right])
                left_win = r_left > threshold
                right_win = r_right > threshold
                total += 1
                if left_win == right_win:
                    agree += 1
                if left_win and not right_win:
                    left_wins_right_fails += 1
                if right_win and not left_win:
                    right_wins_left_fails += 1
                if left_win and right_win:
                    both_win += 1
                if (not left_win) and (not right_win):
                    both_fail += 1

            if total == 0:
                continue
            agree_rate = agree / total
            rows.append(
                {
                    "left_idx": left,
                    "right_idx": right,
                    "left_label": route_labels[left],
                    "right_label": route_labels[right],
                    "n": total,
                    "win_threshold": threshold,
                    "agreement_rate": agree_rate,
                    "disagreement_rate": 1.0 - agree_rate,
                    "left_wins_right_fails_rate": left_wins_right_fails / total,
                    "right_wins_left_fails_rate": right_wins_left_fails / total,
                    "both_win_rate": both_win / total,
                    "both_fail_rate": both_fail / total,
                }
            )

    _write_csv(
        Path(args.output_csv),
        rows,
        [
            "left_idx",
            "right_idx",
            "left_label",
            "right_label",
            "n",
            "win_threshold",
            "agreement_rate",
            "disagreement_rate",
            "left_wins_right_fails_rate",
            "right_wins_left_fails_rate",
            "both_win_rate",
            "both_fail_rate",
        ],
    )

    for row in rows:
        logger.info(
            "%s vs %s | disagree=%.3f left_win_right_fail=%.3f right_win_left_fail=%.3f both_win=%.3f both_fail=%.3f n=%d",
            row["left_label"],
            row["right_label"],
            row["disagreement_rate"],
            row["left_wins_right_fails_rate"],
            row["right_wins_left_fails_rate"],
            row["both_win_rate"],
            row["both_fail_rate"],
            row["n"],
        )
    logger.info("Wrote pairwise agreement stats to %s", args.output_csv)


if __name__ == "__main__":
    main()

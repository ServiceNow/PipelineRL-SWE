from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _resolve_route_index(rows: list[dict[str, Any]], label: str | None, default_index: int) -> int:
    routes = rows[0].get("routes") or []
    if not routes:
        raise ValueError("Prediction file is missing route labels")
    if label is None:
        if default_index >= len(routes):
            raise ValueError(f"Default route index {default_index} out of range for routes={routes}")
        return default_index
    try:
        return routes.index(label)
    except ValueError as exc:
        raise ValueError(f"Route label {label!r} not found in routes={routes}") from exc


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_ranks = float(np.sum(ranks[y_true == 1]))
    return (pos_ranks - positives * (positives + 1) / 2.0) / (positives * negatives)


def _maybe_plot(output_path: Path, sweep_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        logger.info("matplotlib unavailable; skipping threshold plot")
        return

    thresholds = [float(row["threshold"]) for row in sweep_rows]
    avg_rewards = [float(row["avg_reward"]) for row in sweep_rows]
    choose_secondary = [float(row["secondary_fraction"]) for row in sweep_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, avg_rewards, label="avg reward", color="#1f77b4")
    plt.plot(thresholds, choose_secondary, label="secondary fraction", color="#ff7f0e", linestyle="--")
    plt.xlabel("threshold")
    plt.ylabel("value")
    plt.title("Threshold routing from ridge predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate threshold routing from ridge probe predictions.")
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-route-label", type=str, default=None)
    parser.add_argument("--secondary-route-label", type=str, default=None)
    parser.add_argument("--threshold-min", type=float, default=0.0)
    parser.add_argument("--threshold-max", type=float, default=1.0)
    parser.add_argument("--num-thresholds", type=int, default=201)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    rows = _load_rows(args.predictions_jsonl)
    route_labels = rows[0]["routes"]
    primary_idx = _resolve_route_index(rows, args.primary_route_label, 0)
    secondary_idx = _resolve_route_index(rows, args.secondary_route_label, 1)

    pred_primary = np.asarray([float(row["pred_rewards"][primary_idx]) for row in rows], dtype=np.float64)
    true_primary = np.asarray([float(row["true_rewards"][primary_idx]) for row in rows], dtype=np.float64)
    true_secondary = np.asarray([float(row["true_rewards"][secondary_idx]) for row in rows], dtype=np.float64)

    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.num_thresholds)
    sweep_rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        choose_secondary = pred_primary < threshold
        realized = np.where(choose_secondary, true_secondary, true_primary)
        row = {
            "threshold": float(threshold),
            "n_eval": int(len(rows)),
            "avg_reward": float(np.mean(realized)),
            "reward_gain_vs_primary": float(np.mean(realized) - np.mean(true_primary)),
            "reward_gain_vs_secondary": float(np.mean(realized) - np.mean(true_secondary)),
            "primary_fraction": float(np.mean(~choose_secondary)),
            "secondary_fraction": float(np.mean(choose_secondary)),
            "primary_count": int(np.sum(~choose_secondary)),
            "secondary_count": int(np.sum(choose_secondary)),
        }
        sweep_rows.append(row)

    best_row = max(sweep_rows, key=lambda row: float(row["avg_reward"]))
    always_primary = float(np.mean(true_primary))
    always_secondary = float(np.mean(true_secondary))
    oracle = float(np.mean(np.maximum(true_primary, true_secondary)))
    better_secondary = (true_secondary > true_primary).astype(np.int64)
    auc = _safe_auc(better_secondary, -pred_primary)

    summary = {
        "n_eval": int(len(rows)),
        "routes": route_labels,
        "primary_route_label": route_labels[primary_idx],
        "secondary_route_label": route_labels[secondary_idx],
        "always_primary_avg_reward": always_primary,
        "always_secondary_avg_reward": always_secondary,
        "oracle_avg_reward": oracle,
        "best_threshold": float(best_row["threshold"]),
        "best_avg_reward": float(best_row["avg_reward"]),
        "best_reward_gain_vs_primary": float(best_row["reward_gain_vs_primary"]),
        "best_reward_gain_vs_secondary": float(best_row["reward_gain_vs_secondary"]),
        "best_primary_fraction": float(best_row["primary_fraction"]),
        "best_secondary_fraction": float(best_row["secondary_fraction"]),
        "auc_secondary_better_from_neg_pred_primary": auc,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "threshold_sweep.csv",
        sweep_rows,
        [
            "threshold",
            "n_eval",
            "avg_reward",
            "reward_gain_vs_primary",
            "reward_gain_vs_secondary",
            "primary_fraction",
            "secondary_fraction",
            "primary_count",
            "secondary_count",
        ],
    )
    with (output_dir / "summary.json").open("w") as sink:
        json.dump(summary, sink, indent=2, sort_keys=True)
    _maybe_plot(output_dir / "reward_vs_threshold.png", sweep_rows)

    logger.info(
        "Done. primary=%s secondary=%s best_threshold=%.4f best_avg_reward=%.4f gain_vs_primary=%.4f",
        summary["primary_route_label"],
        summary["secondary_route_label"],
        summary["best_threshold"],
        summary["best_avg_reward"],
        summary["best_reward_gain_vs_primary"],
    )


if __name__ == "__main__":
    main()

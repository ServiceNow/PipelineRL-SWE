#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_split_rows(split_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wanted_columns = [
        "problem_id",
        "route_rewards",
        "performance_targets",
        "route_successes",
        "route_prompt_tokens",
        "route_output_tokens",
        "route_failure_types",
    ]
    for shard_path in sorted(split_dir.glob("*.parquet")):
        schema = pq.read_schema(shard_path)
        columns = [column for column in wanted_columns if column in schema.names]
        if not columns:
            continue
        rows.extend(pq.read_table(shard_path, columns=columns).to_pylist())
    return rows


def _numeric_vector(row: dict[str, Any], primary_key: str, fallback_key: str | None = None) -> list[float] | None:
    values = row.get(primary_key)
    if not isinstance(values, list) and fallback_key is not None:
        values = row.get(fallback_key)
    if not isinstance(values, list):
        return None
    result: list[float] = []
    for value in values:
        parsed = _safe_float(value)
        if parsed is None:
            return None
        result.append(parsed)
    return result


def _bool_vector(row: dict[str, Any], key: str) -> list[bool] | None:
    values = row.get(key)
    if not isinstance(values, list):
        return None
    return [bool(value) for value in values]


def _split_summary(rows: list[dict[str, Any]], route_labels: list[str]) -> dict[str, Any]:
    route_count = len(route_labels)
    reward_values: list[list[float]] = [[] for _ in range(route_count)]
    success_values: list[list[float]] = [[] for _ in range(route_count)]
    prompt_token_values: list[list[float]] = [[] for _ in range(route_count)]
    output_token_values: list[list[float]] = [[] for _ in range(route_count)]
    deltas: list[float] = []
    primary_wins = 0
    secondary_wins = 0
    ties = 0
    invalid_reward_rows = 0

    for row in rows:
        rewards = _numeric_vector(row, "route_rewards", "performance_targets")
        successes = _bool_vector(row, "route_successes")
        prompt_tokens = _numeric_vector(row, "route_prompt_tokens")
        output_tokens = _numeric_vector(row, "route_output_tokens")

        if rewards is None or len(rewards) < route_count:
            invalid_reward_rows += 1
            continue

        for route_idx in range(route_count):
            reward_values[route_idx].append(float(rewards[route_idx]))
            if successes is not None and len(successes) > route_idx:
                success_values[route_idx].append(1.0 if successes[route_idx] else 0.0)
            if prompt_tokens is not None and len(prompt_tokens) > route_idx:
                prompt_token_values[route_idx].append(float(prompt_tokens[route_idx]))
            if output_tokens is not None and len(output_tokens) > route_idx:
                output_token_values[route_idx].append(float(output_tokens[route_idx]))

        if route_count >= 2:
            delta = float(rewards[0] - rewards[1])
            deltas.append(delta)
            if delta > 0.0:
                primary_wins += 1
            elif delta < 0.0:
                secondary_wins += 1
            else:
                ties += 1

    by_route = {}
    for route_idx, label in enumerate(route_labels):
        by_route[label] = {
            "reward_mean": _mean(reward_values[route_idx]),
            "reward_min": min(reward_values[route_idx]) if reward_values[route_idx] else None,
            "reward_max": max(reward_values[route_idx]) if reward_values[route_idx] else None,
            "success_rate": _mean(success_values[route_idx]),
            "prompt_tokens_mean": _mean(prompt_token_values[route_idx]),
            "output_tokens_mean": _mean(output_token_values[route_idx]),
        }

    pairwise = None
    if route_count >= 2:
        valid_pairs = len(deltas)
        pairwise = {
            "primary_label": route_labels[0],
            "secondary_label": route_labels[1],
            "valid_pairs": valid_pairs,
            "primary_wins": primary_wins,
            "secondary_wins": secondary_wins,
            "ties": ties,
            "primary_win_rate": float(primary_wins / valid_pairs) if valid_pairs else None,
            "secondary_win_rate": float(secondary_wins / valid_pairs) if valid_pairs else None,
            "tie_rate": float(ties / valid_pairs) if valid_pairs else None,
            "delta_mean": _mean(deltas),
            "delta_min": min(deltas) if deltas else None,
            "delta_max": max(deltas) if deltas else None,
        }

    return {
        "n_rows": len(rows),
        "n_invalid_reward_rows": invalid_reward_rows,
        "by_route": by_route,
        "primary_minus_secondary": pairwise,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize route reward distributions in an offline-router collection.")
    parser.add_argument("--dataset-dir", required=True, help="Offline-router collection directory.")
    parser.add_argument("--output-json", default=None, help="Optional path to write the JSON summary.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing offline-router metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    route_labels = list(metadata.get("route_labels") or [])
    if not route_labels:
        raise ValueError(f"{metadata_path} is missing route_labels")

    summary = {
        "schema_version": 1,
        "dataset_dir": str(dataset_dir),
        "route_labels": route_labels,
        "splits": {},
    }
    for split_name in ("train", "eval"):
        summary["splits"][split_name] = _split_summary(_load_split_rows(dataset_dir / split_name), route_labels)

    output_json = Path(args.output_json) if args.output_json else dataset_dir / "route_distribution_summary.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()

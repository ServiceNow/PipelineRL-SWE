#!/usr/bin/env python
"""Export aligned SWE-Bench prediction files for real-label collection.

The input is a router collection parquet directory containing multiple route
outputs per problem. This script samples problem ids once, then writes one
SWE-Bench harness predictions.jsonl file per requested route over the exact
same selected ids. It is intended for collecting real pass/fail labels on a
budget without losing route alignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DEFAULT_COLLECT_DIR = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_bench_train_all_16k_verified_eval_collect_5route_4b_scout_oss20_qwen30_oss120_gemini/collect"
)
DEFAULT_DATASET_PATH = "/mnt/llmd/data/swebench/all_16k/ds_train"
DEFAULT_DATASET_NAME = "swe_bench_train"
DEFAULT_ROUTES = [
    "scout:Qwen/Qwen3-4B-Instruct-2507",
    "solver:openai/gpt-oss-20b",
    "solver:Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "solver:openai/gpt-oss-120b",
]
ROUTE_RUN_IDS = {
    "scout:Qwen/Qwen3-4B-Instruct-2507": "qwen3_4b_instruct_2507",
    "solver:openai/gpt-oss-20b": "gpt_oss_20b",
    "solver:Qwen/Qwen3-Coder-30B-A3B-Instruct": "qwen3_coder_30b_a3b",
    "solver:openai/gpt-oss-120b": "gpt_oss_120b",
}
ROUTE_MODEL_NAMES = {
    "scout:Qwen/Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "solver:openai/gpt-oss-20b": "openai/gpt-oss-20b",
    "solver:Qwen/Qwen3-Coder-30B-A3B-Instruct": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "solver:openai/gpt-oss-120b": "openai/gpt-oss-120b",
}


def _parse_route_labels(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        return list(DEFAULT_ROUTES)
    labels: list[str] = []
    for raw in raw_values:
        for piece in re.split(r"\s*,\s*", raw.strip()):
            if piece:
                labels.append(piece)
    if not labels:
        raise ValueError("No route labels parsed from --route-label.")
    return labels


def _read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sanitize_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._-") or "route"


def _parquet_files(split_dir: Path) -> list[Path]:
    files = sorted(split_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards found under {split_dir}")
    return files


def _to_pylist(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _iter_collection_rows(collect_dir: Path, split: str) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq  # lazy import; available in the training env

    split_dir = collect_dir / split
    for parquet_path in _parquet_files(split_dir):
        table = pq.read_table(parquet_path)
        columns = table.column_names
        for row_idx in range(table.num_rows):
            yield {name: _to_pylist(table[name][row_idx]) for name in columns}


def _load_dataset_by_id(dataset_path: str, dataset_name: str | None, seed: int) -> dict[str, dict[str, Any]]:
    from pipelinerl.swe.load_datasets import load_local_swe_dataset
    from pipelinerl.swe.scripts.offline_router.common import problem_id_from_item

    rows = load_local_swe_dataset(
        dataset_names=[dataset_name] if dataset_name else [],
        dataset_path=dataset_path,
        shuffle=False,
        seed=seed,
        dataset_label=dataset_name,
        max_samples=None,
    )
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = problem_id_from_item(row)
        if row_id and row_id not in by_id:
            by_id[row_id] = row
    if not by_id:
        raise ValueError(f"No dataset rows loaded from {dataset_path}")
    return by_id


def _as_float_list(value: Any, route_indices: list[int]) -> list[float]:
    values = _to_pylist(value) or []
    return [float(values[idx] or 0.0) for idx in route_indices]


def _git_blob_hash(text: str) -> str:
    data = text.encode("utf-8", errors="surrogateescape")
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()[:7]


def _build_git_patch_real_hash(file_contents: dict[str, str], repair_output: str) -> tuple[str, int, str | None]:
    from pipelinerl.swe.scripts.repair_eval_utils import extract_search_replace_edits
    from pipelinerl.swe.utils.repair_utils import apply_edits_to_files, get_normalized_patch

    edits = extract_search_replace_edits(repair_output)
    if not edits:
        return "", 0, "no_edits"
    modified_contents = apply_edits_to_files(file_contents, edits, silent=False)
    patch_dict = get_normalized_patch(file_contents, modified_contents)
    if not patch_dict:
        return "", len(edits), "empty_patch"

    diff_parts: list[str] = []
    for file_path, patch in patch_dict.items():
        old_text = str(file_contents.get(file_path) or "")
        new_text = str(modified_contents.get(file_path) or "")
        old_hash = _git_blob_hash(old_text)
        new_hash = _git_blob_hash(new_text)
        diff_parts.append(f"diff --git a/{file_path} b/{file_path}")
        diff_parts.append(f"index {old_hash}..{new_hash} 100644")
        diff_parts.append(f"--- a/{file_path}")
        diff_parts.append(f"+++ b/{file_path}")
        diff_parts.append(patch)
    text = "\n".join(diff_parts)
    if text and not text.endswith("\n"):
        text += "\n"
    return text, len(edits), None


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    rewards_by_id: dict[str, list[float]],
    n_instances: int,
    random_fraction: float,
    disagreement_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    if n_instances <= 0:
        raise ValueError("--n-instances must be positive")
    if n_instances > len(rows):
        logger.warning("Requested %s instances but only %s rows exist; using all rows", n_instances, len(rows))
        n_instances = len(rows)

    n_random = round(n_instances * random_fraction)
    n_disagreement = round(n_instances * disagreement_fraction)
    n_oracle = n_instances - n_random - n_disagreement
    if n_oracle < 0:
        raise ValueError("random_fraction + disagreement_fraction must be <= 1.0")

    route_count = len(next(iter(rewards_by_id.values())))
    means = [0.0] * route_count
    for rewards in rewards_by_id.values():
        for idx, reward in enumerate(rewards):
            means[idx] += reward
    means = [value / max(len(rewards_by_id), 1) for value in means]
    best_global_route_idx = max(range(route_count), key=lambda idx: means[idx])

    row_by_id = {str(row["problem_id"]): row for row in rows}
    all_ids = [str(row["problem_id"]) for row in rows]
    rng = random.Random(seed)

    selected: list[str] = []
    source_by_id: dict[str, str] = {}

    random_ids = list(all_ids)
    rng.shuffle(random_ids)
    for problem_id in random_ids[:n_random]:
        selected.append(problem_id)
        source_by_id[problem_id] = "random"

    remaining = [problem_id for problem_id in all_ids if problem_id not in source_by_id]
    disagreement_ranked = sorted(
        remaining,
        key=lambda problem_id: (max(rewards_by_id[problem_id]) - min(rewards_by_id[problem_id]), problem_id),
        reverse=True,
    )
    for problem_id in disagreement_ranked[:n_disagreement]:
        selected.append(problem_id)
        source_by_id[problem_id] = "high_proxy_disagreement"

    remaining = [problem_id for problem_id in all_ids if problem_id not in source_by_id]
    oracle_ranked = sorted(
        remaining,
        key=lambda problem_id: (
            max(rewards_by_id[problem_id]) - rewards_by_id[problem_id][best_global_route_idx],
            problem_id,
        ),
        reverse=True,
    )
    for problem_id in oracle_ranked[:n_oracle]:
        selected.append(problem_id)
        source_by_id[problem_id] = "high_proxy_oracle_gain"

    if len(selected) < n_instances:
        remaining = [problem_id for problem_id in all_ids if problem_id not in source_by_id]
        rng.shuffle(remaining)
        for problem_id in remaining[: n_instances - len(selected)]:
            selected.append(problem_id)
            source_by_id[problem_id] = "fill_random"

    selected = selected[:n_instances]
    selected_rows = [row_by_id[problem_id] for problem_id in selected]
    diagnostics = {
        "n_instances": len(selected_rows),
        "requested_n_instances": n_instances,
        "requested_counts": {
            "random": n_random,
            "high_proxy_disagreement": n_disagreement,
            "high_proxy_oracle_gain": n_oracle,
        },
        "actual_counts": dict(Counter(source_by_id[problem_id] for problem_id in selected)),
        "route_proxy_means": means,
        "best_global_route_index": best_global_route_idx,
    }
    return selected_rows, source_by_id, diagnostics


def _write_runner(output_dir: Path, run_ids: list[str], dataset_name_for_harness: str, dataset_split: str) -> None:
    run_lines = "\n".join(f"run_eval {run_id}" for run_id in run_ids)
    text = f"""#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="${{DATASET_NAME:-{dataset_name_for_harness}}}"
DATASET_SPLIT="${{DATASET_SPLIT:-{dataset_split}}}"
MAX_WORKERS="${{MAX_WORKERS:-24}}"
RESULTS_DIR="${{RESULTS_DIR:-$HOME/swebench_train_real_label_subset_results}}"
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
mkdir -p "$RESULTS_DIR"

run_eval() {{
  local run_id="$1"
  local predictions_path="$SCRIPT_DIR/predictions/$run_id/predictions.jsonl"
  local report_path="$RESULTS_DIR/${{run_id}}_results.json"

  if [[ -s "$report_path" ]]; then
    echo "[skip] $run_id already has $report_path"
    return 0
  fi

  local split_args=()
  if [[ -n "$DATASET_SPLIT" ]]; then
    split_args=(--split "$DATASET_SPLIT")
  fi

  echo "[eval] $run_id"
  python -m swebench.harness.run_evaluation \\
    --dataset_name "$DATASET_NAME" \\
    "${{split_args[@]}}" \\
    --predictions_path "$predictions_path" \\
    --max_workers "$MAX_WORKERS" \\
    --run_id "$run_id" \\
    --report_path "$report_path"
}}

{run_lines}

echo "Reports written to $RESULTS_DIR"
"""
    runner = output_dir / "run_all_swebench_train_subset.sh"
    runner.write_text(text)
    runner.chmod(0o755)


def _write_readme(output_dir: Path, run_ids: list[str]) -> None:
    run_list = "\n".join(f"- `{run_id}`" for run_id in run_ids)
    text = f"""# SWE-Bench Train Real-Label Subset

This package contains aligned SWE-Bench harness prediction files for a shared
subset of train instances. Every route file contains the same instance ids in
the same order, so per-instance pass/fail labels can be joined directly across
routes.

Routes:
{run_list}

Typical AWS command:

```bash
cd /path/to/this/package
MAX_WORKERS=24 RESULTS_DIR=$HOME/swebench_train_real_label_subset_results \\
  bash run_all_swebench_train_subset.sh
```

The runner defaults to `DATASET_NAME=SWE-bench/SWE-bench` and
`DATASET_SPLIT=train`. Override those environment variables if your installed
SWE-bench harness expects a different dataset id or split behavior.
"""
    (output_dir / "README.md").write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-dir", default=DEFAULT_COLLECT_DIR)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-name-for-harness", default="SWE-bench/SWE-bench")
    parser.add_argument("--dataset-split-for-harness", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-instances", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-fraction", type=float, default=0.5)
    parser.add_argument("--disagreement-fraction", type=float, default=0.3)
    parser.add_argument(
        "--route-label",
        action="append",
        help="Route label to export. Can be repeated or comma-separated. Defaults to the 4-route no-Gemini mix.",
    )
    parser.add_argument("--make-tarball", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    collect_dir = Path(args.collect_dir)
    output_dir = Path(args.output_dir)
    route_labels = _parse_route_labels(args.route_label)

    summary_path = collect_dir / "route_distribution_summary.json"
    summary = _read_json(summary_path)
    all_route_labels = list(summary.get("route_labels") or [])
    if not all_route_labels:
        raise ValueError(f"Missing route_labels in {summary_path}")
    missing = [label for label in route_labels if label not in all_route_labels]
    if missing:
        raise ValueError(f"Requested route labels missing from collection: {missing}; available: {all_route_labels}")
    route_indices = [all_route_labels.index(label) for label in route_labels]

    logger.info("Loading %s/%s", collect_dir, args.split)
    rows: list[dict[str, Any]] = []
    rewards_by_id: dict[str, list[float]] = {}
    for row in _iter_collection_rows(collect_dir, args.split):
        problem_id = str(row.get("problem_id") or "")
        if not problem_id:
            continue
        rewards = _as_float_list(row.get("route_rewards") or row.get("performance_targets"), route_indices)
        rows.append(row)
        rewards_by_id[problem_id] = rewards
    if not rows:
        raise ValueError(f"No rows read from {collect_dir / args.split}")
    logger.info("Read %s rows", len(rows))

    selected_rows, source_by_id, selection_diag = _select_rows(
        rows,
        rewards_by_id=rewards_by_id,
        n_instances=args.n_instances,
        random_fraction=args.random_fraction,
        disagreement_fraction=args.disagreement_fraction,
        seed=args.seed,
    )
    selected_ids = [str(row["problem_id"]) for row in selected_rows]
    logger.info("Selected %s aligned instances: %s", len(selected_ids), selection_diag["actual_counts"])

    logger.info("Loading local SWE dataset from %s", args.dataset_path)
    dataset_by_id = _load_dataset_by_id(args.dataset_path, args.dataset_name, args.seed)
    missing_dataset = [problem_id for problem_id in selected_ids if problem_id not in dataset_by_id]
    if missing_dataset:
        raise ValueError(
            f"{len(missing_dataset)} selected ids missing from local dataset; first: {missing_dataset[:5]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    run_ids: list[str] = []
    route_summaries: list[dict[str, Any]] = []

    for label, route_idx in zip(route_labels, route_indices):
        run_id = ROUTE_RUN_IDS.get(label) or _sanitize_slug(label)
        model_name = ROUTE_MODEL_NAMES.get(label) or label.split(":", 1)[-1]
        run_ids.append(run_id)
        run_dir = predictions_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "predictions.jsonl"

        n_empty = 0
        n_no_edits = 0
        n_patch_errors = 0
        n_with_patch = 0
        total_proxy_reward = 0.0
        with out_path.open("w") as handle:
            for row in selected_rows:
                problem_id = str(row["problem_id"])
                problem = dataset_by_id[problem_id]
                route_outputs = _to_pylist(row.get("route_outputs")) or []
                repair_output = str(route_outputs[route_idx] or "") if route_idx < len(route_outputs) else ""
                try:
                    model_patch, _n_edits, patch_status = _build_git_patch_real_hash(
                        problem.get("file_contents") or {},
                        repair_output,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    model_patch = ""
                    patch_status = f"reconstruction_error:{type(exc).__name__}"
                    n_patch_errors += 1
                if not model_patch:
                    n_empty += 1
                else:
                    n_with_patch += 1
                if patch_status == "no_edits":
                    n_no_edits += 1
                total_proxy_reward += rewards_by_id[problem_id][route_indices.index(route_idx)]
                handle.write(
                    json.dumps(
                        {
                            "instance_id": problem_id,
                            "model_patch": model_patch,
                            "model_name_or_path": model_name,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

        route_summaries.append(
            {
                "route_label": label,
                "route_index": route_idx,
                "run_id": run_id,
                "model_name_or_path": model_name,
                "predictions_path": str(out_path.relative_to(output_dir)),
                "n_predictions": len(selected_rows),
                "n_with_patch": n_with_patch,
                "n_empty_patch": n_empty,
                "n_no_edits": n_no_edits,
                "n_patch_reconstruction_errors": n_patch_errors,
                "mean_proxy_reward_on_selected": total_proxy_reward / max(len(selected_rows), 1),
            }
        )
        logger.info("Wrote %s (%s empty patches)", out_path, n_empty)

    (output_dir / "instance_ids.txt").write_text("\n".join(selected_ids) + "\n")
    rows_manifest = [
        {
            "instance_id": problem_id,
            "selection_bucket": source_by_id[problem_id],
            "route_proxy_rewards": {
                label: rewards_by_id[problem_id][idx]
                for idx, label in enumerate(route_labels)
            },
        }
        for problem_id in selected_ids
    ]
    _write_json(output_dir / "selected_instances_manifest.json", rows_manifest)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_collect_dir": str(collect_dir),
        "source_split": args.split,
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "dataset_name_for_harness": args.dataset_name_for_harness,
        "dataset_split_for_harness": args.dataset_split_for_harness,
        "seed": args.seed,
        "route_labels": route_labels,
        "route_indices": route_indices,
        "selection": selection_diag,
        "routes": route_summaries,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_runner(output_dir, run_ids, args.dataset_name_for_harness, args.dataset_split_for_harness)
    _write_readme(output_dir, run_ids)

    if args.make_tarball:
        tar_path = output_dir.with_suffix(".tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(output_dir, arcname=output_dir.name)
        logger.info("Wrote tarball: %s", tar_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

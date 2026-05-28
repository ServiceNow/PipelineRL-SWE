#!/usr/bin/env python
"""Recompute proxy rewards and SWE-Bench prediction packages from saved discovery outputs.

This is useful when the SEARCH/REPLACE parser changes. By default it writes
sidecar files and leaves the original collection artifacts untouched.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from pipelinerl.swe.load_datasets import load_local_swe_dataset
from pipelinerl.swe.scripts.offline_router.common import problem_id_from_item, sanitize_for_json
from pipelinerl.swe.scripts.offline_router.collect_model_discovery_candidates import (
    _build_git_patch,
    _derive_failure_type,
    _write_json,
)
from pipelinerl.swe.scripts.repair_eval_utils import extract_search_replace_edits
from pipelinerl.swe.utils.repair_utils import calculate_precise_reward

logger = logging.getLogger(__name__)


def _load_rows(dataset_path: str, dataset_name: str | None, seed: int) -> dict[str, dict[str, Any]]:
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
        if row_id:
            by_id[row_id] = row
    if not by_id:
        raise ValueError(f"No rows loaded from {dataset_path}")
    return by_id


def _iter_model_dirs(collect_dir: Path, model_slug: str | None) -> list[Path]:
    models_dir = collect_dir / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Missing models dir: {models_dir}")
    if model_slug:
        dirs = [models_dir / model_slug]
    else:
        dirs = sorted(path for path in models_dir.iterdir() if path.is_dir())
    missing = [str(path) for path in dirs if not (path / "outputs.jsonl").exists()]
    if missing:
        raise FileNotFoundError(f"Missing outputs.jsonl for: {', '.join(missing)}")
    return dirs


def _write_predictions(model_dir: Path, output_prefix: str, rows: list[dict[str, Any]]) -> None:
    predictions_list = [
        {
            "instance_id": row["instance_id"],
            "model_patch": row.get("model_patch") or "",
            "model_name_or_path": row.get("model_name") or "",
        }
        for row in rows
    ]
    predictions_by_instance = {
        row["instance_id"]: {
            "model_patch": row.get("model_patch") or "",
            "model_name_or_path": row.get("model_name") or "",
        }
        for row in rows
    }
    _write_json(model_dir / f"{output_prefix}_predictions.json", predictions_by_instance)
    _write_json(model_dir / f"{output_prefix}_predictions_by_instance.json", predictions_by_instance)
    _write_json(model_dir / f"{output_prefix}_predictions_list.json", predictions_list)
    with (model_dir / f"{output_prefix}_predictions.jsonl").open("w") as handle:
        for row in predictions_list:
            handle.write(json.dumps(row) + "\n")


def _summary(model_dir: Path, output_prefix: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(row.get("proxy_reward") or 0.0) for row in rows]
    prompt_tokens = [int(row.get("prompt_tokens") or 0) for row in rows]
    output_tokens = [int(row.get("output_tokens") or 0) for row in rows]
    return {
        "status": "complete",
        "source_outputs": str(model_dir / "outputs.jsonl"),
        "output_prefix": output_prefix,
        "model_name": str(rows[0].get("model_name") or "") if rows else "",
        "model_slug": model_dir.name,
        "n_completed": len(rows),
        "n_request_errors": sum(1 for row in rows if row.get("request_error")),
        "n_empty_patches": sum(1 for row in rows if not row.get("model_patch")),
        "n_format_failures": sum(1 for row in rows if row.get("failure_type") == "format"),
        "mean_proxy_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
        "proxy_success_rate": (sum(1 for row in rows if row.get("proxy_success")) / len(rows)) if rows else 0.0,
        "mean_prompt_tokens": (sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else 0.0,
        "mean_output_tokens": (sum(output_tokens) / len(output_tokens)) if output_tokens else 0.0,
        "total_prompt_tokens": sum(prompt_tokens),
        "total_output_tokens": sum(output_tokens),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _reparse_model(
    *,
    model_dir: Path,
    dataset_by_id: dict[str, dict[str, Any]],
    output_prefix: str,
    success_threshold: float,
) -> dict[str, Any]:
    reparsed_rows: list[dict[str, Any]] = []
    for line in (model_dir / "outputs.jsonl").open():
        if not line.strip():
            continue
        row = json.loads(line)
        instance_id = str(row.get("instance_id") or "")
        problem = dataset_by_id.get(instance_id)
        if problem is None:
            raise KeyError(f"{instance_id} from {model_dir / 'outputs.jsonl'} missing in dataset")

        request_error = row.get("request_error")
        reward_metadata: dict[str, Any]
        if request_error:
            reward = 0.0
            reward_metadata = {
                "request_error": True,
                "error": request_error,
                "reparsed": True,
            }
            model_patch = ""
            n_edits = 0
            patch_status = "request_error"
        else:
            output_text = str(row.get("output_text") or "")
            edits = extract_search_replace_edits(output_text)
            reward, reward_metadata = calculate_precise_reward(
                problem.get("file_contents") or {},
                str(problem.get("patch") or ""),
                edits,
            )
            reward_metadata = dict(reward_metadata)
            reward_metadata["reparsed"] = True
            try:
                model_patch, n_edits, patch_status = _build_git_patch(
                    problem.get("file_contents") or {},
                    output_text,
                )
            except Exception as exc:  # pylint: disable=broad-except
                model_patch = ""
                n_edits = 0
                patch_status = "reconstruction_error"
                reward_metadata["patch_reconstruction_error"] = repr(exc)

        new_row = dict(row)
        new_row.update(
            {
                "proxy_reward": float(reward or 0.0),
                "proxy_success": bool(reward and reward > success_threshold),
                "failure_type": _derive_failure_type(reward_metadata, request_error, patch_status),
                "reward_metadata": sanitize_for_json(reward_metadata),
                "model_patch": model_patch,
                "n_edits": int(n_edits),
                "patch_status": patch_status,
                "reparsed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        reparsed_rows.append(sanitize_for_json(new_row))

    outputs_path = model_dir / f"{output_prefix}_outputs.jsonl"
    with outputs_path.open("w") as handle:
        for row in reparsed_rows:
            handle.write(json.dumps(row) + "\n")
    _write_predictions(model_dir, output_prefix, reparsed_rows)
    summary = _summary(model_dir, output_prefix, reparsed_rows)
    _write_json(model_dir / f"{output_prefix}_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-dir", required=True, help="Discovery collect dir containing models/*/outputs.jsonl")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-name", default="swebench_verified")
    parser.add_argument("--model-slug", default=None, help="Optional single model slug under collect-dir/models")
    parser.add_argument("--output-prefix", default="reparsed", help="Prefix for sidecar files in each model dir")
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")
    collect_dir = Path(args.collect_dir)
    dataset_by_id = _load_rows(args.dataset_path, args.dataset_name, args.seed)
    summaries: dict[str, dict[str, Any]] = {}
    for model_dir in _iter_model_dirs(collect_dir, args.model_slug):
        logger.info("Reparsing %s", model_dir.name)
        summaries[model_dir.name] = _reparse_model(
            model_dir=model_dir,
            dataset_by_id=dataset_by_id,
            output_prefix=args.output_prefix,
            success_threshold=args.success_threshold,
        )
    _write_json(collect_dir / f"{args.output_prefix}_manifest.json", summaries)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

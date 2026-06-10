#!/usr/bin/env python
"""Sample fixed instance-id files for real-label routing collections."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_ids(dataset_path: str, dataset_name: str | None, seed: int) -> list[str]:
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
    ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        problem_id = problem_id_from_item(row)
        if problem_id and problem_id not in seen:
            seen.add(problem_id)
            ids.append(problem_id)
    if not ids:
        raise ValueError(f"No ids loaded from {dataset_path}")
    return sorted(ids)


def _read_id_file(path: str | Path) -> set[str]:
    value = Path(path)
    if not value.exists():
        raise FileNotFoundError(f"Missing exclude id file: {value}")
    return {line.strip() for line in value.read_text().splitlines() if line.strip()}


def _load_excluded(paths: list[str] | None) -> set[str]:
    excluded: set[str] = set()
    for raw in paths or []:
        for piece in str(raw).replace(",", " ").split():
            if piece.strip():
                excluded.update(_read_id_file(piece.strip()))
    return excluded


def _sample(ids: list[str], n: int, seed: int, excluded: set[str] | None = None) -> list[str]:
    if n <= 0:
        raise ValueError("sample size must be positive")
    excluded = excluded or set()
    available_ids = [instance_id for instance_id in ids if instance_id not in excluded]
    if n > len(available_ids):
        logger.warning(
            "Requested %s ids from only %s available after excluding %s; using all",
            n,
            len(available_ids),
            len(excluded),
        )
        n = len(available_ids)
    rng = random.Random(seed)
    selected = rng.sample(available_ids, n)
    return sorted(selected)


def _write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset-path", default="/mnt/llmd/data/swe_smith/ds_train")
    parser.add_argument("--train-dataset-name", default="swe_smith_train")
    parser.add_argument("--eval-dataset-path", default="/mnt/llmd/data/swe_smith/ds_test")
    parser.add_argument("--eval-dataset-name", default="swe_smith_test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-n", type=int, default=1500)
    parser.add_argument("--eval-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-train-ids-path",
        action="append",
        default=[],
        help="File(s) of train IDs to exclude. Can be repeated or comma/space separated.",
    )
    parser.add_argument(
        "--exclude-eval-ids-path",
        action="append",
        default=[],
        help="File(s) of eval IDs to exclude. Can be repeated or comma/space separated.",
    )
    parser.add_argument(
        "--exclude-ids-path",
        action="append",
        default=[],
        help="File(s) of IDs to exclude from both train and eval. Can be repeated or comma/space separated.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SWE-Smith train ids from %s", args.train_dataset_path)
    train_ids_all = _load_ids(args.train_dataset_path, args.train_dataset_name, args.seed)
    logger.info("Loading SWE-Smith eval ids from %s", args.eval_dataset_path)
    eval_ids_all = _load_ids(args.eval_dataset_path, args.eval_dataset_name, args.seed)

    shared_excluded = _load_excluded(args.exclude_ids_path)
    train_excluded = shared_excluded | _load_excluded(args.exclude_train_ids_path)
    eval_excluded = shared_excluded | _load_excluded(args.exclude_eval_ids_path)
    if train_excluded:
        logger.info("Excluding %d train ids", len(train_excluded))
    if eval_excluded:
        logger.info("Excluding %d eval ids", len(eval_excluded))

    train_ids = _sample(train_ids_all, args.train_n, args.seed, train_excluded)
    # Offset eval seed so identical identifiers cannot be selected by identical RNG state if splits overlap.
    eval_ids = _sample(eval_ids_all, args.eval_n, args.seed + 1009, eval_excluded)

    train_path = output_dir / f"swe_smith_train_{len(train_ids)}_ids.txt"
    eval_path = output_dir / f"swe_smith_eval_{len(eval_ids)}_ids.txt"
    _write_ids(train_path, train_ids)
    _write_ids(eval_path, eval_ids)

    manifest: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "train": {
            "dataset_path": args.train_dataset_path,
            "dataset_name": args.train_dataset_name,
            "available_ids": len(train_ids_all),
            "excluded_ids": len(train_excluded),
            "available_after_exclusion": len([x for x in train_ids_all if x not in train_excluded]),
            "selected_ids": len(train_ids),
            "ids_path": str(train_path),
        },
        "eval": {
            "dataset_path": args.eval_dataset_path,
            "dataset_name": args.eval_dataset_name,
            "available_ids": len(eval_ids_all),
            "excluded_ids": len(eval_excluded),
            "available_after_exclusion": len([x for x in eval_ids_all if x not in eval_excluded]),
            "selected_ids": len(eval_ids),
            "ids_path": str(eval_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from pipelinerl.swe.load_datasets import load_local_swe_dataset
from pipelinerl.swe.scripts.offline_router.common import problem_id_from_item
from pipelinerl.swe.utils.repair_utils import compute_change_similarities, get_filelevel_diff

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for minimal envs
    tqdm = None


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_dataset_by_id(path: str, name: str, ids: set[str]) -> dict[str, dict[str, Any]]:
    log(f"Loading {name} from {path} for {len(ids)} ids")
    rows = load_local_swe_dataset(
        dataset_names=[name],
        dataset_path=path,
        shuffle=False,
        seed=42,
        dataset_label=name,
        max_samples=None,
    )
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        pid = problem_id_from_item(row)
        if pid in ids:
            out[pid] = row
    missing = ids - set(out)
    if missing:
        raise ValueError(f"Missing {len(missing)} ids from {path}; first={sorted(missing)[:5]}")
    return out


def score_patch(pred_patch_text: str, oracle_patch: dict[str, str], pred_cache: dict[str, dict[str, str]]) -> float:
    pred_key = pred_patch_text or ""
    pred = pred_cache.get(pred_key)
    if pred is None:
        pred = get_filelevel_diff(pred_key)
        pred_cache[pred_key] = pred
    similarities = compute_change_similarities(pred, oracle_patch)
    if not similarities:
        if not pred and not oracle_patch:
            return 1.0
        return 0.0
    return sum(float(s["similarity"]) for s in similarities) / len(similarities)


def count_jsonl_rows(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute SWE-Smith proxy similarity for an eval package with tqdm ETA.")
    p.add_argument("--package", required=True)
    p.add_argument("--train-dataset-path", default="/mnt/llmd/data/swe_smith_bugged_context/ds_train")
    p.add_argument("--train-dataset-name", default="swe_smith_train_bugged_context")
    p.add_argument("--eval-dataset-path", default="/mnt/llmd/data/swe_smith_bugged_context/ds_test")
    p.add_argument("--eval-dataset-name", default="swe_smith_test_bugged_context")
    p.add_argument("--output-json", default=None, help="Optional path to write the JSON report as well as stdout.")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.package)
    manifest = json.loads((root / "manifest.json").read_text())
    dataset_args = {
        "train1500": (args.train_dataset_path, args.train_dataset_name),
        "eval500": (args.eval_dataset_path, args.eval_dataset_name),
    }

    total_predictions = 0
    for sinfo in manifest["splits"].values():
        for rinfo in sinfo["routes"].values():
            total_predictions += count_jsonl_rows(root / rinfo["predictions_path"])

    pbar = None
    if not args.no_progress and tqdm is not None:
        pbar = tqdm(total=total_predictions, unit="pred", dynamic_ncols=True, desc="score proxy similarity")
    elif not args.no_progress:
        log("tqdm is not installed; running without a progress bar")

    out: dict[str, Any] = {"package": str(root), "splits": {}}
    try:
        for split, sinfo in manifest["splits"].items():
            ids_path = root / sinfo["ids_path"]
            ids = {x.strip() for x in ids_path.read_text().splitlines() if x.strip()}
            dpath, dname = dataset_args[split]
            dataset = load_dataset_by_id(dpath, dname, ids)
            log(f"Parsing oracle patches for {split}")
            oracle_by_id = {
                pid: get_filelevel_diff(row.get("patch") or "")
                for pid, row in dataset.items()
            }
            out["splits"][split] = {"routes": {}}
            for route, rinfo in sinfo["routes"].items():
                pred_cache: dict[str, dict[str, str]] = {}
                scores: list[float] = []
                nonempty = 0
                pred_path = root / rinfo["predictions_path"]
                if pbar is not None:
                    pbar.set_postfix_str(f"{split}/{route}")
                for row in iter_jsonl(pred_path):
                    patch = row.get("patch") or ""
                    if patch:
                        nonempty += 1
                    scores.append(score_patch(patch, oracle_by_id[row["instance_id"]], pred_cache))
                    if pbar is not None:
                        pbar.update(1)
                out["splits"][split]["routes"][route] = {
                    "n": len(scores),
                    "nonempty": nonempty,
                    "empty": len(scores) - nonempty,
                    "mean_similarity": mean(scores) if scores else 0.0,
                    "success_at_0_8": sum(1 for s in scores if s >= 0.8) / len(scores) if scores else 0.0,
                    "success_at_0_5": sum(1 for s in scores if s >= 0.5) / len(scores) if scores else 0.0,
                }
    finally:
        if pbar is not None:
            pbar.close()

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")
        log(f"Wrote {output_path}")
    print(text)


if __name__ == "__main__":
    main()

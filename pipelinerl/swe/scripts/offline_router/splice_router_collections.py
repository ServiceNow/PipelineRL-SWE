#!/usr/bin/env python
import argparse
import json
import os
import shutil
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _link_split(output_dir: Path, split: str, source_dir: Path) -> None:
    target = output_dir / split
    source = source_dir / split
    if not source.exists():
        raise FileNotFoundError(f"Missing source split: {source}")
    if target.exists():
        if target.is_symlink() and Path(os.readlink(target)) == source:
            return
        raise FileExistsError(f"{target} already exists and is not the expected symlink")
    os.symlink(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-collect-dir", required=True)
    parser.add_argument("--eval-collect-dir", required=True)
    parser.add_argument("--output-collect-dir", required=True)
    args = parser.parse_args()

    train_dir = Path(args.train_collect_dir)
    eval_dir = Path(args.eval_collect_dir)
    output_dir = Path(args.output_collect_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _link_split(output_dir, "train", train_dir)
    _link_split(output_dir, "eval", eval_dir)

    train_meta = _load_json(train_dir / "metadata.json")
    eval_meta = _load_json(eval_dir / "metadata.json")
    combined_meta = dict(train_meta)
    combined_meta["combined_from"] = {
        "train_collect": str(train_dir),
        "eval_collect": str(eval_dir),
    }
    combined_meta["split_summaries"] = [
        summary for summary in train_meta.get("split_summaries", []) if summary.get("split") == "train"
    ] + [
        summary for summary in eval_meta.get("split_summaries", []) if summary.get("split") == "eval"
    ]
    _write_json(output_dir / "metadata.json", combined_meta)

    train_summary_path = train_dir / "route_distribution_summary.json"
    eval_summary_path = eval_dir / "route_distribution_summary.json"
    if train_summary_path.exists() and eval_summary_path.exists():
        train_summary = _load_json(train_summary_path)
        eval_summary = _load_json(eval_summary_path)
        train_summary["dataset_dir"] = str(output_dir)
        train_summary["combined_from"] = {
            "train_collect": str(train_dir),
            "eval_collect": str(eval_dir),
        }
        train_summary.setdefault("splits", {})["eval"] = eval_summary.get("splits", {}).get("eval")
        _write_json(output_dir / "route_distribution_summary.json", train_summary)

    config_path = train_dir / "collection_config.json"
    if config_path.exists():
        shutil.copy2(config_path, output_dir / "collection_config.json")

    print(output_dir)


if __name__ == "__main__":
    main()

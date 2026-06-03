#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

DEFAULT_BASE = "/mnt/llmd/results/exps/aristides/reason"
ROUTES = [
    {"run_id": "qwen3_4b_instruct_2507", "model_slug": "Qwen_Qwen3-4B-Instruct-2507", "job_slug": "qwen3_4b_instruct_2507", "model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507"},
    {"run_id": "gpt_oss_20b", "model_slug": "openai_gpt-oss-20b", "job_slug": "gpt_oss_20b", "model_name_or_path": "openai/gpt-oss-20b"},
    {"run_id": "qwen3_coder_30b_a3b", "model_slug": "Qwen_Qwen3-Coder-30B-A3B-Instruct", "job_slug": "qwen3_coder_30b_a3b", "model_name_or_path": "Qwen/Qwen3-Coder-30B-A3B-Instruct"},
    {"run_id": "gpt_oss_120b", "model_slug": "openai_gpt-oss-120b", "job_slug": "gpt_oss_120b", "model_name_or_path": "openai/gpt-oss-120b"},
]
SPLITS = ["train1500", "eval500"]


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def collection_dir(base: Path, split: str, route: dict[str, str], timestamp: str) -> Path:
    return base / f"offline_router_swe_smith_real_{split}_collect_{route['job_slug']}_{timestamp}" / "collect"


def convert_predictions(src: Path, dst: Path, model_name: str) -> dict[str, Any]:
    total = 0
    nonempty = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            patch = row.get("patch")
            if patch is None:
                patch = row.get("model_patch") or ""
            if patch:
                nonempty += 1
            fout.write(json.dumps({
                "instance_id": row["instance_id"],
                "patch": patch,
                "model_patch": patch,
                "model_name_or_path": row.get("model_name_or_path") or model_name,
            }, sort_keys=True) + "\n")
    return {"n_predictions": total, "n_nonempty_patches": nonempty, "n_empty_patches": total - nonempty}


MATERIALIZER = r'''#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datasets import load_dataset

REQUIRED = {"instance_id", "patch", "FAIL_TO_PASS", "PASS_TO_PASS", "image_name", "repo", "problem_statement"}
DEFAULT_DATASETS = [
    "SWE-bench/SWE-smith",
    "SWE-bench/SWE-smith-py",
    "SWE-bench/SWE-smith-java",
    "SWE-bench/SWE-smith-rs",
    "SWE-bench/SWE-smith-go",
]


def parse_dataset_names(raw: str) -> list[str]:
    values: list[str] = []
    for item in (raw or "").split(","):
        item = item.strip()
        if item and item not in values:
            values.append(item)
    for item in DEFAULT_DATASETS:
        if item not in values:
            values.append(item)
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--dataset-name",
        default="SWE-bench/SWE-smith",
        help="Comma-separated HF dataset names to try first; language-specific SWE-Smith datasets are always used as fallback.",
    )
    p.add_argument("--split", default="train")
    args = p.parse_args()

    ids = [x.strip() for x in Path(args.ids).read_text().splitlines() if x.strip()]
    wanted = set(ids)
    by_id = {}
    datasets = parse_dataset_names(args.dataset_name)
    for dataset_name in datasets:
        remaining = wanted - set(by_id)
        if not remaining:
            break
        print(f"Loading {dataset_name}:{args.split}; remaining ids={len(remaining)}")
        ds = load_dataset(dataset_name, split=args.split)
        found_here = 0
        for row in ds:
            instance_id = row.get("instance_id")
            if instance_id in remaining:
                by_id[instance_id] = dict(row)
                found_here += 1
        print(f"  found {found_here} ids in {dataset_name}")

    missing = [x for x in ids if x not in by_id]
    if missing:
        raise SystemExit(f"Missing {len(missing)} ids after trying {datasets}; first: {missing[:10]}")

    selected = [by_id[x] for x in ids]
    if selected:
        missing_keys = sorted(REQUIRED - set(selected[0]))
        if missing_keys:
            raise SystemExit(f"Dataset rows missing required keys: {missing_keys}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected, indent=2))
    print(f"Wrote {len(selected)} rows to {out}")


if __name__ == "__main__":
    main()
'''


def write_materializer(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MATERIALIZER)
    path.chmod(0o755)


def write_runner(output_dir: Path, splits: list[str], route_ids: list[str]) -> None:
    split_words = " ".join(splits)
    route_words = " ".join(route_ids)
    text = f'''#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
WORKERS="${{WORKERS:-10}}"
HF_DATASET="${{HF_DATASET:-SWE-bench/SWE-smith}}"
HF_SPLIT="${{HF_SPLIT:-train}}"
INSTALL_SWESMITH="${{INSTALL_SWESMITH:-0}}"
if [[ "${{INSTALL_SWESMITH}}" == "1" ]]; then
  python -m pip install -U 'git+https://github.com/SWE-bench/SWE-smith.git'
fi
for split in {split_words}; do
  ids_path="$SCRIPT_DIR/ids/${{split}}.txt"
  dataset_path="$SCRIPT_DIR/datasets/swe_smith_${{split}}.json"
  if [[ ! -s "$dataset_path" ]]; then
    python "$SCRIPT_DIR/scripts/materialize_swesmith_subset.py" --ids "$ids_path" --output "$dataset_path" --dataset-name "$HF_DATASET" --split "$HF_SPLIT"
  fi
  for route in {route_words}; do
    predictions_path="$SCRIPT_DIR/predictions/${{split}}/${{route}}/predictions.jsonl"
    run_id="swe_smith_${{split}}_${{route}}"
    echo "[eval] $run_id"
    python -m swesmith.harness.eval --dataset_path "$dataset_path" --predictions_path "$predictions_path" --run_id "$run_id" --workers "$WORKERS"
  done
done
echo "Reports are under logs/run_evaluation/<run_id>/report.json"
'''
    runner = output_dir / "run_all_swe_smith.sh"
    runner.write_text(text)
    runner.chmod(0o755)


def write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text('''# SWE-Smith Real-Label Eval Package

Prediction rows include both `patch` and `model_patch` aliases plus `instance_id` and `model_name_or_path`, because SWE-Smith versions differ on the prediction key.
The runner first materializes selected SWE-Smith task instances from HuggingFace, then runs:

```bash
python -m swesmith.harness.eval --dataset_path datasets/swe_smith_<split>.json --predictions_path predictions/<split>/<route>/predictions.jsonl --run_id swe_smith_<split>_<route> --workers 10
```

On AWS:

```bash
INSTALL_SWESMITH=1 WORKERS=10 bash run_all_swe_smith.sh
```

Reports are written under `logs/run_evaluation/<run_id>/report.json`.
''')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--timestamp", required=True)
    p.add_argument("--base-dir", default=DEFAULT_BASE)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", action="append", choices=SPLITS)
    p.add_argument("--make-tarball", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    splits = args.split or list(SPLITS)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_materializer(output_dir / "scripts" / "materialize_swesmith_subset.py")
    route_ids = [r["run_id"] for r in ROUTES]
    write_runner(output_dir, splits, route_ids)
    write_readme(output_dir)
    manifest: dict[str, Any] = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "timestamp": args.timestamp, "format": "SWE-Smith predictions: instance_id, patch/model_patch, model_name_or_path", "splits": {}, "routes": ROUTES}
    for split in splits:
        split_manifest: dict[str, Any] = {"routes": {}}
        ids_written = False
        for route in ROUTES:
            cdir = collection_dir(base, split, route, args.timestamp)
            src_pred = cdir / "models" / route["model_slug"] / "predictions.jsonl"
            src_ids = cdir / "instance_ids.txt"
            cmanifest = cdir / "manifest.json"
            if not src_pred.exists() or not src_ids.exists() or not cmanifest.exists():
                raise FileNotFoundError(f"Missing artifacts for {split}/{route['run_id']} under {cdir}")
            if not ids_written:
                ids_dst = output_dir / "ids" / f"{split}.txt"
                ids_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_ids, ids_dst)
                split_manifest["ids_path"] = str(ids_dst.relative_to(output_dir))
                split_manifest["n_ids"] = len([x for x in ids_dst.read_text().splitlines() if x.strip()])
                ids_written = True
            dst_pred = output_dir / "predictions" / split / route["run_id"] / "predictions.jsonl"
            pred_summary = convert_predictions(src_pred, dst_pred, route["model_name_or_path"])
            cm = read_json(cmanifest)
            summary = next(iter((cm.get("model_summaries") or {}).values()), {})
            split_manifest["routes"][route["run_id"]] = {"source_collect_dir": str(cdir), "predictions_path": str(dst_pred.relative_to(output_dir)), "mean_proxy_reward": summary.get("mean_proxy_reward"), **pred_summary}
        manifest["splits"][split] = split_manifest
    write_json(output_dir / "manifest.json", manifest)
    if args.make_tarball:
        tar_path = output_dir.with_suffix(".tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(output_dir, arcname=output_dir.name)
        print(f"Wrote tarball: {tar_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import tarfile
from pathlib import Path
from typing import Any

from datasets import load_from_disk

FAKE_INDEX_RE = re.compile(r"^index 0{7,40}\.\.1{7,40}(?: \d+)?$")
DIFF_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")


def parse_file_contents(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    return {}


def problem_id_from_item(item: dict[str, Any]) -> str:
    for key in ("instance_id", "issue_id", "id"):
        value = item.get(key)
        if value:
            return str(value)
    raise ValueError("Missing instance id")


def git_blob_hash(text: str) -> str:
    data = text.encode("utf-8", errors="surrogateescape")
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()[:7]


def load_dataset_by_id(path: str, name: str, ids: set[str]) -> dict[str, dict[str, Any]]:
    _ = name
    dataset = load_from_disk(path)
    out: dict[str, dict[str, Any]] = {}
    for item in dataset:
        row = dict(item)
        try:
            pid = problem_id_from_item(row)
        except ValueError:
            continue
        if pid not in ids:
            continue
        file_contents = parse_file_contents(row.get("repair_file_contents", row.get("gold_file_contents", "{}")))
        out[pid] = {**row, "file_contents": file_contents}
    missing = ids - set(out)
    if missing:
        raise ValueError(f"Missing {len(missing)} selected ids from {path}; first={sorted(missing)[:5]}")
    return out


def paths_from_patch(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        match = DIFF_RE.match(line)
        if match:
            path = match.group(2)
            if path not in paths:
                paths.append(path)
    return paths


def strip_fake_index_lines(patch: str) -> str:
    lines = [line for line in patch.splitlines() if not FAKE_INDEX_RE.match(line.strip())]
    text = "\n".join(lines)
    if text and patch.endswith("\n"):
        text += "\n"
    return text


def apply_patch_to_files(file_contents: dict[str, str], patch: str, touched_paths: list[str]) -> dict[str, str] | None:
    if not patch.strip():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for path in touched_paths:
            if path not in file_contents:
                return None
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file_contents[path], encoding="utf-8", errors="surrogateescape")
        patch_path = root / "prediction.patch"
        patch_path.write_text(strip_fake_index_lines(patch), encoding="utf-8", errors="surrogateescape")
        result = subprocess.run(
            ["patch", "-p1", "--batch", "--forward", "--reject-file=-", "-i", str(patch_path)],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            return None
        return {path: (root / path).read_text(encoding="utf-8", errors="surrogateescape") for path in touched_paths}


def replace_fake_index_lines(patch: str, old_contents: dict[str, str], new_contents: dict[str, str]) -> tuple[str, int, int]:
    current_path: str | None = None
    replaced = 0
    fake_remaining = 0
    out: list[str] = []
    for line in patch.splitlines():
        diff_match = DIFF_RE.match(line)
        if diff_match:
            current_path = diff_match.group(2)
            out.append(line)
            continue
        if FAKE_INDEX_RE.match(line.strip()):
            if current_path and current_path in old_contents and current_path in new_contents:
                out.append(f"index {git_blob_hash(old_contents[current_path])}..{git_blob_hash(new_contents[current_path])} 100644")
                replaced += 1
            else:
                out.append(line)
                fake_remaining += 1
            continue
        out.append(line)
    text = "\n".join(out)
    if text and patch.endswith("\n"):
        text += "\n"
    return text, replaced, fake_remaining


def repair_patch(patch: str, file_contents: dict[str, str]) -> tuple[str, dict[str, int]]:
    stats = {
        "fake_index_lines": 0,
        "replaced_index_lines": 0,
        "stripped_uncomputable_index_lines": 0,
        "unrepaired_fake_index_lines": 0,
        "apply_failures": 0,
    }
    if not patch:
        return "", stats
    stats["fake_index_lines"] = sum(1 for line in patch.splitlines() if FAKE_INDEX_RE.match(line.strip()))
    if stats["fake_index_lines"] == 0:
        return patch, stats
    touched = paths_from_patch(patch)
    new_contents = apply_patch_to_files(file_contents, patch, touched)
    if new_contents is None:
        # If the patch does not apply even after ignoring synthetic index lines,
        # there is no well-defined post-patch blob hash to compute. Keep the
        # actual diff content, but remove fake index lines so the harness reports
        # a real hunk/apply failure instead of a misleading create-file failure.
        stats["apply_failures"] = 1
        stats["stripped_uncomputable_index_lines"] = stats["fake_index_lines"]
        return strip_fake_index_lines(patch), stats
    repaired, replaced, remaining = replace_fake_index_lines(patch, file_contents, new_contents)
    stats["replaced_index_lines"] = replaced
    stats["unrepaired_fake_index_lines"] = remaining
    return repaired, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replace synthetic patch index hashes in a SWE-Smith eval package with real blob hashes.")
    p.add_argument("--package-dir", required=True)
    p.add_argument("--make-tarball", action="store_true")
    p.add_argument("--train-dataset-path", default="/mnt/llmd/data/swe_smith_bugged_context/ds_train")
    p.add_argument("--train-dataset-name", default="swe_smith_train_bugged_context")
    p.add_argument("--eval-dataset-path", default="/mnt/llmd/data/swe_smith_bugged_context/ds_test")
    p.add_argument("--eval-dataset-name", default="swe_smith_test_bugged_context")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pkg = Path(args.package_dir)
    manifest_path = pkg / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    def dataset_info_for_split(split: str) -> tuple[str, str]:
        split_lower = str(split).lower()
        if split_lower.startswith("train"):
            return args.train_dataset_path, args.train_dataset_name
        if split_lower.startswith("eval") or split_lower.startswith("test"):
            return args.eval_dataset_path, args.eval_dataset_name
        raise ValueError(f"Cannot infer dataset for split={split!r}; expected train*/eval*/test*")

    def iter_manifest_splits() -> dict[str, dict[str, Any]]:
        if isinstance(manifest.get("splits"), dict):
            return manifest["splits"]
        split = str(manifest.get("split") or "")
        ids_path = manifest.get("ids_path")
        routes = manifest.get("routes")
        if not split or not ids_path or not isinstance(routes, list):
            raise ValueError("Unsupported manifest shape: expected either manifest['splits'] or top-level split/ids_path/routes")
        return {
            split: {
                "ids_path": ids_path,
                "routes": {str(route["route_id"]): route for route in routes},
            }
        }

    totals = {
        "prediction_files_changed": 0,
        "prediction_rows_changed": 0,
        "fake_index_lines": 0,
        "replaced_index_lines": 0,
        "stripped_uncomputable_index_lines": 0,
        "unrepaired_fake_index_lines": 0,
        "apply_failures": 0,
    }
    repair_manifest: dict[str, Any] = {"splits": {}}
    for split, split_info in iter_manifest_splits().items():
        ids_path = pkg / split_info["ids_path"]
        ids = {line.strip() for line in ids_path.read_text().splitlines() if line.strip()}
        dataset_path, dataset_name = dataset_info_for_split(split)
        dataset = load_dataset_by_id(dataset_path, dataset_name, ids)
        repair_manifest["splits"][split] = {"routes": {}}
        for route, route_info in split_info["routes"].items():
            pred_path = pkg / route_info["predictions_path"]
            rows = []
            route_stats = {key: 0 for key in totals if key not in {"prediction_files_changed"}}
            changed_file = False
            for line in pred_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                pid = row["instance_id"]
                patch = row.get("patch") or row.get("model_patch") or ""
                repaired, stats = repair_patch(patch, dataset[pid].get("file_contents") or {})
                for key, value in stats.items():
                    route_stats[key] += value
                    totals[key] += value
                if repaired != patch:
                    changed_file = True
                    route_stats["prediction_rows_changed"] += 1
                    totals["prediction_rows_changed"] += 1
                row["patch"] = repaired
                row["model_patch"] = repaired
                rows.append(row)
            if changed_file:
                pred_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
                totals["prediction_files_changed"] += 1
            repair_manifest["splits"][split]["routes"][route] = route_stats
    manifest["patch_hash_repair"] = repair_manifest
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (pkg / "patch_hash_repair_manifest.json").write_text(json.dumps(repair_manifest, indent=2, sort_keys=True) + "\n")
    if args.make_tarball:
        tar_path = pkg.with_suffix(".tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(pkg, arcname=pkg.name)
        print(f"Wrote tarball: {tar_path}")
    print(json.dumps(totals, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

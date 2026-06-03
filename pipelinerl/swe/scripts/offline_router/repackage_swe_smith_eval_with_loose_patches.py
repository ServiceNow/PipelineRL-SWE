#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from pipelinerl.swe.load_datasets import load_local_swe_dataset
from pipelinerl.swe.scripts.offline_router.common import problem_id_from_item
from pipelinerl.swe.scripts.repair_eval_utils import extract_search_replace_edits
from pipelinerl.swe.utils.repair_utils import FormatError, apply_edits_to_files, get_normalized_patch


def git_blob_hash(text: str) -> str:
    data = text.encode("utf-8", errors="surrogateescape")
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()[:7]


def build_git_patch(old_contents: dict[str, str], new_contents: dict[str, str]) -> str:
    patch_dict = get_normalized_patch(old_contents, new_contents)
    parts: list[str] = []
    for file_path, body in patch_dict.items():
        old = old_contents.get(file_path, "")
        new = new_contents.get(file_path, "")
        parts.extend([
            f"diff --git a/{file_path} b/{file_path}",
            f"index {git_blob_hash(old)}..{git_blob_hash(new)} 100644",
            f"--- a/{file_path}",
            f"+++ b/{file_path}",
            body,
        ])
    text = "\n".join(parts)
    return text + ("\n" if text and not text.endswith("\n") else "")


def resolve_path(file_path: str, file_contents: dict[str, str]) -> str | None:
    raw = str(file_path or "").strip().strip("` ").lstrip("./")
    if raw in file_contents:
        return raw
    candidates = [p for p in file_contents if p.endswith("/" + raw) or raw.endswith("/" + p)]
    if len(candidates) == 1:
        return candidates[0]
    basename = Path(raw).name
    if basename:
        candidates = [p for p in file_contents if Path(p).name == basename]
        if len(candidates) == 1:
            return candidates[0]
    return None


def loose_reconstruct(output_text: str, file_contents: dict[str, str]) -> tuple[str, dict[str, Any]]:
    edits = extract_search_replace_edits(output_text or "")
    stats: dict[str, Any] = {
        "parsed_edits": len(edits),
        "skipped_noop": 0,
        "skipped_bad_path": 0,
        "skipped_apply_error": 0,
        "applied_edits": 0,
    }
    if not edits:
        return "", stats
    current = dict(file_contents)
    for edit in edits:
        search = str(edit.get("search") or "")
        replace = str(edit.get("replace") or "")
        if search == replace:
            stats["skipped_noop"] += 1
            continue
        resolved = resolve_path(str(edit.get("file_path") or ""), current)
        if not resolved:
            stats["skipped_bad_path"] += 1
            continue
        one = {"file_path": resolved, "search": search, "replace": replace}
        try:
            current = apply_edits_to_files(current, [one], silent=False)
        except FormatError:
            stats["skipped_apply_error"] += 1
            continue
        except Exception:
            stats["skipped_apply_error"] += 1
            continue
        stats["applied_edits"] += 1
    if stats["applied_edits"] <= 0:
        return "", stats
    return build_git_patch(file_contents, current), stats


def load_dataset_by_id(path: str, name: str, ids: set[str]) -> dict[str, dict[str, Any]]:
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
        raise ValueError(f"Missing {len(missing)} selected ids from {path}; first={sorted(missing)[:5]}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repackage SWE-Smith eval package with conservative loose patch reconstruction.")
    p.add_argument("--source-package", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--make-tarball", action="store_true")
    p.add_argument("--train-dataset-path", default="/mnt/llmd/data/swe_smith_bugged_context/ds_train")
    p.add_argument("--train-dataset-name", default="swe_smith_train_bugged_context")
    p.add_argument("--eval-dataset-path", default="/mnt/llmd/data/swe_smith_bugged_context/ds_test")
    p.add_argument("--eval-dataset-name", default="swe_smith_test_bugged_context")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.source_package)
    dst = Path(args.output_dir)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    manifest = json.loads((dst / "manifest.json").read_text())
    recovery_manifest: dict[str, Any] = {"splits": {}}
    split_dataset_info = {
        "train1500": (args.train_dataset_path, args.train_dataset_name),
        "eval500": (args.eval_dataset_path, args.eval_dataset_name),
    }
    for split, split_info in manifest["splits"].items():
        ids_path = dst / split_info["ids_path"]
        ids = {line.strip() for line in ids_path.read_text().splitlines() if line.strip()}
        dataset_path, dataset_name = split_dataset_info[split]
        dataset = load_dataset_by_id(dataset_path, dataset_name, ids)
        recovery_manifest["splits"][split] = {"routes": {}}
        for route, route_info in split_info["routes"].items():
            source_collect_dir = Path(route_info["source_collect_dir"])
            # Locate the single outputs.jsonl under this model collection.
            outputs_paths = list((source_collect_dir / "models").glob("*/outputs.jsonl"))
            if len(outputs_paths) != 1:
                raise ValueError(f"Expected one outputs.jsonl under {source_collect_dir}, found {outputs_paths}")
            pred_path = dst / route_info["predictions_path"]
            old_by_id = {}
            for line in pred_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    old_by_id[row["instance_id"]] = row
            recovered = 0
            attempted = 0
            still_empty = 0
            stats_totals = {"parsed_edits": 0, "skipped_noop": 0, "skipped_bad_path": 0, "skipped_apply_error": 0, "applied_edits": 0}
            new_rows = []
            for line in outputs_paths[0].open():
                if not line.strip():
                    continue
                out_row = json.loads(line)
                pid = out_row["instance_id"]
                pred_row = old_by_id.get(pid) or {"instance_id": pid, "model_name_or_path": out_row.get("model_name") or route, "patch": ""}
                patch = pred_row.get("patch") or pred_row.get("model_patch") or ""
                if not patch:
                    attempted += 1
                    patch, stats = loose_reconstruct(out_row.get("output_text") or "", dataset[pid].get("file_contents") or {})
                    for key in stats_totals:
                        stats_totals[key] += int(stats.get(key) or 0)
                    if patch:
                        recovered += 1
                if not patch:
                    still_empty += 1
                new_rows.append({"instance_id": pid, "patch": patch, "model_patch": patch, "model_name_or_path": pred_row.get("model_name_or_path") or out_row.get("model_name") or route})
            with pred_path.open("w") as handle:
                for row in new_rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            recovery_manifest["splits"][split]["routes"][route] = {
                "attempted_empty": attempted,
                "recovered": recovered,
                "still_empty": still_empty,
                "previous_empty": route_info.get("n_empty_patches"),
                "n_predictions": len(new_rows),
                **stats_totals,
            }
            route_info["n_empty_patches_loose"] = still_empty
            route_info["n_recovered_loose"] = recovered
            route_info["n_nonempty_patches_loose"] = len(new_rows) - still_empty
    manifest["loose_reconstruction"] = recovery_manifest
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (dst / "loose_reconstruction_manifest.json").write_text(json.dumps(recovery_manifest, indent=2, sort_keys=True) + "\n")
    if args.make_tarball:
        tar_path = dst.with_suffix(".tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(dst, arcname=dst.name)
        print(f"Wrote tarball: {tar_path}")
    print(json.dumps(recovery_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

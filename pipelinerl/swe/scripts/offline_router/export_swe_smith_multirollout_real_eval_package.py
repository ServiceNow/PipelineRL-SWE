#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

ROUTES = [
    {"base_route_id": "qwen3_4b_instruct_2507", "model_slug": "Qwen_Qwen3-4B-Instruct-2507", "model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507"},
    {"base_route_id": "gpt_oss_20b", "model_slug": "openai_gpt-oss-20b", "model_name_or_path": "openai/gpt-oss-20b"},
    {"base_route_id": "qwen3_coder_30b_a3b", "model_slug": "Qwen_Qwen3-Coder-30B-A3B-Instruct", "model_name_or_path": "Qwen/Qwen3-Coder-30B-A3B-Instruct"},
    {"base_route_id": "gpt_oss_120b", "model_slug": "openai_gpt-oss-120b", "model_name_or_path": "openai/gpt-oss-120b"},
]

def route_rollouts_for_profile(profile: str, route: dict[str, str], n_rollouts: int) -> list[int]:
    base_route_id = route["base_route_id"]
    if profile == "full":
        return list(range(n_rollouts))
    if profile == "20b_resample_120b_escalate":
        if base_route_id == "gpt_oss_20b":
            return list(range(n_rollouts))
        if base_route_id == "gpt_oss_120b":
            return [0]
        return []
    raise ValueError(f"Unknown export profile: {profile}")

def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]

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
            fout.write(json.dumps({"instance_id": row["instance_id"], "patch": patch, "model_patch": patch, "model_name_or_path": row.get("model_name_or_path") or model_name}, sort_keys=True) + "\n")
    return {"n_predictions": total, "n_nonempty_patches": nonempty, "n_empty_patches": total - nonempty}

def copy_helper(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing helper source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    dst.chmod(0o755)

def write_runner(output_dir: Path, split: str, route_ids: list[str]) -> None:
    route_words = " ".join(route_ids)
    text = "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        "WORKERS=\"${WORKERS:-10}\"",
        "ULIMIT_NOFILE=\"${ULIMIT_NOFILE:-65535}\"",
        "ulimit -n \"$ULIMIT_NOFILE\" 2>/dev/null || true",
        "HF_DATASET=\"${HF_DATASET:-SWE-bench/SWE-smith}\"",
        "HF_SPLIT=\"${HF_SPLIT:-train}\"",
        "INSTALL_SWESMITH=\"${INSTALL_SWESMITH:-0}\"",
        "if [[ \"${INSTALL_SWESMITH}\" == \"1\" ]]; then",
        "  python -m pip install -U 'git+https://github.com/SWE-bench/SWE-smith.git'",
        "fi",
        f"split=\"{split}\"",
        "ids_path=\"$SCRIPT_DIR/ids/${split}.txt\"",
        "dataset_path=\"$SCRIPT_DIR/datasets/swe_smith_${split}.json\"",
        "if [[ ! -s \"$dataset_path\" ]]; then",
        "  python \"$SCRIPT_DIR/scripts/materialize_swesmith_subset.py\" --ids \"$ids_path\" --output \"$dataset_path\" --dataset-name \"$HF_DATASET\" --split \"$HF_SPLIT\"",
        "fi",
        f"for route in {route_words}; do",
        "  predictions_path=\"$SCRIPT_DIR/predictions/${split}/${route}/predictions.jsonl\"",
        "  run_id=\"swe_smith_${split}_${route}\"",
        "  echo \"[eval] $run_id\"",
        "  python \"$SCRIPT_DIR/scripts/run_swesmith_eval_with_prune.py\" --dataset_path \"$dataset_path\" --predictions_path \"$predictions_path\" --run_id \"$run_id\" --workers \"$WORKERS\"",
        "done",
        "echo \"Reports are under logs/run_evaluation/<run_id>/report.json\"",
        "",
    ])
    runner = output_dir / "run_all_swe_smith.sh"
    runner.write_text(text)
    runner.chmod(0o755)

def write_readme(output_dir: Path) -> None:
    name = output_dir.name
    text = "\n".join([
        "# SWE-Smith Multi-Rollout Real Eval Package",
        "",
        "On AWS:",
        "",
        "```bash",
        f"tar -xzf {name}.tar.gz",
        f"cd {name}",
        "INSTALL_SWESMITH=1 WORKERS=10 DOCKER_CLEANUP=1 DOCKER_CLEANUP_BATCH_SIZE=16 bash run_all_swe_smith.sh 2>&1 | tee swe_smith_eval.log",
        "```",
        "",
    ])
    (output_dir / "README.md").write_text(text)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a SWE-Smith multi-rollout collection root into an AWS eval package.")
    p.add_argument("--source-root", required=True)
    p.add_argument("--split", default="eval300")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rollouts", type=int, default=3)
    p.add_argument(
        "--profile",
        choices=["full", "20b_resample_120b_escalate"],
        default="full",
        help=(
            "Which route/rollout subset to export. "
            "20b_resample_120b_escalate exports gpt-oss-20b r0/r1/r2 "
            "plus gpt-oss-120b r0 for stop/resample/escalate controller analysis."
        ),
    )
    p.add_argument("--make-tarball", action="store_true")
    p.add_argument("--helper-source-dir", default="router_analysis/aws_eval_packages/swe_smith_multirollout_eval150_1781382734/scripts")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    split = str(args.split)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    helper_source = Path(args.helper_source_dir)
    copy_helper(helper_source / "materialize_swesmith_subset.py", output_dir / "scripts" / "materialize_swesmith_subset.py")
    copy_helper(helper_source / "run_swesmith_eval_with_prune.py", output_dir / "scripts" / "run_swesmith_eval_with_prune.py")
    route_ids: list[str] = []
    routes_manifest: list[dict[str, Any]] = []
    reference_ids: list[str] | None = None
    for route in ROUTES:
        for rollout_idx in route_rollouts_for_profile(args.profile, route, int(args.rollouts)):
            route_id = f"{route['base_route_id']}_r{rollout_idx}"
            route_ids.append(route_id)
            collect_dir = source_root / split / route["base_route_id"] / f"rollout_{rollout_idx}" / "collect"
            pred_src = collect_dir / "models" / route["model_slug"] / "predictions.jsonl"
            ids_src = collect_dir / "instance_ids.txt"
            manifest_src = collect_dir / "manifest.json"
            if not pred_src.exists() or not ids_src.exists() or not manifest_src.exists():
                raise FileNotFoundError(f"Missing artifacts for {route_id} under {collect_dir}")
            ids = read_ids(ids_src)
            if reference_ids is None:
                reference_ids = ids
                ids_dst = output_dir / "ids" / f"{split}.txt"
                ids_dst.parent.mkdir(parents=True, exist_ok=True)
                ids_dst.write_text("".join(instance_id + "\n" for instance_id in ids))
            elif ids != reference_ids:
                raise ValueError(f"ID mismatch for {route_id}: {ids_src}")
            pred_dst = output_dir / "predictions" / split / route_id / "predictions.jsonl"
            pred_summary = convert_predictions(pred_src, pred_dst, route["model_name_or_path"])
            collect_manifest = read_json(manifest_src)
            model_summary = (collect_manifest.get("model_summaries") or {}).get(route["model_slug"], {})
            routes_manifest.append({"route_id": route_id, "base_route_id": route["base_route_id"], "rollout": rollout_idx, "model_name_or_path": route["model_name_or_path"], "predictions_path": str(pred_dst.relative_to(output_dir)), "source_collect_dir": str(collect_dir), "mean_proxy_reward": model_summary.get("mean_proxy_reward"), "proxy_success_rate": model_summary.get("proxy_success_rate"), "mean_output_tokens": model_summary.get("mean_output_tokens"), "n_format_failures": model_summary.get("n_format_failures"), "n_request_errors": model_summary.get("n_request_errors"), **pred_summary})
    if reference_ids is None:
        raise ValueError("No routes exported")
    write_runner(output_dir, split, route_ids)
    write_readme(output_dir)
    manifest = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "format": "SWE-Smith predictions; rows include instance_id, patch, model_patch, model_name_or_path", "source_root": str(source_root), "split": split, "ids_path": f"ids/{split}.txt", "n_ids": len(reference_ids), "n_rollouts": int(args.rollouts), "profile": args.profile, "routes": routes_manifest}
    write_json(output_dir / "manifest.json", manifest)
    if args.make_tarball:
        tar_path = output_dir.with_suffix(".tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(output_dir, arcname=output_dir.name)
        print(f"Wrote tarball: {tar_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SWE-Smith SEARCH/REPLACE text -> unified diffs, WITHOUT importing the training stack.

Identical logic to openrouter_sweep/convert_text_to_patches.py (make_git_diff), but that module imports
pipelinerl.swe.utils.repair_utils, whose module-level imports pull in pipelinerl.rollouts -> transformers ->
triton, which crashes on a CPU node ("0 active drivers"). Here only the needed definitions are loaded from
repair_utils' source: apply_edits_to_files, generate_unified_diff and their private helpers. Rewrites
model_patch in place in every *.jsonl under --predictions-dir, keeping the raw text in raw_output.
"""
from __future__ import annotations
import argparse, ast, difflib, json, logging, re, typing
from pathlib import Path

from pipelinerl.swe.scripts.repair_eval_utils import extract_search_replace_edits

_SRC = Path(__file__).resolve().parents[2] / "utils" / "repair_utils.py"
_WANT = {"apply_edits_to_files", "generate_unified_diff", "_find_unique_normalized_block",
         "_reindent_replacement", "FormatError"}
_ns = {"difflib": difflib, "re": re, "logging": logging, "logger": logging.getLogger("repair_utils"),
       **{k: getattr(typing, k) for k in ("Dict", "List", "Tuple", "Optional", "Any")}}
_tree = ast.parse(_SRC.read_text())
_nodes = [n for n in _tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in _WANT]
missing = _WANT - {n.name for n in _nodes}
if missing:
    raise SystemExit(f"definitions not found in {_SRC}: {missing}")
exec(compile(ast.Module(body=_nodes, type_ignores=[]), str(_SRC), "exec"), _ns)
apply_edits_to_files, generate_unified_diff = _ns["apply_edits_to_files"], _ns["generate_unified_diff"]


def make_git_diff(file_contents: dict[str, str], raw_text: str) -> str:
    edits = extract_search_replace_edits(raw_text)
    if not edits:
        return ""
    try:
        new_contents = apply_edits_to_files(file_contents, edits, silent=True)
    except Exception:
        return ""
    parts = []
    for path, new_code in new_contents.items():
        old_code = file_contents.get(path, "")
        if old_code == new_code:
            continue
        hunks = generate_unified_diff(old_code, new_code)
        if not hunks:
            continue
        parts.append(f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n{hunks}")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions-dir", required=True)
    ap.add_argument("--dataset-path", default="/mnt/llmd/data/swe_smith_bugged_context/ds_train")
    a = ap.parse_args()
    from datasets import load_from_disk
    lookup = {}
    for row in load_from_disk(a.dataset_path):
        iid = row.get("instance_id") or row.get("id"); fc = row.get("gold_file_contents") or row.get("file_contents")
        if iid and fc:
            if isinstance(fc, str):
                try:
                    fc = json.loads(fc)
                except Exception:
                    continue
            lookup[iid] = fc
    print(f"{len(lookup)} instances with file_contents")
    for f in sorted(Path(a.predictions_dir).glob("*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        n_ok = 0
        for r in rows:
            raw = r.get("raw_output", r.get("model_patch", ""))
            r["raw_output"] = raw
            r["model_patch"] = make_git_diff(lookup.get(r["instance_id"], {}), raw) if raw.strip() else ""
            n_ok += bool(r["model_patch"])
        f.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{f.name}: {n_ok}/{len(rows)} converted to a diff")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert a collection's route outputs into unified diffs the SWE-bench evaluator can apply.

The Daytona evaluator wants `model_patch` to be an applicable unified diff. The Verified 5-route
collection does not contain diffs: two routes answer in prose, two in SEARCH/REPLACE blocks, one
in markdown. Feeding those straight through -- which is what the Opus launcher does, since Opus
happened to emit diffs -- scores prose against a test suite and returns ~0% resolved for every
route. That is not a harness failure and not a labelling failure; it is a format mismatch, and it
is the third distinct way SWE-bench evaluation has broken in this project.

The edits are recoverable: `prompt_text` embeds the file contents the model was shown, in the
`### path` + fenced-block form that `_format_file_context` writes, and
`extract_search_replace_edits` already parses every route's output. So apply the edits to the
shown contents and diff the result.

A SEARCH block that does not occur verbatim is dropped and counted, never fuzzy-matched: a
mis-applied edit produces a patch that fails the tests and is indistinguishable from a model that
got the answer wrong, which is exactly how this project once reported a 10% oracle rate on
Verified.
"""
from __future__ import annotations
import argparse, difflib, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from pipelinerl.swe.scripts.repair_eval_utils import extract_search_replace_edits  # noqa: E402

FILE_BLOCK = re.compile(r"### ([^\n]+)\n```[a-zA-Z]*\n(.*?)\n```", re.S)


def files_from_prompt(prompt_text: str) -> dict[str, str]:
    return {p.strip(): c for p, c in FILE_BLOCK.findall(prompt_text or "")}


def to_patch(raw: str, files: dict[str, str]) -> tuple[str, str]:
    """Return (unified_diff, reason_if_empty)."""
    if not (raw or "").strip():
        return "", "empty output"
    if raw.lstrip().startswith(("diff --git", "--- ", "Index:")):
        return raw, ""                       # already a diff (e.g. the Opus collection)
    edits = extract_search_replace_edits(raw)
    if not edits:
        return "", "no parsable edits"
    updated = dict(files)
    applied = dropped = 0
    for e in edits:
        path, search, replace = e.get("file_path"), e.get("search"), e.get("replace")
        if path is None and len(files) == 1:
            path = next(iter(files))         # single-file context: the target is unambiguous
        if path not in updated or search is None:
            dropped += 1
            continue
        if search not in updated[path]:
            dropped += 1                     # verbatim or not at all
            continue
        updated[path] = updated[path].replace(search, replace or "", 1)
        applied += 1
    if not applied:
        return "", f"no edit applied ({dropped} dropped)"
    out = []
    for path, after in updated.items():
        before = files[path]
        if before == after:
            continue
        out.append("".join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}", n=3)))
    return "".join(out), ("" if out else "edits were no-ops")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection-dir", required=True)
    ap.add_argument("--route-idx", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="route")
    a = ap.parse_args()

    import pandas as pd
    paths = sorted(Path(a.collection_dir).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquets in {a.collection_dir}")
    df = pd.concat([pd.read_parquet(p) for p in paths])

    n_ok = 0
    reasons: dict[str, int] = {}
    with open(a.out, "w") as fh:
        for _, row in df.iterrows():
            iid = str(row.get("problem_id") or row.get("instance_id") or "").strip()
            outs = row.get("route_outputs")
            raw = "" if outs is None or len(outs) <= a.route_idx else str(outs[a.route_idx] or "")
            patch, why = to_patch(raw, files_from_prompt(row.get("prompt_text", "")))
            if patch:
                n_ok += 1
            else:
                reasons[why] = reasons.get(why, 0) + 1
            fh.write(json.dumps({"instance_id": iid, "model_patch": patch,
                                 "model": f"{a.model_name}_route{a.route_idx}"}) + "\n")
    print(f"route {a.route_idx}: {n_ok}/{len(df)} rows produced a patch")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"   {v:4d}  {k}")


if __name__ == "__main__":
    main()

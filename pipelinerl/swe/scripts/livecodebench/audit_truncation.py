#!/usr/bin/env python3
"""Is the pool-unsolved set real, or is it truncation at the 4096-token cap?

The multi-draw LCB collection ran with `--max-tokens 4096`. Measured on the resulting
tensor, 62.7% of gpt-oss-120b's failed draws end within 50 tokens of that cap, rising to
73.0% on the 124 problems that no route solves at any depth. A truncated generation has no
extractable program, so it is graded as a failure regardless of whether the model was on
its way to a correct solution.

That matters far beyond bookkeeping. The pool-unsolved set is what drives the project's
largest measured quantity -- those problems consume ~82% of realized spend, and refusing to
start on them is ~63% of all oracle headroom. If a large share of the set is a cap artifact
rather than a capability limit, the stopping result is measuring the collection, not the
models.

  --dump-unsolved   write the pool-unsolved problem ids from a tensor bundle, plus the
                    truncation statistics that motivate the re-collection.
  --compare         join a fresh large-cap collection against those ids and report how many
                    fall, with the implied correction to the impossible fraction.

Nothing here mutates the existing tensors. The re-collection lands in its own directory and
is compared against the old labels rather than overwriting them.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

CAP_TOLERANCE = 50


def load_bundle(tensors_dir: Path):
    t = np.load(tensors_dir / "tensors.npz", allow_pickle=True)
    problems = [json.loads(l) for l in open(tensors_dir / "problems.jsonl") if l.strip()]
    return t, problems


def dump_unsolved(tensors_dir: Path, out_path: Path, old_cap: int) -> None:
    t, problems = load_bundle(tensors_dir)
    slots = list(t["model_slots"])
    solved = t["final_outcome"] & t["valid"]
    valid = t["valid"]
    completion = t["completion_tokens"]
    unsolved = ~solved.any(axis=(1, 2))
    pids = [str(p) for p in t["problem_ids"]]
    split_of = {str(p["problem_id"]): p.get("source_temporal_split") for p in problems}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids = [pid for pid, u in zip(pids, unsolved) if u]
    out_path.write_text("\n".join(ids) + "\n")

    summary = {
        "tensors_dir": str(tensors_dir),
        "n_problems": len(pids),
        "n_pool_unsolved": int(unsolved.sum()),
        "pool_unsolved_fraction": float(unsolved.mean()),
        "by_split": {},
        "old_cap": old_cap,
        "per_route": {},
        "ids_file": str(out_path),
    }
    for split in sorted({v for v in split_of.values() if v}):
        summary["by_split"][split] = sum(1 for pid in ids if split_of.get(pid) == split)
    for i, slot in enumerate(slots):
        failed = valid[:, i, :] & ~solved[:, i, :]
        at_cap = completion[:, i, :] >= old_cap - CAP_TOLERANCE
        uns_valid = valid[unsolved][:, i, :]
        summary["per_route"][slot] = {
            "failed_draws_at_cap": float(at_cap[failed].mean()) if failed.any() else None,
            "unsolved_draws_at_cap": float(at_cap[unsolved][uns_valid].mean()) if uns_valid.any() else None,
            "mean_completion_tokens_on_unsolved": float(completion[unsolved][:, i, :][uns_valid].mean()) if uns_valid.any() else None,
        }
    print(json.dumps(summary, indent=2))
    (out_path.parent / "unsolved_summary.json").write_text(json.dumps(summary, indent=2))


def compare(tensors_dir: Path, recollect_dir: Path, route_label: str, ids_file: Path, new_cap: int) -> None:
    t, problems = load_bundle(tensors_dir)
    solved = t["final_outcome"] & t["valid"]
    pids = [str(p) for p in t["problem_ids"]]
    idx = {pid: i for i, pid in enumerate(pids)}
    unsolved_ids = [l.strip() for l in ids_file.read_text().splitlines() if l.strip()]
    target = set(unsolved_ids)

    files = sorted(glob.glob(str(recollect_dir / f"{route_label}_*_d*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no {route_label}_*_d*.jsonl under {recollect_dir}")
    per_problem: dict[str, list[dict]] = {}
    for path in files:
        for line in open(path):
            if not line.strip():
                continue
            row = json.loads(line)
            pid = str(row["problem_id"])
            if pid in target:
                per_problem.setdefault(pid, []).append(row)

    covered = sorted(per_problem)
    now_solved = [pid for pid in covered if any(bool(r.get("resolved")) for r in per_problem[pid])]
    draws = [r for pid in covered for r in per_problem[pid]]
    at_new_cap = [r for r in draws if (r.get("completion_tokens") or 0) >= new_cap - CAP_TOLERANCE]
    over_old = [r for r in draws if (r.get("completion_tokens") or 0) > 4096]

    n_all = len(pids)
    n_uns = len(unsolved_ids)
    rescued = len(now_solved)
    # Every rescued problem leaves the impossible set, so the corrected fraction is a
    # straightforward re-count. Problems not re-collected stay unsolved by assumption,
    # which makes this a LOWER bound on the correction.
    corrected = (n_uns - rescued) / n_all

    report = {
        "route_label": route_label,
        "new_cap": new_cap,
        "n_problems_total": n_all,
        "n_pool_unsolved_before": n_uns,
        "n_recollected": len(covered),
        "n_draws": len(draws),
        "n_now_solved": rescued,
        "rescue_rate_of_recollected": rescued / len(covered) if covered else None,
        "impossible_fraction_before": n_uns / n_all,
        "impossible_fraction_after": corrected,
        "draws_exceeding_old_4096_cap": len(over_old) / len(draws) if draws else None,
        "draws_at_new_cap": len(at_new_cap) / len(draws) if draws else None,
        "mean_completion_tokens": float(np.mean([r.get("completion_tokens") or 0 for r in draws])) if draws else None,
        "rescued_ids": now_solved,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "rescued_ids"}, indent=2))
    print(f"\nVERDICT: {rescued}/{len(covered)} previously-unsolvable problems fall at a {new_cap}-token cap.")
    print(f"Impossible set {n_uns}/{n_all} ({100*n_uns/n_all:.1f}%) -> {n_uns-rescued}/{n_all} ({100*corrected:.1f}%).")
    if len(covered):
        rate = rescued / len(covered)
        if rate >= 0.20:
            print("=> The labels are cap-limited. Stopping-headroom numbers must be re-derived before use.")
        elif rate >= 0.05:
            print("=> A modest artifact. Report the corrected impossible fraction; conclusions likely survive.")
        else:
            print("=> The pool-unsolved set is real. Every existing stopping result stands as measured.")
    (recollect_dir / "truncation_audit.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tensors-dir", required=True)
    p.add_argument("--dump-unsolved", action="store_true")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--ids-file", required=True)
    p.add_argument("--recollect-dir", default="")
    p.add_argument("--route-label", default="oss120_32k")
    p.add_argument("--old-cap", type=int, default=4096)
    p.add_argument("--new-cap", type=int, default=32768)
    args = p.parse_args()
    if args.dump_unsolved:
        dump_unsolved(Path(args.tensors_dir), Path(args.ids_file), args.old_cap)
    if args.compare:
        if not args.recollect_dir:
            raise SystemExit("--compare needs --recollect-dir")
        compare(Path(args.tensors_dir), Path(args.recollect_dir), args.route_label,
                Path(args.ids_file), args.new_cap)


if __name__ == "__main__":
    main()

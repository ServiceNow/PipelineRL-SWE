#!/usr/bin/env python3
"""Build the MDP tensor bundle for the recollected pool (pool_v2), which the v2 builder cannot.

Three things changed with the recollection and each breaks the old builder:
  * The rungs are no longer {scout, oss20, oss120}: they are effort/price tiers across two
    vendors, named at the command line, and the 4B is a PROBE ONLY -- it contributes prefill
    features, never a draw.
  * Depth is RAGGED on purpose -- 16 draws where a draw costs 0.012c, 3 where it costs 0.5c --
    so K is the max over rungs and `valid` carries the rest.
  * Some rows are infrastructure outcomes, not model outcomes. A dropped response
    (ClientPayloadError, empty output, no provider) is marked INVALID rather than counted as a
    failure; 120 such rows appeared in this collection, concentrated in the longest-generation
    rungs, which is the same bias direction as the three serving artifacts before it
    (PAPER_OUTLINE 3b-lxxxv/lxxxvi/lxxxix).

The split is inherited from the source collection's temporal split so the recollected pool is
comparable to everything measured before it.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np


def _rows(pool: Path, label: str):
    out = {}
    for f in sorted(pool.glob(f"{label}_*_d*.jsonl")):
        m = re.match(rf"{re.escape(label)}_(train|eval)_d(\d+)\.jsonl$", f.name)
        if not m:
            continue
        draw = int(m.group(2))
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                out[(str(r["problem_id"]), draw)] = r
    return out


def _is_infra_failure(r: dict) -> bool:
    """A transport/serving outcome carries no evidence about the model."""
    if str(r.get("full_output") or "").strip():
        return False
    msg = str((r.get("eval_metadata") or {}).get("error_message", ""))
    if r.get("finish_reason") == "length":
        return False                       # a real budget exhaustion IS a model outcome
    return (not r.get("provider")) or "PayloadError" in msg or "ClientConnection" in msg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--source-collection-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rungs", required=True,
                    help="comma-separated label:draws, e.g. oss20lo:16,oss20md:12,oss120md:6")
    ap.add_argument("--calibration-fraction", type=float, default=0.2, help=(
        "fraction of the non-test problems held out for calibration. 0.2, not 0.5: Platt needs "
        "only ~100 points, while the belief head is DATA-STARVED for the tail. Measured on "
        "oss20lo, the spike-and-slab's low-belief calls went from 40% precision at 275 train "
        "problems to 69% at 441 -- identifying hopeless problems from the prompt is limited by "
        "how many hopeless examples it has seen, not by the model class."))
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--split-mode", choices=["temporal", "random"], default="random", help=(
        "temporal keeps the source collection's era split (train = earlier contests). random "
        "shuffles all problems. RANDOM IS THE DEFAULT and is the right primary evaluation: the "
        "object of study is the ROUTER, whose task is to predict whether model m solves problem "
        "x. If m memorised x then q really is high and the router should say so -- contamination "
        "moves the base rates, not the prediction task. The temporal split instead imposes an era "
        "shift that is NOT the phenomenon under study, and it is what made the prompt-only head "
        "read +7 to +12pt optimistic on test: it was calibrated on an easier era. Keep temporal "
        "as a robustness check against the reviewer question 'did it just learn memorisation'."))
    a = ap.parse_args()

    pool, src, out = Path(a.pool_dir), Path(a.source_collection_dir), Path(a.output_dir)
    spec = [(s.split(":")[0], int(s.split(":")[1])) for s in a.rungs.split(",") if s.strip()]
    slots = [s for s, _ in spec]
    K = max(k for _, k in spec)

    split_ids = {}
    for sp, name in [("train", "train"), ("eval", "test")]:
        ids = [str(json.loads(l)["problem_id"]) for l in open(src / f"scout_{sp}.jsonl") if l.strip()]
        split_ids[name] = ids
    pids = split_ids["train"] + split_ids["test"]
    pidx = {p: i for i, p in enumerate(pids)}
    P, M = len(pids), len(slots)

    final = np.zeros((P, M, K), bool); valid = np.zeros((P, M, K), bool)
    ptok = np.zeros((P, M, K), np.float32); ctok = np.zeros((P, M, K), np.float32)
    records, infra, missing = [], 0, 0
    for mi, (label, ndraw) in enumerate(spec):
        got = _rows(pool, label)
        for pi, pid in enumerate(pids):
            for k in range(ndraw):
                r = got.get((pid, k))
                if r is None:
                    missing += 1
                    continue
                if _is_infra_failure(r):
                    infra += 1
                    continue                       # stays invalid: not evidence about the model
                valid[pi, mi, k] = True
                final[pi, mi, k] = bool(r.get("resolved"))
                ptok[pi, mi, k] = float(r.get("prompt_tokens", 0))
                ctok[pi, mi, k] = float(r.get("completion_tokens", 0))
                # Field names match the v2 bundle: the state renderer and the reachable-dataset
                # builder read `full_execution_feedback` and `final_outcome` by those names.
                rc = r.get("result_codes") or []
                npass = sum(1 for c in rc if c is True)
                records.append({"problem_id": pid, "model_slot": label, "draw_index": k,
                                "code": r.get("code", ""),
                                "final_outcome": bool(r.get("resolved")),
                                "weak_verifier_outcome": bool(r.get("public_resolved")),
                                "full_result_codes": rc,
                                "full_execution_feedback": (
                                    f"Full execution: {'PASSED' if r.get('resolved') else 'FAILED'}; "
                                    f"passed={npass}/{len(rc)}" if rc else
                                    f"Full execution: {'PASSED' if r.get('resolved') else 'FAILED'}"),
                                "provider": r.get("provider"),
                                "prompt_tokens": r.get("prompt_tokens", 0),
                                "completion_tokens": r.get("completion_tokens", 0)})
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "tensors.npz", final_outcome=final, execution_outcome=final,
                        weak_verifier_outcome=final, valid=valid, prompt_tokens=ptok,
                        completion_tokens=ctok, problem_ids=np.array(pids),
                        model_slots=np.array(slots), schema_version=np.array(3))
    rng = np.random.default_rng(a.split_seed)
    if a.split_mode == "random":
        allp = list(pids); rng.shuffle(allp)
        ntest = len(split_ids["test"])
        test_ids, rest = allp[:ntest], allp[ntest:]
    else:
        test_ids, rest = list(split_ids["test"]), list(split_ids["train"])
        rng.shuffle(rest)
    ncal = int(round(len(rest) * a.calibration_fraction))
    (out / "split_manifest.json").write_text(json.dumps(
        {"train_problem_ids": sorted(rest[ncal:]), "calibration_problem_ids": sorted(rest[:ncal]),
         "test_problem_ids": sorted(test_ids),
         "split_mode": "random" if a.split_mode == "random" else "source_temporal"}, indent=1))
    # problems.jsonl: downstream probes read difficulty/statement from here, so the bundle has to
    # be self-contained rather than sending them back to the source collection.
    src_prob = {}
    for sp in ("train", "eval"):
        for line in open(src / f"scout_{sp}.jsonl"):
            if line.strip():
                r = json.loads(line)
                src_prob[str(r["problem_id"])] = r
    with open(out / "problems.jsonl", "w") as f:
        for pid in pids:
            r = src_prob.get(pid, {})
            f.write(json.dumps({"problem_id": pid,
                                "platform": r.get("platform", pid.split("_")[0]),
                                "contest_date": r.get("contest_date", ""),
                                "difficulty": r.get("difficulty", ""),
                                "problem_statement": r.get("question_content",
                                                           r.get("problem_statement", ""))}) + "\n")
    with open(out / "draw_records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"{P} problems x {M} rungs x up to {K} draws")
    for mi, (label, ndraw) in enumerate(spec):
        v = valid[:, mi, :ndraw]
        print(f"  {label:<10} {v.sum():>6}/{P*ndraw:<6} valid ({v.sum()/(P*ndraw)*100:>5.1f}%)  "
              f"pass@1 {final[:, mi, :ndraw][v].mean()*100:>5.1f}%")
    print(f"dropped {infra} infrastructure failures (marked invalid), {missing} not yet collected")
    print(f"split ({a.split_mode}): {len(rest)-ncal} train / {ncal} calibration / {len(test_ids)} test")
    print("wrote", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prompts for the history-conditioned probe: the problem plus the attempts that already failed.

The deployed belief is theta_m(x) * kappa / (kappa + n_m): the probe reads the problem once and
every later update uses only the COUNT of failures on the same route. The count carries almost
nothing once theta is known (held-out log-loss 0.36886 -> 0.36875), but among problems whose
observed draws all failed the fraction of tests passed predicts pool solvability -- so the signal,
if any, is in what a failure produced. Re-prefilling the scout on the problem plus the failed
attempt is one cheap forward pass (the problem's prefix is cached), and it lets a failure on one
route inform every route's belief, which the count update cannot.

Examples written, per problem (all splits; fitting uses train, Platt uses calibration, test once):
  * the prompt alone (history "") -- the matched baseline, extracted by the same job;
  * every failed draw on its own:                          "<slot><k>";
  * up to --pairs-per-problem scout-then-other failures:   "scout<k1>+<slot><k2>",
    the order the scout-first protocol actually visits.

Feedback: the failed code and "It passed k of N tests." Test contents never enter a prompt.
(A public-tests-only variant is kept behind --variant public; not used for now.)

Outputs --out-dir/<variant>_shard<i>.jsonl ({problem_id: example id, prompt}) for
pool_activation_probe.py --phase extract, and --out-dir/<variant>_manifest.jsonl with
{example_id, problem_id, history: [[slot, draw_index], ...]} for building labels.
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path

ROUTE_NAME = {"scout": "a small 4B model", "oss20": "a medium 20B model",
              "oss120": "a large 120B model"}


def attempt_text(rec: dict, variant: str, max_code_chars: int) -> str:
    code = rec.get("code") or ""
    if len(code) > max_code_chars:
        half = max_code_chars // 2
        code = code[:half] + "\n# ... (truncated) ...\n" + code[-half:]
    if variant == "public":
        fb = "It passed the public example tests." if rec.get("weak_verifier_outcome") \
            else "It failed the public example tests."
    else:
        res = rec.get("full_result_codes") or []
        fb = f"It passed {sum(bool(x) for x in res)} of {len(res)} tests."
    return (f"A previous attempt by {ROUTE_NAME[rec['model_slot']]} was judged incorrect. "
            f"{fb}\n```python\n{code}\n```")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--base-prompts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", choices=["public", "count"], default="count")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--pairs-per-problem", type=int, default=4)
    ap.add_argument("--max-code-chars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    base = {json.loads(l)["problem_id"]: json.loads(l)["prompt"]
            for l in open(a.base_prompts) if l.strip()}
    recs = defaultdict(dict)
    for l in open(Path(a.tensors_dir) / "draw_records.jsonl"):
        if l.strip():
            r = json.loads(l)
            recs[r["problem_id"]][(r["model_slot"], int(r["draw_index"]))] = r
    rng = random.Random(a.seed)
    rows, manifest = [], []
    missing = 0
    for pid in sorted(recs):
        if pid not in base:
            missing += 1
            continue
        fails = sorted(k for k, r in recs[pid].items() if not r.get("final_outcome"))
        hists = [[]] + [[k] for k in fails]
        scout_f = [k for k in fails if k[0] == "scout"]
        other_f = [k for k in fails if k[0] != "scout"]
        pairs = [[s, o] for s in scout_f for o in other_f]
        rng.shuffle(pairs)
        hists += pairs[: a.pairs_per_problem]
        for h in hists:
            eid = pid + "||" + "+".join(f"{s}{d}" for s, d in h)
            text = base[pid]
            if h:
                text += "\n\n" + "\n\n".join(attempt_text(recs[pid][k], a.variant,
                                                          a.max_code_chars) for k in h)
            rows.append({"problem_id": eid, "prompt": text})
            manifest.append({"example_id": eid, "problem_id": pid,
                             "history": [list(k) for k in h]})
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for i in range(a.shards):
        with open(out / f"{a.variant}_shard{i}.jsonl", "w") as f:
            for r in rows[i::a.shards]:
                f.write(json.dumps(r) + "\n")
    with open(out / f"{a.variant}_manifest.jsonl", "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    n_single = sum(len(m["history"]) == 1 for m in manifest)
    n_pair = sum(len(m["history"]) == 2 for m in manifest)
    print(f"{a.variant}: {len(rows)} examples ({len(base) - missing} prompt-only, {n_single} "
          f"single failures, {n_pair} pairs) in {a.shards} shards -> {out}; "
          f"{missing} problems lacked a base prompt")


if __name__ == "__main__":
    main()

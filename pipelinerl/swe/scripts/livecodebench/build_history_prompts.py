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

Variants, which separate three different signals a history can carry:
  traj   -- trajectory only: which model failed, nothing it produced ("A previous attempt by a
            small 4B model was judged incorrect."). What the policy already knows as counts.
  code   -- trajectory + the failed code, no test result.
  count  -- trajectory + code + "It passed k of N tests.": adds graded verifier feedback, i.e.
            how CLOSE the attempt came. Assumes a verifier that reports counts.
  public -- code + public-test verdict only (kept; not used for now).
Test contents never enter a prompt.

--deep: histories 1..--max-depth failures deep, scout first, INCLUDING repeated failures on the
same route, rendered as a one-line trajectory summary plus the most recent failed attempt's code.
Everything the count decay uses is in the text, so a probe trained on these can learn the decay
itself; the test is whether it can then run with no decay at all. --samples-per-problem histories
per problem.

Outputs --out-dir/<variant>_shard<i>.jsonl ({problem_id: example id, prompt}) for
pool_activation_probe.py --phase extract, and --out-dir/<variant>_manifest.jsonl with
{example_id, problem_id, history: [[slot, draw_index], ...]} for building labels.
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path

COUNT_WORD = {1: "once", 2: "twice"}


def summary_line(hist: list) -> str:
    parts = []
    for slot in ("scout", "oss20", "oss120"):
        n = sum(1 for s, _ in hist if s == slot)
        who = ROUTE_NAME[slot].replace("a ", "the ", 1)
        parts.append(f"{who} has not been tried" if n == 0 else
                     f"{who} has failed {COUNT_WORD.get(n, f'{n} times')}")
    return "So far " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."


# The probe is told WHICH route produced the failed attempt, because the router knows that at
# decision time. Effort tier is part of a route's identity here: the same weights thinking briefly
# and thinking hard are different arms with different prices, and the pilot measured them 8.7
# points apart.
ROUTE_NAME = {"scout": "a small 4B model", "oss20": "a medium 20B model",
              "oss120": "a large 120B model",
              "oss20lo": "a 20B model given a small thinking budget",
              "oss20md": "a 20B model given a moderate thinking budget",
              "oss20hi": "a 20B model given a large thinking budget",
              "oss120lo": "a 120B model given a small thinking budget",
              "oss120md": "a 120B model given a moderate thinking budget",
              "oss120hi": "a 120B model given a large thinking budget",
              "dsv4f": "a large mixture-of-experts model"}


def route_name(slot: str) -> str:
    return ROUTE_NAME.get(slot, f"the {slot} model")


def attempt_text(rec: dict, variant: str, max_code_chars: int) -> str:
    code = rec.get("code") or ""
    if len(code) > max_code_chars:
        half = max_code_chars // 2
        code = code[:half] + "\n# ... (truncated) ...\n" + code[-half:]
    if variant == "traj":
        return f"A previous attempt by {route_name(rec['model_slot'])} was judged incorrect."
    if variant == "code":
        return (f"A previous attempt by {route_name(rec['model_slot'])} was judged incorrect.\n"
                f"```python\n{code}\n```")
    if variant == "public":
        fb = "It passed the public example tests." if rec.get("weak_verifier_outcome") \
            else "It failed the public example tests."
    else:
        res = rec.get("full_result_codes") or []
        fb = f"It passed {sum(bool(x) for x in res)} of {len(res)} tests."
    return (f"A previous attempt by {route_name(rec['model_slot'])} was judged incorrect. "
            f"{fb}\n```python\n{code}\n```")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--base-prompts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", choices=["traj", "code", "count", "public", "deep"], default="count")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--pairs-per-problem", type=int, default=4)
    ap.add_argument("--max-code-chars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--samples-per-problem", type=int, default=12)
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
        if a.deep:
            by = {s_: [k for k in fails if k[0] == s_] for s_ in ("scout", "oss20", "oss120")}
            hists, seen = [[]], set()
            for _ in range(a.samples_per_problem * 3):
                if len(hists) > a.samples_per_problem:
                    break
                pools = {s_: rng.sample(v, len(v)) for s_, v in by.items()}
                depth = rng.randint(1, a.max_depth)
                h = []
                if pools["scout"]:
                    h.append(pools["scout"].pop())
                while len(h) < depth and any(pools.values()):
                    s_ = rng.choice([x for x in pools if pools[x]])
                    h.append(pools[s_].pop())
                key = tuple(h)
                if h and key not in seen:
                    seen.add(key); hists.append(h)
            for h in hists:
                eid = pid + "||" + "+".join(f"{s}{d}" for s, d in h)
                text = base[pid]
                if h:
                    last = recs[pid][h[-1]]
                    text += ("\n\n" + summary_line(h) + " The most recent failed attempt:\n"
                             + f"```python\n{(last.get('code') or '')[:a.max_code_chars]}\n```")
                rows.append({"problem_id": eid, "prompt": text})
                manifest.append({"example_id": eid, "problem_id": pid,
                                 "history": [list(k) for k in h]})
            continue
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

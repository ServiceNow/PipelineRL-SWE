#!/usr/bin/env python3
"""Prompts for EVERY state the policy can reach on the test split, for the deep-history belief.

The deep-history probe reads the problem, a one-line trajectory summary and the most recent failed
attempt, and outputs each route's belief with no decay at all (build_history_prompts.py --deep).
To run it inside the replay, every reachable state needs its own prefill, because the text depends
on the failure counts AND on which stored draw failed last.

Reachable states are enumerable: the replay draws route m's draws in the order
`rng.permutation(K)` (one permutation per problem x ordering x route, from np.random.default_rng
(seed)), skipping invalid draws, and an episode only survives while every draw so far has failed.
So for each (problem, seed, ordering) the reachable states are the count vectors n with
n_m <= (leading failures of route m in that ordering), and the last failure is the n_r-th effective
draw of whichever route was drawn last. Keys are deduplicated across orderings and seeds.

Writes <out-dir>/<variant>_shard<i>.jsonl for pool_activation_probe.py and a manifest with
{example_id, problem_id, counts, last: [slot, draw]}; example ids are
"<pid>||S<n0>-<n1>-<n2>|<slot><draw>" (and "<pid>||" for the no-history state).
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np

from pipelinerl.swe.scripts.livecodebench.build_history_prompts import summary_line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--base-prompts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", default="state")
    ap.add_argument("--seeds", default="0", help="comma-separated replay seeds to cover")
    ap.add_argument("--num-orderings", type=int, default=5)
    ap.add_argument("--split", default="test")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--max-code-chars", type=int, default=6000)
    ap.add_argument("--count-only", action="store_true", help="just report how many states")
    a = ap.parse_args()

    t = np.load(Path(a.tensors_dir) / "tensors.npz", allow_pickle=True)
    ok = (t["final_outcome"] & t["valid"]).astype(bool); valid = t["valid"].astype(bool)
    pids = [str(p) for p in t["problem_ids"]]; slots = [str(s) for s in t["model_slots"]]
    P, M, K = ok.shape
    man = json.loads((Path(a.tensors_dir) / "split_manifest.json").read_text())
    keep = {str(x) for x in man[f"{a.split}_problem_ids"]}
    base = {json.loads(l)["problem_id"]: json.loads(l)["prompt"]
            for l in open(a.base_prompts) if l.strip()}
    recs = {}
    for l in open(Path(a.tensors_dir) / "draw_records.jsonl"):
        if l.strip():
            r = json.loads(l); recs[(r["problem_id"], r["model_slot"], int(r["draw_index"]))] = r

    # states[pid] = {(counts, last_slot, last_draw)}; reproduce the replay's orderings exactly
    states: dict[str, set] = {p: set() for p in pids if p in keep}
    for seed in [int(x) for x in a.seeds.split(",") if x.strip()]:
        rng = np.random.default_rng(seed)
        orderings = np.array([[[rng.permutation(K) for _ in range(M)]
                               for _ in range(a.num_orderings)] for _ in range(P)])
        for pi, pid in enumerate(pids):
            if pid not in keep:
                continue
            for oi in range(a.num_orderings):
                eff = [[int(d) for d in orderings[pi, oi, m] if valid[pi, m, d]] for m in range(M)]
                lead = []                      # leading failures per route: the episode dies later
                for m in range(M):
                    n = 0
                    while n < len(eff[m]) and not ok[pi, m, eff[m][n]]:
                        n += 1
                    lead.append(n)
                for counts in itertools.product(*[range(n + 1) for n in lead]):
                    if sum(counts) == 0:
                        continue
                    for r in range(M):        # whichever route was drawn last
                        if counts[r] >= 1:
                            states[pid].add((counts, slots[r], eff[r][counts[r] - 1]))
    n_states = sum(len(v) for v in states.values())
    print(f"{len(states)} {a.split} problems, {n_states} distinct states "
          f"({n_states / max(1, len(states)):.0f} per problem) over seeds {a.seeds} "
          f"x {a.num_orderings} orderings")
    if a.count_only:
        return

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows, manifest = [], []
    for pid in sorted(states):
        rows.append({"problem_id": pid + "||", "prompt": base[pid]})
        manifest.append({"example_id": pid + "||", "problem_id": pid,
                         "counts": [0] * M, "last": None})
        for counts, slot, draw in sorted(states[pid]):
            hist = [(s, 0) for s, n in zip(slots, counts) for _ in range(n)]   # summary needs counts
            code = (recs[(pid, slot, draw)].get("code") or "")[: a.max_code_chars]
            eid = f"{pid}||S{'-'.join(map(str, counts))}|{slot}{draw}"
            rows.append({"problem_id": eid,
                         "prompt": base[pid] + "\n\n" + summary_line(hist)
                         + " The most recent failed attempt:\n" + f"```python\n{code}\n```"})
            manifest.append({"example_id": eid, "problem_id": pid, "counts": list(counts),
                             "last": [slot, draw]})
    for i in range(a.shards):
        with open(out / f"{a.variant}_shard{i}.jsonl", "w") as f:
            for r in rows[i::a.shards]:
                f.write(json.dumps(r) + "\n")
    with open(out / f"{a.variant}_manifest.jsonl", "w") as f:
        for m_ in manifest:
            f.write(json.dumps(m_) + "\n")
    print(f"wrote {len(rows)} prompts in {a.shards} shards -> {out}")


if __name__ == "__main__":
    main()

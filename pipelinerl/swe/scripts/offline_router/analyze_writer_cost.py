#!/usr/bin/env python3
"""Cost view of test-writer choice (pilot data): can a cheap writer be used when it suffices?

Accuracy = expected correctness of the submitted pool patch (submit among patches the chosen test
passes; ties uniform; no usable test -> uniform among all applied patches). Cost = the writers'
real per-instance token cost (logged) at OpenRouter list prices; execution cost excluded here.
Label-free escalation: buy writers cheapest-first, stop at the first test that is INFORMATIVE:
fails on the unpatched repo AND splits the applied patches (accepts some, rejects some).
"""
import argparse, json
from pathlib import Path
import numpy as np

PRICE = {"oss20": (0.018, 0.09), "dsv4f": (0.04704, 0.09408), "qcoder30": (0.07, 0.28),
         "oss120": (0.15, 0.6), "devstral": (0.4, 2.0)}   # $/M in, out (OpenRouter, 2026-09-25)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot-dir", required=True)
    ap.add_argument("--results-root", default="/mnt/llmd/results/exps/aristides/reason")
    a = ap.parse_args()
    O = Path(a.pilot_dir); rng = np.random.default_rng(0)
    runs = json.loads((O / "patch_runs.json").read_text())
    truth = {k: {json.loads(l)["instance_id"]: bool(json.loads(l)["resolved"]) for l in
                 open(Path(a.results_root) / f"opus_verified_daytona_eval_{r}/predictions/predictions_opus_verified.results.jsonl")}
             for k, r in runs.items()}
    cost = {w: {json.loads(l)["instance_id"]: (json.loads(l)["prompt_tokens"] * p[0] + json.loads(l)["completion_tokens"] * p[1]) / 1e6 * 100
                for l in open(O / f"scripts/scripts_{w}.jsonl")} for w, p in PRICE.items()}
    rows = [json.loads(l) for l in open(O / "exec_clean.jsonl")]
    W = sorted(PRICE, key=lambda w: np.median(list(cost[w].values())))

    def accepted(r, w):
        cands = [p for p in runs if r["apply"].get(p)]
        if r["base"].get(w) in (0, None):
            return cands, False
        acc = [p for p in cands if r["patches"].get(p, {}).get(w) == 0]
        informative = 0 < len(acc) < len(cands)
        return (acc or cands), informative

    def score(r, pool):
        return float(np.mean([truth[p][r["instance_id"]] for p in pool]))

    res = {}
    for r in rows:
        iid = r["instance_id"]; cands = [p for p in runs if r["apply"].get(p)]
        if not cands:
            continue
        res.setdefault("no test", []).append((score(r, cands), 0.0))
        per = {}
        for w in W:
            pool, info = accepted(r, w); per[w] = (score(r, pool), cost[w].get(iid, 0.0), info, pool)
            res.setdefault(f"always {w}", []).append(per[w][:2])
        # vote of all valid writers
        valid = [w for w in W if r["base"].get(w) not in (0, None)]
        if valid:
            sc = {p: sum(r["patches"].get(p, {}).get(w) == 0 for w in valid) for p in cands}
            top = max(sc.values()); pool = [p for p in cands if sc[p] == top]
        else:
            pool = cands
        res.setdefault("vote, all 5 writers", []).append((score(r, pool), sum(cost[w].get(iid, 0) for w in W)))
        # label-free cheap-first escalation
        for ladder_name, ladder in [("cheap-first: oss20 -> dsv4f", ["oss20", "dsv4f"]),
                                    ("cheap-first: oss20 -> dsv4f -> oss120", ["oss20", "dsv4f", "oss120"]),
                                    ("cheap-first: all 5 by price", W)]:
            spent = 0.0; chosen = None
            for w in ladder:
                spent += cost[w].get(iid, 0.0)
                if per[w][2]:
                    chosen = w; break
            acc_ = per[chosen][0] if chosen else score(r, cands)
            res.setdefault(ladder_name, []).append((acc_, spent))
        # oracle: cheapest writer whose pick is as good as the best writer's (upper bound, uses labels)
        best = max(per[w][0] for w in W)
        w_or = min((w for w in W if per[w][0] >= best - 1e-9), key=lambda w: cost[w].get(iid, 0))
        res.setdefault("oracle cheapest sufficient writer", []).append((best, cost[w_or].get(iid, 0)))
    print(f"{len(rows)} instances; cost = test-writing tokens only (cents per instance)")
    print(f"   {'policy':<40}{'accuracy':>9}{'writing cost':>14}")
    for k, v in res.items():
        v = np.array(v)
        print(f"   {k:<40}{v[:,0].mean()*100:>8.1f}%{v[:,1].mean():>12.4f}c")


if __name__ == "__main__":
    main()

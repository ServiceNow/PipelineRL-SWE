#!/usr/bin/env python3
"""Compare attempt readers on the SAME attempts: which one picks the correct draw?

Each reader file has one row per attempt with example_id "<pid>||<slot><draw>" and a score
(p_correct for the trained 4B probe, p_yes for asked readers). Only attempts every listed reader
scored are used. Per model: within-problem AUC, across-problem AUC, and pick-1-of-k accuracy
against random and perfect selection.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reader", action="append", required=True, help="name=path.jsonl")
    ap.add_argument("--routes", default="oss20lo,oss20md,dsv4f,oss120md")
    ap.add_argument("--ks", default="2,4")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    readers = {}
    for spec in a.reader:
        name, path = spec.split("=", 1)
        sc = {}
        for line in open(path):
            r = json.loads(line)
            s = r.get("p_yes", r.get("p_correct"))
            if s is not None:
                sc[r["example_id"]] = float(s)
        readers[name] = sc
    # ground truth from any asked-reader file (they carry "correct")
    truth = {}
    for spec in a.reader:
        for line in open(spec.split("=", 1)[1]):
            r = json.loads(line)
            if "correct" in r:
                truth[r["example_id"]] = bool(r["correct"])
    common = set(truth).intersection(*[set(s) for s in readers.values()])
    ks = [int(k) for k in a.ks.split(",")]
    print(f"{len(common)} attempts scored by every reader: {', '.join(readers)}")
    for route in a.routes.split(","):
        by = defaultdict(list)
        for e in common:
            pid, suf = e.split("||")
            if suf.rstrip("0123456789") == route:
                by[pid].append(e)
        probs = {p: v for p, v in by.items() if len(v) >= max(ks)}
        if not probs:
            continue
        y_all = np.array([truth[e] for v in probs.values() for e in v])
        print(f"\n== {route}: {len(probs)} problems, {len(y_all)} attempts, pass@1 {y_all.mean()*100:.1f}%")
        print(f"   {'reader':<14}{'withinAUC':>10}{'acrossAUC':>10}" + "".join(f"{'pick@'+str(k):>9}" for k in ks))
        samples = {k: [(p, rng.choice(len(v), k, replace=False)) for p, v in probs.items() for _ in range(20)]
                   for k in ks}
        for name, sc in readers.items():
            within = []
            for v in probs.values():
                y = np.array([truth[e] for e in v]); s = np.array([sc[e] for e in v])
                if 0 < y.sum() < len(y):
                    within.append(roc_auc_score(y, s))
            across = roc_auc_score(y_all, [sc[e] for v in probs.values() for e in v])
            picks = [np.mean([truth[probs[p][S[np.argmax([sc[probs[p][i]] for i in S])]]] for p, S in samples[k]]) * 100
                     for k in ks]
            print(f"   {name:<14}{np.mean(within):>10.3f}{across:>10.3f}" + "".join(f"{x:>8.1f}%" for x in picks))
        for name, f in [("random", lambda y: y[0]), ("perfect", lambda y: y.max())]:
            picks = [np.mean([f(np.array([truth[probs[p][i]] for i in S])) for p, S in samples[k]]) * 100 for k in ks]
            print(f"   {name:<14}{'':>10}{'':>10}" + "".join(f"{x:>8.1f}%" for x in picks))


if __name__ == "__main__":
    main()

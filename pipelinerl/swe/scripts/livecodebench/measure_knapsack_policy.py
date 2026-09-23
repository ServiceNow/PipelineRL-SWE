#!/usr/bin/env python3
"""Spend a FIXED BUDGET by marginal value per dollar, instead of abstaining on a price threshold.

Motivation. The deployed rule is greedy on Q = p*R - c, which stops as soon as p < c/R. Measured
(PAPER_OUTLINE 3b-lxxxvii) that rule UNDER-SPENDS badly: at R=10c it used 0.091c against the best
fixed plan's 0.237c and lost 1.9pt of accuracy, because beliefs estimated from a few draws cannot
separate 0.95 from 1.0 and so a problem that looks certain buys one draw. Matched on spend it wins,
which says the threshold, not the policy, was wrong.

So parameterise by the budget instead of by the price. The k-th draw of route m on problem i has
marginal success probability q_im (1-q_im)^{k-1} and marginal cost c_m, hence ratio
    r(i, m, k) = q_im (1 - q_im)^{k-1} / c_m
which is decreasing in k, so the separable-concave knapsack is solved exactly by taking draws in
order of r until the budget runs out (its threshold ratio is the Lagrange multiplier 1/R -- same
frontier, but reached by fixing spend rather than price, which is what a deployment actually has).

GLOBAL spends one budget across the whole test set, so an easy problem can subsidise a hard one.
PERPROB gives every problem the same budget, which is what a per-episode cap does. FIXED is the
best single plan for everyone at that budget (the Zero Router, and a stronger form of it than a
cascade). All three are evaluated on HELD-OUT draws, so no policy can exploit noise in its own
beliefs.
"""
from __future__ import annotations
import argparse, glob, itertools, json, os, re
from collections import defaultdict
import numpy as np


def load(pool_dir: str, rungs: dict, clean_from: dict):
    D = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(pool_dir, "*_d*.jsonl")):
        m = re.match(r"(.+?)_(train|eval)_d(\d+)\.jsonl$", os.path.basename(f))
        if not m:
            continue
        lab, _, dr = m.groups()
        if lab in rungs and int(dr) >= clean_from.get(lab, 0):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    D[lab][str(r["problem_id"])].append(bool(r.get("resolved")))
    return D


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", default="/mnt/llmd/results/exps/aristides/reason/pool_pilot_lcb")
    ap.add_argument("--max-draws", type=int, default=6, help="per rung, per problem")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    C = {"oss20lo": 0.011, "oss20md": 0.042, "dsv4f": 0.158, "oss120md": 0.106, "oss120hi": 0.569}
    CLEAN = {"oss20md": 10, "dsv4f": 10, "oss120hi": 10}
    D = load(a.pool_dir, C, CLEAN)
    RUNGS = list(C)
    P = sorted(set.intersection(*[{p for p, v in D[m].items() if len(v) >= 4} for m in RUNGS]))
    rng = np.random.default_rng(a.seed)
    QA, QB = [], []
    for p in P:
        aa, bb = [], []
        for m in RUNGS:
            v = list(D[m][p]); rng.shuffle(v); h = max(1, len(v) // 2)
            aa.append(np.mean(v[:h])); bb.append(np.mean(v[h:]))
        QA.append(aa); QB.append(bb)
    QA, QB = np.array(QA), np.array(QB)          # beliefs / truth
    n, M = QA.shape
    cost = np.array([C[m] for m in RUNGS])
    K = a.max_draws
    print(f"{n} problems, {M} rungs, up to {K} draws each; beliefs from half the draws, "
          f"evaluated on the other half\n")

    # every candidate marginal draw, ranked by believed value per dollar
    cand = [(QA[i, m] * (1 - QA[i, m]) ** k / cost[m], i, m) for i in range(n)
            for m in range(M) for k in range(K)]
    cand.sort(key=lambda x: -x[0])

    def solve_global(budget_total):
        take = np.zeros((n, M), int); spent = 0.0
        for _, i, m in cand:
            if spent + cost[m] > budget_total:
                continue
            take[i, m] += 1; spent += cost[m]
        return take, spent

    def solve_perproblem(budget_each):
        take = np.zeros((n, M), int); spent = np.zeros(n)
        for _, i, m in cand:
            if spent[i] + cost[m] <= budget_each:
                take[i, m] += 1; spent[i] += cost[m]
        return take, spent.sum()

    def solved(take):     # truth = held-out q
        return np.mean(1 - np.prod((1 - QB) ** take, axis=1))

    PLANS = np.array([k for k in itertools.product(*[range(K + 1)] * M) if 0 < sum(k) <= 8])
    PCOST = PLANS @ cost
    print(f"{'budget/problem':>15}{'GLOBAL knapsack':>18}{'per-problem':>14}{'best FIXED plan':>18}"
          f"{'gain vs fixed':>15}")
    for b in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
        tg, sg = solve_global(b * n)
        tp, sp = solve_perproblem(b)
        ok = [j for j in range(len(PLANS)) if PCOST[j] <= b]
        if not ok:
            continue
        bestj = max(ok, key=lambda j: np.mean(1 - np.prod((1 - QB) ** PLANS[j], axis=1)))
        f = np.mean(1 - np.prod((1 - QB) ** PLANS[bestj], axis=1))
        print(f"{b:>14.2f}c{solved(tg)*100:>17.1f}%{solved(tp)*100:>13.1f}%{f*100:>17.1f}%"
              f"{(solved(tg)-f)*100:>+14.1f}pt")
    print("\n(GLOBAL may exceed a per-episode cap on individual problems; per-problem obeys one.)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Batch-budget allocation: is a cheap probe worth 2% of the budget?

Every experiment in this line so far uses a PER-QUERY budget, inherited from RoR.
That is an evaluation convenience: real deployments hold one budget across a batch
of queries, and under a per-query cap abstaining on a doomed query saves only that
query's money. Under a batch budget it directly funds a solvable one, which is the
setting where stopping quality -- measured at 63.5% of headroom -- actually pays.

Two allocation policies over the same batch and the same budget:

  no-probe             Never run the scout. Rank from problem text (AUC 0.5521) and
                       buy expert calls down the ranking until the budget is gone.

  probe + text ranking Run the scout on every query, keep what it solves, then rank
                       the REMAINDER by the same text-only score.

  probe + post ranking Identical, but rank the remainder by the post-scout score
                       (AUC 0.7693).

The probe pays in two separable ways, and conflating them overstates the claim:
it resolves the easy tail outright (worth a lot, and obvious -- a cheap model is a
cheap model), and it upgrades the ordering governing the rest of the spend (the
actual "scout before you route" claim). Only the third-minus-second contrast
isolates the second. The first-to-second gap is the solver value, which no
information argument is needed to predict.

Reported across budget fractions rather than at one point: a tight budget only ever
uses the very top of the ranking, so an AUC gain spread over the middle could leave
the binding regime untouched. That has to be visible rather than averaged away.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCOUT_COST = 0.000553
EXPERT_COST = 0.029725


def _load_preds(path: Path) -> dict[str, tuple[float, bool]]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return {r["problem_id"]: (float(r["p_yes"]), bool(r["resolved"])) for r in rows}


def allocate(scores: np.ndarray, solved: np.ndarray, costs: np.ndarray,
             budget: float, prepaid: float = 0.0,
             already: np.ndarray | None = None) -> tuple[float, float]:
    """Greedy knapsack by score/cost density; returns (accuracy, spend).

    Density rather than raw score because a query already resolved by the probe
    costs nothing more, and cheap queries should outrank equally-promising dear ones.
    """
    n = len(scores)
    got = np.zeros(n, dtype=bool) if already is None else already.copy()
    spend = prepaid
    order = np.argsort(-(scores / np.maximum(costs, 1e-12)))
    for i in order:
        if got[i]:
            continue
        if spend + costs[i] > budget:
            continue
        spend += costs[i]
        got[i] = solved[i]
    return got.mean(), spend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-only-preds", required=True)
    parser.add_argument("--post-scout-preds", required=True)
    parser.add_argument("--scout-eval", required=True,
                        help="scout_eval.jsonl carrying scout_correct per problem")
    parser.add_argument("--fractions", default="0.05,0.1,0.2,0.3,0.5,0.75,1.0",
                        help="Budget as a fraction of buying an expert call for every query")
    args = parser.parse_args()

    io_preds = _load_preds(Path(args.input_only_preds))
    ps_preds = _load_preds(Path(args.post_scout_preds))
    scout = {
        json.loads(line)["problem_id"]: bool(json.loads(line)["scout_correct"])
        for line in open(args.scout_eval) if line.strip()
    }
    ids = sorted(set(io_preds) & set(ps_preds) & set(scout))
    if not ids:
        raise SystemExit("No overlapping problem ids across the three inputs")

    io_score = np.array([io_preds[i][0] for i in ids])
    ps_score = np.array([ps_preds[i][0] for i in ids])
    expert_ok = np.array([io_preds[i][1] for i in ids])
    scout_ok = np.array([scout[i] for i in ids])
    # Labels must agree across the two predictor runs, or the comparison is invalid.
    assert all(io_preds[i][1] == ps_preds[i][1] for i in ids), "label mismatch between runs"

    n = len(ids)
    costs = np.full(n, EXPERT_COST)
    full = n * EXPERT_COST
    probe_fee = n * SCOUT_COST

    print(f"{n} queries | scout solves {scout_ok.mean()*100:.1f}% | "
          f"expert solves {expert_ok.mean()*100:.1f}%")
    print(f"probe fee = {probe_fee/full*100:.2f}% of buying an expert call for every query\n")
    print(f"{'budget':>8} {'no-probe':>10} {'probe+text':>12} {'probe+post':>12} "
          f"{'solver':>9} {'INFO':>8}")

    for frac in [float(x) for x in args.fractions.split(",")]:
        budget = frac * full
        acc_none, _ = allocate(io_score, expert_ok, costs, budget)
        if budget < probe_fee:
            print(f"{frac:8.2f} {acc_none*100:9.2f}%   probe unaffordable")
            continue
        acc_pt, _ = allocate(io_score, expert_ok, costs, budget,
                             prepaid=probe_fee, already=scout_ok)
        acc_pp, _ = allocate(ps_score, expert_ok, costs, budget,
                             prepaid=probe_fee, already=scout_ok)
        print(f"{frac:8.2f} {acc_none*100:9.2f}% {acc_pt*100:11.2f}% {acc_pp*100:11.2f}% "
              f"{(acc_pt-acc_none)*100:+8.2f} {(acc_pp-acc_pt)*100:+7.2f}pt")


if __name__ == "__main__":
    main()

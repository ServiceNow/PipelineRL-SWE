#!/usr/bin/env python3
"""Exact finite-horizon routing with priced, perfect verification.

The policy has no learned action head. At each state it solves the remaining
draw horizon by dynamic programming, using per-problem success priors and either
global or learned generation costs. Only verified failures update beliefs.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from scipy.special import betaln
from pathlib import Path

import numpy as np

from pipelinerl.swe.scripts.livecodebench.mdp_utils import load_split_manifest, split_indices
from pipelinerl.swe.scripts.livecodebench.replay_priced_verification import (
    choose_mix, read_predictions,
)


def spike_slab_mean(pi0: float, a: float, b: float, f: int) -> float:
    """P(next draw succeeds | f verified failures) under a spike-and-slab Beta belief (exact posterior)."""
    if f == 0:
        return (1.0 - pi0) * a / (a + b)
    w = (1.0 - pi0) * np.exp(betaln(a, b + f) - betaln(a, b))
    return (w / (pi0 + w)) * a / (a + b + f)


def build_solver(prior: np.ndarray, costs: np.ndarray, value: float, verify_cost: float,
                 capacities: tuple[int, ...], horizon: int, pseudo: float = 2.0, dist=None):
    """Return cached optimal values/actions for the finite remaining-draw problem.
    dist: optional (M, 3) spike-and-slab params per route; replaces the hyperbolic decay prior*pseudo/(pseudo+f)."""
    def belief(m: int, f: int) -> float:
        if dist is not None:
            return spike_slab_mean(float(dist[m, 0]), float(dist[m, 1]), float(dist[m, 2]), f)
        return float(prior[m]) * pseudo / (pseudo + f)
    @lru_cache(maxsize=None)
    def solve(failures: tuple[int, ...], remaining: tuple[int, ...], h: int) -> float:
        if h <= 0 or not any(remaining):
            return 0.0
        best = 0.0
        for m, n_left in enumerate(remaining):
            if n_left <= 0:
                continue
            q = belief(m, failures[m])
            next_remaining = list(remaining)
            next_remaining[m] -= 1
            next_failures = list(failures)
            next_failures[m] += 1
            continuation = solve(tuple(next_failures), tuple(next_remaining), h - 1)
            verify_delta = -verify_cost + (1.0 - q) * continuation
            action_value = q * value - float(costs[m]) + max(0.0, verify_delta)
            best = max(best, action_value)
        return best

    def action(failures: tuple[int, ...], remaining: tuple[int, ...], h: int):
        best_value, best_route, best_verify = 0.0, None, False
        for m, n_left in enumerate(remaining):
            if n_left <= 0:
                continue
            q = belief(m, failures[m])
            nr, nf = list(remaining), list(failures)
            nr[m] -= 1
            nf[m] += 1
            continuation = solve(tuple(nf), tuple(nr), h - 1)
            verify_delta = -verify_cost + (1.0 - q) * continuation
            action_value = q * value - float(costs[m]) + max(0.0, verify_delta)
            # On a free-check tie, verify. This recovers the ordinary perfect-
            # verifier regime even when a failure has no profitable continuation.
            verify = verify_delta > 1e-15 or (verify_cost == 0.0 and verify_delta >= -1e-15)
            if action_value > best_value + 1e-15:
                best_value, best_route, best_verify = action_value, m, verify
        return best_value, best_route, best_verify

    return solve, action


def run_problem(pi: int, ordering: np.ndarray, valid: np.ndarray, truth: np.ndarray,
                realized: np.ndarray, solve, action, horizon: int,
                verify_cost: float):
    counts = tuple(int(valid[pi, m].sum()) for m in range(valid.shape[1]))
    failures = tuple(0 for _ in counts)
    remaining = counts
    ptr = np.zeros(len(counts), dtype=np.int16)
    correct, spent, checks, draws = 0.0, 0.0, 0, 0
    h = horizon
    while h > 0 and any(remaining):
        objective, route, verify = action(failures, remaining, h)
        if route is None or objective <= 0.0:
            break
        while ptr[route] < ordering.shape[1] and not valid[pi, route, ordering[route, ptr[route]]]:
            ptr[route] += 1
        if ptr[route] >= ordering.shape[1]:
            break
        draw = int(ordering[route, ptr[route]])
        ptr[route] += 1
        rem = list(remaining)
        rem[route] -= 1
        remaining = tuple(rem)
        draws += 1
        spent += float(realized[pi, route, draw])
        h -= 1
        is_correct = bool(truth[pi, route, draw])
        if not verify:
            correct = float(is_correct)
            break
        checks += 1
        spent += verify_cost
        if is_correct:
            correct = 1.0
            break
        fail = list(failures)
        fail[route] += 1
        failures = tuple(fail)
    return np.asarray([correct, spent * 100.0, checks, draws], dtype=float)


def run_split(indices: np.ndarray, orders: np.ndarray, valid: np.ndarray, truth: np.ndarray,
              realized: np.ndarray, prior: np.ndarray, costs: np.ndarray,
              value: float, verify_cost: float, horizon: int, pseudo: float, dist=None) -> np.ndarray:
    out = np.zeros((len(indices), 4), dtype=float)
    for row, pi in enumerate(indices):
        capacities = tuple(int(valid[int(pi), m].sum()) for m in range(valid.shape[1]))
        solve, action = build_solver(prior[int(pi)], costs[int(pi)], value,
                                    verify_cost, capacities, horizon, pseudo,
                                    None if dist is None else dist[int(pi)])
        for ordering in orders[int(pi)]:
            out[row] += run_problem(int(pi), ordering, valid, truth, realized,
                                    solve, action, horizon, verify_cost)
    return out / orders.shape[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-orderings", type=int, default=3)
    ap.add_argument("--value-grid-points", type=int, default=14)
    ap.add_argument("--max-draws", type=int, default=8,
                    help="exact finite horizon; 8 covers observed replay trajectories")
    ap.add_argument("--budget-cents", default="0.05,0.1,0.15")
    ap.add_argument("--v-multipliers", default="0,1,3")
    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dist-preds", default="", help="fit_entry_distribution.py output: adds the 'dist' belief family")
    ap.add_argument("--beliefs", default="content", help="comma list from {content, dist}")
    args = ap.parse_args()

    folder = Path(args.tensors_dir)
    data = np.load(folder / "tensors.npz", allow_pickle=True)
    ids = [str(x) for x in data["problem_ids"]]
    slots = [str(x) for x in data["model_slots"]]
    valid = data["valid"].astype(bool)
    truth = data["execution_outcome"].astype(bool)
    price = {k: float(v) for k, v in (item.split("=", 1) for item in args.prices.split(","))}
    if set(price) != set(slots):
        raise ValueError(f"prices must name {slots}")
    realized = np.stack([(data["prompt_tokens"][:, m] + data["completion_tokens"][:, m])
                         * price[slot] / 1e6 for m, slot in enumerate(slots)], axis=1)
    train, cal, test = split_indices(load_split_manifest(folder / "split_manifest.json", ids), ids)
    global_prior = np.asarray([truth[train, m][valid[train, m]].mean()
                               for m in range(len(slots))])
    global_cost = np.asarray([realized[train, m][valid[train, m]].mean()
                              for m in range(len(slots))])
    prior = read_predictions(folder / "content_preds.jsonl", ids, "p_successes")[:, :len(slots)]
    learned_cost = read_predictions(folder / "cost_preds.jsonl", ids, "expected_costs")[:, :len(slots)]
    costs = {"global": np.broadcast_to(global_cost, learned_cost.shape),
             "learned": learned_cost}
    beliefs = {}
    for name in args.beliefs.split(","):
        if name == "content":
            beliefs["content"] = None
        elif name == "global":
            beliefs["global"] = "global"    # no prefill: each route's TRAIN pass rate for every problem
        elif name == "dist":
            rows = {json.loads(l)["problem_id"]: json.loads(l)["params"] for l in open(args.dist_preds)}
            beliefs["dist"] = np.asarray([rows[i] for i in ids], dtype=float)[:, :len(slots)]
    rng = np.random.default_rng(args.seed)
    orders = np.asarray([[[rng.permutation(valid.shape[2]) for _ in slots]
                          for _ in range(args.num_orderings)] for _ in ids])
    values = np.geomspace(.4 * np.min(global_cost / global_prior),
                          150 * np.max(global_cost / global_prior), args.value_grid_points)
    budgets = [float(x) for x in args.budget_cents.split(",")]
    multipliers = [float(x) for x in args.v_multipliers.split(",")]
    results = []
    for multiplier in multipliers:
        verify_cost = multiplier * global_cost[0]
        for (belief_name, dist), (cost_name, cost_matrix) in [(b, c) for b in beliefs.items() for c in costs.items()]:
            prior_used = np.broadcast_to(global_prior, prior.shape) if isinstance(dist, str) else prior
            dist = None if isinstance(dist, str) else dist
            print(f"v={multiplier:g}x cheapest draw; belief={belief_name}; cost={cost_name}", flush=True)
            cal_grid, test_grid = [], []
            for i, value in enumerate(values):
                cal_grid.append(run_split(cal, orders, valid, truth, realized, prior_used,
                                          cost_matrix, float(value), verify_cost,
                                          args.max_draws, 2.0, dist))
                test_grid.append(run_split(test, orders, valid, truth, realized, prior_used,
                                           cost_matrix, float(value), verify_cost,
                                           args.max_draws, 2.0, dist))
                print(f"  R {i+1}/{len(values)} complete", flush=True)
            for budget in budgets:
                left, right, weight = choose_mix(
                    np.asarray([x[:, 1].mean() for x in cal_grid]),
                    np.asarray([x[:, 0].mean() for x in cal_grid]), budget)
                zeros = np.zeros((len(test), 4), dtype=float)
                out = ((1-weight) * (test_grid[left] if left >= 0 else zeros)
                       + weight * (test_grid[right] if right >= 0 else zeros))
                results.append({
                    "v_multiplier": multiplier, "v_cents": verify_cost*100, "belief": belief_name,
                    "cost": cost_name, "budget_cents": budget,
                    "calibration_mix": [left, right, weight],
                    "test_accuracy": float(out[:, 0].mean()),
                    "test_cost_cents": float(out[:, 1].mean()),
                    "test_checks": float(out[:, 2].mean()),
                    "test_draws": float(out[:, 3].mean()),
                    "test_accuracy_by_problem": out[:, 0].tolist(),
                    "test_cost_cents_by_problem": out[:, 1].tolist(),
                })
                print(f"  budget {budget:g}c -> {out[:,0].mean():.3%} at "
                      f"{out[:,1].mean():.4f}c", flush=True)
    output = {"dataset": str(folder), "slots": slots, "train_n": len(train),
              "cal_n": len(cal), "test_n": len(test),
              "test_problem_ids": [ids[int(i)] for i in test],
              "global_priors": global_prior.tolist(),
              "global_costs_usd": global_cost.tolist(),
              "values_usd": values.tolist(), "max_draws": args.max_draws,
              "num_orderings": args.num_orderings, "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2)+"\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

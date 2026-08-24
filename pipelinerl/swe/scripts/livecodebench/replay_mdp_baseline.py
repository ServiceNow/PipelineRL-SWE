#!/usr/bin/env python3
"""
RoR-faithful offline replay over the LCB multi-draw MDP tensors.

Policy class: at each step, pick argmax_m p_hat(m)/cost(m) where
    p_hat(m) = (S * prior(m) + w_m) / (S + n_m),
w_m = verifier successes (public tests) among n_m draws of model m so far.
Early-stop and submit when any draw passes the verifier.
Final correctness = full-suite `resolved` of the first verifier-passed draw
(or 0 if none passed when budget exhausts).

Baselines: always-X, best-of-K scout, sequential cascade, random-at-matched-budget,
oracle cheapest-success. Budget-B sweep; results averaged over randomized draw orderings.

Usage:
  python replay_mdp_baseline.py --tensors-dir .../mdp_tensors_v1 --output-dir .../mdp_replay_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# Recorded AWS prices ($/M output tokens) — analysis/real_swebench_verified_5route_policy_sim_1780039546_aws_cost
USD_PER_M_TOKENS = {"scout": 0.278, "oss20": 1.299, "oss120": 11.13}
COST_WEIGHTS = {"scout": 1.0, "oss20": 5.0, "oss120": 30.0}


def load_tensors(tensors_dir: Path):
    d = np.load(tensors_dir / "tensors.npz", allow_pickle=True)
    return d


def replay_problem(
    R_pub: np.ndarray,        # (M,K) public outcomes available to the policy
    R_full: np.ndarray,       # (M,K) ground-truth outcomes
    C: np.ndarray,            # (M,) cost per draw
    orderings: np.ndarray,    # (M,K) permutation: draw index consumed at step t for model m
    budget: float,
    prior: np.ndarray,        # (M,)
    S: float,
    policy: str = "greedy",
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Run one episode under a fixed draw-ordering. Returns spend + correctness."""
    M = len(C)
    ptr = np.zeros(M, dtype=int)          # next draw index per model (into ordering)
    n = np.zeros(M, dtype=int)
    w = np.zeros(M, dtype=float)
    spent = 0.0
    submitted_idx: tuple[int, int] | None = None   # (model, draw) of first public pass

    while True:
        # early stop: verifier already accepted a draw
        if submitted_idx is not None:
            break
        # affordable candidate models with remaining draws
        avail = [m for m in range(M) if ptr[m] < R_pub.shape[1] and spent + C[m] <= budget]
        if not avail:
            break
        if policy == "greedy":
            values = [( (S * prior[m] + w[m]) / (S + n[m]) ) / C[m] for m in avail]
            m_star = avail[int(np.argmax(values))]
        elif policy == "ucb":
            total = sum(n)
            bonus = [np.sqrt(np.log(total + 1) / max(1, n[m])) for m in avail]
            values = [((S * prior[m] + w[m]) / (S + n[m])) / C[m] + 0.1 * bonus[i]
                      for i, m in enumerate(avail)]
            m_star = avail[int(np.argmax(values))]
        else:
            raise ValueError(policy)

        k = int(orderings[m_star][ptr[m_star]])
        ptr[m_star] += 1
        n[m_star] += 1
        spent += C[m_star]
        pub = bool(R_pub[m_star][k])
        w[m_star] += float(pub)
        if pub and submitted_idx is None:
            submitted_idx = (m_star, k)

    if submitted_idx is not None:
        m, k = submitted_idx
        correct = bool(R_full[m][k])
    else:
        correct = False
    return {"spend": spent, "correct": correct}


def run_fixed_policy(
    R_pub: np.ndarray, R_full: np.ndarray, C: np.ndarray, orderings: np.ndarray,
    kind: str,
) -> dict[str, Any]:
    """Fixed-sequence policies with early stop on verifier pass."""
    plans = {
        "always_scout": [0], "always_oss20": [1], "always_oss120": [2],
        "cascade": [0, 1, 2],
    }
    models = plans[kind]
    spent = 0.0
    counts = {m: 0 for m in range(MODEL_COUNT)}
    first_pass = None
    for m in models * 10:  # up to 10 draws per model in sequence
        if counts[m] >= R_pub.shape[1]:
            continue
        k = int(orderings[m][counts[m]])
        counts[m] += 1
        spent += C[m]
        if R_pub[m][k]:
            first_pass = (m, k)
            break  # submit first verifier-passed draw
    if first_pass is not None:
        m, k = first_pass
        correct = bool(R_full[m][k])
    else:
        correct = False
    return {"spend": spent, "correct": correct}


MODEL_COUNT = 3

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensors-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-orderings", type=int, default=20)
    parser.add_argument("--pseudo-count", type=float, default=2.0)
    parser.add_argument("--cost-mode", choices=["usd", "weights"], default="weights")
    parser.add_argument("--verifier", choices=["public", "full"], default="public",
                        help="'full' = oracle verifier (diagnostic upper bound, not deployable)")
    args = parser.parse_args()

    d = load_tensors(Path(args.tensors_dir))
    R_full = d["resolved"].astype(bool)      # (P,M,K)
    R_pub = d["public"].astype(bool)
    if args.verifier == "full":
        print("WARNING: --verifier full is an oracle diagnostic, not deployable")
        R_pub = R_full.copy()
    V = d["valid"]
    slots = [str(s) for s in d["model_slots"]]
    M = len(slots)

    C = (
        np.array([USD_PER_M_TOKENS[s] for s in slots])
        if args.cost_mode == "usd"
        else np.array([COST_WEIGHTS[s] for s in slots])
    )

    rng = np.random.default_rng(0)
    P = R_full.shape[0]
    K = R_full.shape[2]

    # calibration/test halves (by problem) for priors — RoR protocol
    perm = rng.permutation(P)
    cal_idx, test_idx = perm[: P // 2], perm[P // 2:]

    # offline priors: public-test success rate on calibration half
    prior = np.zeros(M)
    for m in range(M):
        vals = []
        for pi in cal_idx:
            v = V[pi, m]
            if v.any():
                vals.extend(R_pub[pi, m][v].tolist())
        prior[m] = float(np.mean(vals)) if vals else 0.5
    print(f"priors (public, cal half): {dict(zip(slots, prior.round(3)))}")

    # pre-generate orderings: (P, num_orderings, M, K)
    orders = np.array([
        [[rng.permutation(K) for _ in range(M)] for _ in range(args.num_orderings)]
        for _ in range(test_idx.size)
    ])  # (N_test, O, M, K)

    c_max = C.max()
    budgets = sorted({
        round(x, 2)
        for x in np.linspace(0.25 * c_max, 3.0 * c_max, 12).tolist()
        + [C[0], C.sum(), 2 * C[2]]
    })

    rows_out = []

    def evaluate(policy_name: str, budget: float, runner) -> dict[str, Any]:
        corrs, spends = [], []
        for ni in range(test_idx.size):
            pi = test_idx[ni]
            for oi in range(args.num_orderings):
                out = runner(R_pub[pi], R_full[pi], C, orders[ni, oi], budget)
                corrs.append(out["correct"])
                spends.append(out["spend"])
        row = {
            "policy": policy_name, "budget": budget,
            "correctness": float(np.mean(corrs)),
            "mean_cost": float(np.mean(spends)),
            "n_episodes": len(corrs),
        }
        rows_out.append(row)
        print(f"{policy_name:28s} B={budget:8.2f} corr={row['correctness']:.4f} cost={row['mean_cost']:.2f}")
        return row

    for B in budgets:
        for pol in ["greedy", "ucb"]:
            evaluate(
                f"RoR_{pol}", B,
                lambda rp, rf, c, o, b: replay_problem(rp, rf, c, o, b, prior, args.pseudo_count, pol),
            )
        evaluate("cascade", B, lambda rp, rf, c, o, b: run_fixed_policy(rp, rf, c, o, "cascade"))
        evaluate("always_scout", B, lambda rp, rf, c, o, b: run_fixed_policy(rp, rf, c, o, "always_scout"))
        evaluate("always_oss20", B, lambda rp, rf, c, o, b: run_fixed_policy(rp, rf, c, o, "always_oss20"))
        evaluate("always_oss120", B, lambda rp, rf, c, o, b: run_fixed_policy(rp, rf, c, o, "always_oss120"))
        # random allocation at matched budget is computed downstream from RoR route mixes

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "replay_results.json", "w") as f:
        json.dump({
            "slots": slots, "costs": dict(zip(slots, C.tolist())),
            "priors": dict(zip(slots, prior.tolist())),
            "pseudo_count": args.pseudo_count, "cost_mode": args.cost_mode,
            "budgets": budgets, "results": rows_out,
        }, f, indent=2)
    print(f"\nwrote {out/'replay_results.json'}")


if __name__ == "__main__":
    main()

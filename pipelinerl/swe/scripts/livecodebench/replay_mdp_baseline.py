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
    tau_abstain: float | None = None,
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
        # abstain arm: give up when estimated P(any model succeeds) <= tau
        if tau_abstain is not None:
            p_each = np.array([(S * prior[m] + w[m]) / (S + n[m]) for m in range(M)])
            p_any = 1.0 - np.prod(1.0 - p_each)
            if p_any <= tau_abstain:
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
    parser.add_argument("--content-preds", default=None,
                        help="adapted eval_predictions.jsonl with p_successes per problem "
                             "(static content prior); enables the content cells")
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

    # pre-generate orderings: (P, num_orderings, M, K), indexed by global problem id
    orders = np.array([
        [[rng.permutation(K) for _ in range(M)] for _ in range(args.num_orderings)]
        for _ in range(P)
    ])  # (P, O, M, K)

    c_max = C.max()
    budgets = sorted({
        round(x, 2)
        for x in np.linspace(0.25 * c_max, 3.0 * c_max, 12).tolist()
        + [C[0], C.sum(), 2 * C[2]]
    })

    rows_out = []

    # static content prior: per-problem p vector from the trained router (if provided)
    content = None
    if args.content_preds:
        content = {}
        with open(args.content_preds) as f:
            for line in f:
                r = json.loads(line)
                content[str(r["problem_id"])] = np.array(r["p_successes"], dtype=float)
        print(f"loaded content priors for {len(content)} problems")

    def episode(pi_local, oi_local, B, use_content, tau):
        prior_vec = (
            content.get(str(d["problem_ids"][pi_local]), prior)
            if use_content else prior
        )
        return replay_problem(
            R_pub[pi_local], R_full[pi_local], C, orders[pi_local, oi_local], B,
            prior_vec, S=args.pseudo_count, policy="greedy", tau_abstain=tau)

    def run_split(split_idx, B, use_content, tau):
        idx = cal_idx if split_idx == "cal" else test_idx
        corrs, spends = [], []
        for pi_local in idx:
            for oi_local in range(args.num_orderings):
                out = episode(int(pi_local), oi_local, B, use_content, tau)
                corrs.append(out["correct"]); spends.append(out["spend"])
        return float(np.mean(corrs)), float(np.mean(spends))

    def evaluate(cell_name, B, use_content, tau):
        corr_t, spend_t = run_split("test", B, use_content, tau)
        row = {"cell": cell_name, "budget": B, "tau": tau,
               "correctness": corr_t, "mean_cost": spend_t,
               "n_episodes": len(test_idx) * args.num_orderings}
        rows_out.append(row)
        print(f"{cell_name:24s} B={B:7.2f} tau={str(tau):6s} corr={corr_t:.4f} cost={spend_t:.2f}")

    tau_grid = [round(x, 2) for x in np.linspace(0.1, 0.9, 17)]

    # abstention rule: retain >= RETAIN_FRAC of no-abstain calibration correctness,
    # then minimize spend. (Maximizing correctness alone never abstains.)
    RETAIN_FRAC = 0.95

    for B in budgets:
        # references without abstention
        evaluate("counts", B, False, None)
        if content is not None:
            evaluate("content", B, True, None)

        # counts + abstain
        base_corr_c, _ = run_split("cal", B, False, None)
        cand = []
        for tau in tau_grid:
            corr_c, spend_c = run_split("cal", B, False, tau)
            cand.append((tau, corr_c, spend_c))
        ok = [c for c in cand if c[1] >= RETAIN_FRAC * base_corr_c]
        tau_star = min(ok, key=lambda c: c[2])[0] if ok else None
        evaluate("counts_abstain", B, False, tau_star)

        # content + abstain
        if content is not None:
            base_corr_cc, _ = run_split("cal", B, True, None)
            cand = []
            for tau in tau_grid:
                corr_c, spend_c = run_split("cal", B, True, tau)
                cand.append((tau, corr_c, spend_c))
            ok = [c for c in cand if c[1] >= RETAIN_FRAC * base_corr_cc]
            tau_star_c = min(ok, key=lambda c: c[2])[0] if ok else None
            evaluate("content_abstain", B, True, tau_star_c)

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

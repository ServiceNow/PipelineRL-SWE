#!/usr/bin/env python3
"""Does per-problem abstention beat giving up on everything at a constant depth?

The project's headline mechanism is that query-conditioned beliefs plus an explicit stop
action buy the savings: abstention on-versus-off was measured at 8.45% against 51.5%, and
"no policy without a give-up arm finds a cheap ceiling point" is load-bearing.

But a truncated fixed schedule IS a give-up rule -- a constant one. It buys a
problem-independent sequence of (model, draw) calls, stops on success, and quits after k
purchases whatever the problem looks like. It never identifies a hopeless instance, yet the
instances still alive at depth k are overwhelmingly the hopeless ones, so uniform give-up
collects the same waste without predicting anything. On the temporal test split the fixed
schedule reaches the 73.10% ceiling at $0.0792 against the learned abstaining policy's
$0.1359, which is the opposite of what the headline claims.

This compares them directly and cost-matched, with problem-clustered paired bootstraps:

  UNIFORM     greedy marginal-coverage-per-dollar schedule fitted on the 551 train problems,
              executed identically on every test problem, truncated at each length k. No
              model, no beliefs, no per-problem decision.
  LEARNED     the value-rule arms from the existing replay traces, whose abstention is
              per-problem and threshold-free (stop when max_m p_m*R - c_m <= 0).

If uniform truncation matches, the abstention result collapses to choosing a constant and
the central mechanism must be restated. If the learned arm wins, it is the first clean
evidence here that per-problem give-up prediction is worth anything.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

USD_PER_M_TOKENS = {"scout": 0.278, "oss20": 1.299, "oss120": 11.13}


def load_tensors(tensors_dir: Path):
    t = np.load(tensors_dir / "tensors.npz", allow_pickle=True)
    problems = [json.loads(l) for l in open(tensors_dir / "problems.jsonl") if l.strip()]
    split = {str(p["problem_id"]): p.get("source_temporal_split") for p in problems}
    slots = [str(s) for s in t["model_slots"]]
    pids = [str(p) for p in t["problem_ids"]]
    ok = t["final_outcome"] & t["valid"]
    cost = np.zeros_like(t["prompt_tokens"], dtype=float)
    for i, s in enumerate(slots):
        cost[:, i, :] = (t["prompt_tokens"][:, i, :] + t["completion_tokens"][:, i, :]) * USD_PER_M_TOKENS[s] / 1e6
    return pids, slots, ok, t["valid"], cost, split


def fit_schedule(ok, valid, cost, max_len, n_perm, seed):
    """Greedy purchase order over (model, next-draw) by marginal coverage per dollar.

    Fitted on TRAIN only. Cost is the expected marginal spend, i.e. charged only on problems
    still alive, because a schedule pays for a call only until something passes.
    """
    rng = np.random.default_rng(seed)
    P, M, K = ok.shape
    perms = [np.stack([rng.permutation(K) for _ in range(P)]) for _ in range(n_perm)]
    alive = [np.ones(P, bool) for _ in range(n_perm)]
    used = [np.zeros(M, int) for _ in range(n_perm)]
    schedule: list[int] = []
    for _ in range(max_len):
        best, best_density = None, -1.0
        for m in range(M):
            gains, costs = [], []
            for q in range(n_perm):
                j = used[q][m]
                if j >= K:
                    break
                d = perms[q][:, j]
                hit = ok[np.arange(P), m, d] & valid[np.arange(P), m, d]
                c = cost[np.arange(P), m, d] * valid[np.arange(P), m, d]
                gains.append((alive[q] & hit).mean())
                costs.append((alive[q] * c).mean())
            if not gains:
                continue
            g, c = float(np.mean(gains)), float(np.mean(costs))
            if c <= 0:
                continue
            if g / c > best_density:
                best, best_density = m, g / c
        if best is None:
            break
        schedule.append(best)
        for q in range(n_perm):
            j = used[q][best]
            d = perms[q][:, j]
            hit = ok[np.arange(len(d)), best, d] & valid[np.arange(len(d)), best, d]
            alive[q] &= ~hit
            used[q][best] += 1
    return schedule


def run_schedule(ok, valid, cost, schedule, n_perm, seed):
    """Execute every truncation of the schedule. Returns per-(k, problem) cost and correct."""
    rng = np.random.default_rng(seed)
    P, M, K = ok.shape
    out_cost = np.zeros((len(schedule) + 1, P))
    out_ok = np.zeros((len(schedule) + 1, P), float)
    for _ in range(n_perm):
        perm = np.stack([rng.permutation(K) for _ in range(P)])
        alive = np.ones(P, bool)
        used = np.zeros(M, int)
        spend = np.zeros(P)
        solved = np.zeros(P, bool)
        for step, m in enumerate(schedule, 1):
            j = used[m]
            if j < K:
                d = perm[:, j]
                idx = np.arange(P)
                v = valid[idx, m, d]
                spend += alive * cost[idx, m, d] * v
                hit = alive & ok[idx, m, d] & v
                solved |= hit
                alive &= ~hit
                used[m] += 1
            out_cost[step] += spend
            out_ok[step] += solved
    out_cost /= n_perm
    return out_cost, out_ok / n_perm


def learned_points(traces_path: Path, policies: set[str], test_pids: list[str]):
    """Per-problem cost and correctness for each (policy, R), averaged over draw orderings."""
    agg: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with open(traces_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pol = r.get("policy")
            if pol not in policies or r.get("value_of_correct") is None:
                continue
            agg[(pol, float(r["value_of_correct"]))][str(r["problem_id"])].append(
                (float(r["realized_spend"]), bool(r["correct"]), bool(r.get("abstained")))
            )
    idx = {p: i for i, p in enumerate(test_pids)}
    points = []
    for (pol, R), per_p in agg.items():
        if set(per_p) != set(test_pids):
            continue
        c = np.zeros(len(test_pids))
        a = np.zeros(len(test_pids))
        ab = np.zeros(len(test_pids))
        for p, vals in per_p.items():
            i = idx[p]
            c[i] = np.mean([v[0] for v in vals])
            a[i] = np.mean([v[1] for v in vals])
            ab[i] = np.mean([v[2] for v in vals])
        points.append({"policy": pol, "R": R, "cost": c, "acc": a, "abstain": float(ab.mean())})
    return points


def cheapest_at(points, target):
    ok_pts = [p for p in points if p["acc"].mean() + 1e-12 >= target]
    return min(ok_pts, key=lambda p: p["cost"].mean()) if ok_pts else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--traces", required=True, help="episode_traces.jsonl from a replay with value arms")
    ap.add_argument("--policies", default="sequential_value,sequential_decay_value")
    ap.add_argument("--max-len", type=int, default=24)
    ap.add_argument("--fit-perms", type=int, default=10)
    ap.add_argument("--eval-perms", type=int, default=30)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", default="0.55,0.60,0.65,0.68,0.70,0.730")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pids, slots, ok, valid, cost, split = load_tensors(Path(args.tensors_dir))
    idx = {p: i for i, p in enumerate(pids)}
    test_pids = sorted({str(json.loads(l)["problem_id"]) for l in open(args.traces) if l.strip()})
    train_rows = np.array([idx[p] for p in pids if split.get(p) == "train" and p not in set(test_pids)])
    test_rows = np.array([idx[p] for p in test_pids])
    print(f"train {len(train_rows)} problems, test {len(test_rows)} problems, routes {slots}")

    schedule = fit_schedule(ok[train_rows], valid[train_rows], cost[train_rows],
                            args.max_len, args.fit_perms, args.seed)
    print("fitted schedule:", " -> ".join(slots[m] for m in schedule))

    sc_cost, sc_acc = run_schedule(ok[test_rows], valid[test_rows], cost[test_rows],
                                   schedule, args.eval_perms, args.seed + 1)
    uniform = [{"policy": f"uniform_truncate_k{k}", "R": None, "cost": sc_cost[k],
                "acc": sc_acc[k], "abstain": 0.0, "k": k} for k in range(1, len(schedule) + 1)]
    print(f"\n{'k':>3} {'schedule prefix':<46} {'acc':>7} {'cost':>10}")
    for u in uniform:
        k = u["k"]
        print(f'{k:3d} {" ".join(slots[m] for m in schedule[:k])[:46]:<46} '
              f'{100*u["acc"].mean():6.2f}% {u["cost"].mean():10.5f}')

    learned = learned_points(Path(args.traces), set(args.policies.split(",")), test_pids)
    print(f"\n{len(learned)} learned operating points across {args.policies}")

    rng = np.random.default_rng(args.seed + 2)
    n = len(test_rows)
    targets = [float(x) for x in args.targets.split(",")]
    report = {"schedule": [slots[m] for m in schedule], "n_test": n, "comparisons": []}
    print(f"\n{'target':>7} {'uniform $':>10} {'learned $':>10} {'learned-abst':>12} "
          f"{'uniform saving':>15} {'95% CI':>20} {'P(uniform cheaper)':>19}")
    for tgt in targets:
        bu, bl = cheapest_at(uniform, tgt), cheapest_at(learned, tgt)
        if bu is None or bl is None:
            print(f"{100*tgt:6.1f}%  unreachable by {'uniform' if bu is None else 'learned'}")
            continue
        diff = bl["cost"] - bu["cost"]          # positive => uniform is cheaper
        boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(args.boot)])
        rel = 100 * diff.mean() / bl["cost"].mean()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rel_lo, rel_hi = 100 * lo / bl["cost"].mean(), 100 * hi / bl["cost"].mean()
        print(f"{100*tgt:6.1f}% {bu['cost'].mean():10.5f} {bl['cost'].mean():10.5f} "
              f"{100*bl['abstain']:11.1f}% {rel:+14.1f}% [{rel_lo:+6.1f},{rel_hi:+6.1f}] "
              f"{(boots>0).mean():18.2f}")
        report["comparisons"].append({
            "target": tgt, "uniform_k": bu["k"], "uniform_cost": float(bu["cost"].mean()),
            "learned_policy": bl["policy"], "learned_R": bl["R"],
            "learned_cost": float(bl["cost"].mean()), "learned_abstain_rate": bl["abstain"],
            "uniform_saving_pct": float(rel), "ci95": [float(rel_lo), float(rel_hi)],
            "p_uniform_cheaper": float((boots > 0).mean()),
        })
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

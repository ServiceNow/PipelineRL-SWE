#!/usr/bin/env python3
"""Cost at matched accuracy along the achievable (randomised) frontier.

"Cheapest arm reaching T" evaluates a policy family at ONE swept operating point, so it is
sensitive to whether the value grid happens to place a point near T. When it does not, the
comparison silently switches to a different arm of the family, and the reported number moves
for reasons that have nothing to do with the policy. That is what made three LiveCodeBench
targets swing with the draw-ordering seed (sd 10-12) while their neighbours stayed at sd 4.

Any two operating points can be mixed by flipping a coin per episode, so the genuinely
achievable set is the convex hull of the family's (cost, accuracy) points -- the standard
convexification for a constrained MDP, and the same construction that makes randomised tests
admissible in Neyman-Pearson. Comparing hulls compares the policy classes; comparing single
grid points compares the grids.

This keeps the units of the frontier (percent cost saved at matched accuracy) and removes the
artifact: seed sd falls 2-4x at the unstable targets, and it lowers the flattering numbers too,
because mixtures are available to the baseline as well.
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path

SUFFIXES = ("_value_frozen", "_value", "_abstain")


def policy_family(policy: str) -> str:
    for suf in SUFFIXES:
        if policy.endswith(suf):
            return policy[: -len(suf)]
    return policy


def load(run_dir: str) -> list[dict]:
    return json.loads(Path(run_dir, "replay", "replay_results.json").read_text())["results"]


def hull(results: list[dict], family: str) -> list[tuple[float, float]]:
    """Upper-left convex envelope as (cost, accuracy) vertices, cheapest first."""
    pts = sorted({(r["mean_realized_cost"], r["correctness"])
                  for r in results if policy_family(r["policy"]) == family})
    keep, best = [], -1.0
    for c, a in pts:                       # drop points beaten on both axes
        if a > best:
            keep.append((c, a)); best = a
    v: list[tuple[float, float]] = []      # upper concave envelope of acc(cost)
    for p in keep:
        while len(v) >= 2:
            (c1, a1), (c2, a2) = v[-2], v[-1]
            if (a2 - a1) * (p[0] - c1) <= (p[1] - a1) * (c2 - c1):
                v.pop()                    # v[-1] sits below the chord: a mixture beats it
            else:
                break
        v.append(p)
    return v


def cost_at(vertices: list[tuple[float, float]], target: float) -> float | None:
    """Cheapest randomised mixture reaching `target` accuracy."""
    if not vertices or vertices[-1][1] < target:
        return None
    for i, (c, a) in enumerate(vertices):
        if a >= target:
            if i == 0:
                return c
            c1, a1 = vertices[i - 1]
            if a == a1:
                return min(c1, c)
            return c1 + (c - c1) * (target - a1) / (a - a1)
    return None


def advantage(results: list[dict], base: str, ours: str, target: float) -> float | None:
    cb = cost_at(hull(results, base), target)
    co = cost_at(hull(results, ours), target)
    return None if cb is None or co is None else 100.0 * (cb - co) / cb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="replay output dirs, one per seed")
    ap.add_argument("--base", default="counts")
    ap.add_argument("--ours", default="content_decay_qcost")
    ap.add_argument("--targets", default="0.50,0.60,0.65,0.70,0.75,0.80")
    ap.add_argument("--scan", nargs=2, type=float, metavar=("LO", "HI"),
                    help="instead of targets, sweep this accuracy range densely and report "
                         "the worst mean advantage -- the strict-improvement check")
    ap.add_argument("--scan-points", type=int, default=400)
    a = ap.parse_args()

    runs = [load(d) for d in a.run_dirs]

    if a.scan:
        lo, hi = a.scan
        neg, worst = 0, None
        for k in range(a.scan_points + 1):
            t = lo + (hi - lo) * k / a.scan_points
            vals = [advantage(r, a.base, a.ours, t) for r in runs]
            if any(v is None for v in vals):
                continue
            m = st.mean(vals)
            neg += m < 0
            if worst is None or m < worst[1]:
                worst = (t, m)
        print(f"swept {a.scan_points + 1} accuracy levels over {lo:.1%}-{hi:.1%}")
        print(f"  levels where the mean advantage is negative: {neg}")
        print(f"  worst mean advantage anywhere: {worst[1]:+.2f}% at accuracy {worst[0]:.1%}")
        return

    print(f"| target | advantage over {a.base} |")
    print("|---|---|")
    for t in (float(x) for x in a.targets.split(",")):
        vals = [v for v in (advantage(r, a.base, a.ours, t) for r in runs) if v is not None]
        if not vals:
            print(f"| {t:.0%} | n/a |"); continue
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        print(f"| {t:.0%} | **{st.mean(vals):+.1f}% ± {sd:.1f}** |")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read out a peer-pool screen and decide, per model, go or no-go.

The screen exists because `qwen/qwen3.5-plus-20260420` returned 336/341 empty outputs in
the gate-1 collection -- a reasoning model whose whole 4096-token budget went to hidden
reasoning. Three of the four candidates are reasoning models, so the same trap is live.

Three questions, in the order that matters:

  1. Does the model emit extractable code? An empty-output rate above a few percent means
     the token cap or the reasoning handling is wrong, and every number downstream is
     meaningless rather than merely noisy.
  2. What does a call actually cost on competitive-programming problems? The pool was
     picked using $/call estimates that assume 1800 completion tokens. Reasoning models
     will exceed that, and the full-collection budget is a multiple of the real figure.
  3. Is the solve rate inside the peer band? The pool was selected on CodeRouterBench's
     HumanEval-style tasks. If a model falls out of the band on LiveCodeBench, the pool is
     a ladder here and has little routable structure -- which is the whole premise.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from pathlib import Path

# OpenRouter list prices, USD per token, as of 2026-09-01. Dated deliberately: our
# historical cost model sits 14-99x above current prices, so any figure derived from
# prices must carry the date it was taken.
PRICES = {
    "qwen/qwen3-max": (0.78e-6, 3.90e-6),
    "z-ai/glm-5": (0.60e-6, 1.92e-6),
    "z-ai/glm-5.3": (1.40e-6, 4.40e-6),
    "moonshotai/kimi-k2.5": (0.45e-6, 2.25e-6),
    "minimax/minimax-m2.7": (0.30e-6, 1.20e-6),
    "openai/gpt-oss-120b": (0.03e-6, 0.17e-6),
}


def routable_structure(rows_by_label: dict[str, list[dict]], labels: list[str]) -> None:
    """Rank candidate sub-pools by the only thing that decides whether routing can pay.

    A peer band is a proxy; the quantity itself is c(1) = union minus best single, the
    accuracy a router could gain over always calling the pool's strongest member. Alongside
    it, the disagreement rate is the fraction of problems on which any routing decision
    changes the outcome, and hence the effective sample size for training one.

    Single draw and ~50 problems, so these are coarse. They are for choosing a pool, never
    for reporting: single-draw complementarity is inflated in a ladder, which is exactly the
    correction this project measured (19.3% -> 7.4% over ten draws).
    """
    from itertools import combinations

    solved: dict[str, dict[str, bool]] = {
        label: {str(r["problem_id"]): bool(r.get("resolved")) for r in rows}
        for label in labels
        for rows in [rows_by_label[label]]
    }
    print(f"\n{'sub-pool':<34} {'n':>4} {'best':>7} {'union':>7} {'c(1)':>8} {'disagree':>9}")
    ranked = []
    for size in range(2, len(labels) + 1):
        for combo in combinations(sorted(labels), size):
            shared = sorted(set.intersection(*[set(solved[m]) for m in combo]))
            if len(shared) < 10:
                continue
            per_model = {m: [solved[m][p] for p in shared] for m in combo}
            best = max(sum(v) for v in per_model.values()) / len(shared)
            union = sum(any(per_model[m][i] for m in combo) for i in range(len(shared))) / len(shared)
            disagree = sum(
                len({per_model[m][i] for m in combo}) > 1 for i in range(len(shared))
            ) / len(shared)
            ranked.append((100 * (union - best), "+".join(combo), len(shared), best, union, disagree))
    for c1, name, n, best, union, dis in sorted(ranked, reverse=True):
        print(f"{name:<34} {n:4d} {100*best:6.1f}% {100*union:6.1f}% {c1:+7.2f}pt {100*dis:8.1f}%")
    if ranked:
        top = max(ranked)
        print(f"\n  best routable structure: {top[1]} at {top[0]:+.2f}pt over its strongest member, "
              f"{100*top[5]:.1f}% of problems contested")
        print("  reference: this project's cost ladder holds +0.88pt at ten draws; "
              "RouterBench's typical task ~+2.8pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", required=True)
    parser.add_argument("--max-empty-frac", type=float, default=0.05,
                        help="Above this empty-output rate a model is a hard no-go.")
    parser.add_argument("--band-pt", type=float, default=15.0,
                        help="Max solve-rate spread, in points, for the pool to be peers.")
    args = parser.parse_args()

    rows_by_label: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(args.screen_dir, "*_screen.jsonl"))):
        label = os.path.basename(path).split("_")[0]
        rows_by_label.setdefault(label, [])
        for line in open(path):
            if line.strip():
                rows_by_label[label].append(json.loads(line))
    if not rows_by_label:
        raise SystemExit(f"No *_screen.jsonl under {args.screen_dir}")

    print(f"{'model':>9} {'n':>4} {'empty':>7} {'no code':>8} {'solve':>7} "
          f"{'prompt tok':>11} {'compl tok':>10} {'think ch':>9} {'$/call':>9}  verdict")
    summary = {}
    for label, rows in sorted(rows_by_label.items()):
        n = len(rows)
        empty = sum(1 for r in rows if not str(r.get("full_output") or "").strip())
        nocode = sum(1 for r in rows if not str(r.get("code") or "").strip())
        solved = sum(1 for r in rows if str(r.get("resolved")) == "True" or r.get("resolved") is True)
        ptok = statistics.mean([float(r.get("prompt_tokens") or 0) for r in rows])
        ctok = statistics.mean([float(r.get("completion_tokens") or 0) for r in rows])
        think = statistics.mean([float(r.get("thinking_chars") or 0) for r in rows])
        model = rows[0].get("model", "")
        pin, pout = PRICES.get(model, (float("nan"), float("nan")))
        cost = ptok * pin + ctok * pout
        bad = empty / n > args.max_empty_frac
        verdict = "NO-GO: empty outputs" if bad else "ok"
        summary[label] = dict(model=model, solve=solved / n, cost=cost, empty=empty / n)
        print(f"{label:>9} {n:4d} {empty/n*100:6.1f}% {nocode/n*100:7.1f}% {solved/n*100:6.1f}% "
              f"{ptok:11.0f} {ctok:10.0f} {think:9.0f} ${cost:8.5f}  {verdict}")

    good = {k: v for k, v in summary.items() if v["empty"] <= args.max_empty_frac}
    print()
    if len(good) < len(summary):
        for k, v in summary.items():
            if k not in good:
                print(f"  DROP {k} ({v['model']}): {v['empty']*100:.1f}% empty outputs -- raise "
                      f"--max-tokens or handle its reasoning field, then re-screen.")
    if len(good) >= 2:
        rates = [v["solve"] for v in good.values()]
        spread = (max(rates) - min(rates)) * 100
        print(f"  surviving pool: {', '.join(sorted(good))}")
        print(f"  solve-rate spread {spread:.1f}pt "
              f"({'PEER band' if spread <= args.band_pt else 'LADDER -- little routable structure'})")
        mean_cost = statistics.mean([v["cost"] for v in good.values()])
        print(f"  mean ${mean_cost:.5f}/call at 2026-09-01 list prices")
        for n_problems in (892, 2892):
            print(f"    full collection, {len(good)} models x {n_problems} problems x 1 draw: "
                  f"~${len(good)*n_problems*mean_cost:.2f}")
    else:
        print("  fewer than two models survived: re-screen before spending anything.")

    if len(good) >= 2:
        routable_structure(rows_by_label, sorted(good))


if __name__ == "__main__":
    main()

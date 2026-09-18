#!/usr/bin/env python3
"""Cost-accuracy frontier of our method against RouterBench's Zero Router analogue: the convex hull
of every fixed plan (one route, one route resampled up to 6 times, scout-first cascades). One seed
per pool; linear cost axis (log-x linearises the frontier's concavity and hides the gap)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/mnt/llmd/results/exps/aristides/reason/gridmatch/fixedacct2")
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else "overleaf/figures")
INK, INK2, RULE, BLUE, GREEN = "#1d2129", "#5b616e", "#c9ccd2", "#2a78d6", "#1baf7a"
OURS = ("content_decay_qcost_value", "content_decay_qcost_cappedvalue")
FIXED = {"single_scout": "scout ×1", "single_oss20": "20B ×1", "single_oss120": "120B ×1",
         "scout_then_oss20": "scout→20B", "scout_then_oss120": "scout→120B",
         "single_pass_cascade": "scout→20B→120B", "best_of_6_scout": "scout ×6",
         "best_of_6_oss120": "120B ×6"}
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.5,
                     "legend.frameon": False, "legend.fontsize": 7.5})


def hull(pts):
    keep, best = [], -1.0
    for c, a in sorted(set(pts)):
        if a > best: keep.append((c, a)); best = a
    v = []
    for q in keep:
        while len(v) >= 2:
            (c1, a1), (c2, a2) = v[-2], v[-1]
            if (a2 - a1) * (q[0] - c1) <= (q[1] - a1) * (c2 - c1): v.pop()
            else: break
        v.append(q)
    return v


def cost_at(v, t):
    if v[-1][1] < t: return np.nan
    if v[0][1] >= t: return v[0][0]
    for (c1, a1), (c2, a2) in zip(v, v[1:]):
        if a1 < t <= a2: return c1 + (c2 - c1) * (t - a1) / (a2 - a1)


OFFSET = {  # (pool, plan): label offset in points, placed by hand to avoid collisions
    ("lcb", "scout_then_oss120"): (6, -3), ("lcb", "single_oss120"): (6, -9),
    ("lcb", "single_pass_cascade"): (4, 5), ("lcb", "single_oss20"): (5, -8),
    ("lcb", "scout_then_oss20"): (5, 2), ("lcb", "best_of_6_scout"): (5, -3),
    ("lcb", "single_scout"): (5, -3), ("lcb", "best_of_6_oss120"): (-5, 4),
    ("taco", "single_pass_cascade"): (5, 3), ("taco", "scout_then_oss120"): (5, -3),
    ("taco", "single_oss120"): (5, -3), ("taco", "scout_then_oss20"): (5, -6),
    ("taco", "single_oss20"): (5, -5), ("taco", "best_of_6_scout"): (5, -3),
    ("taco", "single_scout"): (5, -3), ("taco", "best_of_6_oss120"): (-5, 4),
}
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.9), gridspec_kw={"wspace": 0.22})
for ax, (pool, name) in zip(axes, (("lcb", "LiveCodeBench"), ("taco", "TACO"))):
    rows = json.loads((RUN / f"{pool}_s{SEED}" / "replay_results.json").read_text())["results"]
    ours = hull([(r["mean_realized_cost"], r["correctness"]) for r in rows if r["policy"] in OURS])
    fixed = {r["policy"]: (r["mean_realized_cost"], r["correctness"]) for r in rows if r["policy"] in FIXED}
    zr = hull(list(fixed.values()))
    ax.plot([100 * c for c, _ in zr], [100 * a for _, a in zr], "-", color=GREEN, lw=1.6,
            label="Zero Router (hull of fixed plans)", zorder=2)
    for p, (c, a) in fixed.items():
        ax.plot(100 * c, 100 * a, "o", ms=4.5, color=GREEN, mfc="white", mew=1.2, zorder=3)
        dx, dy = OFFSET.get((pool, p), (4, 3))
        ax.annotate(FIXED[p], (100 * c, 100 * a), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.5, color=INK2, ha="right" if dx < 0 else "left")
    ax.plot([100 * c for c, _ in ours], [100 * a for _, a in ours], "-", color=BLUE, lw=2.0,
            label="ours", zorder=4)
    # read-off at the 120B's own accuracy
    t = fixed["single_oss120"][1]
    co, cz = cost_at(ours, t), cost_at(zr, t)
    ax.annotate("", xy=(100 * co, 100 * t), xytext=(100 * cz, 100 * t),
                arrowprops=dict(arrowstyle="<|-", color=INK, lw=1.0, shrinkA=0, shrinkB=0), zorder=5)
    saving = 100 * (1 - co / cz)
    ax.set_xlabel("mean cost per problem (US cents)")
    ax.set_title(f"{name} (seed {SEED}): {saving:.0f}% cheaper than the\nZero Router at gpt-oss-120b's accuracy (arrow)",
                 loc="left", color=INK, fontsize=7.5)
    ax.set_xlim(0, 100 * max(max(c for c, _ in fixed.values()), ours[-1][0]) * 1.08)
axes[0].set_ylabel("accuracy over all problems (%)")
axes[1].legend(loc="center right")
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"fig_frontier_zero_router.{ext}", **({"dpi": 160} if ext == "png" else {}))
print("wrote", OUT / "fig_frontier_zero_router.pdf")

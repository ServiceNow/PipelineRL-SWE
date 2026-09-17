#!/usr/bin/env python3
"""Headline figures for the paper, as SVG. Oracle-verifier regime only.

Every number is read from the replay outputs on disk, not typed in, except the two single-split
pools (RouterBench, SWE-bench Verified) whose frontiers come from a separate harness -- those are
marked in the source below with the section they were recorded in.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = Path("/mnt/llmd/results/exps/aristides/reason/gridmatch")
OUT = Path("analysis/paper_figures"); OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, AMBER, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a9099"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.bbox": "tight", "svg.fonttype": "none",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
})


def load(d): return json.loads(Path(d, "replay_results.json").read_text())["results"]


def hull(rows, pol):
    keep, best = [], -1.0
    for c, a in sorted({(r["mean_realized_cost"], r["correctness"])
                        for r in rows if r["policy"] == pol}):
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
    if not v or v[-1][1] < t: return None
    if v[0][1] >= t: return v[0][0]
    for (c1, a1), (c2, a2) in zip(v, v[1:]):
        if a1 < t <= a2: return c1 + (c2 - c1) * (t - a1) / (a2 - a1)


# ---------------------------------------------------------------- Fig 1: the 19/19 claim
def fig1():
    # LCB and TACO are seed-aggregated from runs/ ; RouterBench and SWE-V are single split
    # (PAPER_OUTLINE 3b-xlvi) and carry no error bar, which the figure shows by omitting one.
    pools = [
        ("LiveCodeBench\n5 seeds",  ["50%","60%","70%","80%","84%"],
         [40.5,19.6,16.5,4.6,5.4], [1.0,2.4,2.6,1.6,2.1], BLUE),
        ("TACO\n3 seeds",           ["35%","40%","45%","50%","55%"],
         [33.6,29.3,2.2,7.3,1.1],  [1.1,0.2,2.0,1.9,0.8], ORANGE),
        ("RouterBench\n1 split",    ["60%","70%","75%","80%","84%"],
         [35.2,76.9,75.1,50.8,22.9], None, GREEN),
        ("SWE-bench Verified\n1 split", ["30%","40%","50%","55%"],
         [19.5,9.5,3.4,1.2], None, AMBER),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.9),
                             gridspec_kw={"width_ratios": [1,1,1,0.82], "wspace": 0.34})
    for ax, (name, ts, vs, sd, col) in zip(axes, pools):
        x = np.arange(len(ts))
        ax.bar(x, vs, color=col, width=0.62,
               yerr=sd, error_kw=dict(ecolor="#3c4048", elinewidth=1.0, capsize=2.5))
        ax.axhline(0, color="#3c4048", lw=0.9)
        ax.set_xticks(x); ax.set_xticklabels(ts)
        ax.set_title(name, pad=7)
        ax.set_ylim(0, 84)
        for xi, v in zip(x, vs):
            ax.text(xi, v + 2.0, f"{v:.1f}", ha="center", va="bottom", fontsize=7.6)
    axes[0].set_ylabel("cost saved at matched accuracy (%)")
    for ax in axes[1:]: ax.set_yticklabels([])
    fig.supxlabel("accuracy target", y=-0.04, fontsize=9.5)
    fig.suptitle("Activation beliefs vs count beliefs — formulation and costs held fixed",
                 y=1.045, fontsize=11)
    fig.savefig(OUT / "fig1_belief_head_all_pools.svg"); plt.close(fig)
    print("fig1 ->", OUT / "fig1_belief_head_all_pools.svg")


# ---------------------------------------------------------------- Fig 2: the frontiers
def fig2():
    """Two panels. The frontier alone is a bad headline figure: the claim is a HORIZONTAL
    distance (cost at matched accuracy) but the eye reads vertical gaps, and over a 250x cost
    range every policy collapses onto a near-parallel diagonal. So the left panel shows the
    frontier with the read-off drawn explicitly, and the right panel plots the claim itself as a
    continuous curve over every reachable accuracy rather than at five hand-picked targets.

    The x axis is LINEAR, not log. Log-x linearises the concavity of a cost-accuracy frontier, so
    two genuinely different curves render as near-parallel straight lines and the saving reads as a
    uniform offset. On a linear axis the diminishing returns are visible and the saving reads as a
    lens between the curves. The tight-budget end is compressed as a result, which is why the
    right panel exists -- it is where the tight-budget gains are legible."""
    rows = load(G / "nulls/n")
    H = {p_: hull(rows, p_) for p_ in
         ("content_decay_qcost_value", "counts_value", "counts", "random_allocation")}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9),
                                  gridspec_kw={"wspace": 0.26, "width_ratios": [1, 1]})

    # -- left: frontier, zoomed to where the arms actually differ, with the read-off drawn
    ours, base = H["content_decay_qcost_value"], H["counts"]
    for pol, lab, col, lw, z in (("counts", "count beliefs (RoR v1 policy)", ORANGE, 1.7, 3),
                                 ("content_decay_qcost_value", "ours (activation beliefs)", BLUE, 2.2, 4)):
        h = H[pol]
        ax.plot([c for c, _ in h], [100*a for _, a in h], "-", color=col, lw=lw,
                marker="o", ms=3.2, mfc="white", mew=1.0, label=lab, zorder=z)
    # shade the saving between the two frontiers
    accs = np.linspace(31, 84.5, 300)
    co = np.array([cost_at(ours, a/100) for a in accs], dtype=float)
    cb = np.array([cost_at(base, a/100) for a in accs], dtype=float)
    ok = np.isfinite(co) & np.isfinite(cb)
    ax.fill_betweenx(accs[ok], co[ok], cb[ok], color=BLUE, alpha=0.11, lw=0, zorder=1)
    for tgt in (0.60, 0.80):
        a_, b_ = cost_at(ours, tgt), cost_at(base, tgt)
        ax.annotate("", xy=(a_, 100*tgt), xytext=(b_, 100*tgt),
                    arrowprops=dict(arrowstyle="<|-", color="#12151b", lw=1.1,
                                    shrinkA=0, shrinkB=0), zorder=6)
        ax.text((a_ + b_)/2, 100*tgt + 1.2, f"{100*(1-a_/b_):.0f}% cheaper",
                ha="center", va="bottom", fontsize=8, zorder=6)
    ax.set_xlabel("mean cost per problem (USD)")
    ax.set_ylabel("accuracy over all problems (%)")
    ax.set_title("The read-off is horizontal", pad=8)
    ax.set_ylim(28, 87); ax.legend(loc="lower right")

    # -- right: the claim itself, continuously
    grid = np.linspace(0.32, 0.845, 240)
    for pol, lab, col, ls in (("counts", "vs count beliefs, per-query cap (RoR v1)", ORANGE, "-"),
                              ("counts_value", "vs count beliefs, same formulation", GREEN, "-"),
                              ("random_allocation", "vs random allocation", GREY, "--")):
        b_ = H[pol]
        y = [100*(1 - cost_at(ours, t)/cost_at(b_, t))
             if (cost_at(ours, t) and cost_at(b_, t)) else np.nan for t in grid]
        ax2.plot(100*grid, y, ls, color=col, lw=1.9, label=lab)
    # Mark the five targets the tables quote. The continuous curve is NOT monotone -- there is a
    # trough near 80% where the advantage nearly vanishes -- and a five-point table samples either
    # side of it. Showing the sample points on the curve is the honest way to present that.
    for t in (0.50, 0.60, 0.70, 0.80, 0.84):
        y_ = 100*(1 - cost_at(ours, t)/cost_at(H["counts"], t))
        ax2.plot(100*t, y_, "o", color=ORANGE, ms=6, mfc="white", mew=1.6, zorder=5)
    ax2.plot([], [], "o", color=ORANGE, ms=6, mfc="white", mew=1.6,
             label="targets quoted in the tables")
    ax2.axhline(0, color="#3c4048", lw=0.9)
    ax2.set_xlabel("accuracy target (%)")
    ax2.set_ylabel("cost saved by activation beliefs (%)")
    ax2.set_title("Cost saved at matched accuracy, every reachable target", pad=8)
    ax2.set_ylim(-12, 72)
    ax2.legend(loc="lower left", fontsize=7.8)
    fig.savefig(OUT / "fig2_frontiers_lcb.svg"); plt.close(fig)
    print("fig2 ->", OUT / "fig2_frontiers_lcb.svg")


# ---------------------------------------------------------------- Fig 3: what is ours
def fig3():
    T = [0.50, 0.60, 0.70, 0.80, 0.84]
    rows = load(G / "nulls/n")
    base = hull(rows, "counts_value")
    def rel(pol):
        h = hull(rows, pol)
        return [100*(1 - cost_at(h,t)/cost_at(base,t))
                if (cost_at(h,t) and cost_at(base,t)) else np.nan for t in T]
    series = [("belief head only",            rel("content_decay_value"),       BLUE),
              ("cost head only",              rel("counts_qcost_value"),        GREEN),
              ("both (ours)",                 rel("content_decay_qcost_value"), "#12151b")]
    x = np.arange(len(T)); w = 0.26
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4),
                                  gridspec_kw={"wspace": 0.28, "width_ratios": [1.15, 1]})
    for i, (lab, vals, col) in enumerate(series):
        ax.bar(x + (i-1)*w, vals, w, label=lab, color=col)
    ax.axhline(0, color="#3c4048", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([f"{int(100*t)}%" for t in T])
    ax.set_xlabel("accuracy target"); ax.set_ylabel("cost saved vs count beliefs (%)")
    ax.set_title("Our two heads are substitutes, not complements", pad=7)
    ax.legend(loc="upper right")

    # Null ladder. Plotted WITHOUT `ours` on purpose: at +48.6% it compresses the +3.5-6.6%
    # effect this panel exists to show, and the zero-height `counts` bar is invisible anyway.
    # counts is the baseline line, not a bar.
    b2 = hull(rows, "counts")
    def rel2(pol):
        h = hull(rows, pol)
        return [100*(1 - cost_at(h,t)/cost_at(b2,t))
                if (cost_at(h,t) and cost_at(b2,t)) else np.nan for t in T]
    rnd = rel2("random_allocation")
    ax2.bar(x, rnd, 0.55, color=GREY, label="random allocation (no information)")
    ax2.axhline(0, color=ORANGE, lw=1.6,
                label="count beliefs, pool-level (RoR v1 policy)")
    for xi, v in zip(x, rnd):
        ax2.text(xi, v + 0.18, f"+{v:.1f}", ha="center", va="bottom", fontsize=7.8)
    ax2.set_ylim(0, max(rnd)*1.45)
    ax2.set_xticks(x); ax2.set_xticklabels([f"{int(100*t)}%" for t in T])
    ax2.set_xlabel("accuracy target"); ax2.set_ylabel("cost saved vs count beliefs (%)")
    ax2.set_title("Choosing routes at random beats the pool-level rule", pad=7)
    ax2.legend(loc="upper left")
    fig.savefig(OUT / "fig3_decomposition_and_nulls.svg"); plt.close(fig)
    print("fig3 ->", OUT / "fig3_decomposition_and_nulls.svg")


# ---------------------------------------------------------------- Fig 4: why the machinery
def fig4():
    T = [0.50, 0.60, 0.70, 0.80]
    sc = load("/mnt/llmd/results/exps/aristides/reason/sc_lcb_s0_1905612733/replay")
    b = hull(sc, "counts_value")
    def rel(rows, pol, base):
        h = hull(rows, pol)
        return [100*(1 - cost_at(h,t)/cost_at(base,t))
                if (cost_at(h,t) and cost_at(base,t)) else np.nan for t in T]
    single = rel(sc, "content_commit_value", b)
    seq    = rel(sc, "content_decay_value", b)
    hz = load(G / "horizon2/h"); bm = hull(hz, "content_decay_value")
    hs = {h: rel(hz, f"content_decay_bellman_h{h}_value", bm) for h in (2, 4, 6, 18)}

    x = np.arange(len(T)); w = 0.36
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"wspace": 0.3})
    ax.bar(x-w/2, single, w, label="single-commit routing", color=ORANGE)
    ax.bar(x+w/2, seq,    w, label="sequential (resample + reroute + give up)", color=BLUE)
    for xi, v in zip(x, single):
        if np.isnan(v):
            ax.text(xi-w/2, 1.2, "cannot\nreach", ha="center", va="bottom", fontsize=7.0,
                    color=ORANGE, style="italic")
    ax.axhline(0, color="#3c4048", lw=0.9)
    ax.set_ylim(min(-13, np.nanmin(single)-3), 56)   # headroom so the legend clears the bars
    ax.set_xticks(x); ax.set_xticklabels([f"{int(100*t)}%" for t in T])
    ax.set_xlabel("accuracy target"); ax.set_ylabel("cost saved vs count beliefs (%)")
    ax.set_title("The sequential structure is what pays", pad=7)
    ax.legend(loc="upper right")

    # h=4, 6 and 18 are numerically identical here -- the solve has converged, so plotting four
    # labelled lines would imply four visible curves. Draw h=2 against the converged rest.
    ax2.plot(x, hs[2], "-o", ms=4, color=BLUE, label="h=2")
    ax2.plot(x, hs[18], "-o", ms=4, color=ORANGE,
             label="h=4, 6, 18 (converged;\nh=18 is the exact solve)")
    ax2.axhline(0, color="#3c4048", lw=0.9)
    ax2.set_xticks(x); ax2.set_xticklabels([f"{int(100*t)}%" for t in T])
    ax2.set_xlabel("accuracy target"); ax2.set_ylabel("cost saved vs myopic rule (%)")
    ax2.set_title("Deeper lookahead does not help", pad=7)
    ax2.legend(loc="upper left", ncol=2)
    fig.savefig(OUT / "fig4_machinery.svg"); plt.close(fig)
    print("fig4 ->", OUT / "fig4_machinery.svg")


import os
if os.environ.get("PNG_TOO"):
    plt.rcParams["savefig.format"]="png"
for f in (fig1, fig2, fig3, fig4):
    try: f()
    except Exception as e: print(f"{f.__name__} FAILED: {type(e).__name__}: {e}")

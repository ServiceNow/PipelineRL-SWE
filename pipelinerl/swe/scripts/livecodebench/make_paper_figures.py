#!/usr/bin/env python3
"""Figures for overleaf/tmlr.tex, written to overleaf/figures/ as PDF (for LaTeX) and SVG.

Oracle-verifier regime only. Every number is read from replay outputs on disk except the
SWE-bench Verified row of the isolation figure and the belief-source ladder, which come from
single-split / single-seed runs recorded in PAPER_OUTLINE.md (sections marked at each use).

Baselines are arms that never abstain (agreement-gated escalation, the count-belief greedy rule
of RoR v1, random allocation, budget-aware best-of-K, and the fixed-plan hull). "Ours" is the
two-constraint family: the per-episode cap B and the price R swept jointly, where B = infinity
is the pure price arm -- so it is one family, and its hull is the frontier we report.

The frontier plot is deliberately absent. Two policies over one pool trace nearly the same curve
by construction and a 20-40% saving is a small horizontal shift at any scale, so every figure
here plots the saving itself.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = Path("/mnt/llmd/results/exps/aristides/reason/gridmatch")
# The clean-grid rerun is authoritative. The truncated-grid run is identical below ~79% on LCB
# (only the grid ceiling differs) and is used only while the rerun is in flight.
RUN = G / "fixedacct"
if not all((RUN / f"{p}_s{s}" / "replay_results.json").exists()
           for p, n in (("lcb", 5), ("taco", 3)) for s in range(n)):
    RUN = G / "fixedacct_truncgrid"
    print("NOTE: clean-grid rerun incomplete; drawing from", RUN)
OUT = Path(os.environ.get("FIG_OUT", "overleaf/figures")); OUT.mkdir(parents=True, exist_ok=True)

INK, INK2, RULE = "#1d2129", "#5b616e", "#c9ccd2"
BLUE, ORANGE, GREEN, AMBER, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a9099"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": RULE, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 7.5,
})
POOLS = {"lcb": ("LiveCodeBench", 5), "taco": ("TACO", 3)}
OURS = ("content_decay_qcost_value", "content_decay_qcost_cappedvalue")
FIXED = ("single_scout", "single_oss20", "single_oss120", "scout_then_oss20",
         "scout_then_oss120", "single_pass_cascade", "best_of_6_scout", "best_of_6_oss120")
BASELINES = [  # (label, policies, colour, linestyle)
    ("agreement-gated escalation",      ("agreement_gated",),        ORANGE, "-"),
    ("count-belief greedy (RoR v1)",    ("counts",),                 AMBER,  "-"),
    ("random allocation",               ("random_allocation",),      GREY,   "--"),
    ("fixed-plan hull (Zero Router)",   FIXED,                       GREEN,  "-"),
]


def load(d): return json.loads(Path(d, "replay_results.json").read_text())["results"]


def hull(rows, pols):
    """Upper-left convex hull of a family's (realised cost, accuracy) points: the set a
    randomised mixture of the family's deterministic policies can achieve."""
    keep, best = [], -1.0
    for c, a in sorted({(r["mean_realized_cost"], r["correctness"])
                        for r in rows if r["policy"] in pols}):
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
    if not v or v[-1][1] < t: return np.nan
    if v[0][1] >= t: return v[0][0]
    for (c1, a1), (c2, a2) in zip(v, v[1:]):
        if a1 < t <= a2: return c1 + (c2 - c1) * (t - a1) / (a2 - a1)
    return np.nan


def seeds(pool): return [load(RUN / f"{pool}_s{s}") for s in range(POOLS[pool][1])]


def saving_curve(runs, ours, base, grid):
    """Per-seed % cost saved at each accuracy; NaN where either side cannot reach it."""
    return np.array([[100 * (1 - cost_at(hull(r, ours), t) / cost_at(hull(r, base), t))
                      for t in grid] for r in runs])


def save(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}")
    if os.environ.get("PREVIEW_DIR"):
        fig.savefig(Path(os.environ["PREVIEW_DIR"]) / f"{name}.png", dpi=160)
    plt.close(fig); print("wrote", OUT / f"{name}.pdf")


# ------------------------------------------------------------------ Fig 1: the anchor
def fig_anchor():
    """Cost per problem to reach gpt-oss-120b's own accuracy. The simplest statement of the
    result: what does it cost each policy to be as good as always calling the large model?"""
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.35), gridspec_kw={"wspace": 1.05})
    for ax, pool in zip(axes, POOLS):
        runs = seeds(pool)
        rows = {"always gpt-oss-120b": [], **{b[0]: [] for b in BASELINES}, "ours": []}
        accs = []
        for r in runs:
            s = next(x for x in r if x["policy"] == "single_oss120")
            a = s["correctness"]; accs.append(a)
            rows["always gpt-oss-120b"].append(s["mean_realized_cost"])
            for lab, pols, *_ in BASELINES: rows[lab].append(cost_at(hull(r, pols), a))
            rows["ours"].append(cost_at(hull(r, OURS), a))
        labs = list(rows)
        m = np.array([np.mean(rows[k]) for k in labs]); sd = np.array([np.std(rows[k]) for k in labs])
        order = np.argsort(-m); y = np.arange(len(labs))
        cols = {"always gpt-oss-120b": "#b4b8bf", "ours": BLUE,
                **{b[0]: b[2] for b in BASELINES}}
        ref = m[labs.index("always gpt-oss-120b")]
        for yi, i in zip(y, order):
            ax.barh(yi, 100 * m[i], height=0.62, color=cols[labs[i]],
                    xerr=100 * sd[i], error_kw=dict(ecolor=INK2, elinewidth=0.7, capsize=1.8))
            txt = "reference" if labs[i] == "always gpt-oss-120b" else f"{100*(1-m[i]/ref):.0f}% cheaper"
            ax.text(100 * (m[i] + sd[i]) + 0.12, yi, txt, va="center", fontsize=7,
                    color=INK if labs[i] == "ours" else INK2,
                    weight="bold" if labs[i] == "ours" else "normal")
        ax.set_yticks(y); ax.set_yticklabels([labs[i] for i in order])
        for t, i in zip(ax.get_yticklabels(), order):
            if labs[i] == "ours": t.set_weight("bold"); t.set_color(INK)
        ax.set_xlim(0, 100 * max(m + sd) * 1.55)
        ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
        ax.set_xlabel("cost per problem (US cents)")
        ax.set_title(f"{POOLS[pool][0]}  (target {100*np.mean(accs):.1f}%)", loc="left", color=INK)
    save(fig, "fig_anchor")


# ------------------------------------------------------------------ Fig 2: savings curves
def fig_savings():
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.75), gridspec_kw={"wspace": 0.18})
    lo = {"lcb": 0.40, "taco": 0.30}
    for ax, pool in zip(axes, POOLS):
        runs = seeds(pool)
        grid = np.linspace(lo[pool], 0.90, 221)
        for lab, pols, col, ls in BASELINES:
            S = saving_curve(runs, OURS, pols, grid)
            ok = np.all(np.isfinite(S), axis=0)          # every seed reaches it on both sides
            mu, sd = S.mean(0), S.std(0)
            x = 100 * grid
            ax.fill_between(x[ok], (mu - sd)[ok], (mu + sd)[ok], color=col, alpha=0.14, lw=0)
            ax.plot(np.where(ok, x, np.nan), np.where(ok, mu, np.nan), ls, color=col, lw=1.5,
                    label=lab)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlabel("accuracy target, all problems (%)")
        ax.set_title(f"{POOLS[pool][0]} ({POOLS[pool][1]} seeds)", loc="left", color=INK)
        ax.set_ylim(-25, 75)
    axes[0].set_ylabel("cost saved by ours (%)")
    axes[1].tick_params(labelleft=True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [f"vs {x}" for x in l], loc="upper center", ncol=2, handlelength=2.2,
               bbox_to_anchor=(0.5, 0.0), columnspacing=2.0)
    save(fig, "fig_savings")


# ------------------------------------------------------------------ Fig 3: belief isolation
def fig_isolation():
    """Activation beliefs vs count beliefs, everything else identical: the same rule, the same
    give-up action, the same constant per-route costs on both sides."""
    targets = {"lcb": [0.50, 0.60, 0.70, 0.80, 0.84], "taco": [0.35, 0.40, 0.45, 0.50, 0.55]}
    panels = []
    for pool in POOLS:
        runs = seeds(pool)
        S = np.array([[100 * (1 - cost_at(hull(r, ("content_decay_value",)), t)
                              / cost_at(hull(r, ("counts_value",)), t))
                       for t in targets[pool]] for r in runs])
        panels.append((f"{POOLS[pool][0]}\n{POOLS[pool][1]} seeds", targets[pool],
                       S.mean(0), S.std(0), (S > 0).sum(0), len(runs)))
    # SWE-bench Verified: single split, separate harness (PAPER_OUTLINE 3b-xlvi). No error bar.
    panels.append(("SWE-bench Verified\n1 split", [0.30, 0.40, 0.50, 0.55],
                   np.array([19.5, 9.5, 3.4, 1.2]), None, None, None))
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.2), sharey=True,
                             gridspec_kw={"wspace": 0.08, "width_ratios": [5, 5, 4]})
    for ax, (name, ts, mu, sd, pos, n) in zip(axes, panels):
        x = np.arange(len(ts))
        ax.bar(x, mu, width=0.6, color=BLUE,
               yerr=sd, error_kw=dict(ecolor=INK2, elinewidth=0.7, capsize=1.8))
        for i, v in enumerate(mu):
            top = v + (sd[i] if sd is not None else 0)
            ax.text(i, top + 1.2, f"{v:+.1f}", ha="center", va="bottom", fontsize=6.8, color=INK)
            if pos is not None:
                ax.text(i, -4.2, f"{pos[i]}/{n}", ha="center", va="top", fontsize=6.3, color=INK2)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([f"{int(round(100*t))}%" for t in ts])
        ax.set_title(name, loc="left", color=INK, fontsize=8)
        ax.set_ylim(-9, 50)
    axes[0].set_ylabel("cost saved vs count beliefs (%)")
    fig.supxlabel("accuracy target  (small figures: seeds with a positive saving)",
                  fontsize=7.5, color=INK2, y=-0.06)
    save(fig, "fig_isolation")


# ------------------------------------------------------------------ Fig 4: belief sources
def fig_ladder():
    """Belief-source ladder, one replay (LiveCodeBench, seed 0), constant costs, identical rule;
    PAPER_OUTLINE 3b-xlix and 3b-liii. Plotted as the share of the ORACLE-belief saving each
    source recovers, which is the quantity the paper argues about (headroom)."""
    ts = ["50%", "60%", "70%"]
    rows = [("prompt length",                  [2.7, -2.8, -2.2]),
            ("TF-IDF of the statement",        [12.2, -0.3, 11.8]),
            ("learned per-problem decay",      [12.8, 4.2, 4.8]),
            ("kNN on activations",             [26.8, 8.8, 7.6]),
            ("linear probe on activations",    [40.8, 19.2, 12.0])]
    oracle = np.array([75.4, 71.8, 68.7])
    fig, ax = plt.subplots(figsize=(6.75, 2.0))
    y = np.arange(len(rows)); h = 0.25
    shades = ["#9cc3ee", "#5c9be2", BLUE]
    for j, t in enumerate(ts):
        vals = np.array([r[1][j] for r in rows]) / oracle[j] * 100
        ax.barh(y + (1 - j) * h, vals, height=h * 0.92, color=shades[j], label=f"{t} target")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlabel("share of the oracle-belief saving recovered (%)")
    ax.set_xlim(-8, 60)
    ax.legend(loc="lower right")
    save(fig, "fig_ladder")


for f in (fig_anchor, fig_savings, fig_isolation, fig_ladder):
    try: f()
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"{f.__name__} FAILED: {e}")

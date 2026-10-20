"""Route-share figure (draft): share of test queries sent to each route at the middle of the shared accuracy band, for median-length
pricing, our cost probe and realized length (same success predictions; paper split). Routes stacked from cheapest to most expensive
per query. Read from analysis/cost_headroom/route_shares_pinned.json (route_shares.py).
Usage: python make_shares_fig.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; S = json.load(open(HERE.parent / "analysis/cost_headroom/route_shares_pinned.json"))
INK, MUTED, SURF = "#263240", "#5A6470", "#FFFFFF"
ROUTES = [("oss20lo", "gpt-oss-20b, low", "#2a78d6"), ("oss20md", "gpt-oss-20b, medium", "#eb6834"), ("oss120md", "gpt-oss-120b, medium", "#1baf7a"),
          ("dsv4f", "deepseek-v4-flash", "#eda100"), ("oss120hi", "gpt-oss-120b, high", "#e87ba4")]
ARMS = [("median", "Median pricing"), ("ours", "Ours"), ("realized", "Realized length")]
PANELS = [("LCB", "LiveCodeBench"), ("Omni", "Omni-MATH"), ("MMLU-Pro", "MMLU-Pro"), ("AIME", "AIME"), ("APPS", "APPS"),
          ("CC", "CodeContests"), ("SuperGPQA", "SuperGPQA"), ("BBEH", "BIG-Bench Extra Hard"), ("BCB", "BigCodeBench")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(3, 3, figsize=(7.0, 4.4), sharex=True, gridspec_kw=dict(left=.115, right=.985, bottom=.06, top=.85, wspace=.16, hspace=.75))
for ax, (p, title) in zip(axes.flat, PANELS):
    o = S[p]; sl = o["slots"]
    for y, (a, lab) in enumerate(ARMS):
        left = 0.
        for key, _, col in ROUTES:
            w = o[a]["shares"][sl.index(key)]
            if w > 0:
                ax.barh(y, w, left=left, height=.72, color=col, edgecolor=SURF, linewidth=1.0)
                if w >= .1:
                    ax.text(left + w / 2, y, f"{w * 100:.0f}", ha="center", va="center", fontsize=5.6, color=INK)
            left += w
        ax.text(1.01, y, f"${o[a]['cost_per_1k']:.2f}", ha="left", va="center", fontsize=5.6, color=MUTED)
    ax.set_yticks(range(3)); ax.set_yticklabels([l for _, l in ARMS] if ax in axes[:, 0] else [], fontsize=6.2)
    ax.invert_yaxis(); ax.set_xlim(0, 1.13); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(["0", "50", "100%"], fontsize=5.8)
    ax.text(0, 1.30, title, transform=ax.transAxes, fontsize=7, fontweight="bold", va="bottom")
    ax.text(0, 1.06, f"at {o['accuracy'] * 100:.0f}% accuracy; ours saves {o['saving_vs_median'] * 100:.0f}%", transform=ax.transAxes, fontsize=6, color=MUTED, va="bottom")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in ROUTES]
fig.legend(h, [l for _, l, _ in ROUTES], ncol=5, frameon=False, fontsize=6.4, loc="upper center", bbox_to_anchor=(.54, 1.0), handlelength=1.2, columnspacing=1.2)
fig.text(.54, .935, "Share of test queries per route (%), routes ordered by cost per query; right: spend per 1,000 queries", ha="center", fontsize=6.2, color=MUTED)
for s in ("pdf", "png"):
    fig.savefig(HERE / "figures" / f"route_shares.{s}", dpi=220)
print("saved")

"""Figure 2 (frontier + forest; the nine-dataset saving vs median moved to Figure 1, make_hero_fig.py). Left: LiveCodeBench cost-accuracy
frontiers at billed prices, ours vs median-length pricing (same success predictions, one training set;
paper_nowai/data/fresh_billed_curves_pinned.json), arrows at matched accuracy. Right: cost saved by ours relative
to ZeroRouter as published (its own success model and pricing), paired intervals, from zr_faithful_draws_summary_pinned.json (one matched
run). Each interval is drawn against its own zero line, so no interval invites a comparison it cannot support.
Usage: python make_headline_fig.py
"""
import json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CUR = json.load(open(HERE.parent / "paper_nowai/data/fresh_billed_curves_pinned.json"))["datasets"]["lcb"]
SUM = json.load(open(HERE.parent / "analysis/cost_headroom/zr_faithful_draws_summary_pinned.json"))
MAIN = json.load(open(HERE.parent / "analysis/cost_headroom/draws_summary_pinned.json"))["draws"]
INK, MUTED, GRID = "#263240", "#5A6470", "#E6EBF0"; OURS, MED, ZR, HEAD = "#2166AC", "#D67C27", "#6250D6", "#7F8C99"
NAMES = {"LCB": "LiveCodeBench", "Omni": "Omni-MATH", "MMLU-Pro": "MMLU-Pro", "AIME": "AIME", "APPS": "APPS", "BCB": "BigCodeBench",
         "CC": "CodeContests", "SuperGPQA": "SuperGPQA", "BBEH": "BIG-Bench Extra Hard"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, (a1, sp_, a3) = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1, .42, .8], wspace=.1, left=.065, right=.99, bottom=.25, top=.88))
sp_.axis("off")                                                                   # room for the dataset labels

# left: frontiers
for arm, col, lab in (("median", MED, "Median pricing"), ("learned", OURS, "Ours (cost probes)")):
    c = CUR["curves"][arm]
    a1.scatter(np.array(c["mean_cost_usd"]) * 1000, np.array(c["accuracy"]) * 100, s=1.2, color=col, alpha=.25, lw=0, zorder=2)
    a1.plot(np.array(c["hull_cost_usd"]) * 1000, np.array(c["hull_accuracy"]) * 100, color=col, lw=1.5, label=lab, zorder=3)
for k in CUR["connectors"]:
    y, x0, x1 = k["accuracy"] * 100, k["median_cost_usd"] * 1000, k["ours_cost_usd"] * 1000
    a1.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=INK, lw=.7, mutation_scale=6, shrinkA=0, shrinkB=1), zorder=4)
    a1.text(x0 + .012, y - .6, f"−{k['saved'] * 100:.0f}%", fontsize=6, color=INK, ha="left", va="top")
a1.set_xlim(0, 1.0); a1.set_ylim(48, 92); a1.set_xlabel("Billed spend ($ / 1,000 queries)"); a1.set_ylabel("Accuracy (%)")
a1.set_title("LiveCodeBench: same accuracy, lower cost", loc="left", fontsize=7.5, fontweight="bold", pad=4)
a1.legend(frameon=False, fontsize=6.3, loc="lower right")

# right: ours vs ZeroRouter as published (paired), datasets in Figure 1's order
order = sorted(SUM, key=lambda p: MAIN[p]["ours"]["vs_median"][0])
for y, p in enumerate(order):
    g = SUM[p]["zr_pub"]["ours_vs_it"]; sig = g[1] > 0
    a3.plot([g[1], g[2]], [y] * 2, color=ZR, lw=1.4, solid_capstyle="round", zorder=4)
    a3.plot(g[0], y, "o", color=ZR, mfc=ZR if sig else "white", mew=1.1, ms=4.5, zorder=5)
a3.set_yticks(range(len(order))); a3.set_yticklabels([NAMES[p] for p in order], fontsize=6.4); a3.set_xlabel("Cost saved vs. ZeroRouter as published (%)")
a3.set_title("Ours vs. ZeroRouter, all nine datasets", loc="left", fontsize=7.5, fontweight="bold", pad=4)
for ax in (a3,):
    ax.axvline(0, color=INK, lw=.6); ax.grid(axis="x", color=GRID, lw=.6); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    ax.set_ylim(-.6, len(order) - .4); ax.spines["left"].set_visible(False)
a3.set_xlim(-42, 52)
for ax in (a1, a3):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
hs = [plt.Line2D([], [], color=ZR, marker="o", ms=4.5, lw=1.4), plt.Line2D([], [], color=ZR, marker="o", mfc="white", ms=4.5, lw=1.4)]
fig.legend(hs, ["Paired 95% interval excludes 0", "Interval includes 0"],
           frameon=False, fontsize=6.2, loc="lower center", bbox_to_anchor=(.78, 0), ncol=2, handlelength=1.4, columnspacing=1.2)
for s in ("pdf", "png"):
    fig.savefig(HERE / "figures" / f"headline.{s}", dpi=220)
print("saved")

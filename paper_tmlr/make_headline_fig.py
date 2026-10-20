"""Figure 1 (draft, frontier + forest). Left: LiveCodeBench cost-accuracy frontiers at billed prices, ours vs median-length pricing
(same success predictions, one training set; paper_nowai/data/fresh_billed_curves_pinned.json), arrows at matched accuracy. Right: cost
saved relative to median pricing on all nine datasets, mean over ten training sets with 95% intervals of the pooled test resamples:
ours and headroom (realized length) from the main runs (draws_summary_pinned.json, as in the tables), ZeroRouter as published
(its own success model and pricing) and the paired gap ours - ZeroRouter from zr_faithful_draws_summary_pinned.json (one matched run).
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
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=.42, left=.07, right=.95, bottom=.25, top=.9))

# left: frontiers
for arm, col, lab in (("median", MED, "Median pricing"), ("learned", OURS, "Ours (prefill cost probes)")):
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

# right: forest over the nine datasets
order = sorted(SUM, key=lambda p: MAIN[p]["ours"]["vs_median"][0])
for y, p in enumerate(order):
    r = SUM[p]; o, z, h = MAIN[p]["ours"]["vs_median"], r["zr_pub"]["vs_median"], MAIN[p]["oracle"]["vs_median"]
    a2.plot(z[0], y, "^", color=ZR, ms=4, zorder=4)
    a2.plot([o[1], o[2]], [y] * 2, color=OURS, lw=1.4, solid_capstyle="round", zorder=4)
    a2.plot(o[0], y, "o", color=OURS, ms=4.5, zorder=5)
    g = r["zr_pub"]["ours_vs_it"]; sig = g[1] > 0                     # paired: ours minus ZeroRouter, pooled test resamples
    a2.text(1.02, y, f"{g[0]:+.0f}", transform=a2.get_yaxis_transform(), ha="right" if False else "left", va="center", fontsize=6.3,
            color=INK if sig else MUTED, fontweight="bold" if sig else "normal")
    a2.plot(h[0], y, "D", mfc="white", mec=HEAD, mew=.9, ms=3.8, zorder=3)
a2.axvline(0, color=INK, lw=.6); a2.set_yticks(range(len(order))); a2.set_yticklabels([NAMES[p] for p in order], fontsize=6.4)
a2.set_xlabel("Cost saved relative to median pricing (%)"); a2.grid(axis="x", color=GRID, lw=.6); a2.set_axisbelow(True)
a2.set_title("All nine datasets (ten training sets)", loc="left", fontsize=7.5, fontweight="bold", pad=4)
a2.tick_params(axis="y", length=0)
a2.text(1.02, len(order) - .55, "gap", transform=a2.get_yaxis_transform(), ha="left", va="bottom", fontsize=5.8, color=MUTED, linespacing=.95)
for ax in (a1, a2):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
a2.spines["left"].set_visible(False)
hs = [plt.Line2D([], [], color=OURS, marker="o", ms=4.5, lw=1.4), plt.Line2D([], [], color=ZR, marker="^", ms=4, lw=0),
      plt.Line2D([], [], color=HEAD, marker="D", mfc="white", ms=3.8, lw=0)]
a2.legend(hs, ["Ours", "ZeroRouter (as published)", "Headroom (realized length)"], frameon=False, fontsize=6.2, loc="upper center",
          bbox_to_anchor=(.36, -.2), ncol=3, handlelength=1.4, columnspacing=1.0)
for s in ("pdf", "png"):
    fig.savefig(HERE / "figures" / f"headline.{s}", dpi=220)
print("saved")

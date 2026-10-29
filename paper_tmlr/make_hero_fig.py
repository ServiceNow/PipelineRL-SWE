"""Figure 1 (hero + breadth). Left: where Omni-MATH queries go at one matched accuracy (the middle of the shared accuracy band, paper
split): share of test queries per route under median-length pricing, our cost probes and realized length, with the spend per 1,000
queries (analysis/cost_headroom/route_shares_pinned.json, as fig:shares). Right: cost saved by ours relative to median pricing on all
nine datasets, mean over ten training sets with 95% intervals, and the headroom (draws_summary_pinned.json, as the tables).
Usage: python make_hero_fig.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S = json.load(open(HERE.parent / "analysis/cost_headroom/route_shares_pinned.json"))
MAIN = json.load(open(HERE.parent / "analysis/cost_headroom/draws_summary_pinned.json"))["draws"]
INK, MUTED, GRID, SURF = "#263240", "#5A6470", "#E6EBF0", "#FFFFFF"; OURS, HEAD = "#2166AC", "#7F8C99"
ROUTES = [("oss20lo", "gpt-oss-20b, low", "#2a78d6"), ("oss20md", "gpt-oss-20b, medium", "#eb6834"), ("oss120md", "gpt-oss-120b, medium", "#1baf7a"),
          ("dsv4f", "deepseek-v4-flash", "#eda100"), ("oss120hi", "gpt-oss-120b, high", "#e87ba4")]
ARMS = [("median", "Median pricing"), ("ours", "Ours"), ("realized", "Realized length")]
NAMES = {"LCB": "LiveCodeBench", "Omni": "Omni-MATH", "MMLU-Pro": "MMLU-Pro", "AIME": "AIME", "APPS": "APPS", "BCB": "BigCodeBench",
         "CC": "CodeContests", "SuperGPQA": "SuperGPQA", "BBEH": "BIG-Bench Extra Hard"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, (a1, sp_, a2) = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw=dict(width_ratios=[1.25, .36, .8], wspace=.08, left=.13, right=.99, bottom=.3, top=.86))
sp_.axis("off")

# left: route shares on Omni-MATH
o = S["Omni"]; sl = o["slots"]
for y, (a, lab) in enumerate(ARMS):
    left = 0.
    for key, _, col in ROUTES:
        w = o[a]["shares"][sl.index(key)]
        if w > 0:
            a1.barh(y, w, left=left, height=.68, color=col, edgecolor=SURF, linewidth=1.2)
            if w >= .1:
                a1.text(left + w / 2, y, f"{w * 100:.0f}%", ha="center", va="center", fontsize=6.4, color=INK)
        left += w
    a1.text(1.015, y, f"${o[a]['cost_per_1k']:.2f}", ha="left", va="center", fontsize=7, color=INK, fontweight="bold" if a == "ours" else "normal")
a1.text(1.015, -.58, "$ / 1k queries", ha="left", va="bottom", fontsize=5.8, color=MUTED)
a1.set_yticks(range(3)); a1.set_yticklabels([l for _, l in ARMS], fontsize=7); a1.invert_yaxis()
a1.set_xlim(0, 1.16); a1.set_xticks([]); a1.tick_params(length=0)
for sp in a1.spines.values():
    sp.set_visible(False)
a1.set_title(f"Omni-MATH at {o['accuracy'] * 100:.0f}% accuracy: where the queries go", loc="left", fontsize=7.5, fontweight="bold", pad=10)
hs = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in ROUTES]
a1.legend(hs, [l for _, l, _ in ROUTES], ncol=3, frameon=False, fontsize=5.9, loc="upper left", bbox_to_anchor=(-.01, -.04), handlelength=1.0,
          columnspacing=.9, handletextpad=.4)

# right: saving relative to median pricing on all nine datasets
order = sorted(MAIN, key=lambda p: MAIN[p]["ours"]["vs_median"][0])
for y, p in enumerate(order):
    g, h = MAIN[p]["ours"]["vs_median"], MAIN[p]["oracle"]["vs_median"]
    a2.plot([g[1], g[2]], [y] * 2, color=OURS, lw=1.4, solid_capstyle="round", zorder=4)
    a2.plot(g[0], y, "o", color=OURS, ms=4.5, zorder=5)
    a2.plot(h[0], y, "D", mfc="white", mec=HEAD, mew=.9, ms=3.8, zorder=3)
a2.set_yticks(range(len(order))); a2.set_yticklabels([NAMES[p] for p in order], fontsize=6.4); a2.tick_params(axis="y", length=0)
a2.axvline(0, color=INK, lw=.6); a2.grid(axis="x", color=GRID, lw=.6); a2.set_axisbelow(True); a2.set_xlim(-12, 65); a2.set_ylim(-.6, len(order) - .4)
a2.set_xlabel("Cost saved vs. median pricing (%)")
for sp in ("top", "right", "left"):
    a2.spines[sp].set_visible(False)
a2.set_title("All nine datasets", loc="left", fontsize=7.5, fontweight="bold", pad=10)
a2.legend([plt.Line2D([], [], color=OURS, marker="o", ms=4.5, lw=1.4), plt.Line2D([], [], color=HEAD, marker="D", mfc="white", ms=3.8, lw=0)],
          ["Ours, 95% interval", "Perfect cost knowledge"], ncol=2, frameon=False, fontsize=5.9, loc="upper center", bbox_to_anchor=(.42, -.3),
          handlelength=1.4, columnspacing=.9)
for s in ("pdf", "png"):
    fig.savefig(HERE / "figures" / f"hero.{s}", dpi=220)
print("saved")

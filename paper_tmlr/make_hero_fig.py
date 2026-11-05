"""Figure 1 (breadth + hero). Right: where LiveCodeBench queries go at one matched accuracy (77%, from route_snapshot.py; paper
split): share of test queries per route under median-length pricing, our cost probes and realized length, with the spend per 1,000
queries (analysis/cost_headroom/route_shares_pinned.json, as fig:shares). Left: cost saved by ours relative to median pricing on all
nine datasets at each of the 12 matched accuracies, mean over ten training sets (per_accuracy_savings_pinned.json), with the band average and
the headroom (draws_summary_pinned.json, as the tables).
Usage: python make_hero_fig.py [POOL]  (default LCB)
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import sys

HERE = Path(__file__).resolve().parent; POOL = sys.argv[1] if len(sys.argv) > 1 else "LCB"
S = json.load(open(HERE.parent / "analysis/cost_headroom/route_shares_pinned.json"))   # middle of the band (fig:shares)
S.update(json.load(open(HERE.parent / "analysis/cost_headroom/route_snapshot_pinned.json")))  # chosen snapshot (LCB at 77%), route_snapshot.py
MAIN = json.load(open(HERE.parent / "analysis/cost_headroom/draws_summary_pinned.json"))["draws"]
PER = json.load(open(HERE.parent / "analysis/cost_headroom/per_accuracy_savings_pinned.json")); DY = .06   # per_accuracy_savings.py
INK, MUTED, GRID, SURF = "#263240", "#5A6470", "#E6EBF0", "#FFFFFF"; OURS, HEAD = "#2166AC", "#7F8C99"
ROUTES = [("oss20lo", "gpt-oss-20b, low", "#2a78d6"), ("oss20md", "gpt-oss-20b, medium", "#eb6834"), ("oss120md", "gpt-oss-120b, medium", "#1baf7a"),
          ("dsv4f", "deepseek-v4-flash", "#eda100"), ("oss120hi", "gpt-oss-120b, high", "#e87ba4")]
ARMS = [("median", "Median pricing"), ("ours", "Ours"), ("realized", "Actual length")]
NAMES = {"LCB": "LiveCodeBench", "Omni": "Omni-MATH", "MMLU-Pro": "MMLU-Pro", "AIME": "AIME", "APPS": "APPS", "BCB": "BigCodeBench",
         "CC": "CodeContests", "SuperGPQA": "SuperGPQA", "BBEH": "BIG-Bench Extra Hard"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, (a2, sp_, a1) = plt.subplots(1, 3, figsize=(7.0, 2.85), gridspec_kw=dict(width_ratios=[.7, .36, 1.2], wspace=.08, left=.155, right=.9, bottom=.29, top=.87))
sp_.axis("off")

# right: route shares on one dataset
o = S[POOL]; sl = o["slots"]
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
a1.set_title(f"{NAMES[POOL]} at {o['accuracy'] * 100:.0f}% accuracy: where the queries go", loc="left", fontsize=7.5, fontweight="bold", pad=15)
hs = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in ROUTES]
a1.legend(hs, [l for _, l, _ in ROUTES], ncol=3, frameon=False, fontsize=5.9, loc="upper left", bbox_to_anchor=(-.01, -.04), handlelength=1.0,
          columnspacing=.9, handletextpad=.4)

# left: saving relative to median pricing at each of the 12 matched accuracies, on all nine datasets
order = sorted(MAIN, key=lambda p: MAIN[p]["ours"]["vs_median"][0])
CM = LinearSegmentedColormap.from_list("acc", ["#B5D3EE", "#08306B"])
for y, p in enumerate(order):
    v, h = PER[p]["saved"], MAIN[p]["oracle"]["vs_median"][0]
    yy = y + (np.arange(12) - 5.5) * DY                                                # low accuracy lower in the row
    a2.plot(v, yy, color="#A8B4C0", lw=.5, zorder=4)
    a2.scatter(v, yy, c=np.arange(12) / 11, cmap=CM, vmin=0, vmax=1, s=6, lw=0, zorder=5)
    a2.plot([MAIN[p]["ours"]["vs_median"][0]] * 2, [y - .38, y + .38], color=INK, lw=1.8, solid_capstyle="butt", zorder=6)
    a2.plot(h, y, "D", mfc="white", mec=HEAD, mew=.9, ms=3.8, zorder=3)
a2.set_yticks(range(len(order))); a2.set_yticklabels([NAMES[p] for p in order], fontsize=6.4); a2.tick_params(axis="y", length=0)
a2.axvline(0, color=INK, lw=.6); a2.grid(axis="x", color=GRID, lw=.6); a2.set_axisbelow(True); a2.set_xlim(-15, 65); a2.set_ylim(-.6, len(order) - .4)
a2.set_xlabel("Cost saved vs. median pricing (%)")
for sp in ("top", "right", "left"):
    a2.spines[sp].set_visible(False)
a2.set_title("All nine datasets", loc="left", fontsize=7.5, fontweight="bold", pad=15)
a2.legend([plt.Line2D([], [], color=INK, marker="|", ms=6.5, mew=1.8, lw=0), plt.Line2D([], [], color=HEAD, marker="D", mfc="white", ms=3.8, lw=0)],
          ["Ours, average", "Perfect cost knowledge"], ncol=2, frameon=False, fontsize=5.9, loc="upper center", bbox_to_anchor=(.42, -.2),
          handlelength=1.0, columnspacing=.9)
# the colour legend: position in each dataset's range of matched accuracies
bb = a2.get_position(); cb = fig.add_axes([bb.x0 + .21 * bb.width, .06, .42 * bb.width, .02])
cb.imshow(np.linspace(0, 1, 256)[None], aspect="auto", cmap=CM, extent=(0, 1, 0, 1)); cb.set_xticks([]); cb.set_yticks([])
for sp in cb.spines.values():
    sp.set_visible(False)
cb.text(-.04, .5, "Lowest accuracy", transform=cb.transAxes, ha="right", va="center", fontsize=5.9)
cb.text(1.04, .5, "highest", transform=cb.transAxes, ha="left", va="center", fontsize=5.9)
for s in ("pdf", "png"):
    fig.savefig(HERE / "figures" / f"hero.{s}", dpi=220)
print("saved")

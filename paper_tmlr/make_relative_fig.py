"""Frontiers relative to deepseek-v4-flash alone (candidate figure, not in the paper): on LiveCodeBench, the cost of reaching each share of
deepseek-v4-flash's accuracy, as a share of calling it on every query, for every cost estimator of tab:allest, ZeroRouter's whole router
and perfect cost knowledge; geometric mean over ten training sets where all ten frontiers reach that accuracy
(analysis/cost_headroom/relative_frontiers_pinned.json).
Usage: python make_relative_fig.py [POOL]
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent; POOL = sys.argv[1] if len(sys.argv) > 1 else "LCB"
R = json.load(open(HERE.parent / "analysis/cost_headroom/relative_frontiers_pinned.json"))[POOL]
INK, MUTED, GRID, OTHER = "#263240", "#5A6470", "#E6EBF0", "#C3CAD2"; OURS, MED, ZR, HEAD = "#2166AC", "#D67C27", "#6250D6", "#7F8C99"
NAMES = {"LCB": "LiveCodeBench", "Omni": "Omni-MATH", "MMLU-Pro": "MMLU-Pro"}
MAIN = [("oracle", "Perfect cost knowledge", HEAD, (0, (3, 2))), ("median", "Median pricing", MED, "-"), ("zr_4b", "ZeroRouter (4B prefill)", ZR, "-"),
        ("zr_pub", "ZeroRouter (as published)", ZR, (0, (1, 1.5))), ("ours", "Ours (cost probe)", OURS, "-")]
OTHERS = ["mean", "gbm", "mixllm", "rtr", "zr-bins", "fromsuccess", "scx"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42})
x = np.array(R["share_of_accuracy"]) * 100
y = lambda a: np.array([np.nan if v is None else v * 100 for v in R["arms"][a]["cost_share"]])
fig, ax = plt.subplots(figsize=(3.4, 2.6)); fig.subplots_adjust(left=.155, right=.97, bottom=.17, top=.9)
for a in OTHERS:
    ax.plot(x, y(a), color=OTHER, lw=.9, zorder=2)
hs = [plt.Line2D([], [], color=OTHER, lw=.9)]; labs = ["Other cost estimators (7)"]
for a, lab, col, ls in MAIN:
    h, = ax.plot(x, y(a), color=col, lw=1.6 if a == "ours" else 1.2, ls=ls, zorder=4 if a == "ours" else 3); hs.append(h); labs.append(lab)
h, = ax.plot(100, 100, "D", mfc="white", mec=INK, mew=.9, ms=4.5, zorder=5, ls=""); hs.append(h); labs.append("deepseek-v4-flash alone")
rd = []
for s in (80, 90):
    i = int(np.argmin(np.abs(x - s))); yo, ym = y("ours")[i], y("median")[i]; rd.append(f"At {s}%: ours {yo:.0f}%, median {ym:.0f}%")
    ax.plot([s, s], [yo, ym], color=INK, lw=.6, zorder=6); ax.plot([s, s], [yo, ym], "o", color=INK, ms=2.2, zorder=6)
    ax.text(s - .6, (yo + ym) / 2, f"{s}%", fontsize=5.6, color=INK, ha="right", va="center")
ax.text(100.8, 3, "Cost at a share of its accuracy\n" + "\n".join(rd), fontsize=5.6, color=INK, ha="right", va="bottom", linespacing=1.35)
ax.set_xlim(60, 101.5); ax.set_ylim(0, 105); ax.set_xticks([60, 70, 80, 90, 100]); ax.set_yticks([0, 25, 50, 75, 100])
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%"); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.set_xlabel("Accuracy (% of deepseek-v4-flash alone)"); ax.set_ylabel("Cost (% of deepseek-v4-flash alone)")
ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.set_title(f"{NAMES.get(POOL, POOL)}: routers vs. deepseek-v4-flash alone", loc="left", fontsize=7.2, fontweight="bold", pad=4)
ax.legend(hs, labs, frameon=False, fontsize=5.6, loc="upper left", handlelength=2.0, borderaxespad=.2)
for s in ("pdf", "png"):
    fig.savefig(HERE / "figures" / f"relative_{POOL.lower().replace('-', '')}.{s}", dpi=240)
print("saved")

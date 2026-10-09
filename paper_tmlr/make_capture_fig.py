"""Headroom x capture map (TMLR sec:when): per pool, x = headroom (oracle per-query cost vs median), y = cost saved vs median by our
readouts (filled) and by the best external estimator on that pool (hollow: mean, prompt GBM, MixLLM-style, ZeroRouter); dotted line =
full capture. Pinned, billed, test splits. Reads analysis/cost_headroom/{pool_baselines_<P>,fresh_baselines}_pinned.json; Omni-MATH /
MMLU-Pro / LCB headroom from headroom_routes (NEW_PATH 4.A.60), since those suite outputs have no oracle arm. Usage: python make_capture_fig.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; A = HERE.parent / "analysis" / "cost_headroom"
INK, GRID, OURS, EXT = "#263240", "#E6EBF0", "#2166AC", "#B2182B"
NAMES = {"LCB": "LiveCodeBench", "APPS": "APPS", "BCB": "BigCodeBench", "CC": "CodeContests", "AIME": "AIME", "Omni": "Omni-MATH",
         "MMLU-Pro": "MMLU-Pro", "SuperGPQA": "SuperGPQA", "BBEH": "BBEH"}
HEADROOM_FRESH = {"Omni": .555, "MMLU-Pro": .595, "LCB": .453}   # suites without an oracle arm
EXTERNAL = ("mean", "gbm", "mixllm", "zerorouter")
OFF = {"AIME": (-24, 3), "APPS": (4, -2), "SuperGPQA": (5, -8), "MMLU-Pro": (4, 2), "Omni": (-14, 5), "LCB": (-62, 2), "BBEH": (-30, 2)}
rows = {}
for p in ("LCB", "APPS", "AIME", "BCB", "CC", "SuperGPQA", "BBEH"):
    rows[p] = json.load(open(A / f"pool_baselines_{p}_pinned.json"))[p]
rows.update(json.load(open(A / "fresh_baselines_pinned.json")))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.4, 2.5), gridspec_kw=dict(left=.15, right=.97, bottom=.16, top=.97))
ax.plot([0, 65], [0, 65], ":", color=INK, lw=.7); ax.text(15, 17.5, "full capture", rotation=44, fontsize=6, color=INK)
for p, r in rows.items():
    h = (r["oracle"]["vs_median"] if "oracle" in r else HEADROOM_FRESH[p]) * 100; o = r["ours"]["vs_median"] * 100
    best = max((r[k]["vs_median"] * 100 for k in EXTERNAL if k in r), default=None)
    ax.plot([h, h], [best, o], color=GRID, lw=1.5, zorder=1)
    ax.scatter([h], [best], s=16, facecolors="white", edgecolors=EXT, lw=.9, zorder=2)
    ax.scatter([h], [o], s=18, color=OURS, zorder=3)
    ax.annotate(NAMES[p], (h, o), xytext=OFF.get(p, (4, 2)), textcoords="offset points", fontsize=6)
ax.scatter([], [], s=18, color=OURS, label="our readouts"); ax.scatter([], [], s=16, facecolors="white", edgecolors=EXT, label="best external estimator")
ax.legend(frameon=False, fontsize=6, loc="upper left"); ax.set_xlim(0, 72); ax.set_ylim(-5, 45)
ax.set_xlabel("headroom: perfect per-query cost vs median (%)"); ax.set_ylabel("cost saved vs median (%)")
ax.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.savefig(HERE / "figures" / "capture_map.pdf"); fig.savefig(HERE / "figures" / "capture_map.png", dpi=220)

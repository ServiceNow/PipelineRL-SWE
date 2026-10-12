"""Price-gap figure (NEW_PATH 4.A.86): cost saved vs median pricing as the spread of per-token prices between routes widens (flat to
~270x output-price ratio; billed spread 3.1x marked), per pool: headroom (perfect cost knowledge, dashed gray), our cost readout
(blue), pricing from the success readouts (red). Read from reason_pinned_logs/tmlr_strengthen_pricegap_<pool>_pinned.json.
BigCodeBench is omitted: its matched-accuracy band degenerates at some spreads (headroom 12.6% at billed prices).
Usage: python make_pricegap_fig.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent; L = Path("/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs")
INK, GRID, OURS, ABL, HEAD = "#263240", "#E6EBF0", "#2166AC", "#B2182B", "#7F8C99"
PANELS = [("LiveCodeBench", "lcb"), ("Omni-MATH", "omni"), ("MMLU-Pro", "mmlupro"), ("AIME", "aime"), ("APPS", "apps"),
          ("CodeContests", "cc"), ("SuperGPQA", "supergpqa"), ("BIG-Bench Extra Hard", "bbeh")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.2), sharex=True, sharey=True,
                         gridspec_kw=dict(left=.08, right=.99, bottom=.12, top=.84, wspace=.1, hspace=.32))
for ax, (title, p) in zip(axes.flat, PANELS):
    rows = json.load(open(L / f"tmlr_strengthen_pricegap_{p}_pinned.json"))["rows"]
    x = np.array([r["out_price_ratio"] for r in rows])
    for key, c, ls, lab in (("oracle", HEAD, "--", "headroom (perfect cost knowledge)"), ("ours", OURS, "-", "dedicated cost readout"),
                            ("fromsuccess", ABL, "-", "from success readouts")):
        ax.plot(x, [r[key]["vs_median"] * 100 for r in rows], ls, marker="o", color=c, lw=1.2, ms=2.5, label=lab)
    ax.axvline(3.1, color=INK, lw=.5, ls=":"); ax.axhline(0, color=INK, lw=.5)
    ax.set_xscale("log"); ax.set_title(title, fontsize=7.5); ax.set_ylim(-10, 65); ax.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
for ax in axes.flat:
    ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1x", "10x", "100x"])
fig.supxlabel("output-price ratio between the most and least expensive route (billed: 3.1x, dotted)", fontsize=7, y=.01)
for ax in axes[:, 0]:
    ax.set_ylabel("cost saved vs median (%)")
h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=6.5, loc="upper center", ncol=3, bbox_to_anchor=(.54, 1.0))
fig.savefig(HERE / "figures" / "price_gap.pdf"); fig.savefig(HERE / "figures" / "price_gap.png", dpi=200)
print("saved")

"""Label-efficiency figure (NEW_PATH 4.A.67/4.A.70): cost saved vs median pricing as a function of the number of labelled training
problems, dedicated cost readout vs pricing from the success readouts; mean over 10 training draws (2 random subsets each), band = +-1
sd over draws (NEW_PATH 4.A.92). Reads analysis/cost_headroom/draws_summary_pinned.json (draws_aggregate.py). Usage: python make_label_fig.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent; L = Path("/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs")
INK, GRID, OURS, ABL = "#263240", "#E6EBF0", "#2166AC", "#B2182B"
PANELS = [("LiveCodeBench", "LCB"), ("Omni-MATH", "Omni"), ("MMLU-Pro", "MMLU-Pro"), ("BIG-Bench Extra Hard", "BBEH"), ("SuperGPQA", "SuperGPQA")]
LAB = json.load(open(HERE.parent / "analysis" / "cost_headroom" / "draws_summary_pinned.json"))["labels"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42})
panels = [(t, np.array([[int(n), r["ours"]["vs_median"], r["ours"]["sd"], r["fromsuccess"]["vs_median"], r["fromsuccess"]["sd"]]
                        for n, r in sorted(LAB[k].items(), key=lambda x: int(x[0]) if x[0].isdigit() else -1) if n.isdigit()])) for t, k in PANELS if k in LAB]
fig, axes = plt.subplots(1, len(panels), figsize=(1.45 * len(panels) + .3, 1.75), sharey=True,
                         gridspec_kw=dict(left=.07, right=.99, bottom=.24, top=.86, wspace=.12))
for ax, (title, a) in zip(np.atleast_1d(axes), panels):
    n = a[:, 0]
    for col, sd, c, lab in ((1, 2, OURS, "dedicated cost probe"), (3, 4, ABL, "from success probes")):
        ax.plot(n, a[:, col], "o-", color=c, lw=1.2, ms=2.5, label=lab)
        ax.fill_between(n, a[:, col] - a[:, sd], a[:, col] + a[:, sd], color=c, alpha=.15, lw=0)
    ax.set_xscale("log"); ax.set_title(title, fontsize=7.5); ax.axhline(0, color=INK, lw=.5)
    ax.set_ylim(-25, 40); ax.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True); ax.set_xlabel("labelled training problems")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
np.atleast_1d(axes)[0].set_ylabel("cost saved vs median (%)")
np.atleast_1d(axes)[0].legend(frameon=False, fontsize=6, loc="lower right")
fig.savefig(HERE / "figures" / "label_efficiency.pdf"); fig.savefig(HERE / "figures" / "label_efficiency.png", dpi=200)
print("panels:", [t for t, _ in panels])

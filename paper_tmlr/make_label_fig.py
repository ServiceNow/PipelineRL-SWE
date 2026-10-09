"""Label-efficiency figure (NEW_PATH 4.A.67/4.A.70): cost saved vs median pricing as a function of the number of labelled training
problems, dedicated cost readout vs pricing from the success readouts; mean over 5 seeds, band = +-1 sd. Parsed from the job logs
(reason_pinned_logs/<pool>_label_efficiency.txt / label_efficiency_<pool>.txt). Usage: python make_label_fig.py
"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent; L = Path("/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs")
INK, GRID, OURS, ABL = "#263240", "#E6EBF0", "#2166AC", "#B2182B"
PANELS = [("LiveCodeBench", "label_efficiency_lcb.txt"), ("Omni-MATH", "label_efficiency_omni.txt"), ("MMLU-Pro", "label_efficiency_mmlupro.txt"),
          ("BIG-Bench Extra Hard", "bbeh_label_efficiency.txt"), ("SuperGPQA", "supergpqa_label_efficiency.txt")]
PAT = re.compile(r"n=\s*(\d+): dedicated\s+([-+.\d]+)% \(sd ([.\d]+)\)\s+from-success\s+([-+.\d]+)% \(sd ([.\d]+)\)")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42})
panels = [(t, np.array([[float(x) for x in m.groups()] for m in PAT.finditer((L / f).read_text())])) for t, f in PANELS if (L / f).exists()]
fig, axes = plt.subplots(1, len(panels), figsize=(1.45 * len(panels) + .3, 1.75), sharey=True,
                         gridspec_kw=dict(left=.07, right=.99, bottom=.24, top=.86, wspace=.12))
for ax, (title, a) in zip(np.atleast_1d(axes), panels):
    n = a[:, 0]
    for col, sd, c, lab in ((1, 2, OURS, "dedicated cost readout"), (3, 4, ABL, "from success readouts")):
        ax.plot(n, a[:, col], "o-", color=c, lw=1.2, ms=2.5, label=lab)
        ax.fill_between(n, a[:, col] - a[:, sd], a[:, col] + a[:, sd], color=c, alpha=.15, lw=0)
    ax.set_xscale("log"); ax.set_title(title, fontsize=7.5); ax.axhline(0, color=INK, lw=.5)
    ax.set_ylim(-20, 40); ax.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True); ax.set_xlabel("labelled training problems")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
np.atleast_1d(axes)[0].set_ylabel("cost saved vs median (%)")
np.atleast_1d(axes)[0].legend(frameon=False, fontsize=6, loc="lower right")
fig.savefig(HERE / "figures" / "label_efficiency.pdf"); fig.savefig(HERE / "figures" / "label_efficiency.png", dpi=200)
print("panels:", [t for t, _ in panels])

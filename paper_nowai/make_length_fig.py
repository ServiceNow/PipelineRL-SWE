"""Motivation figure: per-problem output length by route on the three test sets (pinned deepseek-v4-flash).
Box = interquartile range, whiskers = 5th-95th percentile, line = median; label above each box = p90/p10 ratio.
LCB: mean over a problem's draws on its temporal test split; Omni-MATH / MMLU-Pro: one draw per test problem.
Usage: REASON_ROOT=/mnt/llmd/results/exps/aristides/reason_pinned python make_length_fig.py
"""
import json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = Path(os.environ.get("REASON_ROOT", "/mnt/llmd/results/exps/aristides/reason"))
INK, MUTED, GRID, FILL, EDGE = "#263240", "#5B6676", "#E6EBF0", "#C9DCEF", "#2166AC"
NAMES = {"oss20lo": "20b\nlow", "oss20md": "20b\nmed", "dsv4f": "DS-V4\nflash", "oss120md": "120b\nmed", "oss120hi": "120b\nhigh"}
SETS = [("LiveCodeBench test set", "pool_v2_tensors_5rung", None)]                # 4-pager: LCB only (single column)


def lengths(pool, ds):
    if ds is None:
        t = np.load(R / pool / "tensors.npz", allow_pickle=True); sp = json.loads((R / pool / "split_manifest.json").read_text())
        idx = {str(p): i for i, p in enumerate(t["problem_ids"])}; ev = np.array([idx[str(p)] for p in sp["test_problem_ids"]])
    else:
        t = np.load(R / "expanded_eval_20261001" / ds / "tensors.npz", allow_pickle=True)
        n_old = len(np.load(R / pool / "tensors.npz", allow_pickle=True)["problem_ids"]); ev = np.arange(n_old, len(t["problem_ids"]))
    v = t["valid"].astype(bool); L = np.where(v, t["completion_tokens"], 0).sum(2) / np.maximum(v.sum(2), 1)
    slots = list(map(str, t["model_slots"]))
    return {s: L[ev, k][v[ev, k].any(1)] for k, s in enumerate(slots)}


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax0 = plt.subplots(1, 1, figsize=(3.3, 1.9), gridspec_kw=dict(left=.17, right=.99, bottom=.22, top=.88)); axes = [ax0]
report = {}
for ax, (title, pool, ds) in zip(axes, SETS):
    d = lengths(pool, ds); order = [s for s in NAMES if s in d]; data = [np.maximum(d[s], 1) for s in order]
    ax.boxplot(data, whis=(5, 95), widths=.55, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=FILL, edgecolor=EDGE, lw=.8), medianprops=dict(color=EDGE, lw=1.4),
               whiskerprops=dict(color=EDGE, lw=.8), capprops=dict(color=EDGE, lw=.8))
    ax.set_yscale("log"); ax.set_ylim(40, 1.6e5)
    ratios = []
    for i, x in enumerate(data):
        r = np.percentile(x, 90) / np.percentile(x, 10); ratios.append(float(r))
        ax.text(i + 1, np.percentile(x, 95) * 1.35, f"{r:.0f}×", ha="center", va="bottom", fontsize=6.2, color=MUTED)
    report[title] = dict(zip(order, ratios))
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels([NAMES[s] for s in order], fontsize=6)
    ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold", pad=3)
    ax.grid(axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("Output tokens per problem")
for sfx in ("pdf", "png"):
    fig.savefig(HERE / "figures" / f"length_spread.{sfx}", dpi=250, bbox_inches="tight", pad_inches=.04)
print(json.dumps({k: {s: round(r, 1) for s, r in v.items()} for k, v in report.items()}))

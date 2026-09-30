"""Paper figures from saved analysis JSONs; PDF vectors plus PNG previews.

Run from any directory with the pipeline-rl Python environment. The manifest
records source rows and derived paired differences; no numerical values are
copied from the outline. Figures deliberately omit pending experiments.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "data"
if not SOURCE.exists():
    SOURCE = HERE.parent / "analysis" / "cost_headroom"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, INK, GREY = "#2166AC", "#D67C27", "#14866D", "#263240", "#87929E"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#B5BEC7", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.axisbelow": True,
    "grid.color": "#E8EDF2", "grid.linewidth": .6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
manifest = {}


def load(name):
    return json.loads((SOURCE / name).read_text())


def row(filename, pool, cost):
    return next(r for r in load(filename) if r["pool"] == pool+" [market]" and r["cost_file"] == cost)


def save(fig, name):
    for suffix in ["pdf", "png"]:
        fig.savefig(OUT / f"{name}.{suffix}", dpi=250, bbox_inches="tight", pad_inches=.05)
    plt.close(fig)
    print(name, flush=True)


def headroom():
    specs = [
        ("MMLU-Pro", "baselines_simple.json", "mmlupro_tensors", "cost_preds_probe_instruct.jsonl"),
        ("LiveCodeBench", "baselines_simple.json", "pool_v2_tensors_5rung", "cost_preds_probe.jsonl"),
        ("Omni-MATH", "baselines_simple.json", "omni500_tensors", "cost_preds_probe_thinking.jsonl"),
        ("TACO", "head_to_head.json", "taco_tensors_ha", "cost_preds_probe.jsonl"),
        ("CodeContests", "baselines_simple.json", "cc_tensors", "cost_preds_probe.jsonl"),
        ("BigCodeBench", "head_to_head.json", "bcb_tensors_5r", "cost_preds_probe.jsonl"),
        ("RouterBench", "routerbench.json", "routerbench_tensors", "cost_preds_market.jsonl"),
    ]
    fig, ax = plt.subplots(figsize=(3.45, 2.72))
    for i, (label, filename, pool, cost) in enumerate(specs):
        r = row(filename, pool, cost)
        manifest[label] = dict(source=filename, pool=r["pool"], cost_file=cost,
            headroom=r["headroom"], headroom_ci=r["headroom_ci"], gain=r["learned_gain"], gain_ci=r["learned_ci"], n_test=r["n_test"])
        y = len(specs)-1-i
        ax.plot([r["learned_gain"]*100, r["headroom"]*100], [y, y], color="#D8E2EA", lw=1.5, zorder=1)
        for value, bounds, offset, color, marker in [
            (r["headroom"], r["headroom_ci"], .11, GREY, "o"),
            (r["learned_gain"], r["learned_ci"], -.11, BLUE, "s")]:
            v, lo, hi = value*100, bounds[0]*100, bounds[1]*100
            ax.errorbar(v, y+offset, xerr=[[v-lo], [hi-v]], fmt=marker, color=color,
                        ms=4.4, capsize=2, lw=1.1, zorder=3)
    ax.set_yticks(range(len(specs)), [s[0] for s in reversed(specs)], fontsize=8)
    ax.axvline(0, color=INK, ls=":", lw=.8)
    ax.grid(axis="x")
    ax.set_xlim(-18, 77)
    ax.set_xticks([-10, 0, 20, 40, 60])
    ax.set_xlabel("Cost saved vs median-length pricing (%)", fontsize=8)
    ax.plot([], [], "o", color=GREY, ms=4.4, label="Oracle cost (headroom)")
    ax.plot([], [], "s", color=BLUE, ms=4.4, label="Prefill cost prediction")
    ax.legend(loc="lower right", fontsize=7.2, frameon=False, handlelength=1, borderaxespad=.2)
    ax.set_ylim(-.65, 6.65)
    save(fig, "headroom_and_capture")


def cost_ablation():
    fig, axs = plt.subplots(1, 2, figsize=(6.98, 2.22), gridspec_kw={"width_ratios": [1, 1.1]})
    names = [("LCB", "pool_v2_tensors_5rung", "cost_preds_probe.jsonl"),
             ("Omni", "omni500_tensors", "cost_preds_probe_thinking.jsonl"),
             ("MMLU-Pro", "mmlupro_tensors", "cost_preds_probe_instruct.jsonl")]
    deltas = {}
    ax = axs[0]
    for i, (label, pool, cost) in enumerate(names):
        ours = row("baselines_simple.json", pool, cost)
        baseline = row("baselines_simple.json", pool, "cost_preds_fromsuccess.jsonl")
        boot = 100 * (np.array(ours["boot"])-np.array(baseline["boot"]))
        v = 100*(ours["learned_gain"]-baseline["learned_gain"])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        deltas[label] = dict(difference_points=float(v), ci95=[float(lo), float(hi)])
        ax.errorbar(v, 2-i, xerr=[[v-lo], [hi-v]], fmt="o", color=BLUE, capsize=3, ms=5)
        ax.text(v+.7, 2-i+.16, f"{v:+.1f}", fontsize=8)
    ax.set_yticks([0, 1, 2], ["MMLU-Pro", "Omni", "LCB"])
    ax.set_xlim(-10, 35)
    ax.set_ylim(-.5, 2.7)
    ax.axvline(0, color=GREY, ls=":", lw=.9)
    ax.grid(axis="x")
    ax.set_xlabel("Extra savings from dedicated cost head (pp)", fontsize=8)
    ax.set_title("(a) Cost cannot always be read from success", fontsize=8.5, loc="left", pad=10)
    manifest["cost_from_success_paired"] = deltas
    ax = axs[1]
    x = np.arange(3)
    width = .24
    comparison = {}
    specs = [("LCB", "head_to_head.json", "pool_v2_tensors_5rung", "cost_preds_probe.jsonl"),
             ("Omni", "baselines_lit.json", "omni500_tensors", "cost_preds_probe_thinking.jsonl"),
             ("MMLU-Pro", "baselines_lit.json", "mmlupro_tensors", "cost_preds_probe_instruct.jsonl")]
    series = [("cost_preds_mixllm.jsonl", ORANGE, "Embedding ensemble"),
              ("cost_preds_gbm.jsonl", GREY, "Prompt-feature GBM"),
              (None, BLUE, "Shared prefill + ridge")]
    for j, (cost, color, label) in enumerate(series):
        vals = []
        for pool_label, filename, pool, ours_cost in specs:
            r = row(filename, pool, cost or ours_cost)
            vals.append(r["learned_gain"]*100)
            comparison.setdefault(pool_label, {})[label] = dict(source=filename, cost_file=cost or ours_cost,
                gain=r["learned_gain"], ci95=r["learned_ci"])
        bars = ax.bar(x+(j-1)*width, vals, width, color=color, label=label, zorder=3)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, value+.7, f"{value:.1f}", ha="center", va="bottom", rotation=90, fontsize=7)
    ax.set_xticks(x, ["LCB", "Omni", "MMLU-Pro"])
    ax.set_ylim(0, 60)
    ax.grid(axis="y")
    ax.set_ylabel("Cost saved vs median rule (%)", fontsize=8)
    ax.set_title("(b) Hold success fixed; compare cost estimators", fontsize=8.1, loc="left", pad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=6.8, ncol=1)
    manifest["cost_estimator_comparison"] = comparison
    fig.subplots_adjust(wspace=.4)
    save(fig, "cost_signal_ablation")



if __name__ == "__main__":
    headroom()
    cost_ablation()
    (OUT / "data_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

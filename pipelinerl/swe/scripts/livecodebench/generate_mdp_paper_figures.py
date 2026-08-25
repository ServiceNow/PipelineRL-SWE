#!/usr/bin/env python3
"""
Generate paper figures for the thread-(a) MDP results and two-domain verdicts.

Outputs to analysis/mdp_paper_figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "analysis" / "mdp_paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

CELL_STYLE = {
    "counts": dict(color="#888888", marker="o", label="Counts (RoR-faithful)"),
    "content": dict(color="#1f77b4", marker="s", label="Content prior"),
    "counts_abstain": dict(color="#ff7f0e", marker="^", label="Counts + abstain"),
    "content_abstain": dict(color="#d62728", marker="D", label="Content + abstain (ours)"),
}


def load(path):
    return json.load(open(path))


# ── Figure 1: MDP frontier, 2x2 factorial + fixed baselines ─────────────────
def fig_frontier():
    d = load("/mnt/llmd/results/exps/aristides/reason/mdp_replay_v1_2x2/replay_results.json")
    fig, ax = plt.subplots(figsize=(7, 5))

    by_cell = {}
    for r in d["results"]:
        by_cell.setdefault(r["cell"], []).append(r)

    for cell, style in CELL_STYLE.items():
        pts = sorted(by_cell.get(cell, []), key=lambda r: r["mean_cost"])
        if not pts:
            continue
        ax.plot([p["mean_cost"] for p in pts], [100 * p["correctness"] for p in pts],
                marker=style["marker"], color=style["color"], label=style["label"], lw=1.8, ms=5)

    # fixed-policy reference points (budget-independent)
    refs = {
        "always_scout": ("Always scout", "#aaaaaa", "v"),
        "always_oss20": ("Always oss-20b", "#8c564b", "<"),
        "always_oss120": ("Always oss-120b", "#2ca02c", ">"),
        "cascade": ("Cascade 4B>20B>120B", "#9467bd", "*"),
    }
    for cell, (label, color, marker) in refs.items():
        pts = [r for r in d["results"] if r.get("policy") == cell or r.get("cell") == cell]
        if not pts:
            continue
        p = pts[0]
        ax.scatter([p["mean_cost"]], [100 * p["correctness"]], marker=marker, s=90,
                   color=color, edgecolors="k", linewidths=0.5, zorder=5, label=label)

    ax.set_xlabel("Mean inference spend per problem (weighted tokens)")
    ax.set_ylabel("Resolve rate (%)")
    ax.set_title("MDP replay frontier: content beliefs + abstention vs counting (public verifier)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mdp_frontier_2x2.png")
    plt.close(fig)


# ── Figure 2: verifier regime gap (public vs oracle) ────────────────────────
def fig_verifier_gap():
    pub = load("/mnt/llmd/results/exps/aristides/reason/mdp_replay_v1/replay_results.json")
    ora = load("/mnt/llmd/results/exps/aristides/reason/mdp_replay_v1_oracleverif/replay_results.json")

    fig, ax = plt.subplots(figsize=(7, 5))
    for d, label, ls, color in [
        (pub, "Public-test verifier (deployable)", "-", "#d62728"),
        (ora, "All-tests verifier (oracle bound)", "--", "#2ca02c"),
    ]:
        ucb = sorted([r for r in d["results"] if r.get("policy") == "RoR_ucb"],
                     key=lambda r: r["mean_cost"])
        ax.plot([r["mean_cost"] for r in ucb], [100 * r["correctness"] for r in ucb],
                ls=ls, color=color, marker="o", ms=4, label=f"RoR-UCB, {label}")

    # always-oss120 anchors
    for d, label, color in [(pub, "deployable", "#d62728"), (ora, "oracle", "#2ca02c")]:
        p = [r for r in d["results"] if r.get("policy") == "always_oss120"]
        if p:
            ax.scatter([p[0]["mean_cost"]], [100 * p[0]["correctness"]], color=color,
                       marker=">", s=90, edgecolors="k", linewidths=0.5, zorder=5)

    ax.annotate("the gap the content\npolicy must close",
                xy=(30, 74), xytext=(45, 55),
                arrowprops=dict(arrowstyle="->", color="k"), fontsize=10)

    ax.set_xlabel("Mean inference spend per problem (weighted tokens)")
    ax.set_ylabel("Resolve rate (%)")
    ax.set_title("Same algorithm, two verifiers: false-accepts collapse counting performance")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_verifier_regime_gap.png")
    plt.close(fig)


# ── Figure 3: two-domain scout delta with error bars ────────────────────────
def fig_two_domain():
    # means/sds transcribed from PAPER_PLAN n=10 tables
    domains = ["LCB\n(codegen)", "SWE-Smith\n(in-domain)", "SWE → Verified\n(transfer)"]
    io_mean = [0.784, 0.720, 0.606]; io_sd = [0.067, 0.021, 0.015]
    ps_mean = [0.900, 0.717, 0.625]; ps_sd = [0.017, 0.020, 0.048]

    x = np.arange(len(domains)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, io_mean, w, yerr=io_sd, capsize=4, label="Input-only (problem text)",
           color="#888888", error_kw=dict(alpha=0.7))
    ax.bar(x + w/2, ps_mean, w, yerr=ps_sd, capsize=4, label="Post-scout + evidence",
           color="#1f77b4", error_kw=dict(alpha=0.7))
    deltas = [(p - i) * 100 for p, i in zip(ps_mean, io_mean)]
    for xi, dv in zip(x, deltas):
        ax.annotate(f"{dv:+.1f}pp", xy=(xi + w/2, max(ps_mean[xi], io_mean[xi]) + 0.03),
                    ha="center", fontweight="bold",
                    color="#d62728" if dv > 5 else ("#555555"))
    ax.set_xticks(x); ax.set_xticklabels(domains)
    ax.set_ylabel("AUC predicting oss-120b success (n=10 seeds)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Scout evidence helps short-form codegen; nothing on agentic repair")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_two_domain_verdict.png")
    plt.close(fig)


# ── Figure 4: seed-variance stabilization ───────────────────────────────────
def fig_seed_variance():
    io_runs = [0.790, 0.848, 0.732, 0.828, 0.787, 0.811, 0.642, 0.725, 0.817, 0.858]
    psf_runs = [0.915, 0.905, 0.891, 0.893, 0.916, 0.927, 0.869, 0.887, 0.894, 0.905]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    positions = [1, 2]
    bp = ax.boxplot([io_runs, psf_runs], positions=positions, widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor="#dddddd"))
    rng = np.random.default_rng(0)
    for pos, runs, color in [(1, io_runs, "#888888"), (2, psf_runs, "#1f77b4")]:
        ax.scatter(np.full(len(runs), pos) + rng.uniform(-0.08, 0.08, len(runs)),
                   runs, color=color, s=18, zorder=3, alpha=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Input-only\n{np.mean(io_runs):.3f}±{np.std(io_runs, ddof=1):.3f}",
                        f"Post-scout+fb\n{np.mean(psf_runs):.3f}±{np.std(psf_runs, ddof=1):.3f}"])
    ax.set_ylabel("AUC across 10 training seeds (LCB)")
    ax.set_title("Scout grounding stabilizes training:\n4x lower seed variance, no catastrophic runs")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_seed_variance_stabilization.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_frontier()
    fig_verifier_gap()
    fig_two_domain()
    fig_seed_variance()
    print(f"figures written to {OUT}")
    for f in sorted(OUT.glob("*.png")):
        print(" ", f.name)

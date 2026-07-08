"""
Unified routing + abstention frontier.

The routing Pareto (routing_pareto.py) varies λ at α=0 (no abstention).
Here we also vary the abstention rate α, giving a 2D sweep (α, λ).

Key metric: fraction of ALL tasks solved locally vs cost incurred locally.
This is the honest combined metric — abstained tasks contribute 0 reward
and 0 local cost.

Also shows utility (frac_solved - λ_display × frac_cost) vs abstention rate
for a fixed routing λ.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

REAL_LABELS_DIR = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect/"
)
PS_PATH = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_10epoch_1781112916/"
    "train_qwen3_embedding_8b_lora_reward_bce_10epoch/eval_predictions.jsonl"
)
IO_PATH = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_input_only_10epoch_1781112942/"
    "train_qwen3_embedding_8b_lora_reward_bce_10epoch/eval_predictions.jsonl"
)
OUT_PARETO  = "/home/toolkit/PipelineRL-SWE/analysis/unified_pareto.png"
OUT_UTILITY = "/home/toolkit/PipelineRL-SWE/analysis/utility_vs_abstention.png"

# ── load ──────────────────────────────────────────────────────────────────────
dfs = [pd.read_parquet(p) for p in Path(REAL_LABELS_DIR + "eval").glob("*.parquet")]
eval_df = pd.concat(dfs).set_index("problem_id")

def load_preds(path):
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["problem_id"]] = np.array(d["pred_rewards"])
    return out

ps_preds = load_preds(PS_PATH)
io_preds = load_preds(IO_PATH)
common = list(eval_df.index.intersection(ps_preds).intersection(io_preds))
N = len(common)

successes  = np.array([eval_df.loc[p, "route_successes"]     for p in common])
out_tokens = np.array([eval_df.loc[p, "route_output_tokens"]  for p in common])
ps_pred    = np.array([ps_preds[p] for p in common])
io_pred    = np.array([io_preds[p] for p in common])

MODEL_SIZES    = np.array([4, 20, 30, 120])
size_per_token = MODEL_SIZES / 4.0
actual_costs   = out_tokens * size_per_token          # (N, 4)

avg_out_tokens    = out_tokens.mean(axis=0)
io_dc             = avg_out_tokens * size_per_token   # (4,) flat prior
ps_dc             = (out_tokens[:, 0] / avg_out_tokens[0])[:, None] * io_dc[None, :]  # (N, 4)

ref_cost = actual_costs[:, 3].mean()   # always-120B mean cost per task

# ── helpers ───────────────────────────────────────────────────────────────────
def get_chosen(pred, dc, lam):
    dc2 = dc if dc.ndim == 2 else dc[np.newaxis, :]
    return (pred - lam * dc2).argmax(axis=1)

def sweep(signal, chosen, alphas):
    """Return frac_solved, frac_cost (fraction of always-120B) for each α."""
    order  = np.argsort(signal)
    solved = successes[np.arange(N), chosen]
    costs  = actual_costs[np.arange(N), chosen]
    fs, fc = [], []
    for a in alphas:
        n_ab  = int(round(a * N))
        keep  = np.ones(N, dtype=bool)
        keep[order[:n_ab]] = False
        fs.append(solved[keep].sum() / N)
        fc.append(costs[keep].sum() / (N * ref_cost))
    return np.array(fs), np.array(fc)

ps_signal = ps_pred.mean(1)
io_signal = io_pred.mean(1)

alphas = np.linspace(0, 0.85, 200)

# oracle: perfect signal (r120b) + cheapest-succeeding route
oracle_signal = successes[:, 3].astype(float)
oracle_chosen = np.array([
    next((r for r in [0, 1, 2, 3] if successes[i, r]), 3)
    for i in range(N)
])
ora_fs, ora_fc = sweep(oracle_signal, oracle_chosen, alphas)

# fixed baselines
names  = ["Scout 4B", "OSS-20B", "Qwen-30B", "OSS-120B"]
fx_r   = [successes[:, r].mean() for r in range(4)]
fx_c   = [actual_costs[:, r].mean() / ref_cost for r in range(4)]

# ── Figure 1: extended Pareto (cost % vs frac_solved %) ──────────────────────
# Sweep both λ_route and α for post-scout; overlay routing-only curve (α=0).
lam_routing_vals  = np.concatenate([[0], np.logspace(-7, -2.5, 120)])
lam_display_set   = [0, 1e-6, 3e-6, 7e-6, 1e-5]
lam_display_labels = ["λ=0", "λ=1e-6", "λ=3e-6", "λ=7e-6", "λ=1e-5"]

fig, ax = plt.subplots(figsize=(9, 5.5))

# Routing-only curve (α=0), same as routing_pareto.py
ps_routing_only = np.array([
    (successes[np.arange(N), get_chosen(ps_pred, ps_dc, l)].mean(),
     actual_costs[np.arange(N), get_chosen(ps_pred, ps_dc, l)].mean() / ref_cost)
    for l in lam_routing_vals
])
io_routing_only = np.array([
    (successes[np.arange(N), get_chosen(io_pred, io_dc, l)].mean(),
     actual_costs[np.arange(N), get_chosen(io_pred, io_dc, l)].mean() / ref_cost)
    for l in lam_routing_vals
])
ax.plot(ps_routing_only[:, 1] * 100, ps_routing_only[:, 0] * 100,
        color="#1f77b4", lw=2.5, label="Post-scout routing only (α=0)")
ax.plot(io_routing_only[:, 1] * 100, io_routing_only[:, 0] * 100,
        color="#ff7f0e", lw=2, ls="--", label="Input-only routing only (α=0)")

# Abstention-extended curves for selected λ values
colors_ab = ["#1f77b4", "#2196F3", "#03A9F4", "#00BCD4", "#009688"]
for lam, label, col in zip(lam_display_set, lam_display_labels, colors_ab):
    ch = get_chosen(ps_pred, ps_dc, lam)
    fs, fc = sweep(ps_signal, ch, alphas)
    ax.plot(fc * 100, fs * 100, color=col, lw=1.2, ls=":",
            alpha=0.8, label=f"PS routing+abstention ({label})")

# oracle
ax.plot(ora_fc * 100, ora_fs * 100, color="gold", lw=2,
        label="Oracle routing+abstention", zorder=6)
ax.scatter([ora_fc[0] * 100], [ora_fs[0] * 100], marker="*",
           color="gold", s=200, zorder=7, edgecolors="black", lw=0.8)

# fixed baselines
markers = ["^", "s", "D", "o"]
bcolors = ["#2ca02c", "#9467bd", "#8c564b", "black"]
for n, r, c, m, bc in zip(names, fx_r, fx_c, markers, bcolors):
    ax.scatter(c * 100, r * 100, marker=m, color=bc, s=80, zorder=5, label=f"Always {n}")

ax.set_xlabel("Local inference cost (% of always-120B)", fontsize=11)
ax.set_ylabel("Fraction of ALL tasks solved locally (%)", fontsize=11)
ax.set_title("Unified routing + abstention frontier\n"
             "(solid = routing only α=0; dotted = routing+abstention at fixed λ)", fontsize=10)
ax.legend(fontsize=7.5, loc="lower right", ncol=1)
ax.set_xlim(0, 105)
ax.set_ylim(0, 60)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
plt.tight_layout()
plt.savefig(OUT_PARETO, dpi=150)
print(f"Saved {OUT_PARETO}")
plt.close()

# ── Figure 2: utility vs abstention rate ─────────────────────────────────────
# For each routing λ, show frac_solved and frac_cost as α varies.
# Utility = frac_solved (no cost penalty) — let the two panels speak for themselves.

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

blues   = plt.cm.Blues(np.linspace(0.45, 0.9, len(lam_display_set)))
oranges = plt.cm.Oranges(np.linspace(0.45, 0.9, len(lam_display_set)))
pct_alpha = alphas * 100

for lam, label, bc, oc in zip(lam_display_set, lam_display_labels, blues, oranges):
    ps_ch = get_chosen(ps_pred, ps_dc, lam)
    io_ch = get_chosen(io_pred, io_dc, lam)

    ps_fs, ps_fc = sweep(ps_signal, ps_ch, alphas)
    io_fs, io_fc = sweep(io_signal, io_ch, alphas)

    axes[0].plot(pct_alpha, ps_fs * 100, color=bc, lw=2,   label=f"PS {label}")
    axes[0].plot(pct_alpha, io_fs * 100, color=oc, lw=1.5, ls="--")
    axes[1].plot(pct_alpha, ps_fc * 100, color=bc, lw=2,   label=f"PS {label}")
    axes[1].plot(pct_alpha, io_fc * 100, color=oc, lw=1.5, ls="--")

# oracle
axes[0].plot(pct_alpha, ora_fs * 100, color="gold", lw=2.5, label="Oracle", zorder=6)
axes[1].plot(pct_alpha, ora_fc * 100, color="gold", lw=2.5, label="Oracle", zorder=6)

# always-120B with PS abstention signal (routing λ=0, route=3 forced)
ch_120 = np.full(N, 3)
fs_120, fc_120 = sweep(ps_signal, ch_120, alphas)
axes[0].plot(pct_alpha, fs_120 * 100, color="black", lw=1.5, ls=":", label="Always-120B + PS signal")
axes[1].plot(pct_alpha, fc_120 * 100, color="black", lw=1.5, ls=":", label="Always-120B + PS signal")

for ax in axes:
    ax.axvline(52.8, color="gray", ls="--", alpha=0.6, lw=1, label="Oracle abstention rate (52.8%)")
    ax.set_xlabel("Abstention rate (%)", fontsize=11)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Fraction of ALL tasks solved locally (%)", fontsize=11)
axes[0].set_title("Local coverage vs abstention rate\n(solid=PS, dashed=IO)", fontsize=10)
axes[0].set_ylim(0, 55)

axes[1].set_ylabel("Cost incurred (% of always-120B)", fontsize=11)
axes[1].set_title("Cost vs abstention rate\n(solid=PS, dashed=IO)", fontsize=10)
axes[1].set_ylim(0, 105)

from matplotlib.lines import Line2D
leg = [
    Line2D([0], [0], color="steelblue",  lw=2,   label="Post-scout (solid)"),
    Line2D([0], [0], color="darkorange", lw=1.5, ls="--", label="Input-only (dashed)"),
    Line2D([0], [0], color="black",      lw=1.5, ls=":",  label="Always-120B + PS signal"),
    Line2D([0], [0], color="gold",       lw=2.5, label="Oracle"),
    Line2D([0], [0], color="gray",       lw=1,   ls="--", label="Oracle α rate (52.8%)"),
    *[Line2D([0],[0], color=bc, lw=2, label=lab)
      for bc, lab in zip(blues, lam_display_labels)],
]
axes[0].legend(handles=leg, fontsize=7.5, loc="upper right")
axes[1].legend(handles=leg, fontsize=7.5, loc="upper right")

plt.tight_layout()
plt.savefig(OUT_UTILITY, dpi=150)
print(f"Saved {OUT_UTILITY}")
plt.close()

# ── numeric summary ───────────────────────────────────────────────────────────
print("\n=== At oracle abstention rate (α≈52.8%) ===")
idx_oracle_alpha = np.argmin(np.abs(alphas - 0.528))
for lam, label in zip(lam_display_set, lam_display_labels):
    ch = get_chosen(ps_pred, ps_dc, lam)
    fs, fc = sweep(ps_signal, ch, alphas)
    print(f"  PS {label}: solved={fs[idx_oracle_alpha]*100:.1f}%  cost={fc[idx_oracle_alpha]*100:.1f}%")
print(f"  Always-120B (no routing): solved={fs_120[idx_oracle_alpha]*100:.1f}%  cost={fc_120[idx_oracle_alpha]*100:.1f}%")
print(f"  Oracle:                   solved={ora_fs[idx_oracle_alpha]*100:.1f}%  cost={ora_fc[idx_oracle_alpha]*100:.1f}%")

print("\n=== Cost at which routing+abstention matches always-120B reward ===")
always120_base_reward = successes[:, 3].mean() * 100
print(f"  Always-120B base reward: {always120_base_reward:.1f}% (at cost=100%)")
for lam, label in zip([0, 3e-6], ["λ=0", "λ=3e-6"]):
    ch = get_chosen(ps_pred, ps_dc, lam)
    fs, fc = sweep(ps_signal, ch, alphas)
    idx = np.argmin(np.abs(fs * 100 - always120_base_reward))
    print(f"  PS routing+abstention {label}: matches 120B reward at cost={fc[idx]*100:.1f}%, α={alphas[idx]*100:.0f}%")

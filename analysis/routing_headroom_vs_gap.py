"""
Figure: Routing headroom scales with capability gap.

For each 'expensive' model (20B, 30B, 120B, Claude Sonnet 4.6),
compute — relative to 4B as the cheap model:
  - Capability gap (solve rate difference)
  - Mixed-outcome fraction (tasks where cheap fails, expensive succeeds)
  - Inter-model Pearson correlation
  - P(expensive=1 | 4B=0) = conditional success rate of expensive on 4B failures

Shows that routing opportunity is bounded by the capability gap and correlation
structure, and that open-model routing lives in a regime of low headroom.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
REAL_LABELS_DIR = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect/"
)
CLAUDE_REPORT = "logs/run_evaluation/claude_sonnet_eval286_daytona/report.json"
OUT = "/home/toolkit/PipelineRL-SWE/analysis/routing_headroom_vs_gap.png"

# ── load data ──────────────────────────────────────────────────────────────────
dfs = [pd.read_parquet(p) for p in Path(REAL_LABELS_DIR + "eval").glob("*.parquet")]
eval_df = pd.concat(dfs).set_index("problem_id")

with open(CLAUDE_REPORT) as f:
    report = json.load(f)
claude_resolved   = set(report["ids_resolved"])
claude_unresolved = set(report["ids_unresolved"])
claude_all = claude_resolved | claude_unresolved

# 217-instance Python-only common set
common = sorted(eval_df.index.intersection(claude_all))
N = len(common)
print(f"Common set: {N} instances")

successes  = np.array([eval_df.loc[p, "route_successes"]    for p in common])  # (N, 4)
out_tokens = np.array([eval_df.loc[p, "route_output_tokens"] for p in common])  # (N, 4)
claude_s   = np.array([p in claude_resolved for p in common], dtype=bool)

MODEL_NAMES  = ["Qwen3-4B", "OSS-20B", "Qwen3-30B", "OSS-120B"]
MODEL_SIZES  = np.array([4, 20, 30, 120])
size_per_tok = MODEL_SIZES / 4.0
actual_costs = out_tokens * size_per_tok
ref_cost     = actual_costs[:, 3].mean()

solve_rates = successes.mean(axis=0)
claude_rate = claude_s.mean()

print(f"\nSolve rates:")
for i, n in enumerate(MODEL_NAMES):
    print(f"  {n}: {solve_rates[i]*100:.1f}%")
print(f"  Claude Sonnet 4.6: {claude_rate*100:.1f}%")

# ── compute headroom metrics ──────────────────────────────────────────────────
cheap_idx = 0  # 4B is the cheap model
cheap_s = successes[:, cheap_idx]

all_models   = list(range(4)) + ["claude"]
model_labels = ["OSS-20B", "Qwen3-30B", "OSS-120B", "Claude Sonnet 4.6"]
model_shortlabels = ["20B", "30B", "120B", "Claude"]
model_colors = ["#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]

gaps         = []
mixed_fracs  = []
corrs        = []
cond_probs   = []

for exp_idx, label in zip([1, 2, 3, "claude"], model_labels):
    if exp_idx == "claude":
        exp_s = claude_s
    else:
        exp_s = successes[:, exp_idx]

    gap = exp_s.mean() - cheap_s.mean()
    # mixed-outcome: cheap fails, expensive succeeds
    mixed = (~cheap_s.astype(bool)) & exp_s.astype(bool)
    mixed_frac = mixed.mean()
    # correlation
    corr = np.corrcoef(cheap_s.astype(float), exp_s.astype(float))[0, 1]
    # P(expensive=1 | cheap=0)
    cheap_fail = ~cheap_s.astype(bool)
    cond = exp_s[cheap_fail].mean() if cheap_fail.sum() > 0 else 0.0

    gaps.append(gap * 100)
    mixed_fracs.append(mixed_frac * 100)
    corrs.append(corr)
    cond_probs.append(cond * 100)

    print(f"\n{label}:")
    print(f"  Capability gap: {gap*100:+.1f}pp")
    print(f"  Mixed-outcome fraction: {mixed_frac*100:.1f}%")
    print(f"  Pearson corr(4B, {label}): {corr:.3f}")
    print(f"  P(exp=1 | 4B=0): {cond*100:.1f}%")

gaps        = np.array(gaps)
mixed_fracs = np.array(mixed_fracs)
corrs       = np.array(corrs)
cond_probs  = np.array(cond_probs)

# ── full correlation matrix ───────────────────────────────────────────────────
print("\n── Full 5×5 Pearson correlation matrix ──")
all_s = np.column_stack([successes, claude_s.astype(float)])
names5 = ["4B", "20B", "30B", "120B", "Claude"]
C = np.corrcoef(all_s.T)
header = f"{'':>8}" + "".join(f"  {n:>8}" for n in names5)
print(header)
for i, n in enumerate(names5):
    row = f"{n:>8}" + "".join(f"  {C[i,j]:>8.3f}" for j in range(5))
    print(row)

# ── joint distribution ────────────────────────────────────────────────────────
print("\n── Joint distribution (n=217) ──")
all_fail  = (~all_s.any(axis=1)).sum()
all_succ  = (all_s.all(axis=1)).sum()
claude_uniq = ((~successes.any(axis=1)) & claude_s).sum()
open_uniq   = ((successes.any(axis=1)) & ~claude_s).sum()
union_any   = (all_s.any(axis=1)).sum()
print(f"  All 5 fail:                  {all_fail} ({all_fail/N*100:.1f}%)")
print(f"  All 5 succeed:               {all_succ} ({all_succ/N*100:.1f}%)")
print(f"  Claude-unique (all open fail): {claude_uniq} ({claude_uniq/N*100:.1f}%)")
print(f"  Open-unique (Claude fails):    {open_uniq} ({open_uniq/N*100:.1f}%)")
print(f"  Union (any of 5 succeeds):     {union_any} ({union_any/N*100:.1f}%)")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── Left: mixed-outcome fraction vs capability gap ────────────────────────────
ax = axes[0]
for i, (lbl, slbl, c) in enumerate(zip(model_labels, model_shortlabels, model_colors)):
    ax.scatter(gaps[i], mixed_fracs[i], color=c, s=120, zorder=5, label=lbl)
    ax.annotate(slbl, (gaps[i], mixed_fracs[i]),
                textcoords="offset points", xytext=(6, 3), fontsize=9)

# Reference: theoretical ceiling given solve rates
# If cheap_rate=p, exp_rate=q, with independence: mixed = (1-p)*q
p_cheap = solve_rates[cheap_idx]
g_range = np.linspace(0, 35, 100)
q_range = p_cheap + g_range / 100
indep_mixed = (1 - p_cheap) * q_range * 100
ax.plot(g_range, indep_mixed, color="gray", lw=1, ls="--", alpha=0.6,
        label="Independence upper bound")

ax.set_xlabel("Capability gap (expensive − 4B solve rate, pp)", fontsize=11)
ax.set_ylabel("Mixed-outcome fraction (%)\n(cheap fails, expensive succeeds)", fontsize=11)
ax.set_title("Routing headroom scales with capability gap\n(cheap model = Qwen3-4B)", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("+%.0f pp"))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.set_xlim(-1, 32)
ax.set_ylim(0, 40)

# ── Right: inter-model correlation vs capability gap ─────────────────────────
ax = axes[1]
for i, (lbl, slbl, c) in enumerate(zip(model_labels, model_shortlabels, model_colors)):
    ax.scatter(gaps[i], corrs[i], color=c, s=120, zorder=5, label=lbl)
    ax.annotate(slbl, (gaps[i], corrs[i]),
                textcoords="offset points", xytext=(6, -4), fontsize=9)

ax.set_xlabel("Capability gap (expensive − 4B solve rate, pp)", fontsize=11)
ax.set_ylabel("Pearson correlation (4B vs expensive model)", fontsize=11)
ax.set_title("Lower correlation → more routing opportunity\n(cheap model = Qwen3-4B)", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("+%.0f pp"))
ax.set_xlim(-1, 32)
ax.set_ylim(0.3, 0.8)

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"\nSaved {OUT}")
plt.close()

# ── table summary ─────────────────────────────────────────────────────────────
print("\n── Routing headroom summary table ──")
print(f"{'Model pair (4B → ?)':>24}  {'Gap':>8}  {'Mixed%':>8}  {'Corr':>6}  {'P(exp|4B=0)':>12}")
for i, (lbl, slbl) in enumerate(zip(model_labels, model_shortlabels)):
    print(f"  4B → {slbl:>8}:  {gaps[i]:>+6.1f}pp  {mixed_fracs[i]:>7.1f}%  {corrs[i]:>6.3f}  {cond_probs[i]:>11.1f}%")

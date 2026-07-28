"""
Experiment: Claude Escalation vs Abstention

When we would normally abstain (route to nothing), what happens if we instead
escalate to Claude Sonnet 4.6?

Shows two things:
1. Solve rate vs abstention rate: pure quality gain from Claude escalation
2. Pareto (solve rate vs cost): at what cost multiple does Claude escalation
   dominate always-120B local routing?

Dataset: 217 Python eval instances (the 69 non-Python instances excluded because
Claude's eval results are unavailable for them due to private Docker images).

Claude results: logs/run_evaluation/claude_sonnet_eval286_daytona/report.json
Open model results: real labels parquet from 4-route SWE-smith eval
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.stats import spearmanr

# ── paths ──────────────────────────────────────────────────────────────────────
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
CLAUDE_REPORT = "logs/run_evaluation/claude_sonnet_eval286_daytona/report.json"
OUT_QUALITY = "/home/toolkit/PipelineRL-SWE/analysis/escalation_claude_quality.png"
OUT_PARETO  = "/home/toolkit/PipelineRL-SWE/analysis/escalation_claude_pareto.png"

# ── load data ──────────────────────────────────────────────────────────────────
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

with open(CLAUDE_REPORT) as f:
    report = json.load(f)
claude_resolved = set(report["ids_resolved"])
claude_unresolved = set(report["ids_unresolved"])
claude_all = claude_resolved | claude_unresolved  # 217 Python instances

# ── align on 217 Python instances with all signals ─────────────────────────────
common = sorted(eval_df.index.intersection(ps_preds).intersection(io_preds).intersection(claude_all))
N = len(common)
print(f"Analysis set: {N} instances (Python only, Claude results available)")

successes   = np.array([eval_df.loc[p, "route_successes"]    for p in common])  # (N, 4)
out_tokens  = np.array([eval_df.loc[p, "route_output_tokens"] for p in common])  # (N, 4)
ps_pred     = np.array([ps_preds[p] for p in common])   # (N, 4)
io_pred     = np.array([io_preds[p] for p in common])   # (N, 4)
claude_s    = np.array([p in claude_resolved for p in common], dtype=bool)       # (N,)
ps_signal   = ps_pred.mean(axis=1)  # scalar abstention signal per task
io_signal   = io_pred.mean(axis=1)

MODEL_SIZES   = np.array([4, 20, 30, 120])
size_per_tok  = MODEL_SIZES / 4.0
actual_costs  = out_tokens * size_per_tok   # (N, 4)
ref_cost      = actual_costs[:, 3].mean()   # always-120B mean cost per task

print(f"\nSolve rates:")
labels = ["Qwen3-4B", "OSS-20B", "Qwen3-30B", "OSS-120B", "Claude Sonnet 4.6"]
for i, lbl in enumerate(labels[:4]):
    print(f"  {lbl}: {successes[:, i].mean()*100:.1f}%")
print(f"  Claude Sonnet 4.6: {claude_s.mean()*100:.1f}%")

# ── K derivation: Claude cost relative to 120B ────────────────────────────────
# Prices from route_output_cost_weights config and run_summary.json
PRICE_120B_OUTPUT   = 1.113e-5   # $/output token  (AWS, from route_output_cost_weights)
PRICE_CLAUDE_OUTPUT = 15.0 / 1e6  # $/output token  (Sonnet 4.6, $15/MTok)
PRICE_CLAUDE_INPUT  =  3.0 / 1e6  # $/input  token  (Sonnet 4.6, $3/MTok)
CLAUDE_TOTAL_INPUT  = 1_959_323   # tokens, from run_summary.json (286 tasks)
CLAUDE_TOTAL_OUTPUT =   854_972   # tokens, from run_summary.json (286 tasks)
CLAUDE_N_TASKS      = 286
claude_mean_out_tok = CLAUDE_TOTAL_OUTPUT / CLAUDE_N_TASKS  # 2991 tok/task

mean_120b_out_tok = out_tokens[:, 3].mean()
cost_claude_out_per_task = claude_mean_out_tok * PRICE_CLAUDE_OUTPUT
cost_claude_in_per_task  = (CLAUDE_TOTAL_INPUT / CLAUDE_N_TASKS) * PRICE_CLAUDE_INPUT
cost_120b_out_per_task   = mean_120b_out_tok * PRICE_120B_OUTPUT
K_output_only = cost_claude_out_per_task / cost_120b_out_per_task
K_total_vs_output = (cost_claude_out_per_task + cost_claude_in_per_task) / cost_120b_out_per_task
print(f"\n── K derivation ──")
print(f"  Claude output tokens/task: {claude_mean_out_tok:.0f}  @ ${PRICE_CLAUDE_OUTPUT*1e6:.2f}/MTok → ${cost_claude_out_per_task*1000:.3f}/ktask")
print(f"  Claude input tokens/task:  {CLAUDE_TOTAL_INPUT/CLAUDE_N_TASKS:.0f}  @ ${PRICE_CLAUDE_INPUT*1e6:.2f}/MTok → ${cost_claude_in_per_task*1000:.3f}/ktask")
print(f"  120B output tokens/task:   {mean_120b_out_tok:.0f}  @ ${PRICE_120B_OUTPUT*1e6:.3f}/MTok → ${cost_120b_out_per_task*1000:.3f}/ktask")
print(f"  K (output-only, consistent with existing framework): {K_output_only:.2f}")
print(f"  K (Claude total vs 120B output-only):                {K_total_vs_output:.2f}")

r, p = spearmanr(ps_signal, claude_s)
r2, _ = spearmanr(ps_signal, successes[:, 3])
print(f"\nPS signal vs Claude solve:  Spearman r={r:.3f} (p={p:.3g})")
print(f"PS signal vs 120B solve:    Spearman r={r2:.3f}")

# ── helper: routing choice ─────────────────────────────────────────────────────
def get_chosen(pred, dc, lam):
    dc2 = dc if dc.ndim == 2 else dc[np.newaxis, :]
    return (pred - lam * dc2).argmax(axis=1)

avg_out_tokens = out_tokens.mean(axis=0)
io_dc = avg_out_tokens * size_per_tok
ps_dc = (out_tokens[:, 0] / avg_out_tokens[0])[:, None] * io_dc[None, :]

# ── Figure 1: solve rate vs abstention rate ───────────────────────────────────
#
# For a given abstention rate α:
#   - Sort tasks by ascending PS signal (hardest first → abstain these)
#   - Non-abstained (top (1-α)%): local open-model routing
#   - Abstained (bottom α%):
#       * vanilla abstention  → 0 reward
#       * Claude escalation   → claude_s[task] reward
#
# Report fraction of ALL N tasks solved.

alphas = np.linspace(0, 0.85, 200)
order_ps = np.argsort(ps_signal)   # ascending: hardest first
order_io = np.argsort(io_signal)

def solve_curve(signal_order, chosen, escalate_to_claude: bool, alphas):
    """Fraction of ALL tasks solved as function of abstention rate."""
    local_solved = successes[np.arange(N), chosen]
    frac = []
    for a in alphas:
        n_ab = int(round(a * N))
        abstained = signal_order[:n_ab]
        kept      = signal_order[n_ab:]
        local_s   = local_solved[kept].sum()
        claude_bonus = claude_s[abstained].sum() if escalate_to_claude else 0
        frac.append((local_s + claude_bonus) / N)
    return np.array(frac)

def cost_curve(signal_order, chosen, claude_k, alphas):
    """Mean cost per task relative to always-120B ref_cost.
    claude_k: Claude cost = claude_k × ref_cost per task."""
    local_costs = actual_costs[np.arange(N), chosen]
    costs = []
    for a in alphas:
        n_ab = int(round(a * N))
        kept     = signal_order[n_ab:]
        local_c  = local_costs[kept].sum() / N
        claude_c = (n_ab * claude_k * ref_cost) / N
        costs.append((local_c + claude_c) / ref_cost)
    return np.array(costs)

# Routes: always-120B and PS-routing (λ=3e-6)
ch_120     = np.full(N, 3)
ch_ps_lam0 = get_chosen(ps_pred, ps_dc, 0)
ch_ps_lam3 = get_chosen(ps_pred, ps_dc, 3e-6)

# --- solve-rate curves ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax_l, ax_r = axes

pct_alpha = alphas * 100

# Left panel: PS-abstention signal
for ch, label, color, ls in [
    (ch_120,     "Always-120B (local only)", "black",    "-"),
    (ch_ps_lam3, "PS-routing λ=3e-6 (local only)", "#1f77b4", "-"),
]:
    vanilla = solve_curve(order_ps, ch, escalate_to_claude=False, alphas=alphas)
    claude  = solve_curve(order_ps, ch, escalate_to_claude=True,  alphas=alphas)
    ax_l.plot(pct_alpha, vanilla * 100, color=color, lw=2, ls="--", label=f"{label} (abstain)")
    ax_l.plot(pct_alpha, claude  * 100, color=color, lw=2, ls=ls,  label=f"{label} + Claude escalation")

# Oracle: abstain exactly the tasks Claude can't solve AND open models can't solve
always_120b_base = successes[:, 3].mean()
ax_l.axhline(always_120b_base * 100, color="gray", lw=1, ls=":", alpha=0.7, label="Always-120B baseline (0% abstention)")
ax_l.axhline(claude_s.mean() * 100,  color="red",  lw=1, ls=":", alpha=0.7, label="Always-Claude baseline")

ax_l.set_xlabel("Abstention rate (%)", fontsize=11)
ax_l.set_ylabel("Fraction of ALL tasks solved (%)", fontsize=11)
ax_l.set_title("Quality: Claude escalation vs vanilla abstention\n(PS signal, n=217 Python instances)", fontsize=10)
ax_l.legend(fontsize=7.5, loc="lower left")
ax_l.set_xlim(0, 85)
ax_l.set_ylim(35, 65)
ax_l.grid(True, alpha=0.3)
ax_l.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax_l.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

# Right panel: IO-abstention signal
for ch, label, color, ls in [
    (ch_120,     "Always-120B (local only)", "black",    "-"),
    (ch_ps_lam3, "PS-routing λ=3e-6 (local only)", "#1f77b4", "-"),
]:
    vanilla = solve_curve(order_io, ch, escalate_to_claude=False, alphas=alphas)
    claude  = solve_curve(order_io, ch, escalate_to_claude=True,  alphas=alphas)
    ax_r.plot(pct_alpha, vanilla * 100, color=color, lw=2, ls="--", label=f"{label} (abstain)")
    ax_r.plot(pct_alpha, claude  * 100, color=color, lw=2, ls=ls,  label=f"{label} + Claude escalation")

ax_r.axhline(always_120b_base * 100, color="gray", lw=1, ls=":", alpha=0.7, label="Always-120B baseline")
ax_r.axhline(claude_s.mean() * 100,  color="red",  lw=1, ls=":", alpha=0.7, label="Always-Claude baseline")

ax_r.set_xlabel("Abstention rate (%)", fontsize=11)
ax_r.set_ylabel("Fraction of ALL tasks solved (%)", fontsize=11)
ax_r.set_title("Quality: Claude escalation vs vanilla abstention\n(IO signal, n=217 Python instances)", fontsize=10)
ax_r.legend(fontsize=7.5, loc="lower left")
ax_r.set_xlim(0, 85)
ax_r.set_ylim(35, 65)
ax_r.grid(True, alpha=0.3)
ax_r.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax_r.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

plt.tight_layout()
plt.savefig(OUT_QUALITY, dpi=150)
print(f"\nSaved {OUT_QUALITY}")
plt.close()

# ── Figure 2: Pareto — solve rate vs cost ─────────────────────────────────────
#
# X-axis: mean per-task cost relative to always-120B (includes Claude costs)
# Y-axis: fraction of ALL tasks solved
#
# K = Claude cost per task / 120B mean cost per task, computed from real pricing:
#   Claude Sonnet 4.6 output: $15.00/MTok, mean 2991 output tokens/task
#     → Claude output cost/task = 2991 × 15e-6 = $0.04487
#   OSS-120B output: $11.13/MTok (from route_output_cost_weights config),
#     mean 1249 output tokens/task → 120B output cost/task = $0.01390
#   K = 0.04487 / 0.01390 = 3.23 (output-only, consistent with existing cost model)
#
# Note: Claude's input tokens add $5.88 for 286 tasks ($0.0206/task), pushing
# total-cost K to ~4.7, but we use output-only K for consistency with the
# existing framework which tracks only output tokens.

CLAUDE_K_EXACT = K_output_only  # derived above from actual API pricing + mean output tokens

ch_io_lam3 = get_chosen(io_pred, io_dc, 3e-6)

fig, ax = plt.subplots(figsize=(9, 5.5))

# Local-only routing curves — sweep λ
lam_vals = np.concatenate([[0], np.logspace(-7, -2.5, 120)])
ps_local = np.array([
    (successes[np.arange(N), get_chosen(ps_pred, ps_dc, l)].mean(),
     actual_costs[np.arange(N), get_chosen(ps_pred, ps_dc, l)].mean() / ref_cost)
    for l in lam_vals
])
io_local = np.array([
    (successes[np.arange(N), get_chosen(io_pred, io_dc, l)].mean(),
     actual_costs[np.arange(N), get_chosen(io_pred, io_dc, l)].mean() / ref_cost)
    for l in lam_vals
])
ax.plot(ps_local[:, 1] * 100, ps_local[:, 0] * 100,
        color="#1f77b4", lw=2.5, label="PS routing only (local)", zorder=5)
ax.plot(io_local[:, 1] * 100, io_local[:, 0] * 100,
        color="#ff7f0e", lw=2, ls="--", label="IO routing only (local)", zorder=5)

# Fixed baselines
bmarkers = ["^", "s", "D", "o"]
bcolors  = ["#2ca02c", "#9467bd", "#8c564b", "black"]
for i, (lbl, bm, bc) in enumerate(zip(["4B", "20B", "30B", "120B"], bmarkers, bcolors)):
    ax.scatter(actual_costs[:, i].mean() / ref_cost * 100,
               successes[:, i].mean() * 100,
               marker=bm, color=bc, s=70, zorder=6, label=f"Always-{lbl}")

# PS escalation curve
ps_sol = solve_curve(order_ps, ch_ps_lam3, escalate_to_claude=True, alphas=alphas)
ps_cst = cost_curve(order_ps, ch_ps_lam3, claude_k=CLAUDE_K_EXACT, alphas=alphas)
ax.plot(ps_cst * 100, ps_sol * 100, color="#1f77b4", lw=2.5,
        label="PS-routing + Claude escalation")

# IO escalation curve (IO signal for abstention ordering, IO routing for retained tasks)
io_sol = solve_curve(order_io, ch_io_lam3, escalate_to_claude=True, alphas=alphas)
io_cst = cost_curve(order_io, ch_io_lam3, claude_k=CLAUDE_K_EXACT, alphas=alphas)
ax.plot(io_cst * 100, io_sol * 100, color="#ff7f0e", lw=2.5,
        label="IO-routing + Claude escalation")

# Random mixing baseline: send fraction α of tasks to Claude, (1-α) to always-120B.
# Traces a straight line between always-120B and always-Claude endpoints.
alphas_rand = np.linspace(0, 1, 200)
rand_cost = (1 - alphas_rand) * 1.0 + alphas_rand * CLAUDE_K_EXACT  # relative to ref
rand_sol  = (1 - alphas_rand) * successes[:, 3].mean() + alphas_rand * claude_s.mean()
ax.plot(rand_cost * 100, rand_sol * 100, color="gray", lw=1.5, ls=":",
        label="Random 120B/Claude mix (no signal)")

# Always-Claude point
ax.scatter([CLAUDE_K_EXACT * 100],
           [claude_s.mean() * 100],
           marker="*", color="red", s=200, zorder=7, label="Always-Claude (54.4%)",
           edgecolors="black", lw=0.8)

ax.set_xlabel("Mean per-task cost (% of always-120B output cost)", fontsize=11)
ax.set_ylabel("Fraction of ALL tasks solved (%)", fontsize=11)
ax.set_title("Pareto: routing + Claude escalation vs local-only routing\n"
             f"Claude cost = {CLAUDE_K_EXACT:.2f}× 120B avg  "
             f"(Sonnet $15/MTok × {claude_mean_out_tok:.0f} tok vs 120B $11.13/MTok × {mean_120b_out_tok:.0f} tok)\n"
             "(n=217 Python eval instances)", fontsize=9)
ax.legend(fontsize=7.5, loc="lower right", ncol=1)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
plt.tight_layout()
plt.savefig(OUT_PARETO, dpi=150)
print(f"Saved {OUT_PARETO}")
plt.close()

# ── numeric summary ────────────────────────────────────────────────────────────
print("\n=== Claude escalation gain at various abstention rates ===")
print(f"(PS signal, PS-routing λ=3e-6 for retained tasks)")
print(f"\n{'Alpha':>7}  {'Local-only':>12}  {'+ Claude esc.':>14}  {'Gain':>6}  {'Claude rate on abstained':>24}")
vanilla_lam3 = solve_curve(order_ps, ch_ps_lam3, escalate_to_claude=False, alphas=alphas)
claude_lam3  = solve_curve(order_ps, ch_ps_lam3, escalate_to_claude=True,  alphas=alphas)
for a in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
    i = np.searchsorted(alphas, a)
    n_ab = int(round(a * N))
    abstained_tasks = order_ps[:n_ab]
    cr_on_ab = claude_s[abstained_tasks].mean() if n_ab > 0 else float("nan")
    print(f"  {a*100:5.0f}%  {vanilla_lam3[i]*100:12.1f}%  {claude_lam3[i]*100:14.1f}%  {(claude_lam3[i]-vanilla_lam3[i])*100:6.1f}%  {cr_on_ab*100:24.1f}%")

# Oracle escalation: for each α, what if we escalated the BEST α% of tasks to Claude?
# (i.e., order tasks by how much Claude beats the local model)
local_solved_lam3 = successes[np.arange(N), ch_ps_lam3]
claude_gain_per_task = claude_s.astype(float) - local_solved_lam3.astype(float)
order_by_claude_gain = np.argsort(-claude_gain_per_task)  # most gain first

print("\n=== Oracle escalation (escalate tasks where Claude helps most) ===")
print(f"{'Alpha':>7}  {'Oracle esc.':>12}  {'PS esc.':>10}  {'Diff':>6}")
for a in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
    i = np.searchsorted(alphas, a)
    n_ab = int(round(a * N))
    oracle_ab = order_by_claude_gain[:n_ab]
    oracle_kept = order_by_claude_gain[n_ab:]
    # Wait, oracle escalation should pick the best tasks to escalate (not abstain by PS)
    # Oracle chooses WHICH tasks to escalate to Claude to maximize solve rate:
    #   - Tasks where Claude succeeds AND local model fails: pure gain
    #   - Tasks where local model already succeeds: escalating wastes budget
    oracle_sol = (local_solved_lam3[oracle_kept].sum() + claude_s[oracle_ab].sum()) / N
    print(f"  {a*100:5.0f}%  {oracle_sol*100:12.1f}%  {claude_lam3[i]*100:10.1f}%  {(oracle_sol - claude_lam3[i])*100:6.1f}%")

print("\n=== Distribution of tasks by open-model vs Claude solvability ===")
any_open_solved = successes.any(axis=1)
print(f"  Claude solves & 120B solves (both succeed):  {(claude_s & successes[:, 3]).sum()} ({(claude_s & successes[:, 3]).mean()*100:.1f}%)")
print(f"  Claude solves, 120B doesn't (Claude-unique): {(claude_s & ~successes[:, 3]).sum()} ({(claude_s & ~successes[:, 3]).mean()*100:.1f}%)")
print(f"  120B solves, Claude doesn't (120B-unique):   {(~claude_s & successes[:, 3]).sum()} ({(~claude_s & successes[:, 3]).mean()*100:.1f}%)")
print(f"  Neither 120B nor Claude solves (both fail):  {(~claude_s & ~successes[:, 3]).sum()} ({(~claude_s & ~successes[:, 3]).mean()*100:.1f}%)")
print(f"  Union (any model or Claude solves):          {(claude_s | successes[:, 3]).sum()} ({(claude_s | successes[:, 3]).mean()*100:.1f}%)")

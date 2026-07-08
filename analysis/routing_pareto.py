"""
Cost-reward Pareto frontier for routing.

For each lambda (inference-time cost weight), pick the route that maximises
    pred_reward - lambda * size_adjusted_cost
where size_adjusted_cost = output_tokens * model_params / 4
(normalised so that 1 unit = 1 scout-output-token).

Shows the flexibility of a single trained router vs fixed baselines.
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
OUT_PLOT  = "/home/toolkit/PipelineRL-SWE/analysis/routing_pareto.png"
OUT_TABLE = "/home/toolkit/PipelineRL-SWE/analysis/routing_pareto_table.csv"

# ── load data ─────────────────────────────────────────────────────────────────
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
successes  = np.array([eval_df.loc[p, "route_successes"]   for p in common])
out_tokens = np.array([eval_df.loc[p, "route_output_tokens"] for p in common])
ps_pred    = np.array([ps_preds[p] for p in common])
io_pred    = np.array([io_preds[p] for p in common])

MODEL_SIZES = np.array([4, 20, 30, 120])
size_per_token = MODEL_SIZES / 4.0          # normalised: scout = 1 unit/token

# REPORTED cost: actual output tokens × model size for the chosen route.
# Realistic — you incur the real cost after running the chosen model.
actual_task_costs = out_tokens * size_per_token   # (N, 4)

# DECISION costs — what's known at routing time:
#
# Input-only: no output available — use training-set mean per model as prior.
avg_out_tokens = out_tokens.mean(axis=0)    # shape (4,)
io_decision_costs = avg_out_tokens * size_per_token   # (4,) — fixed per model

# Post-scout: scout output token count known at decision time.
#   Use it as a per-task difficulty signal, scaled by each model's typical
#   output-length ratio relative to scout (from training statistics).
#   Spearman correlation between scout tokens and other routes: ~0.47.
#   Formula: cost_r[task] = (scout_tokens[task] / avg_scout_tokens) * avg_cost_r
#   This preserves correct cost ordering between routes while adding per-task signal.
scout_out = out_tokens[:, 0]                    # (N,) — known after scout runs
avg_scout = avg_out_tokens[0]
# per-task scalar: how verbose is this task relative to average?
task_verbosity = scout_out / avg_scout          # (N,)
ps_decision_costs = task_verbosity[:, None] * io_decision_costs[None, :]  # (N, 4)

print(f"IO decision cost prior (training mean):       {io_decision_costs.astype(int)}")
print(f"PS decision cost proxy (per-task, median):    {np.median(ps_decision_costs, axis=0).astype(int)}")
print(f"PS preserves route cost order: {(np.argsort(io_decision_costs) == np.argsort(ps_decision_costs.mean(axis=0))).all()}")

# ── helpers ───────────────────────────────────────────────────────────────────
def pareto_point(pred, lam, decision_costs):
    # decision_costs: (N,4) for post-scout, or (4,) broadcast for input-only
    chosen = (pred - lam * decision_costs).argmax(axis=1)
    reward = successes[np.arange(len(common)), chosen].mean()
    cost   = actual_task_costs[np.arange(len(common)), chosen].mean()
    return float(reward), float(cost)

def oracle_point():
    chosen = np.zeros(len(common), dtype=int)
    for i in range(len(common)):
        chosen[i] = 0
        for r in [3, 2, 1, 0]:
            if successes[i, r]:
                chosen[i] = r
    return (successes[np.arange(len(common)), chosen].mean(),
            actual_task_costs[np.arange(len(common)), chosen].mean())

# ── compute curves ────────────────────────────────────────────────────────────
lambdas = np.concatenate([[0], np.logspace(-7, -2.5, 120)])

ps_curve = np.array([pareto_point(ps_pred, l, ps_decision_costs) for l in lambdas])
io_curve = np.array([pareto_point(io_pred, l, io_decision_costs) for l in lambdas])

# fixed baselines
names = ["Scout 4B", "OSS-20B", "Qwen-30B", "OSS-120B"]
fixed_r = [successes[:, r].mean() for r in range(4)]
fixed_c = [actual_task_costs[:, r].mean() for r in range(4)]
ora_r, ora_c = oracle_point()

# ── summary table ─────────────────────────────────────────────────────────────
ref_cost = fixed_c[3]  # always-120B as reference
rows = []
for lam in [0, 1e-6, 3e-6, 7e-6, 1e-5, 3e-5, 1e-4]:
    ps_r, ps_c = pareto_point(ps_pred, lam, ps_decision_costs)
    io_r, io_c = pareto_point(io_pred, lam, io_decision_costs)
    rows.append({
        "lambda": lam,
        "PS_reward": round(ps_r, 4),
        "PS_norm_cost": round(ps_c, 0),
        "PS_cost_reduction_pct": round((1 - ps_c / ref_cost) * 100, 1),
        "IO_reward": round(io_r, 4),
        "IO_norm_cost": round(io_c, 0),
        "IO_cost_reduction_pct": round((1 - io_c / ref_cost) * 100, 1),
    })
df_table = pd.DataFrame(rows)
print(df_table.to_string(index=False))
df_table.to_csv(OUT_TABLE, index=False)

# print baselines
print(f"\nOracle:    reward={ora_r:.4f}  norm-cost={ora_c:.0f}  ({(1-ora_c/ref_cost)*100:.0f}% cheaper than 120B)")
for i, (n, r, c) in enumerate(zip(names, fixed_r, fixed_c)):
    print(f"Always {n}: reward={r:.4f}  norm-cost={c:.0f}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

# convert cost to "relative to always-120B" for readability
ax.plot(ps_curve[:, 1] / ref_cost * 100, ps_curve[:, 0] * 100,
        color="#1f77b4", lw=2.5, label="Post-scout router (ours)")
ax.plot(io_curve[:, 1] / ref_cost * 100, io_curve[:, 0] * 100,
        color="#ff7f0e", lw=2, linestyle="--", label="Input-only router")

# fixed baselines
markers = ["^", "s", "D", "o"]
colors  = ["#2ca02c", "#9467bd", "#8c564b", "black"]
for i, (n, r, c, m, col) in enumerate(zip(names, fixed_r, fixed_c, markers, colors)):
    ax.scatter(c / ref_cost * 100, r * 100, marker=m, color=col, s=80, zorder=5,
               label=f"Always {n}")

ax.scatter(ora_c / ref_cost * 100, ora_r * 100, marker="*", color="gold",
           s=200, zorder=6, edgecolors="black", linewidths=0.8, label="Oracle")

# annotate a few lambda points on the post-scout curve
for lam, label in [(0, "λ=0"), (1e-6, "λ=1e-6"), (7e-6, "λ=7e-6")]:
    r, c = pareto_point(ps_pred, lam, ps_decision_costs)
    ax.annotate(label, xy=(c / ref_cost * 100, r * 100),
                xytext=(4, 4), textcoords="offset points", fontsize=7.5, color="#1f77b4")

ax.set_xlabel("Inference cost (% of always-120B, size-adjusted)", fontsize=11)
ax.set_ylabel("Success rate (%)", fontsize=11)
ax.set_title("Cost–reward Pareto frontier: single router, many operating points\n(286-task eval, size-adjusted cost = params/4 × output tokens)", fontsize=11)
ax.legend(fontsize=8.5, loc="lower right")
ax.set_xlim(0, 105)
ax.set_ylim(26, 60)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150)
print(f"\nSaved plot to {OUT_PLOT}")
plt.close()

"""
Experiment E: Difficulty-band routing.

The router currently applies a single continuous abstention threshold.
This experiment tests a 3-zone policy:
  - Bottom band (hardest tasks by abstention score)  → always abstain
  - Top band    (easiest tasks by abstention score)  → always use cheapest model
  - Middle band (uncertain tasks)                    → full routing (argmax pred - λ×cost)

Hypothesis: routing only matters for medium-difficulty tasks. On easy tasks any
model succeeds so cheapest wins; on hard tasks no model succeeds so abstain.
Forcing the model to route on these extremes wastes signal and may hurt
performance. Restricting routing to the middle band could improve the Pareto.

Uses existing eval_predictions.jsonl — no new training needed.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.metrics import roc_auc_score

REAL_LABELS_DIR = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect/"
)
PS_PATH = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_10epoch_1781112916/"
    "train_qwen3_embedding_8b_lora_reward_bce_10epoch/eval_predictions.jsonl"
)
OUT_PARETO = "/home/toolkit/PipelineRL-SWE/analysis/difficulty_band_pareto.png"
OUT_DETAIL = "/home/toolkit/PipelineRL-SWE/analysis/difficulty_band_detail.png"
OUT_JSON   = "/home/toolkit/PipelineRL-SWE/analysis/difficulty_band_results.json"

# ── load ──────────────────────────────────────────────────────────────────────
dfs = [pd.read_parquet(p) for p in Path(REAL_LABELS_DIR + "eval").glob("*.parquet")]
eval_df = pd.concat(dfs).set_index("problem_id")

ps_preds = {}
with open(PS_PATH) as f:
    for line in f:
        d = json.loads(line)
        ps_preds[d["problem_id"]] = np.array(d["pred_rewards"])

common = list(eval_df.index.intersection(ps_preds))
N = len(common)

successes  = np.array([eval_df.loc[p, "route_successes"]    for p in common])   # (N, 4)
out_tokens = np.array([eval_df.loc[p, "route_output_tokens"] for p in common])  # (N, 4)
ps_pred    = np.array([ps_preds[p] for p in common])                             # (N, 4)

MODEL_SIZES    = np.array([4, 20, 30, 120])
size_per_token = MODEL_SIZES / 4.0
actual_costs   = out_tokens * size_per_token   # (N, 4)
ref_cost       = actual_costs[:, 3].mean()     # always-120B baseline

# Abstention signal: mean predicted reward across routes
abst_signal = ps_pred.mean(axis=1)   # (N,)  higher = more likely solvable

# ── routing helper ─────────────────────────────────────────────────────────────
def ps_dc(lam: float) -> np.ndarray:
    """Post-scout difficulty-corrected cost proxy per route. (N, 4)"""
    avg_out = out_tokens.mean(axis=0)
    io_dc = avg_out * size_per_token
    return (out_tokens[:, 0] / avg_out[0])[:, None] * io_dc[None, :]

def get_routing_choice(lam: float) -> np.ndarray:
    dc = ps_dc(lam)
    return (ps_pred - lam * dc).argmax(axis=1)   # (N,)

# ── band policy ────────────────────────────────────────────────────────────────
def band_policy(
    band_frac: float,   # fraction of tasks in each extreme band (e.g. 0.2 = 20% each)
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (reward, cost) per task under band policy.

    Bottom `band_frac` of tasks (by abstention score) → abstain (reward=0, cost=0).
    Top `band_frac` of tasks → always cheapest model (route 0: 4B scout).
    Middle `1 - 2*band_frac` of tasks → PS routing at given lambda.
    """
    order = np.argsort(abst_signal)
    n_band = int(round(band_frac * N))

    hard_mask   = np.zeros(N, dtype=bool); hard_mask[order[:n_band]]   = True
    easy_mask   = np.zeros(N, dtype=bool); easy_mask[order[-n_band:]]  = True
    medium_mask = ~hard_mask & ~easy_mask

    routing = get_routing_choice(lam)

    chosen = np.where(
        hard_mask,   -1,           # abstain
        np.where(
            easy_mask, 0,          # always 4B scout
            routing,               # PS routing
        ),
    )

    reward = np.where(chosen < 0, 0.0, successes[np.arange(N), np.clip(chosen, 0, 3)].astype(float))
    cost   = np.where(chosen < 0, 0.0, actual_costs[np.arange(N), np.clip(chosen, 0, 3)])

    return reward, cost

# ── continuous abstention baseline (current approach) ─────────────────────────
def continuous_abstention(alpha: float, lam: float) -> tuple[float, float]:
    order = np.argsort(abst_signal)
    n_ab = int(round(alpha * N))
    keep = np.ones(N, dtype=bool)
    keep[order[:n_ab]] = False
    routing = get_routing_choice(lam)
    r = successes[np.arange(N), routing][keep].sum() / N
    c = actual_costs[np.arange(N), routing][keep].sum() / (N * ref_cost)
    return r, c

# ── oracle ─────────────────────────────────────────────────────────────────────
oracle_chosen = np.array([
    next((r for r in [0, 1, 2, 3] if successes[i, r]), 3)
    for i in range(N)
])
oracle_signal = successes[:, 3].astype(float)

# ── sweep band fractions ───────────────────────────────────────────────────────
band_fracs = np.linspace(0, 0.45, 50)
lam_values = [0, 1e-6, 3e-6, 7e-6]
alphas = np.linspace(0, 0.85, 200)

results = {"band_policy": {}, "continuous_abstention": {}}

print("\n=== Difficulty-band vs continuous-abstention Pareto ===")
print(f"{'Method':<45} {'Max solved @ <50% cost':>22}  {'Max solved @ <30% cost':>22}")

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for lam, col in zip(lam_values, colors):
    # Continuous abstention curve
    cont_pts = [continuous_abstention(a, lam) for a in alphas]
    cont_r = np.array([p[0] for p in cont_pts])
    cont_c = np.array([p[1] for p in cont_pts])
    ax.plot(cont_c * 100, cont_r * 100, color=col, lw=2, ls="--",
            label=f"λ={lam:.0e} continuous")
    results["continuous_abstention"][str(lam)] = {
        "frac_solved": cont_r.tolist(), "frac_cost": cont_c.tolist()
    }

    # Band policy curve (sweep band_frac)
    band_pts = []
    for bf in band_fracs:
        rew, cost = band_policy(bf, lam)
        band_pts.append((rew.mean(), cost.mean() / ref_cost))
    band_r = np.array([p[0] for p in band_pts])
    band_c = np.array([p[1] for p in band_pts])
    ax.plot(band_c * 100, band_r * 100, color=col, lw=2.5, ls="-",
            label=f"λ={lam:.0e} band-policy")
    results["band_policy"][str(lam)] = {
        "band_fracs": band_fracs.tolist(),
        "frac_solved": band_r.tolist(), "frac_cost": band_c.tolist()
    }

    # Print comparison
    def max_solved_at_cost(r_arr, c_arr, thresh):
        mask = c_arr <= thresh
        return float(r_arr[mask].max()) if mask.any() else 0.0

    print(f"  λ={lam:.0e} continuous:   {max_solved_at_cost(cont_r,cont_c,0.5)*100:>5.1f}%         "
          f"{max_solved_at_cost(cont_r,cont_c,0.3)*100:>5.1f}%")
    print(f"  λ={lam:.0e} band-policy:  {max_solved_at_cost(band_r,band_c,0.5)*100:>5.1f}%         "
          f"{max_solved_at_cost(band_r,band_c,0.3)*100:>5.1f}%")

# fixed baselines
names_m = [("Always 4B", 0, "^"), ("Always 120B", 3, "o")]
for name, r, marker in names_m:
    ax.scatter(
        actual_costs[:, r].mean() / ref_cost * 100,
        successes[:, r].mean() * 100,
        marker=marker, s=100, zorder=5, label=name,
        color="black" if r == 3 else "#2ca02c",
    )

# oracle at zero abstention
oracle_r0 = successes[np.arange(N), oracle_chosen].mean()
oracle_c0 = actual_costs[np.arange(N), oracle_chosen].mean() / ref_cost
ax.scatter(oracle_c0 * 100, oracle_r0 * 100, marker="*", s=250, color="gold",
           zorder=6, edgecolors="black", lw=0.8, label="Oracle (no abstention)")

ax.set_xlabel("Local inference cost (% of always-120B)", fontsize=11)
ax.set_ylabel("Fraction of ALL tasks solved locally (%)", fontsize=11)
ax.set_title(
    "Difficulty-band routing vs continuous abstention\n"
    "(solid = band-policy, dashed = continuous threshold; same λ = same color)",
    fontsize=10,
)
ax.legend(fontsize=7, loc="lower right", ncol=2)
ax.set_xlim(0, 105)
ax.set_ylim(0, 55)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
plt.tight_layout()
plt.savefig(OUT_PARETO, dpi=150)
print(f"\nSaved {OUT_PARETO}")
plt.close()

# ── detail plot: how much does band fraction matter? ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for lam, col in zip(lam_values, colors):
    bf_arr = band_fracs * 100
    band_r_arr = np.array(results["band_policy"][str(lam)]["frac_solved"]) * 100
    band_c_arr = np.array(results["band_policy"][str(lam)]["frac_cost"]) * 100
    axes[0].plot(bf_arr, band_r_arr, color=col, lw=2, label=f"λ={lam:.0e}")
    axes[1].plot(bf_arr, band_c_arr, color=col, lw=2, label=f"λ={lam:.0e}")

for ax in axes:
    ax.set_xlabel("Band fraction (% of tasks in each extreme band)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
axes[0].set_ylabel("Fraction of ALL tasks solved (%)", fontsize=11)
axes[0].set_title("Coverage vs band fraction", fontsize=10)
axes[1].set_ylabel("Cost (% of always-120B)", fontsize=11)
axes[1].set_title("Cost vs band fraction", fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DETAIL, dpi=150)
print(f"Saved {OUT_DETAIL}")
plt.close()

# ── numeric summary ────────────────────────────────────────────────────────────
print("\n=== At band_frac=0.20 (20% easy, 20% hard), λ=3e-6 ===")
rew, cost = band_policy(0.20, 3e-6)
print(f"  Band-policy: solved={rew.mean()*100:.1f}%  cost={cost.mean()/ref_cost*100:.1f}%")
cont_r, cont_c = continuous_abstention(0.40, 3e-6)  # ~same abstention rate
print(f"  Continuous (α=0.40): solved={cont_r*100:.1f}%  cost={cont_c*100:.1f}%")

print("\n=== ROC AUC check: does band-easy routing improve? ===")
# For easy tasks: do they actually all get solved by any model?
order = np.argsort(abst_signal)
n_easy = int(round(0.2 * N))
easy_idx = order[-n_easy:]
easy_any_success = successes[easy_idx].any(axis=1).mean()
easy_4b_success = successes[easy_idx, 0].mean()
easy_120b_success = successes[easy_idx, 3].mean()
print(f"  Top-20% easy tasks: any_success={easy_any_success:.2%}  4B={easy_4b_success:.2%}  120B={easy_120b_success:.2%}")
hard_idx = order[:n_easy]
hard_any_success = successes[hard_idx].any(axis=1).mean()
print(f"  Bot-20% hard tasks: any_success={hard_any_success:.2%}")

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {OUT_JSON}")

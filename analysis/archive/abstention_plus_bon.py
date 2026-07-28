"""
Combined abstention + best-of-N simulation on the 150-task multirollout eval set.

Route 0 = 4B scout, Route 3 = 120B solver (the expensive one).

Strategies compared (all using 120B route):
  A) Always 120B×1          — no verifier, pick rollout 0
  B) Verifier abstain only  — abstain by mean score, keep 120B×1 on rest
  C) Verifier best-of-3     — no abstention, pick best of 3 rollouts by verifier score
  D) Verifier abstain + B3  — combine abstention AND best-of-3 selection
  E) Random abstain + B3    — random abstention + verifier best-of-3 (isolates abstention value)
  F) Oracle abstain + B3    — perfect hopeless-task filter + verifier best-of-3 (upper bound)

Abstention signal: mean pred_score across all 4 routes × 3 rollouts per task.
Best-of-3 signal: pred_score for the specific route/rollout.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from collections import defaultdict

SCORES_PATH = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "score_swe_smith_multirollout_eval150_proxy_verifier_soft_bce_1781735614/"
    "scores/eval_verifier_scores.jsonl"
)
LOG_DIR = Path(
    "/home/toolkit/PipelineRL-SWE/router_analysis/uploaded_eval_full_20260617/"
    "swe_smith_multirollout_eval150_1781382734/logs/run_evaluation"
)
OUT_PATH = "/home/toolkit/PipelineRL-SWE/router_analysis/abstention_plus_bon.png"

ROUTE_120B = 3  # solver:openai/gpt-oss-120b

# ── load verifier scores ──────────────────────────────────────────────────────
rows = [json.loads(l) for l in open(SCORES_PATH)]
df = pd.DataFrame(rows)

# The cascade verifier sees each solver's OWN output (problem_text + solver_patch).
# For a realistic pipeline:
#   - 4B runs first (cheap scout)
#   - cascade verifier scores the 4B output → abstention signal (pre-120B)
#   - if not abstaining, run 120B×K
#   - cascade verifier scores each 120B rollout → best-of-N signal (post-120B)
#
# So: abstention signal = mean pred_score over 4B rollouts only (route_idx=0)
#     BoN signal        = pred_score per 120B rollout (route_idx=3)

ROUTE_4B = 0

df_4b = df[df["route_idx"] == ROUTE_4B].copy()
task_4b_score = df_4b.groupby("original_problem_id")["pred_score"].mean()  # mean over 3 4B rollouts

# 120B-specific scores per rollout: task → rollout → pred_score
df_120b = df[df["route_idx"] == ROUTE_120B].copy()
task_rollout_score = (
    df_120b.groupby(["original_problem_id", "rollout_idx"])["pred_score"]
    .first()
    .unstack()  # columns = rollout_idx 0,1,2
)

all_tasks = sorted(task_4b_score.index)
n_tasks = len(all_tasks)
print(f"Tasks: {n_tasks}")

# ── load real labels ──────────────────────────────────────────────────────────
# Run names follow: swe_smith_eval150_{model}_{rollout_tag}
# Route 120B run names contain "gpt_oss_120b"
task_any_success = defaultdict(bool)   # any route/rollout
task_120b_rollout_success = defaultdict(lambda: [None, None, None])  # task -> [r0, r1, r2]

route_120b_runs = {
    0: "swe_smith_eval150_gpt_oss_120b_r0",
    1: "swe_smith_eval150_gpt_oss_120b_r1",
    2: "swe_smith_eval150_gpt_oss_120b_r2",
}

for rollout_idx, run_name in route_120b_runs.items():
    report = LOG_DIR / run_name / "report.json"
    d = json.loads(report.read_text())
    for pid in d.get("ids_resolved", []):
        task_120b_rollout_success[pid][rollout_idx] = 1
    for pid in d.get("ids_unresolved", []):
        if task_120b_rollout_success[pid][rollout_idx] is None:
            task_120b_rollout_success[pid][rollout_idx] = 0

# any-route success (for oracle)
for report in LOG_DIR.glob("*/report.json"):
    d = json.loads(report.read_text())
    for pid in d.get("ids_resolved", []):
        task_any_success[pid] = True

# align to common tasks
tasks = [t for t in all_tasks if t in task_120b_rollout_success and t in task_rollout_score.index]
n = len(tasks)
print(f"Tasks with 120B rollout results: {n}")

abst_signal = np.array([task_4b_score[t] for t in tasks])   # 4B cascade verifier score → abstention
hopeless = np.array([not task_any_success[t] for t in tasks])

# 120B per-rollout success: shape (n, 3)
rollout_success = np.array([task_120b_rollout_success[t] for t in tasks], dtype=float)
rollout_success = np.nan_to_num(rollout_success, nan=0.0)

# 120B per-rollout verifier scores: shape (n, 3)
rollout_score = task_rollout_score.loc[tasks].values  # (n, 3)

n_hopeless = hopeless.sum()
oracle_rate = n_hopeless / n
print(f"Hopeless: {n_hopeless}/{n} ({oracle_rate*100:.1f}%)")
print(f"Always-120B×1 accuracy: {rollout_success[:, 0].mean()*100:.1f}%")
print(f"Any-rollout-120B accuracy: {(rollout_success.max(axis=1)>0).mean()*100:.1f}%")

# ── simulation functions ──────────────────────────────────────────────────────
def best_of_k_verifier(rollout_success, rollout_score):
    """For each task pick the rollout with highest verifier score. Return binary success array."""
    best_rollout = np.argmax(rollout_score, axis=1)
    return rollout_success[np.arange(len(rollout_success)), best_rollout]

def simulate(rates, abst_signal, rollout_success, rollout_score, hopeless,
             use_bon=True, n_rnd_seeds=200, rng_seed=42):
    """
    Returns dict of strategy -> accuracy array over rates.
    abstain = sort by abst_signal ascending, drop bottom-r fraction.
    """
    n = len(abst_signal)
    abst_order = np.argsort(abst_signal)          # ascending: lowest score first

    baseline_120b1 = rollout_success[:, 0]
    bon_success = best_of_k_verifier(rollout_success, rollout_score)
    oracle_keep_order = np.argsort(~hopeless)      # hopeless first, so drop them

    rng = np.random.default_rng(rng_seed)

    res = {
        "always_120b1":   [],
        "abst_bon":       [],
        "abst_only":      [],
        "bon_only":       [],
        "rnd_abst_bon":   [],
        "oracle_abst_bon":[],
    }

    for r in rates:
        n_drop = int(round(r * n))
        kept_verifier = abst_order[n_drop:]
        kept_oracle   = oracle_keep_order[n_drop:]

        def acc(idx, arr):
            return arr[idx].mean() if len(idx) > 0 else np.nan

        res["always_120b1"].append(baseline_120b1.mean())          # doesn't change with rate
        res["abst_only"].append(acc(kept_verifier, baseline_120b1))
        res["bon_only"].append(bon_success.mean())                  # doesn't change with rate
        res["abst_bon"].append(acc(kept_verifier, bon_success))
        res["oracle_abst_bon"].append(acc(kept_oracle, bon_success))

        # random abstention + BoN
        rnd_accs = []
        for _ in range(n_rnd_seeds):
            perm = rng.permutation(n)
            kept_rnd = perm[n_drop:]
            rnd_accs.append(bon_success[kept_rnd].mean() if len(kept_rnd) > 0 else np.nan)
        res["rnd_abst_bon"].append(np.nanmean(rnd_accs))

    return {k: np.array(v) for k, v in res.items()}

rates = np.linspace(0, 0.75, 100)
res = simulate(rates, abst_signal, rollout_success, rollout_score, hopeless)

# print summary
print("\n=== Combined abstention + best-of-3 (150 tasks, 120B route) ===")
print(f"{'Rate':>6}  {'120B×1':>8}  {'AbstOnly':>10}  {'BoN(×3)':>9}  {'Abst+BoN':>10}  {'Oracle+BoN':>12}  {'Rnd+BoN':>9}")
for rate in [0.0, 0.20, 0.30, 0.40, 0.50, 0.60]:
    i = np.searchsorted(rates, rate)
    print(
        f"{rate*100:>5.0f}%  "
        f"{res['always_120b1'][i]*100:>8.1f}  "
        f"{res['abst_only'][i]*100:>10.1f}  "
        f"{res['bon_only'][i]*100:>9.1f}  "
        f"{res['abst_bon'][i]*100:>10.1f}  "
        f"{res['oracle_abst_bon'][i]*100:>12.1f}  "
        f"{res['rnd_abst_bon'][i]*100:>9.1f}"
    )

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.5))

r100 = rates * 100

ax.axhline(res["always_120b1"][0] * 100, color="gray", lw=1.2, linestyle=":",
           label=f"Always 120B×1 ({res['always_120b1'][0]*100:.1f}%)")
ax.axhline(res["bon_only"][0] * 100, color="#9467bd", lw=1.2, linestyle="--",
           label=f"120B×3, verifier selects — no abstention ({res['bon_only'][0]*100:.1f}%)")

ax.plot(r100, res["abst_only"] * 100,    color="#ff7f0e", lw=2,
        label="Verifier abstention + 120B×1")
ax.plot(r100, res["rnd_abst_bon"] * 100, color="#2ca02c", lw=1.5, linestyle="--",
        label="Random abstention + 120B×3 verifier")
ax.plot(r100, res["abst_bon"] * 100,     color="#1f77b4", lw=2.5,
        label="Verifier abstention + 120B×3 verifier (combined)")
ax.plot(r100, res["oracle_abst_bon"] * 100, color="black", lw=1.5, linestyle="-.",
        label="Oracle abstention + 120B×3 verifier")

ax.axvline(oracle_rate * 100, color="black", lw=0.8, linestyle=":", alpha=0.5)
ax.text(oracle_rate * 100 + 0.5, ax.get_ylim()[0] + 2 if ax.get_ylim()[0] > 0 else 52,
        f"Oracle rate ({oracle_rate*100:.1f}%)", fontsize=8, color="black", alpha=0.7)

# annotate combined gain at 40%
i40 = np.searchsorted(rates, 0.40)
ax.annotate(
    f"+{(res['abst_bon'][i40] - res['always_120b1'][i40])*100:.1f}pp vs 120B×1",
    xy=(40, res["abst_bon"][i40] * 100),
    xytext=(42, res["abst_bon"][i40] * 100 - 4),
    fontsize=8, color="#1f77b4",
    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8),
)

ax.set_xlabel("Abstention rate (%)", fontsize=12)
ax.set_ylabel("Accuracy on retained tasks (%)", fontsize=12)
ax.set_title(
    "Abstention + best-of-3 selection on 150-task eval set\n"
    "(abstain on 4B score; BoN selects among 120B rollouts; cascade verifier, proxy rewards)",
    fontsize=11
)
ax.legend(fontsize=8.5, loc="upper left")
ax.set_xlim(0, 75)
ax.set_ylim(50, 102)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"\nSaved to {OUT_PATH}")
plt.close()

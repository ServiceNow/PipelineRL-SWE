"""
Multi-baseline abstention comparison:
  - Post-scout verifier (best signal, has 4B cost)
  - Input-only verifier (no scout cost, just problem statement)
  - Prompt length heuristic (negative token count → shorter = "easier")
  - Random baseline (averaged over 200 seeds)

Abstention: we sort tasks by ascending signal strength (lower score → more hopeless)
and refuse to submit for the bottom-X% of tasks.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
POST_SCOUT_PATH = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_10epoch_1781112916/"
    "train_qwen3_embedding_8b_lora_reward_bce_10epoch/eval_predictions.jsonl"
)
INPUT_ONLY_PATH = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_input_only_10epoch_1781112942/"
    "train_qwen3_embedding_8b_lora_reward_bce_10epoch/eval_predictions.jsonl"
)
REAL_LABELS_DIR = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect/"
)
OUT_PATH = "/home/toolkit/PipelineRL-SWE/router_analysis/abstention_baselines.png"

# ── load post-scout verifier predictions ──────────────────────────────────────
def load_preds(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # pred_rewards is a list of per-route predicted rewards
    df["mean_pred"] = df["pred_rewards"].apply(lambda x: np.mean(x) if x is not None else np.nan)
    return df.set_index("problem_id")

ps_preds = load_preds(POST_SCOUT_PATH)
io_preds = load_preds(INPUT_ONLY_PATH)
print(f"Post-scout preds: {len(ps_preds)} tasks")
print(f"Input-only preds: {len(io_preds)} tasks")

# ── load real labels ──────────────────────────────────────────────────────────
dfs = [pd.read_parquet(p) for p in Path(REAL_LABELS_DIR + "eval").glob("*.parquet")]
eval_df = pd.concat(dfs).set_index("problem_id")

# r120b: task is solved if the 120B route (index 3) succeeded — honest deployed metric
eval_df["r120b"] = eval_df["route_successes"].apply(lambda x: x[3])

# prompt tokens: take the first route's token count (all identical for input portion)
eval_df["prompt_tokens"] = eval_df["route_prompt_tokens"].apply(lambda x: x[0] if x is not None and len(x) > 0 else np.nan)

print(f"Eval tasks: {len(eval_df)}, 120B solvable: {eval_df['r120b'].sum()}")

# ── align on common tasks ─────────────────────────────────────────────────────
common = eval_df.index.intersection(ps_preds.index).intersection(io_preds.index)
print(f"Common tasks with all three signals: {len(common)}")

real   = eval_df.loc[common, "r120b"].values.astype(int)
ps_sig = ps_preds.loc[common, "mean_pred"].values       # higher = more confident = don't abstain
io_sig = io_preds.loc[common, "mean_pred"].values
pl_sig = -eval_df.loc[common, "prompt_tokens"].values   # shorter prompt = less context = harder? (inverted)

n_tasks = len(common)
n_hopeless = (real == 0).sum()
oracle_rate = n_hopeless / n_tasks
print(f"Oracle abstention rate: {oracle_rate:.3f}")

# ── abstention curve function ─────────────────────────────────────────────────
def abstention_curve(signal, labels, rates):
    """
    For each abstention rate r: abstain on the r-fraction of tasks with lowest signal.
    Return accuracy on the remaining tasks.
    """
    order = np.argsort(signal)  # ascending → lowest signal first (abstain these)
    accs = []
    for r in rates:
        n_abstain = int(round(r * len(labels)))
        kept = order[n_abstain:]
        if len(kept) == 0:
            accs.append(np.nan)
        else:
            accs.append(labels[kept].mean())
    return np.array(accs)

def random_abstention_curve(labels, rates, n_seeds=200, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    n = len(labels)
    all_accs = []
    for _ in range(n_seeds):
        perm = rng.permutation(n)
        seed_accs = []
        for r in rates:
            n_abstain = int(round(r * n))
            kept = perm[n_abstain:]
            seed_accs.append(labels[kept].mean() if len(kept) > 0 else np.nan)
        all_accs.append(seed_accs)
    return np.nanmean(all_accs, axis=0)

rates = np.linspace(0, 0.85, 100)

ps_acc  = abstention_curve(ps_sig, real, rates)
io_acc  = abstention_curve(io_sig, real, rates)
pl_acc  = abstention_curve(pl_sig, real, rates)
rnd_acc = random_abstention_curve(real, rates)

# oracle: abstain exactly the hopeless ones
oracle_acc = np.array([
    real[np.argsort(real)[::-1][:max(1, int(round((1-r)*n_tasks)))]].mean()
    if int(round(r * n_tasks)) <= n_hopeless else np.nan
    for r in rates
])
# simpler oracle: if you abstain the right ones, remaining are all successes
oracle_given_rate = np.array([
    min(1.0, n_hopeless / max(1, int(round(r * n_tasks)))) * 0 +
    real.sum() / max(1, n_tasks - int(round(r * n_tasks)))
    for r in rates
])
# Clean oracle: abstain bottom r fraction, assuming we pick the r*n least solvable perfectly
def oracle_curve(labels, rates):
    n = len(labels)
    sorted_labels = np.sort(labels)  # worst first → 0s first
    accs = []
    for r in rates:
        n_abstain = int(round(r * n))
        kept = sorted_labels[n_abstain:]
        accs.append(kept.mean() if len(kept) > 0 else np.nan)
    return np.array(accs)

ora_acc = oracle_curve(real, rates)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(rates * 100, ps_acc * 100,  color="#1f77b4", lw=2,   label="Post-scout verifier (ours)")
ax.plot(rates * 100, io_acc * 100,  color="#ff7f0e", lw=2,   label="Input-only verifier (no scout cost)")
ax.plot(rates * 100, pl_acc * 100,  color="#2ca02c", lw=1.5, linestyle="--", label="Prompt length heuristic")
ax.plot(rates * 100, rnd_acc * 100, color="gray",    lw=1.5, linestyle=":",  label="Random baseline")
ax.plot(rates * 100, ora_acc * 100, color="black",   lw=1.5, linestyle="-.", label="Oracle")

# mark oracle abstention rate
ax.axvline(oracle_rate * 100, color="black", lw=0.8, linestyle=":", alpha=0.6)
ax.text(oracle_rate * 100 + 0.5, 52, f"Oracle rate\n({oracle_rate*100:.1f}%)", fontsize=8, color="black", alpha=0.7)

ax.set_xlabel("Abstention rate (%)", fontsize=12)
ax.set_ylabel("Accuracy on retained tasks (%)", fontsize=12)
ax.set_title("Abstention comparison: post-scout vs baselines\n(286-task eval set)", fontsize=13)
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(0, 85)
ax.set_ylim(50, 102)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
ax.grid(True, alpha=0.3)

# annotation box with key numbers
ps_at40 = ps_acc[np.searchsorted(rates, 0.40)]
io_at40 = io_acc[np.searchsorted(rates, 0.40)]
pl_at40 = pl_acc[np.searchsorted(rates, 0.40)]
rnd_at40 = rnd_acc[np.searchsorted(rates, 0.40)]
txt = (
    f"At 40% abstention:\n"
    f"  Post-scout:    {ps_at40*100:.1f}%\n"
    f"  Input-only:    {io_at40*100:.1f}%\n"
    f"  Prompt length: {pl_at40*100:.1f}%\n"
    f"  Random:        {rnd_at40*100:.1f}%"
)
ax.text(0.97, 0.04, txt, transform=ax.transAxes, fontsize=8,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"Saved to {OUT_PATH}")
plt.close()

# ── summary table ─────────────────────────────────────────────────────────────
print("\n=== Abstention accuracy comparison ===")
print(f"{'Rate':>6}  {'Post-scout':>10}  {'Input-only':>10}  {'Prompt len':>10}  {'Random':>8}  {'Oracle':>8}")
for rate in [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
    i = np.searchsorted(rates, rate)
    print(f"{rate*100:>5.0f}%  {ps_acc[i]*100:>10.1f}  {io_acc[i]*100:>10.1f}  {pl_acc[i]*100:>10.1f}  {rnd_acc[i]*100:>8.1f}  {ora_acc[i]*100:>8.1f}")

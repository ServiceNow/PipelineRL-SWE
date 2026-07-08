"""
Abstention AUC analysis for:
  - Post-scout proxy-init + real fine-tune (N=50,100,200,500,1000)
  - Proxy-only baseline (no real fine-tuning)
  - Cascade-real-1500 (scored at route_0 decision point)
  - Existing baselines: post-scout real-1500, input-only real-1500, random

Prints a table of AUC and key abstention rates for the paper.
Also saves a sample-efficiency plot.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
SWEEP_BASE = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_proxy_init_real_finetune_router_sample_efficiency_1783430086"
)
CASCADE_SCORES_PATH = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "score_swe_smith_real_eval286_real_cascade_verifier_soft_bce_1783430056/"
    "scores/eval_verifier_scores.jsonl"
)
POST_SCOUT_1500_PATH = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_10epoch_1781112916/"
    "train_qwen3_embedding_8b_lora_reward_bce_10epoch/eval_predictions.jsonl"
)
INPUT_ONLY_1500_PATH = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_input_only_10epoch_1781112942/"
    "train_qwen3_embedding_8b_lora_reward_bce_10epoch/eval_predictions.jsonl"
)
REAL_LABELS_DIR = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect/"
)
OUT_PLOT = "/home/toolkit/PipelineRL-SWE/analysis/abstention_sample_efficiency.png"
OUT_TABLE = "/home/toolkit/PipelineRL-SWE/analysis/abstention_sample_efficiency_table.csv"

# ── load real labels ──────────────────────────────────────────────────────────
dfs = [pd.read_parquet(p) for p in Path(REAL_LABELS_DIR + "eval").glob("*.parquet")]
eval_df = pd.concat(dfs).set_index("problem_id")
eval_df["r120b"] = eval_df["route_successes"].apply(lambda x: x[3])
print(f"Eval tasks: {len(eval_df)}, 120B solvable: {eval_df['r120b'].sum()}")

# ── helpers ───────────────────────────────────────────────────────────────────
def load_preds_mean(path):
    """Load eval_predictions.jsonl -> {problem_id: mean_pred}"""
    rows = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            pid = d["problem_id"]
            pr = d.get("pred_rewards")
            rows[pid] = float(np.mean(pr)) if pr is not None else np.nan
    return rows

def load_cascade_mean(path):
    """Load eval_verifier_scores.jsonl -> {problem_id: mean pred_score across routes}"""
    by_task = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            pid = d["problem_id"]
            by_task.setdefault(pid, []).append(d["pred_score"])
    return {pid: float(np.mean(scores)) for pid, scores in by_task.items()}

def abstention_curve(signal_dict, labels_series, rates):
    common = labels_series.index.intersection(list(signal_dict.keys()))
    labels = labels_series.loc[common].values.astype(int)
    sig = np.array([signal_dict[pid] for pid in common])
    order = np.argsort(sig)
    accs = []
    for r in rates:
        n_abstain = int(round(r * len(labels)))
        kept = order[n_abstain:]
        accs.append(labels[kept].mean() if len(kept) > 0 else np.nan)
    return np.array(accs), len(common)

def auc(accs, rates):
    valid = ~np.isnan(accs)
    return float(np.trapz(accs[valid], rates[valid]))

def random_curve(labels, rates, n_seeds=200, seed=42):
    rng = np.random.default_rng(seed)
    n = len(labels)
    all_accs = []
    for _ in range(n_seeds):
        perm = rng.permutation(n)
        seed_accs = [labels[perm[int(round(r*n)):]].mean() if int(round(r*n)) < n else np.nan
                     for r in rates]
        all_accs.append(seed_accs)
    return np.nanmean(all_accs, axis=0)

rates = np.linspace(0, 0.85, 200)

# ── load all signals ──────────────────────────────────────────────────────────
print("Loading post-scout real-1500...")
ps1500 = load_preds_mean(POST_SCOUT_1500_PATH)
print("Loading input-only real-1500...")
io1500 = load_preds_mean(INPUT_ONLY_1500_PATH)
print("Loading cascade-real-1500 scores...")
casc_real = load_cascade_mean(CASCADE_SCORES_PATH)

sweep_preds = {}
for n in [50, 100, 200, 500, 1000]:
    pred_path = (f"{SWEEP_BASE}/random_{n}_seed17/train_router_finetune_10epoch/eval_predictions.jsonl")
    print(f"Loading proxy-init real N={n}...")
    sweep_preds[n] = load_preds_mean(pred_path)

proxy_path = f"{SWEEP_BASE}/proxy_pretrain_5epoch/eval_predictions.jsonl"
if Path(proxy_path).exists():
    print("Loading proxy-only (no real FT)...")
    proxy_only = load_preds_mean(proxy_path)
else:
    print("Proxy-only predictions not found, skipping.")
    proxy_only = None

# ── compute curves ────────────────────────────────────────────────────────────
ps_acc, n_ps = abstention_curve(ps1500, eval_df["r120b"], rates)
io_acc, n_io = abstention_curve(io1500, eval_df["r120b"], rates)
casc_acc, n_casc = abstention_curve(casc_real, eval_df["r120b"], rates)

# random baseline over full eval
rnd_labels = eval_df["r120b"].values.astype(int)
rnd_acc = random_curve(rnd_labels, rates)

sweep_accs = {}
for n, preds in sweep_preds.items():
    sweep_accs[n], _ = abstention_curve(preds, eval_df["r120b"], rates)

if proxy_only is not None:
    prx_acc, _ = abstention_curve(proxy_only, eval_df["r120b"], rates)
else:
    prx_acc = None

# ── summary table ─────────────────────────────────────────────────────────────
checkpoints = [0.30, 0.50, 0.70]

def row(name, accs):
    a = auc(accs, rates)
    vals = []
    for cp in checkpoints:
        i = np.searchsorted(rates, cp)
        vals.append(accs[i] * 100 if not np.isnan(accs[i]) else float("nan"))
    return [name, round(a, 3)] + [round(v, 1) for v in vals]

rows = []
rows.append(row("Random", rnd_acc))
rows.append(row("Input-only (real N=1500)", io_acc))
if prx_acc is not None:
    rows.append(row("Post-scout (proxy only, no real FT)", prx_acc))
rows.append(row("Cascade verifier (real N=1500)", casc_acc))
for n in [50, 100, 200, 500, 1000]:
    rows.append(row(f"Post-scout proxy+real (N={n})", sweep_accs[n]))
rows.append(row("Post-scout (real N=1500)", ps_acc))

df_table = pd.DataFrame(rows, columns=["Method", "AUC", "@30%", "@50%", "@70%"])
print("\n" + df_table.to_string(index=False))
df_table.to_csv(OUT_TABLE, index=False)
print(f"\nSaved table to {OUT_TABLE}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

cmap = plt.cm.Blues
sweep_colors = {50: cmap(0.35), 100: cmap(0.45), 200: cmap(0.55), 500: cmap(0.65), 1000: cmap(0.80)}

ax.plot(rates*100, rnd_acc*100,    color="gray",    lw=1.2, ls=":", label="Random")
ax.plot(rates*100, io_acc*100,     color="#ff7f0e", lw=2,   label="Input-only (real N=1500)")
ax.plot(rates*100, casc_acc*100,   color="#2ca02c", lw=2,   label="Cascade verifier (real N=1500)")
if prx_acc is not None:
    ax.plot(rates*100, prx_acc*100, color="#9467bd", lw=1.5, ls="--", label="Post-scout (proxy only)")
for n in [50, 100, 200, 500, 1000]:
    ax.plot(rates*100, sweep_accs[n]*100, color=sweep_colors[n], lw=1.5,
            label=f"Post-scout proxy+real (N={n})")
ax.plot(rates*100, ps_acc*100,     color="#1f77b4", lw=2.5, label="Post-scout (real N=1500)")

ax.set_xlabel("Abstention rate (%)", fontsize=12)
ax.set_ylabel("Accuracy on retained tasks (%)", fontsize=12)
ax.set_title("Sample efficiency: post-scout proxy+real vs baselines\n(286-task eval set)", fontsize=13)
ax.legend(fontsize=8, loc="upper left", ncol=1)
ax.set_xlim(0, 85)
ax.set_ylim(44, 100)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150)
print(f"Saved plot to {OUT_PLOT}")
plt.close()

#!/usr/bin/env python3
"""
Generate publication-quality figures for the post-scout routing paper.
Outputs PDFs and PNGs to overleaf/figures/.
"""

import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
REAL_LABELS_DIR = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect/eval/"
)
PS_PREDS = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_10epoch_1781112916/"
    "train_qwen3_embedding_8b_lora_reward_bce_10epoch/eval_predictions.jsonl"
)
IO_PREDS = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_input_only_10epoch_1781112942/"
    "train_qwen3_embedding_8b_lora_reward_bce_10epoch/eval_predictions.jsonl"
)
CASCADE_SCORES = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "score_swe_smith_real_eval286_real_cascade_verifier_soft_bce_1783430056/scores/eval_verifier_scores.jsonl"
)
SAMPLE_EFF_BASE = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_proxy_init_real_finetune_router_sample_efficiency_1783430086/"
)
FIGURES_DIR = Path(
    "/home/toolkit/PipelineRL-SWE/overleaf/figures"
)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ── Data loading helpers ────────────────────────────────────────────────────────

def load_real_labels():
    files = sorted(glob.glob(REAL_LABELS_DIR + "*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files]).set_index("problem_id")
    return df


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows).set_index("problem_id")


# ── Metric helpers ──────────────────────────────────────────────────────────────

def abstention_curve(signal, labels, max_abs=0.85, n_points=200):
    """
    signal: higher = more confident = retain first.
    Sort ascending: lowest-confidence first => abstain those.
    Returns (rates, accs) over [0, max_abs].
    """
    sorted_idx = np.argsort(signal)  # lowest first => abstained first
    rates = np.linspace(0.0, max_abs, n_points)
    n = len(labels)
    accs = []
    for rate in rates:
        n_abstain = int(round(rate * n))
        retained = sorted_idx[n_abstain:]
        if len(retained) == 0:
            accs.append(np.nan)
        else:
            accs.append(labels[retained].mean())
    return rates, np.array(accs)


def compute_auc(signal, labels, max_abs=0.85, n_points=200):
    """Raw trapz integral (not normalised) — matches the reported metric."""
    rates, accs = abstention_curve(signal, labels, max_abs, n_points)
    valid = ~np.isnan(accs)
    return float(np.trapz(accs[valid], rates[valid]))


def acc_at(signal, labels, target_rate, max_abs=0.85, n_points=200):
    rates, accs = abstention_curve(signal, labels, max_abs, n_points)
    idx = np.argmin(np.abs(rates - target_rate))
    return float(accs[idx])


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Abstention accuracy curves
# ══════════════════════════════════════════════════════════════════════════════

def figure1():
    print("Generating Figure 1: abstention_curves …")

    df_labels = load_real_labels()
    common_pids = sorted(df_labels.index)
    r120b = np.array(
        [df_labels.loc[pid, "route_successes"][3] for pid in common_pids], dtype=float
    )

    # ── Signals ──
    df_ps = load_jsonl(PS_PREDS)
    ps_signal = np.array([np.mean(df_ps.loc[pid, "pred_rewards"]) for pid in common_pids])

    df_io = load_jsonl(IO_PREDS)
    io_signal = np.array([np.mean(df_io.loc[pid, "pred_rewards"]) for pid in common_pids])

    rows_cv = []
    with open(CASCADE_SCORES) as f:
        for line in f:
            rows_cv.append(json.loads(line))
    df_cv = pd.DataFrame(rows_cv)
    df_cv_r0 = df_cv[df_cv["route_idx"] == 0].set_index("problem_id")
    cv_signal = np.array([df_cv_r0.loc[pid, "pred_score"] for pid in common_pids])

    oracle_signal = r120b.copy()  # 1=success→retain, 0=fail→abstain

    # ── Random baseline (200 seeds) ──
    n_seeds = 200
    random_accs_all = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        rand_sig = rng.rand(len(r120b))
        _, accs = abstention_curve(rand_sig, r120b)
        random_accs_all.append(accs)
    random_accs_mean = np.nanmean(random_accs_all, axis=0)

    rates, _ = abstention_curve(ps_signal, r120b)
    _, accs_io = abstention_curve(io_signal, r120b)
    _, accs_ps = abstention_curve(ps_signal, r120b)
    _, accs_cv = abstention_curve(cv_signal, r120b)
    _, accs_oracle = abstention_curve(oracle_signal, r120b)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    pct_rates = rates * 100

    ax.plot(pct_rates, random_accs_mean * 100, color="gray", linestyle="dotted",
            lw=1.5, label="Random baseline", zorder=2)
    ax.plot(pct_rates, accs_io * 100, color="tab:orange", lw=2.0,
            label="Input-only (N=1500 real)", zorder=3)
    ax.plot(pct_rates, accs_cv * 100, color="tab:green", linestyle="dashed", lw=1.8,
            label="Cascade verifier, route-0 (N=1500 real)", zorder=4)
    ax.plot(pct_rates, accs_ps * 100, color="tab:blue", lw=2.5,
            label="Post-scout router (N=1500 real, ours)", zorder=5)
    ax.plot(pct_rates, accs_oracle * 100, color="black", linestyle="dashdot", lw=1.5,
            label="Oracle", zorder=6)

    # ── Oracle rate vertical line ──
    oracle_rate_pct = 52.8
    ax.axvline(x=oracle_rate_pct, color="black", linestyle="dashed", lw=1.0, alpha=0.6)
    ax.text(oracle_rate_pct + 0.8, 32, "Oracle rate\n(52.8%)",
            fontsize=8.5, va="bottom", color="black", alpha=0.75)

    # ── Annotation box @70% ──
    idx70 = np.argmin(np.abs(rates - 0.70))
    v_rand = random_accs_mean[idx70] * 100
    v_io = accs_io[idx70] * 100
    v_cv = accs_cv[idx70] * 100
    v_ps = accs_ps[idx70] * 100
    v_oracle = accs_oracle[idx70] * 100

    box_text = (
        "At 70% abstention:\n"
        f"  Random:     {v_rand:.1f}%\n"
        f"  Input-only: {v_io:.1f}%\n"
        f"  Cascade-r0: {v_cv:.1f}%\n"
        f"  Post-scout: {v_ps:.1f}%\n"
        f"  Oracle:     {v_oracle:.1f}%"
    )
    ax.text(
        0.975, 0.97, box_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8.0,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85),
    )

    ax.set_xlabel("Abstention rate (%)")
    ax.set_ylabel("Accuracy on retained tasks (%)")
    ax.set_xlim(0, 85)
    ax.set_ylim(28, 105)
    ax.set_xticks(range(0, 90, 10))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("Abstention accuracy curves — SWE-Smith eval (286 tasks)")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"abstention_curves.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)

    # Print computed metrics
    print(f"  Computed AUC: PS={compute_auc(ps_signal, r120b):.3f}, "
          f"IO={compute_auc(io_signal, r120b):.3f}, "
          f"CV={compute_auc(cv_signal, r120b):.3f}")
    print(f"  @70%: PS={v_ps:.1f}, IO={v_io:.1f}, CV={v_cv:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Sample efficiency
# ══════════════════════════════════════════════════════════════════════════════

def figure2():
    print("Generating Figure 2: sample_efficiency …")

    df_labels = load_real_labels()
    common_pids = sorted(df_labels.index)
    r120b = np.array(
        [df_labels.loc[pid, "route_successes"][3] for pid in common_pids], dtype=float
    )

    # Compute AUC for each N in proxy+real sweep
    ns = [50, 100, 200, 500, 1000]
    aucs_computed = []
    for N in ns:
        path = (
            f"{SAMPLE_EFF_BASE}random_{N}_seed17/"
            "train_router_finetune_10epoch/eval_predictions.jsonl"
        )
        df = load_jsonl(path)
        pids_here = [pid for pid in common_pids if pid in df.index]
        signal = np.array([np.mean(df.loc[pid, "pred_rewards"]) for pid in pids_here])
        labs = np.array(
            [df_labels.loc[pid, "route_successes"][3] for pid in pids_here], dtype=float
        )
        aucs_computed.append(compute_auc(signal, labs))

    # N=1500 real (full PS, no proxy)
    df_ps = load_jsonl(PS_PREDS)
    ps_signal = np.array([np.mean(df_ps.loc[pid, "pred_rewards"]) for pid in common_pids])
    auc_ps_full = compute_auc(ps_signal, r120b)

    # Input-only N=1500 (reference line)
    df_io = load_jsonl(IO_PREDS)
    io_signal = np.array([np.mean(df_io.loc[pid, "pred_rewards"]) for pid in common_pids])
    auc_io = compute_auc(io_signal, r120b)

    # Random baseline
    random_aucs = []
    for seed in range(200):
        rng = np.random.RandomState(seed)
        rand_sig = rng.rand(len(r120b))
        random_aucs.append(compute_auc(rand_sig, r120b))
    auc_rand = np.mean(random_aucs)

    print(f"  Computed AUCs: {list(zip(ns, [f'{a:.3f}' for a in aucs_computed]))}")
    print(f"  PS full (N=1500 real): {auc_ps_full:.3f}")
    print(f"  IO (N=1500 real): {auc_io:.3f}")
    print(f"  Random: {auc_rand:.3f}")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    all_ns = ns + [1500]
    all_aucs = aucs_computed + [auc_ps_full]

    ax.plot(ns, [a * 100 for a in aucs_computed], "o-",
            color="tab:blue", lw=2.0, ms=7,
            label="Post-scout proxy+real")
    ax.plot(1500, auc_ps_full * 100, "*",
            color="tab:blue", ms=14,
            label=f"Post-scout (N=1500 real only, AUC={auc_ps_full:.3f})",
            zorder=6)

    # Input-only horizontal line
    ax.axhline(y=auc_io * 100, color="tab:orange", linestyle="dashed", lw=2.0,
               label=f"Input-only (N=1500 real, AUC={auc_io:.3f})")

    # Random horizontal line
    ax.axhline(y=auc_rand * 100, color="gray", linestyle="dashed", lw=1.2,
               label=f"Random (AUC={auc_rand:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("Number of real execution labels (N)")
    ax.set_ylabel("Abstention AUC (×100)")
    ax.set_title("Sample efficiency: proxy pre-training + real fine-tuning")

    xtick_vals = [50, 100, 200, 500, 1000, 1500]
    ax.set_xticks(xtick_vals)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticklabels([str(v) for v in xtick_vals])

    # Y axis: show absolute AUC values (multiplied by 100 for readability)
    ax.set_ylim(None, None)

    ax.grid(True, alpha=0.3, which="both", linestyle="--")
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"sample_efficiency.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Cost–reward Pareto frontier
# ══════════════════════════════════════════════════════════════════════════════

def figure3():
    print("Generating Figure 3: routing_pareto …")

    df_labels = load_real_labels()
    common_pids = sorted(df_labels.index)

    successes = np.array(
        [df_labels.loc[pid, "route_successes"] for pid in common_pids], dtype=float
    )
    out_tokens = np.array(
        [df_labels.loc[pid, "route_output_tokens"] for pid in common_pids], dtype=float
    )

    MODEL_SIZES = [4, 20, 30, 120]
    NORM = 4.0

    # Actual size-adjusted cost per task per route
    actual_costs = np.zeros((len(common_pids), 4))
    for r in range(4):
        actual_costs[:, r] = (MODEL_SIZES[r] / NORM) * out_tokens[:, r]

    always120_cost = actual_costs[:, 3].mean()

    # Decision cost helpers
    scout_out_tokens = out_tokens[:, 0]
    avg_scout_out_tokens = scout_out_tokens.mean()
    avg_out_per_route = out_tokens.mean(axis=0)
    avg_io_cost = np.array([MODEL_SIZES[r] / NORM * avg_out_per_route[r] for r in range(4)])

    # Post-scout decision cost: per-task, scaled by scout output length
    ps_decision_cost = np.outer(scout_out_tokens / avg_scout_out_tokens, avg_io_cost)

    # Input-only decision cost: flat per route
    io_decision_cost = np.tile(avg_io_cost, (len(common_pids), 1))

    # Load predictions
    df_ps = load_jsonl(PS_PREDS)
    ps_rewards = np.array([df_ps.loc[pid, "pred_rewards"] for pid in common_pids])

    df_io = load_jsonl(IO_PREDS)
    io_rewards = np.array([df_io.loc[pid, "pred_rewards"] for pid in common_pids])

    # Lambda sweep
    lambdas = np.concatenate([[0.0], np.logspace(-7, -2.5, 200)])
    ps_pareto = []
    io_pareto = []
    for lam in lambdas:
        chosen_ps = np.argmax(ps_rewards - lam * ps_decision_cost, axis=1)
        reward_ps = np.mean(successes[np.arange(len(common_pids)), chosen_ps])
        cost_ps = np.mean(actual_costs[np.arange(len(common_pids)), chosen_ps])
        ps_pareto.append((cost_ps / always120_cost * 100, reward_ps * 100))

        chosen_io = np.argmax(io_rewards - lam * io_decision_cost, axis=1)
        reward_io = np.mean(successes[np.arange(len(common_pids)), chosen_io])
        cost_io = np.mean(actual_costs[np.arange(len(common_pids)), chosen_io])
        io_pareto.append((cost_io / always120_cost * 100, reward_io * 100))

    ps_costs, ps_rewards_pct = zip(*ps_pareto)
    io_costs, io_rewards_pct = zip(*io_pareto)

    # Oracle: cheapest succeeding route per task
    oracle_rewards = []
    oracle_costs = []
    for i in range(len(common_pids)):
        success_routes = np.where(successes[i] == 1)[0]
        if len(success_routes) > 0:
            chosen = success_routes[np.argmin(actual_costs[i, success_routes])]
        else:
            chosen = np.argmin(actual_costs[i])
        oracle_rewards.append(successes[i, chosen])
        oracle_costs.append(actual_costs[i, chosen])
    oracle_reward = np.mean(oracle_rewards) * 100
    oracle_cost = np.mean(oracle_costs) / always120_cost * 100

    # Fixed baselines
    baselines = []
    for r, (name, marker, color) in enumerate([
        ("Always Scout 4B", "^", "tab:purple"),
        ("Always OSS-20B", "s", "tab:brown"),
        ("Always Qwen-30B", "D", "tab:red"),
        ("Always OSS-120B", "o", "tab:green"),
    ]):
        reward = successes[:, r].mean() * 100
        cost = actual_costs[:, r].mean() / always120_cost * 100
        baselines.append((name, marker, color, cost, reward))

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # Curves
    ax.plot(ps_costs, ps_rewards_pct, "-", color="tab:blue", lw=2.2,
            label="Post-scout router (ours)", zorder=4)
    ax.plot(io_costs, io_rewards_pct, "--", color="tab:orange", lw=1.8,
            label="Input-only router", zorder=3)

    # Oracle
    ax.plot(oracle_cost, oracle_reward, "*", color="black", ms=14, zorder=6,
            label=f"Oracle ({oracle_cost:.0f}%, {oracle_reward:.1f}%)")

    # Fixed baselines
    for name, marker, color, cost, reward in baselines:
        ax.plot(cost, reward, marker=marker, color=color, ms=9, zorder=5,
                linestyle="None", label=f"{name} ({cost:.0f}%, {reward:.1f}%)")

    # Annotate lambda points on post-scout curve
    # lambda=0
    ax.annotate(
        f"λ=0\n({ps_costs[0]:.0f}%, {ps_rewards_pct[0]:.1f}%)",
        xy=(ps_costs[0], ps_rewards_pct[0]),
        xytext=(ps_costs[0] - 8, ps_rewards_pct[0] + 3),
        fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", lw=0.8, color="tab:blue"),
        color="tab:blue",
    )
    # lambda ~3e-6
    idx3e6 = np.argmin(np.abs(lambdas - 3e-6))
    ax.annotate(
        f"λ≈3×10⁻⁶\n({ps_costs[idx3e6]:.0f}%, {ps_rewards_pct[idx3e6]:.1f}%)",
        xy=(ps_costs[idx3e6], ps_rewards_pct[idx3e6]),
        xytext=(ps_costs[idx3e6] + 8, ps_rewards_pct[idx3e6] + 3),
        fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", lw=0.8, color="tab:blue"),
        color="tab:blue",
    )
    # lambda ~1e-5
    idx1e5 = np.argmin(np.abs(lambdas - 1e-5))
    ax.annotate(
        f"λ≈10⁻⁵\n({ps_costs[idx1e5]:.0f}%, {ps_rewards_pct[idx1e5]:.1f}%)",
        xy=(ps_costs[idx1e5], ps_rewards_pct[idx1e5]),
        xytext=(ps_costs[idx1e5] + 10, ps_rewards_pct[idx1e5] - 4),
        fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", lw=0.8, color="tab:blue"),
        color="tab:blue",
    )

    ax.set_xlabel("Cost relative to always-120B (%)")
    ax.set_ylabel("Success rate (%)")
    ax.set_xlim(-2, 102)
    ax.set_ylim(26, 60)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.set_title("Cost–reward Pareto frontier (size-adjusted inference cost)")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"routing_pareto.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)

    # Print key numbers
    print(f"  Always-120B cost: {always120_cost:.1f} units")
    print(f"  Oracle: reward={oracle_reward:.1f}%, cost={oracle_cost:.1f}%")
    print(f"  PS λ=0: reward={ps_rewards_pct[0]:.1f}%, cost={ps_costs[0]:.1f}%")
    print(f"  PS λ≈3e-6: reward={ps_rewards_pct[idx3e6]:.1f}%, cost={ps_costs[idx3e6]:.1f}%")
    print(f"  PS λ≈1e-5: reward={ps_rewards_pct[idx1e5]:.1f}%, cost={ps_costs[idx1e5]:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    print("\nAll figures generated.")

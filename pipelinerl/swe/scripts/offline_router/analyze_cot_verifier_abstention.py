#!/usr/bin/env python3
"""
Abstention analysis: CoT autoregressive verifier vs. baselines.

Compares methods that predict whether gpt-oss-120b will succeed on a given instance,
used to decide which instances to abstain on (i.e., skip rather than paying for the
strong model).

Methods compared:
  1. CoT verifier (ours)     -- P(Yes) from train_autoregressive_verifier.py
  2. Post-scout verifier      -- mean_pred from existing Qwen3-8B embedding verifier
  3. Input-only verifier      -- same model, no scout trace
  4. Prompt-length heuristic  -- shorter prompt → easier → less likely to need abstention
  5. Random baseline          -- averaged over 200 seeds
  6. Oracle                   -- perfect knowledge of labels

Usage:
  python analyze_cot_verifier_abstention.py \
    --cot-scores-path /mnt/.../eval_verifier_scores.jsonl \
    --parquet-dir /mnt/.../collect/eval \
    --output-dir /mnt/.../analysis/cot_verifier \
    [--post-scout-preds-path <path>] \
    [--input-only-preds-path <path>]
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Default paths to the existing embedding verifier predictions (from prior experiments)
DEFAULT_POST_SCOUT_PREDS = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_10epoch_1781112916/"
    "train_qwen3_embedding_8b_lora_reward_bce_10epoch/eval_predictions.jsonl"
)
DEFAULT_INPUT_ONLY_PREDS = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_input_only_10epoch_1781112942/"
    "train_qwen3_embedding_8b_lora_reward_bce_10epoch/eval_predictions.jsonl"
)


def load_cot_scores(path: str) -> dict[str, float]:
    """Load per-instance P(Yes) scores from score_autoregressive_verifier.py output."""
    scores: dict[str, float] = {}
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = rec.get("instance_id") or rec.get("problem_id")
            p_yes = rec.get("p_yes")
            if iid and p_yes is not None:
                scores[str(iid)] = float(p_yes)
    logger.info("Loaded %d CoT verifier scores from %s", len(scores), path)
    return scores


def load_embedding_preds(path: str) -> dict[str, float]:
    """Load mean_pred per instance from Qwen3-8B embedding verifier eval_predictions.jsonl."""
    if not path or not Path(path).exists():
        return {}
    preds: dict[str, float] = {}
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("problem_id")
            pred_rewards = rec.get("pred_rewards")
            if pid and pred_rewards is not None:
                preds[str(pid)] = float(np.mean(pred_rewards))
    logger.info("Loaded %d embedding verifier predictions from %s", len(preds), path)
    return preds


def load_parquet_labels(parquet_dir: str, route_idx: int = 3) -> pd.DataFrame:
    """Load per-instance labels + prompt tokens from parquet."""
    paths = sorted(Path(parquet_dir).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")
    df = pd.concat([pd.read_parquet(p) for p in paths]).set_index("problem_id")
    df["resolved"] = df["route_successes"].apply(lambda x: bool(x[route_idx]))
    df["prompt_tokens"] = df["route_prompt_tokens"].apply(
        lambda x: int(x[0]) if x is not None and len(x) > 0 else None
    )
    logger.info(
        "Loaded %d eval instances, %d resolved (route_idx=%d)",
        len(df), df["resolved"].sum(), route_idx,
    )
    return df


def abstention_curve(signal: np.ndarray, labels: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Sort by ascending signal (lowest = most hopeless = abstain first)."""
    order = np.argsort(signal)
    accs = []
    for r in rates:
        n_abstain = int(round(r * len(labels)))
        kept = order[n_abstain:]
        accs.append(labels[kept].mean() if len(kept) > 0 else float("nan"))
    return np.array(accs)


def oracle_curve(labels: np.ndarray, rates: np.ndarray) -> np.ndarray:
    sorted_labels = np.sort(labels)  # 0s first (abstain these)
    accs = []
    for r in rates:
        n_abstain = int(round(r * len(labels)))
        kept = sorted_labels[n_abstain:]
        accs.append(kept.mean() if len(kept) > 0 else float("nan"))
    return np.array(accs)


def random_abstention_curve(labels: np.ndarray, rates: np.ndarray, n_seeds: int = 200) -> np.ndarray:
    rng = np.random.default_rng(42)
    n = len(labels)
    all_accs = []
    for _ in range(n_seeds):
        perm = rng.permutation(n)
        seed_accs = [
            labels[perm[int(round(r * n)):]].mean() if int(round(r * n)) < n else float("nan")
            for r in rates
        ]
        all_accs.append(seed_accs)
    return np.nanmean(all_accs, axis=0)


def auc_abstention(signal: np.ndarray, labels: np.ndarray, max_rate: float = 0.70) -> float:
    """AUC under the abstention curve up to max_rate (normalized)."""
    rates = np.linspace(0, max_rate, 200)
    accs = abstention_curve(signal, labels, rates)
    valid = ~np.isnan(accs)
    if valid.sum() < 2:
        return float("nan")
    return float(np.trapz(accs[valid], rates[valid]) / max_rate)


def roc_auc(signal: np.ndarray, labels: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, signal))
    except Exception:
        return float("nan")


def plot_abstention_curves(
    methods: list[dict[str, Any]],
    labels: np.ndarray,
    out_path: Path,
    title: str = "Abstention analysis: CoT verifier vs. baselines",
) -> None:
    rates = np.linspace(0, 0.85, 200)
    oracle_acc = oracle_curve(labels, rates)
    random_acc = random_abstention_curve(labels, rates)

    n_hopeless = int((labels == 0).sum())
    oracle_rate = n_hopeless / len(labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: full abstention curve ---
    for m in methods:
        curve = abstention_curve(m["signal"], labels, rates)
        ax1.plot(rates * 100, curve * 100,
                 color=m["color"], lw=m.get("lw", 2), linestyle=m.get("ls", "-"),
                 label=m["label"])

    ax1.plot(rates * 100, random_acc * 100, color="gray", lw=1.5, linestyle=":",
             label="Random baseline")
    ax1.plot(rates * 100, oracle_acc * 100, color="black", lw=1.5, linestyle="-.",
             label="Oracle")
    ax1.axvline(oracle_rate * 100, color="black", lw=0.8, linestyle=":", alpha=0.5)
    ax1.text(oracle_rate * 100 + 0.5, labels.mean() * 100 + 2,
             f"Oracle rate\n({oracle_rate*100:.0f}%)", fontsize=8, alpha=0.6)

    ax1.set_xlabel("Abstention rate (%)", fontsize=11)
    ax1.set_ylabel("Accuracy on retained tasks (%)", fontsize=11)
    ax1.set_title("Abstention curves (full range)", fontsize=11)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_xlim(0, 85)
    ax1.set_ylim(labels.mean() * 100 - 5, 102)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax1.grid(True, alpha=0.3)

    # --- Right: zoom 0–40% abstention ---
    zoom_rates = rates[rates <= 0.40]
    for m in methods:
        curve = abstention_curve(m["signal"], labels, zoom_rates)
        ax2.plot(zoom_rates * 100, curve * 100,
                 color=m["color"], lw=m.get("lw", 2), linestyle=m.get("ls", "-"),
                 label=m["label"])
    ax2.plot(zoom_rates * 100, random_abstention_curve(labels, zoom_rates) * 100,
             color="gray", lw=1.5, linestyle=":", label="Random")
    ax2.plot(zoom_rates * 100, oracle_curve(labels, zoom_rates) * 100,
             color="black", lw=1.5, linestyle="-.", label="Oracle")

    ax2.set_xlabel("Abstention rate (%)", fontsize=11)
    ax2.set_ylabel("Accuracy on retained tasks (%)", fontsize=11)
    ax2.set_title("Zoom: 0–40% abstention", fontsize=11)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_xlim(0, 40)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def print_summary_table(methods: list[dict[str, Any]], labels: np.ndarray) -> None:
    checkpoint_rates = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    rates = np.linspace(0, 0.85, 500)

    print("\n=== Abstention accuracy (%) at each rate ===")
    header = f"{'Method':<30}" + "".join(f"  {int(r*100):>4}%" for r in checkpoint_rates)
    print(header)
    print("-" * len(header))
    for m in methods:
        curve = abstention_curve(m["signal"], labels, rates)
        vals = []
        for r in checkpoint_rates:
            idx = np.searchsorted(rates, r)
            vals.append(f"{curve[idx]*100:>5.1f}")
        print(f"{m['label']:<30}" + "  " + "  ".join(vals))

    # Oracle row
    oracle_acc = oracle_curve(labels, rates)
    vals = []
    for r in checkpoint_rates:
        idx = np.searchsorted(rates, r)
        vals.append(f"{oracle_acc[idx]*100:>5.1f}")
    print(f"{'Oracle':<30}" + "  " + "  ".join(vals))

    print("\n=== ROC-AUC (predict strong model success) ===")
    for m in methods:
        auc = roc_auc(m["signal"], labels)
        print(f"  {m['label']:<30}  AUC={auc:.4f}")

    print("\n=== Area under abstention curve (0–70%, normalized) ===")
    for m in methods:
        a = auc_abstention(m["signal"], labels, max_rate=0.70)
        print(f"  {m['label']:<30}  abs-AUC={a:.4f}")
    oracle_a = auc_abstention(np.sort(labels).astype(float), labels, max_rate=0.70)
    print(f"  {'Oracle':<30}  abs-AUC={oracle_a:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CoT verifier abstention analysis")
    parser.add_argument("--cot-scores-path", required=True,
                        help="JSONL from score_autoregressive_verifier.py (instance_id, p_yes, resolved)")
    parser.add_argument("--parquet-dir", required=True,
                        help="Eval parquet dir with route_successes (for labels + prompt tokens)")
    parser.add_argument("--label-route-idx", type=int, default=3,
                        help="Index into route_successes for target model (default: 3 = gpt-oss-120b)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write figures and summary")
    parser.add_argument("--post-scout-preds-path", default=DEFAULT_POST_SCOUT_PREDS,
                        help="eval_predictions.jsonl from post-scout embedding verifier")
    parser.add_argument("--input-only-preds-path", default=DEFAULT_INPUT_ONLY_PREDS,
                        help="eval_predictions.jsonl from input-only embedding verifier")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load all data ---
    cot_scores = load_cot_scores(args.cot_scores_path)
    post_scout_preds = load_embedding_preds(args.post_scout_preds_path)
    input_only_preds = load_embedding_preds(args.input_only_preds_path)
    label_df = load_parquet_labels(args.parquet_dir, args.label_route_idx)

    # --- Align on common instances ---
    common = set(cot_scores.keys()) & set(label_df.index)
    if not common:
        logger.error("No overlap between CoT scores and parquet labels. Check instance ID format.")
        return

    if post_scout_preds:
        common &= set(post_scout_preds.keys())
    if input_only_preds:
        common &= set(input_only_preds.keys())

    common = sorted(common)
    logger.info("Aligned on %d common instances", len(common))

    labels = label_df.loc[common, "resolved"].values.astype(int)
    cot_sig = np.array([cot_scores[pid] for pid in common])
    prompt_tokens = label_df.loc[common, "prompt_tokens"].values.astype(float)
    prompt_sig = -prompt_tokens  # shorter = lower score → abstain shorter prompts first? inverted

    methods: list[dict[str, Any]] = [
        {
            "label": "CoT verifier (ours)",
            "signal": cot_sig,
            "color": "#1f77b4",
            "lw": 2.5,
        },
    ]

    if post_scout_preds:
        ps_sig = np.array([post_scout_preds[pid] for pid in common])
        methods.append({
            "label": "Post-scout embedding verifier",
            "signal": ps_sig,
            "color": "#ff7f0e",
            "lw": 2,
        })

    if input_only_preds:
        io_sig = np.array([input_only_preds[pid] for pid in common])
        methods.append({
            "label": "Input-only embedding verifier",
            "signal": io_sig,
            "color": "#2ca02c",
            "lw": 2,
            "ls": "--",
        })

    methods.append({
        "label": "Prompt-length heuristic",
        "signal": prompt_sig,
        "color": "#9467bd",
        "lw": 1.5,
        "ls": "--",
    })

    # --- Print summary ---
    n_pos = int(labels.sum())
    logger.info(
        "Eval set: %d instances, %d resolved (%.1f%%), %d unresolved",
        len(labels), n_pos, 100 * n_pos / len(labels), len(labels) - n_pos,
    )
    print_summary_table(methods, labels)

    # --- Figures ---
    plot_abstention_curves(
        methods, labels,
        out_dir / "cot_verifier_abstention.png",
        title=f"Abstention analysis: CoT autoregressive verifier vs. baselines\n"
              f"(n={len(labels)} eval instances, target=gpt-oss-120b route_idx={args.label_route_idx})",
    )

    # Save summary JSON
    summary: dict[str, Any] = {
        "n_instances": len(labels),
        "n_resolved": int(labels.sum()),
        "label_route_idx": args.label_route_idx,
        "methods": [],
    }
    rates = np.linspace(0, 0.85, 500)
    for m in methods:
        curve = abstention_curve(m["signal"], labels, rates)
        checkpoints = {}
        for r in [0.10, 0.20, 0.30, 0.40, 0.50]:
            idx = np.searchsorted(rates, r)
            checkpoints[f"acc_at_{int(r*100)}pct"] = float(curve[idx])
        summary["methods"].append({
            "label": m["label"],
            "roc_auc": roc_auc(m["signal"], labels),
            "abs_auc_70pct": auc_abstention(m["signal"], labels, 0.70),
            **checkpoints,
        })

    summary_path = out_dir / "cot_verifier_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()

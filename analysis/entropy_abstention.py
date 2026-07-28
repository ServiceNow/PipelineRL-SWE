"""
Entropy abstention baseline.

Uses teacher-forcing to compute per-token entropy of the saved 4B outputs
(no re-generation needed). The original outputs are already stored in the eval
parquet, so the Daytona eval labels are valid — we're asking: "given the output
the 4B model actually produced, how uncertain was it token-by-token?"

Two modes:

  --mode collect   Load eval parquet, run Qwen3-4B in teacher-forcing mode,
                   save (instance_id, signals) to LOGPROBS_OUT.
                   Requires a GPU. ~10 min for 217 instances on one A100.

  --mode analyze   Load saved logprobs + labels, compute AUROC/Spearman vs
                   PS and IO trained signals. Print table and save figure.

Usage:
  # Step 1 — collect (GPU required):
  python analysis/entropy_abstention.py --mode collect

  # Step 2 — analyze (CPU fine):
  python analysis/entropy_abstention.py --mode analyze

Signals computed per instance:
  mean_token_entropy   — average H(softmax(logits)) across all output tokens
                         (full-vocab entropy, exact via teacher-forcing)
  mean_token_logprob   — average log P(chosen_token); ≈ negative perplexity
  early_entropy        — mean entropy over first 256 output tokens
                         (captures model uncertainty when deciding approach)
  output_length        — number of generated tokens (longer = harder?)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

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

MODEL_NAME   = "Qwen/Qwen3-4B-Instruct-2507"
LOGPROBS_OUT = "/home/toolkit/PipelineRL-SWE/analysis/entropy_logprobs.jsonl"
FIGURE_OUT   = "/home/toolkit/PipelineRL-SWE/analysis/entropy_abstention.png"
RESULTS_OUT  = "/home/toolkit/PipelineRL-SWE/analysis/entropy_abstention_results.json"

EARLY_WINDOW = 256  # tokens for "early entropy" signal


# ── data loading helpers ───────────────────────────────────────────────────────

def load_eval_parquet() -> pd.DataFrame:
    dfs = [pd.read_parquet(p) for p in Path(REAL_LABELS_DIR + "eval").glob("*.parquet")]
    return pd.concat(dfs).set_index("problem_id")


def load_preds(path: str) -> dict[str, np.ndarray]:
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["problem_id"]] = np.array(d["pred_rewards"])
    return out


def load_claude_resolved() -> set[str]:
    with open(CLAUDE_REPORT) as f:
        report = json.load(f)
    return set(report["ids_resolved"])


# ── collect: teacher-forcing logprob extraction ───────────────────────────────

def collect(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    eval_df = load_eval_parquet()

    # Filter to instances that have primary_output_text saved
    valid = eval_df[eval_df["primary_output_text"].apply(
        lambda x: isinstance(x, str) and len(x) > 0
    )]
    print(f"Instances with saved primary_output_text: {len(valid)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()

    # Load already-done IDs for resume
    done: set[str] = set()
    if Path(LOGPROBS_OUT).exists():
        with open(LOGPROBS_OUT) as f:
            for line in f:
                rec = json.loads(line)
                done.add(rec["instance_id"])
        print(f"Resuming — {len(done)} already done")

    out_f = open(LOGPROBS_OUT, "a")
    todo = [(pid, row) for pid, row in valid.iterrows() if pid not in done]
    print(f"Processing {len(todo)} instances...")

    for i, (pid, row) in enumerate(todo):
        prompt_text    = row["prompt_text"]
        output_text    = row["primary_output_text"]

        # Reconstruct the full sequence: prompt + assistant turn
        # prompt_text already ends with <|im_end|> after the user turn
        assistant_prefix = "\n<|im_start|>assistant\n"
        full_text = prompt_text + assistant_prefix + output_text

        # Tokenize
        enc_full   = tokenizer(full_text,   return_tensors="pt", truncation=True, max_length=32768)
        enc_prompt = tokenizer(prompt_text + assistant_prefix, return_tensors="pt", truncation=True, max_length=32768)

        prompt_len  = enc_prompt["input_ids"].shape[1]
        full_ids    = enc_full["input_ids"].to(device)
        output_len  = full_ids.shape[1] - prompt_len

        if output_len <= 0:
            print(f"  [{i+1}/{len(todo)}] {pid}: output truncated to zero, skipping")
            continue

        # Forward pass — teacher forcing
        with torch.no_grad():
            logits = model(input_ids=full_ids).logits  # (1, seq_len, vocab)

        # Logits at position t predict token t+1.
        # Output tokens span positions [prompt_len .. prompt_len+output_len-1] in full_ids.
        # The logits that predict them are at positions [prompt_len-1 .. prompt_len+output_len-2].
        out_logits = logits[0, prompt_len - 1: prompt_len - 1 + output_len, :]  # (output_len, vocab)
        out_ids    = full_ids[0, prompt_len: prompt_len + output_len]            # (output_len,)

        # Per-token log-probability of the actual generated token
        log_probs   = F.log_softmax(out_logits.float(), dim=-1)
        token_lp    = log_probs[torch.arange(output_len), out_ids]  # (output_len,)

        # Per-token entropy: H = -∑ p log p (exact, full vocab)
        probs         = F.softmax(out_logits.float(), dim=-1)
        token_entropy = -(probs * log_probs).sum(dim=-1)             # (output_len,)

        lp_np  = token_lp.cpu().float().numpy()
        ent_np = token_entropy.cpu().float().numpy()

        early_n = min(EARLY_WINDOW, output_len)
        record = {
            "instance_id":       pid,
            "output_length":     int(output_len),
            "mean_token_logprob": float(lp_np.mean()),
            "mean_token_entropy": float(ent_np.mean()),
            "early_entropy":      float(ent_np[:early_n].mean()),
            "cumulative_logprob": float(lp_np.sum()),
        }
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(todo)}] {pid}: "
                  f"len={output_len} mean_lp={record['mean_token_logprob']:.3f} "
                  f"mean_H={record['mean_token_entropy']:.3f}")

    out_f.close()
    print(f"\nSaved logprobs to {LOGPROBS_OUT}")


# ── analyze: compare entropy vs trained signals ───────────────────────────────

def analyze(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # Load logprobs
    logprobs: dict[str, dict] = {}
    with open(LOGPROBS_OUT) as f:
        for line in f:
            rec = json.loads(line)
            logprobs[rec["instance_id"]] = rec

    # Load labels
    eval_df    = load_eval_parquet()
    ps_preds   = load_preds(PS_PATH)
    io_preds   = load_preds(IO_PATH)
    claude_res = load_claude_resolved()

    claude_all = set()
    with open(CLAUDE_REPORT) as f:
        report = json.load(f)
    claude_all = set(report["ids_resolved"]) | set(report["ids_unresolved"])

    common = sorted(
        eval_df.index.intersection(ps_preds).intersection(io_preds)
        .intersection(claude_all).intersection(logprobs)
    )
    N = len(common)
    print(f"Analysis set: {N} instances")

    successes = np.array([eval_df.loc[p, "route_successes"] for p in common])
    ps_signal = np.array([ps_preds[p].mean() for p in common])
    io_signal = np.array([io_preds[p].mean() for p in common])
    s120b     = successes[:, 3].astype(float)
    s_claude  = np.array([p in claude_res for p in common], dtype=float)

    # Entropy signals (negated so that HIGH = model is CONFIDENT = should keep)
    ent_signals = {
        "mean_token_logprob (↑ = confident)":   np.array([logprobs[p]["mean_token_logprob"]   for p in common]),
        "neg_mean_entropy (↑ = confident)":     -np.array([logprobs[p]["mean_token_entropy"]   for p in common]),
        "neg_early_entropy (↑ = confident)":    -np.array([logprobs[p]["early_entropy"]        for p in common]),
        "neg_output_length (↑ = confident)":    -np.array([logprobs[p]["output_length"]        for p in common]),
    }
    trained_signals = {
        "PS trained head": ps_signal,
        "IO trained head": io_signal,
    }

    # ── table: Spearman r and AUROC vs 120B and Claude success ────────────────
    results = {"n": N, "signals": {}}

    print(f"\n{'Signal':<42}  {'Sp-r(120B)':>10}  {'AUC(120B)':>10}  {'Sp-r(Cla)':>10}  {'AUC(Cla)':>10}")
    print("-" * 90)

    all_signals = {**ent_signals, **trained_signals}
    for name, sig in all_signals.items():
        sp120 = float(spearmanr(sig, s120b).statistic)
        sp_cl = float(spearmanr(sig, s_claude).statistic)
        auc120 = float(roc_auc_score(s120b, sig))
        auc_cl = float(roc_auc_score(s_claude, sig))
        print(f"  {name:<40}  {sp120:>10.3f}  {auc120:>10.4f}  {sp_cl:>10.3f}  {auc_cl:>10.4f}")
        results["signals"][name] = {"spearman_120b": sp120, "auc_120b": auc120,
                                    "spearman_claude": sp_cl, "auc_claude": auc_cl}

    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_OUT}")

    # ── figure: abstention curves ──────────────────────────────────────────────
    alphas = np.linspace(0, 0.80, 200)

    def abstention_solve_curve(signal, target_successes, alphas):
        """Fraction of ALL tasks solved (abstained = 0) as function of α."""
        order = np.argsort(signal)  # ascending = worst signal first = abstain first
        fracs = []
        for a in alphas:
            n_ab  = int(round(a * N))
            kept  = order[n_ab:]
            fracs.append(target_successes[kept].sum() / N)
        return np.array(fracs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, (target_name, target_s) in zip(axes, [("120B", s120b), ("Claude", s_claude)]):
        base = target_s.mean()
        ax.axhline(base * 100, color="black", lw=1, ls=":", alpha=0.5,
                   label=f"Always-{target_name} ({base*100:.1f}%)")

        styles = [
            ("PS trained head",                "#1f77b4", "-",  2.5),
            ("IO trained head",                "#ff7f0e", "-",  2.5),
            ("mean_token_logprob (↑ = confident)",   "#2ca02c", "--", 2.0),
            ("neg_mean_entropy (↑ = confident)",     "#d62728", "--", 2.0),
            ("neg_early_entropy (↑ = confident)",    "#9467bd", ":",  1.5),
            ("neg_output_length (↑ = confident)",    "#8c564b", ":",  1.5),
        ]
        for label, color, ls, lw in styles:
            curve = abstention_solve_curve(all_signals[label], target_s, alphas)
            ax.plot(alphas * 100, curve * 100, color=color, ls=ls, lw=lw,
                    label=label.split(" (")[0])

        ax.set_xlabel("Abstention rate (%)", fontsize=11)
        ax.set_ylabel("Fraction of ALL tasks solved (%)", fontsize=11)
        ax.set_title(f"Abstention signal comparison\n(target: {target_name} success, n={N})", fontsize=11)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.set_xlim(0, 80)

    plt.tight_layout()
    plt.savefig(FIGURE_OUT, dpi=150)
    print(f"Saved figure to {FIGURE_OUT}")
    plt.close()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["collect", "analyze"], required=True,
                        help="collect: run teacher-forcing inference (GPU); analyze: compare signals")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"HuggingFace model name/path for collect mode (default: {MODEL_NAME})")
    args = parser.parse_args()

    if args.mode == "collect":
        collect(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()

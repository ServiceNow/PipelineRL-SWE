#!/usr/bin/env python3
"""
Better complementarity visualizations for the OpenRouter diversity sweep.

Produces three figures:
  1. instance_difficulty_heatmap.png  -- instances sorted by solve count (shows nested/staircase structure)
  2. quadrant_breakdown.png           -- per-model stacked bar: both/only-strong/only-weak/neither vs gpt-oss-120b
  3. resolve_vs_phi.png               -- scatter: resolve rate vs phi to gpt-oss-120b (shows 'more capable = more correlated')

Usage:
  python visualize_complementarity.py \
    --daytona-root /mnt/.../filtered/logs/run_evaluation \
    --parquet-dir /mnt/.../collect/eval \
    --output-dir /mnt/.../filtered/analysis
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


STRONG_MODEL = "solver:openai/gpt-oss-120b"

# Human-readable short names for display
SHORT_NAMES = {
    "ibm-granite__granite-4_1-8b": "Granite-4.1-8B",
    "meta-llama__llama-3_1-8b-instruct": "Llama-3.1-8B",
    "microsoft__phi-4": "Phi-4-14B",
    "mistralai__codestral-2508": "Codestral-2508",
    "meta-llama__llama-3_3-70b-instruct": "Llama-3.3-70B",
    "meta-llama__llama-4-scout": "Llama-4-Scout",
    "mistralai__mistral-small-3_2-24b-instruct": "Mistral-Small-24B",
    "qwen__qwen3-32b": "Qwen3-32B",
    "mistralai__devstral-2512": "Devstral-2512",
    "qwen__qwen3-coder-30b-a3b-instruct": "Qwen3-Coder-30B",
    "deepseek__deepseek-v4-flash": "DeepSeek-V4-Flash",
    "google__gemma-4-31b-it": "Gemma-4-31B",
    "deepseek__deepseek-r1-0528": "DeepSeek-R1",
    "deepseek__deepseek-chat-v3_1": "DeepSeek-Chat-V3.1",
    "scout:Qwen/Qwen3-4B-Instruct-2507": "Qwen3-4B (scout)",
    "solver:openai/gpt-oss-20b": "GPT-OSS-20B",
    "solver:Qwen/Qwen3-Coder-30B-A3B-Instruct": "Qwen3-Coder-30B-A3B",
    "solver:openai/gpt-oss-120b": "GPT-OSS-120B",
}


def short(slug: str) -> str:
    return SHORT_NAMES.get(slug, slug.split("__")[-1][:20])


def load_daytona_results(daytona_root: Path, prefix: str = "or_sweep_") -> dict[str, dict[str, bool]]:
    results: dict[str, dict[str, bool]] = {}
    for model_dir in sorted(daytona_root.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.startswith(prefix):
            continue
        slug = model_dir.name[len(prefix):]
        res: dict[str, bool] = {}
        for inst_dir in model_dir.iterdir():
            if not inst_dir.is_dir():
                continue
            rpt = inst_dir / "report.json"
            if rpt.exists():
                with open(rpt) as f:
                    data = json.load(f)
                res[inst_dir.name] = bool(data.get("resolved", False))
        if res:
            results[slug] = res
    return results


def load_parquet_models(parquet_dir: str) -> dict[str, dict[str, bool]]:
    paths = sorted(Path(parquet_dir).glob("*.parquet"))
    if not paths:
        return {}
    df = pd.concat([pd.read_parquet(p) for p in paths])
    results: dict[str, dict[str, bool]] = {}
    route_labels = list(df.iloc[0]["route_labels"])
    for label_idx, label in enumerate(route_labels):
        res: dict[str, bool] = {}
        for _, row in df.iterrows():
            pid = str(row["problem_id"])
            res[pid] = bool(row["route_successes"][label_idx])
        results[label] = res
    return results


def build_matrix(all_results: dict[str, dict[str, bool]]) -> pd.DataFrame:
    all_instances = sorted(set(iid for res in all_results.values() for iid in res))
    data = {}
    for slug, res in all_results.items():
        data[slug] = {iid: float(res.get(iid, float("nan"))) for iid in all_instances}
    return pd.DataFrame(data, index=all_instances)


def phi(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    tp = float(np.sum((a == 1) & (b == 1)))
    tn = float(np.sum((a == 0) & (b == 0)))
    fp = float(np.sum((a == 0) & (b == 1)))
    fn = float(np.sum((a == 1) & (b == 0)))
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return (tp * tn - fp * fn) / denom if denom > 0 else float("nan")


def plot_instance_difficulty_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """
    Sort instances by how many models solve them (difficulty axis).
    Sort models by resolve rate.
    The staircase pattern reveals whether complementarity exists.
    """
    # Only include models that have results for all (or most) instances
    # Drop columns with too many NaNs
    coverage = df.notna().mean()
    df = df[coverage[coverage > 0.9].index]

    resolve_rates = df.mean(skipna=True).sort_values()
    df = df[resolve_rates.index]

    # Sort instances by solve count (ascending = hardest first)
    instance_solve_counts = df.sum(axis=1, skipna=True)
    df = df.loc[instance_solve_counts.sort_values().index]

    n_inst, n_mod = df.shape
    display_names = [short(c) for c in df.columns]

    fig, ax = plt.subplots(figsize=(max(10, n_mod * 0.7), max(6, n_inst * 0.04 + 2)))

    # Use 3 colors: solved (blue), failed (light gray), missing (white)
    colors = np.full((n_inst, n_mod, 3), fill_value=0.92)  # light gray default (failed)
    for j in range(n_mod):
        col = df.iloc[:, j].values
        for i in range(n_inst):
            if np.isnan(col[i]):
                colors[i, j] = [1.0, 1.0, 1.0]  # white = missing
            elif col[i] == 1:
                colors[i, j] = [0.18, 0.46, 0.71]  # blue = solved

    ax.imshow(colors, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(n_mod))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_ylabel(f"Instances (n={n_inst}, sorted easiest→hardest →)", fontsize=9)
    ax.set_title(
        "Solve patterns across models (instances sorted by difficulty)\n"
        "Blue = solved  ·  Gray = failed  ·  Models sorted by resolve rate (weakest left)",
        fontsize=10,
    )

    # Annotate resolve rates at top
    for j, col in enumerate(df.columns):
        rr = resolve_rates[col]
        ax.text(j, -1.5, f"{rr:.2f}", ha="center", va="top", fontsize=7, color="black")

    ax.set_xlim(-0.5, n_mod - 0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_quadrant_breakdown(df: pd.DataFrame, strong_col: str, out_path: Path) -> None:
    """
    For each non-strong model, show the fraction of instances in each quadrant:
      - Both solve (green)  -- routing gain = 0
      - Only strong solves (orange) -- routing headroom
      - Only weak solves (purple) -- routing would hurt (rare)
      - Neither solves (red) -- routing gain = 0
    """
    if strong_col not in df.columns:
        print(f"Strong model {strong_col} not in matrix; skipping quadrant breakdown")
        return

    strong = df[strong_col].values
    other_cols = [c for c in df.columns if c != strong_col]

    # Sort by resolve rate ascending
    resolve_rates = df[other_cols].mean(skipna=True).sort_values()
    other_cols = list(resolve_rates.index)

    rows = []
    for col in other_cols:
        weak = df[col].values
        mask = ~(np.isnan(weak) | np.isnan(strong))
        w, s = weak[mask], strong[mask]
        n = len(w)
        both = float(np.mean((w == 1) & (s == 1)))
        only_strong = float(np.mean((w == 0) & (s == 1)))
        only_weak = float(np.mean((w == 1) & (s == 0)))
        neither = float(np.mean((w == 0) & (s == 0)))
        rows.append({
            "model": col,
            "both": both,
            "only_strong": only_strong,
            "only_weak": only_weak,
            "neither": neither,
            "n": n,
        })

    fig, ax = plt.subplots(figsize=(9, max(5, len(rows) * 0.45 + 1.5)))

    y = np.arange(len(rows))
    colors = {
        "both": "#4CAF50",        # green
        "only_strong": "#FF9800", # orange
        "only_weak": "#9C27B0",   # purple
        "neither": "#EF5350",     # red
    }
    labels_map = {
        "both": "Both solve",
        "only_strong": f"Only {short(strong_col)} solves (headroom)",
        "only_weak": "Only weak model solves",
        "neither": "Neither solves",
    }

    left = np.zeros(len(rows))
    handles = []
    for key in ["both", "only_strong", "only_weak", "neither"]:
        vals = np.array([r[key] for r in rows])
        bars = ax.barh(y, vals, left=left, color=colors[key], label=labels_map[key], height=0.7)
        handles.append(bars)
        # Annotate cells > 5%
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 0.05:
                ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels([short(r["model"]) for r in rows], fontsize=8)
    ax.set_xlabel("Fraction of instances")
    ax.set_xlim(0, 1)
    ax.set_title(
        f"Instance quadrant breakdown: each model vs. {short(strong_col)}\n"
        "(models sorted by resolve rate, weakest at top)",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.axvline(x=1.0, color="black", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_resolve_vs_phi(df: pd.DataFrame, strong_col: str, out_path: Path) -> None:
    """
    Scatter: resolve rate (x) vs phi correlation with strong model (y).
    Shows that as models get better, they become MORE correlated with the strong model —
    harder to predict when routing helps.
    """
    if strong_col not in df.columns:
        print(f"Strong model {strong_col} not in matrix; skipping scatter")
        return

    strong = df[strong_col].values
    other_cols = [c for c in df.columns if c != strong_col]

    rr_vals, phi_vals, headroom_vals, names = [], [], [], []
    for col in other_cols:
        weak = df[col].values
        mask = ~(np.isnan(weak) | np.isnan(strong))
        w, s = weak[mask], strong[mask]
        rr_vals.append(float(np.nanmean(weak)))
        phi_vals.append(phi(weak, strong))
        headroom_vals.append(float(np.mean((w == 0) & (s == 1))))
        names.append(col)

    rr_vals = np.array(rr_vals)
    phi_vals = np.array(phi_vals)
    headroom_vals = np.array(headroom_vals)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: resolve rate vs phi — colored by headroom
    ax = axes[0]
    sc = ax.scatter(rr_vals, phi_vals, c=headroom_vals, cmap="YlOrRd_r",
                    s=80, edgecolors="black", linewidths=0.5, vmin=0, vmax=0.55)
    for i, name in enumerate(names):
        ax.annotate(short(name), (rr_vals[i], phi_vals[i]),
                    fontsize=6.5, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Routing headroom\n(weak fails, strong solves)", fontsize=8)
    ax.set_xlabel(f"Resolve rate of weak model", fontsize=9)
    ax.set_ylabel(f"Phi correlation with {short(strong_col)}", fontsize=9)
    ax.set_title("Better models → more correlated with strong model\n(routing becomes less useful)", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    # Right: resolve rate vs headroom — the abstention operating curve
    ax = axes[1]
    ax.scatter(rr_vals, headroom_vals, c=phi_vals, cmap="RdYlGn_r",
               s=80, edgecolors="black", linewidths=0.5, vmin=0, vmax=1)
    for i, name in enumerate(names):
        ax.annotate(short(name), (rr_vals[i], headroom_vals[i]),
                    fontsize=6.5, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")
    # Add oracle line: if all failures were recoverable
    strong_rr = float(np.nanmean(strong))
    xx = np.linspace(0, strong_rr, 100)
    ax.plot(xx, strong_rr - xx, color="gray", linestyle="--", linewidth=1, label="Oracle (perfect abstention)")
    ax.set_xlabel("Resolve rate of weak model", fontsize=9)
    ax.set_ylabel(f"Headroom (weak fails → {short(strong_col)} succeeds)", fontsize=9)
    ax.set_title("Routing headroom vs. weak model capability\n(dashed = oracle if strong model always recovers)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.65)

    plt.suptitle("OpenRouter sweep: complementarity analysis", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daytona-root", required=True,
                        help="Root dir containing or_sweep_<slug> subdirs with per-instance report.json")
    parser.add_argument("--parquet-dir", default="",
                        help="Eval parquet dir with 4-route successes (optional)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="or_sweep_",
                        help="Prefix to strip from Daytona dir names")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Daytona results...")
    sweep_results = load_daytona_results(Path(args.daytona_root), args.prefix)
    print(f"  {len(sweep_results)} sweep models loaded")

    all_results = dict(sweep_results)
    if args.parquet_dir:
        print("Loading parquet 4-route results...")
        parquet_results = load_parquet_models(args.parquet_dir)
        all_results.update(parquet_results)
        print(f"  {len(parquet_results)} parquet models added")

    print("Building success matrix...")
    df = build_matrix(all_results)
    # Drop instances not covered by sweep (NaN for all sweep models)
    sweep_cols = list(sweep_results.keys())
    df = df.dropna(subset=sweep_cols, how="all")
    print(f"  Matrix: {df.shape[0]} instances × {df.shape[1]} models")

    plot_instance_difficulty_heatmap(df, out_dir / "instance_difficulty_heatmap.png")
    plot_quadrant_breakdown(df, STRONG_MODEL, out_dir / "quadrant_breakdown.png")
    plot_resolve_vs_phi(df, STRONG_MODEL, out_dir / "resolve_vs_phi.png")
    print("\nDone.")


if __name__ == "__main__":
    main()

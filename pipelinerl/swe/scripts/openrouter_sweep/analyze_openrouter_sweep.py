#!/usr/bin/env python3
"""
Analyze the OpenRouter diversity sweep results.

Loads Daytona eval reports for all sweep models plus the existing 4-route
parquet, then computes:
  - Per-model resolve rates
  - Pairwise phi-correlation matrix (how correlated are solve patterns?)
  - Mixed-outcome fractions (model A fails, model B succeeds = routing headroom)
  - Capability-gap adjusted routing headroom

Writes a summary JSON and PNG figures to --output-dir.

Usage:
  python analyze_openrouter_sweep.py \
    --daytona-report-dir /mnt/llmd/results/.../openrouter_sweep/daytona \
    --existing-parquet-dir /mnt/llmd/results/.../collect/eval \
    --output-dir /mnt/llmd/results/.../openrouter_sweep/analysis
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(".", "_")


def load_daytona_results(
    daytona_log_dir: str, run_id_prefix: str
) -> dict[str, dict[str, bool]]:
    """
    Load per-model Daytona results from logs/run_evaluation/<run_id>/report.json.

    Expects:
      <daytona_log_dir>/<run_id_prefix>_<model_slug>/report.json
        {"ids_resolved": [...], "ids_unresolved": [...]}

    Returns: model_slug -> {instance_id: resolved}
    """
    results: dict[str, dict[str, bool]] = {}
    root = Path(daytona_log_dir)

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        if not model_dir.name.startswith(run_id_prefix):
            continue
        # Strip prefix to recover slug
        slug = model_dir.name[len(run_id_prefix):].lstrip("_")
        summary = model_dir / "report.json"

        if summary.exists():
            with summary.open() as fh:
                data = json.load(fh)
            model_results: dict[str, bool] = {}
            for iid in data.get("ids_resolved", []):
                model_results[str(iid)] = True
            for iid in data.get("ids_unresolved", []):
                model_results[str(iid)] = False
            results[slug] = model_results
        else:
            # Fall back to per-instance report subdirs
            model_results = {}
            for instance_dir in model_dir.iterdir():
                if not instance_dir.is_dir():
                    continue
                report = instance_dir / "report.json"
                if report.exists():
                    with report.open() as fh:
                        data = json.load(fh)
                    model_results[instance_dir.name] = bool(data.get("resolved", False))
            if model_results:
                results[slug] = model_results

        n_res = sum(v for v in results.get(slug, {}).values())
        n_tot = len(results.get(slug, {}))
        logger.info("model=%s  resolved=%d / %d", slug, n_res, n_tot)

    return results


def load_existing_route_successes(parquet_dir: str) -> pd.DataFrame | None:
    """Load the existing 4-route success matrix from the router dataset parquet."""
    if not parquet_dir:
        return None
    paths = sorted(Path(parquet_dir).glob("*.parquet"))
    if not paths:
        return None
    df = pd.concat([pd.read_parquet(p) for p in paths])
    if "route_successes" not in df.columns or "problem_id" not in df.columns:
        return None
    df = df.set_index("problem_id")
    # route_successes is a list of bools per row; expand into columns
    route_labels = df.iloc[0]["route_labels"] if "route_labels" in df.columns else None
    success_df = pd.DataFrame(
        df["route_successes"].tolist(),
        index=df.index,
        columns=route_labels if route_labels is not None else [f"route_{i}" for i in range(len(df.iloc[0]["route_successes"]))],
    )
    return success_df.astype(float)


def build_success_matrix(
    daytona_results: dict[str, dict[str, bool]],
    all_instance_ids: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Build an (n_instances x n_models) binary success matrix."""
    rows = {slug: {iid: float(v) for iid, v in res.items()} for slug, res in daytona_results.items()}
    df = pd.DataFrame(rows, index=all_instance_ids)
    # NaN = model didn't run on this instance
    return df, list(df.columns)


def phi_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Phi coefficient (Matthews correlation) for two binary arrays."""
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n == 0:
        return float("nan")
    tp = float(np.sum((a == 1) & (b == 1)))
    tn = float(np.sum((a == 0) & (b == 0)))
    fp = float(np.sum((a == 0) & (b == 1)))
    fn = float(np.sum((a == 1) & (b == 0)))
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return (tp * tn - fp * fn) / denom if denom > 0 else float("nan")


def mixed_outcome_fraction(cheaper: np.ndarray, pricier: np.ndarray) -> float:
    """Fraction of instances where cheaper fails but pricier succeeds."""
    mask = ~(np.isnan(cheaper) | np.isnan(pricier))
    a, b = cheaper[mask], pricier[mask]
    return float(np.mean((a == 0) & (b == 1))) if len(a) > 0 else float("nan")


def compute_pairwise_stats(success_df: pd.DataFrame) -> list[dict[str, Any]]:
    cols = list(success_df.columns)
    rows = []
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if i >= j:
                continue
            a = success_df[ci].values
            b = success_df[cj].values
            rows.append({
                "model_a": ci,
                "model_b": cj,
                "phi": phi_correlation(a, b),
                "mixed_a_fails_b_succeeds": mixed_outcome_fraction(a, b),
                "mixed_b_fails_a_succeeds": mixed_outcome_fraction(b, a),
                "resolve_a": float(np.nanmean(a)),
                "resolve_b": float(np.nanmean(b)),
            })
    return rows


def save_figures(success_df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available; skipping figures")
        return

    cols = list(success_df.columns)
    n = len(cols)
    phi_mat = np.full((n, n), fill_value=float("nan"))
    for i in range(n):
        for j in range(n):
            if i == j:
                phi_mat[i, j] = 1.0
            elif i < j:
                v = phi_correlation(success_df.iloc[:, i].values, success_df.iloc[:, j].values)
                phi_mat[i, j] = v
                phi_mat[j, i] = v

    short_names = [c.split("__")[-1][:25] for c in cols]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.7)))
    sns.heatmap(
        phi_mat,
        xticklabels=short_names,
        yticklabels=short_names,
        vmin=-1, vmax=1, center=0,
        cmap="RdBu_r",
        annot=True, fmt=".2f",
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Phi correlation of solve patterns across models")
    plt.tight_layout()
    fig.savefig(out_dir / "phi_correlation_matrix.png", dpi=150)
    plt.close(fig)
    logger.info("Saved phi_correlation_matrix.png")

    # Per-model resolve rates
    resolve_rates = success_df.mean(skipna=True).sort_values()
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), 5))
    resolve_rates.plot(kind="barh", ax=ax)
    ax.set_xlabel("Resolve rate")
    ax.set_title("Resolve rate by model")
    ax.set_xlim(0, 1)
    for bar, val in zip(ax.patches, resolve_rates.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "resolve_rates.png", dpi=150)
    plt.close(fig)
    logger.info("Saved resolve_rates.png")

    # Mixed-outcome heatmap (routing headroom)
    mixed_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mixed_mat[i, j] = mixed_outcome_fraction(
                success_df.iloc[:, i].values, success_df.iloc[:, j].values
            )
    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.7)))
    sns.heatmap(
        mixed_mat,
        xticklabels=short_names,
        yticklabels=short_names,
        vmin=0, vmax=0.5,
        cmap="YlOrRd",
        annot=True, fmt=".2f",
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Mixed-outcome fraction: row model fails, col model succeeds")
    plt.tight_layout()
    fig.savefig(out_dir / "routing_headroom_matrix.png", dpi=150)
    plt.close(fig)
    logger.info("Saved routing_headroom_matrix.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze OpenRouter diversity sweep")
    parser.add_argument("--daytona-log-dir", required=True,
                        help="Base dir for Daytona logs (typically logs/run_evaluation)")
    parser.add_argument("--run-id-prefix", default="or_sweep",
                        help="Prefix of run_id subdirs to include (default: or_sweep)")
    parser.add_argument("--existing-parquet-dir", default="",
                        help="Dir with existing 4-route eval parquet (optional)")
    parser.add_argument("--output-dir", required=True,
                        help="Dir to write analysis outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daytona_results = load_daytona_results(args.daytona_log_dir, args.run_id_prefix)
    if not daytona_results:
        logger.error("No Daytona results found in %s", args.daytona_report_dir)
        return

    all_instance_ids = sorted(
        set(iid for res in daytona_results.values() for iid in res)
    )
    logger.info("Total unique instances across all models: %d", len(all_instance_ids))

    success_df, model_slugs = build_success_matrix(daytona_results, all_instance_ids)

    # Optionally merge existing 4-route results for comparison
    existing_df = load_existing_route_successes(args.existing_parquet_dir)
    if existing_df is not None:
        existing_df = existing_df.reindex(all_instance_ids)
        success_df = pd.concat([success_df, existing_df], axis=1)
        logger.info("Merged %d existing route columns", len(existing_df.columns))

    per_model = {
        col: {
            "resolve_rate": float(success_df[col].mean(skipna=True)),
            "n_evaluated": int(success_df[col].notna().sum()),
            "n_resolved": int(success_df[col].sum(skipna=True)),
        }
        for col in success_df.columns
    }

    pairwise = compute_pairwise_stats(success_df)

    summary = {
        "n_models": len(success_df.columns),
        "n_instances": len(all_instance_ids),
        "per_model": per_model,
        "pairwise": pairwise,
    }

    summary_path = out_dir / "sweep_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote %s", summary_path)

    # Print a short table
    print("\n=== Per-model resolve rates ===")
    sorted_models = sorted(per_model.items(), key=lambda x: -x[1]["resolve_rate"])
    for slug, stats in sorted_models:
        print(f"  {slug:<50}  {stats['resolve_rate']:.3f}  ({stats['n_resolved']}/{stats['n_evaluated']})")

    print("\n=== Top routing headroom pairs (cheap fails, expensive succeeds) ===")
    top_pairs = sorted(pairwise, key=lambda x: -x["mixed_a_fails_b_succeeds"])[:10]
    for p in top_pairs:
        print(
            f"  {p['model_a'][:30]} -> {p['model_b'][:30]}  "
            f"headroom={p['mixed_a_fails_b_succeeds']:.3f}  phi={p['phi']:.3f}"
        )

    save_figures(success_df, out_dir)


if __name__ == "__main__":
    main()

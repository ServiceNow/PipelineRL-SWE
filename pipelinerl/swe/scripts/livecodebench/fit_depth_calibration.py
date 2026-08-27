#!/usr/bin/env python3
"""Fit per-route, per-failure-depth recalibration for the reachable policy heads.

The learned heads rank well but are badly scaled: on the held-out reachable split
the scout head is 9.28x overconfident while the oss120 head is 0.97x, and the
distortion is monotone in base rate. Because the decision rule compares
probabilities *across* routes with different costs, that per-route scale error
corrupts every routing decision even though within-route ranking is good
(within-depth AUC 0.834 for oss120, 0.855 for the nothing head).

A single global map cannot fix this: the model barely varies with failure depth
(scout 0.180 -> 0.107 across depths 1-10 while the truth goes 0.041 -> 0.000), so
depth structure has to come from the empirical table rather than from the model.
Fitting Platt scaling *within* each (route, depth) bucket injects the missing
level while preserving the ranking the model does supply.

Buckets with too little data or no positives fall back to the smoothed empirical
base rate, which is the correct behaviour where a route provably never succeeds.
Everything is fit on the calibration split only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pipelinerl.swe.scripts.livecodebench.replay_mdp_full_execution import (
    TorchSequentialScorer,
)

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _fit_platt(x: np.ndarray, y: np.ndarray, iters: int = 200) -> tuple[float, float]:
    """Newton-free logistic fit of y ~ sigmoid(a*x + b) via gradient descent."""
    a, b = 1.0, 0.0
    lr = 0.1
    for _ in range(iters):
        p = _sigmoid(a * x + b)
        ga = float(np.mean((p - y) * x))
        gb = float(np.mean(p - y))
        a -= lr * ga
        b -= lr * gb
    return a, b


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="calibration",
                        help="Split to fit on. Must not be the split used to report.")
    parser.add_argument("--min-bucket", type=int, default=40,
                        help="Below this many examples a bucket falls back to its base rate")
    parser.add_argument("--min-positives", type=int, default=5,
                        help="Below this many positives a bucket falls back to its base rate")
    parser.add_argument("--heads", default="scout,oss20,oss120,nothing")
    args = parser.parse_args()

    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    rows = _read_jsonl(Path(args.dataset_dir) / f"{args.split}.jsonl")
    if not rows:
        raise ValueError(f"No rows in {args.split} split")
    scorer = TorchSequentialScorer(Path(args.model_dir))

    preds = np.array([scorer(str(row["text"])) for row in rows], dtype=float)
    targets = np.array([row["targets"] for row in rows], dtype=float)
    depths = np.array([int(row["failure_depth"]) for row in rows])
    if preds.shape[1] < len(heads):
        raise ValueError(f"Model emits {preds.shape[1]} heads, expected {len(heads)}")

    calibration: dict[str, dict[str, Any]] = {}
    report: list[dict[str, Any]] = []
    for hi, head in enumerate(heads):
        calibration[head] = {}
        for depth in sorted(set(depths.tolist())):
            mask = depths == depth
            x = _logit(preds[mask, hi])
            y = targets[mask, hi]
            n = int(mask.sum())
            pos = int(y.sum())
            # Laplace-smoothed base rate: the fallback, and the right answer for
            # buckets where the route provably never succeeds.
            base = float((pos + 0.5) / (n + 1.0))
            if n >= args.min_bucket and pos >= args.min_positives:
                a, b = _fit_platt(x, y)
                entry = {"mode": "platt", "a": a, "b": b, "n": n, "positives": pos}
                fitted = _sigmoid(a * x + b)
            else:
                entry = {"mode": "constant", "value": base, "n": n, "positives": pos}
                fitted = np.full(n, base)
            calibration[head][str(int(depth))] = entry
            report.append({
                "head": head,
                "failure_depth": int(depth),
                "n": n,
                "positives": pos,
                "mean_raw": float(preds[mask, hi].mean()),
                "mean_calibrated": float(np.mean(fitted)),
                "actual": float(y.mean()),
                "mode": entry["mode"],
            })

    out = {
        "schema_version": 1,
        "fit_split": args.split,
        "dataset_dir": str(args.dataset_dir),
        "model_dir": str(args.model_dir),
        "heads": heads,
        "min_bucket": args.min_bucket,
        "min_positives": args.min_positives,
        "calibration": calibration,
        "fit_report": report,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'head':<10} {'depth':>6} {'n':>6} {'raw':>8} {'calib':>8} {'actual':>8}  mode")
    for row in report:
        print(
            f"{row['head']:<10} {row['failure_depth']:6d} {row['n']:6d} "
            f"{row['mean_raw']:8.4f} {row['mean_calibrated']:8.4f} {row['actual']:8.4f}  {row['mode']}"
        )
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()

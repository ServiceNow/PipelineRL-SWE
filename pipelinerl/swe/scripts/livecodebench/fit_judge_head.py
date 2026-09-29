#!/usr/bin/env python3
"""Fit a causal judge for one code attempt from that attempt's own embedding.

The replay consumes a fixed score as soon as an attempt is drawn. Features based on
other attempts cannot be precomputed for this interface: they may include attempts
that have not been drawn yet. A set-aware judge needs a separate online scorer that
recomputes scores from only the attempts observed at each decision.
`replay_prefix_value.py` instead keeps this independent attempt score and learns
continuation value from the scores and routes actually observed so far.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts-dir", required=True)
    ap.add_argument("--act-tag", default="judge")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--readouts", default="last")
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prompts_dir = Path(a.prompts_dir)
    ids: list[str] = []
    features: list[np.ndarray] = []
    for shard in range(a.shards):
        path = prompts_dir / f"act_{a.act_tag}_shard{shard}.npz"
        if not path.exists():
            raise SystemExit(f"missing activation shard: {path}")
        with np.load(path, allow_pickle=True) as data:
            ids.extend(str(x) for x in data["problem_ids"])
            features.append(np.concatenate(
                [data[k].reshape(len(data[k]), -1) for k in a.readouts.split(",")], axis=1
            ))
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate attempt IDs in activation shards")
    X = np.concatenate(features).astype(np.float32)
    manifest_rows = [json.loads(line) for line in open(prompts_dir / "judge_manifest.jsonl")
                     if line.strip()]
    manifest = {row["example_id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise SystemExit("duplicate attempt IDs in judge manifest")
    activation_ids = set(ids)
    missing = manifest.keys() - activation_ids
    unexpected = activation_ids - manifest.keys()
    if missing or unexpected:
        raise SystemExit(f"judge activation/manifest mismatch: {len(missing)} missing, "
                         f"{len(unexpected)} unexpected attempt IDs")

    split = json.loads((Path(a.tensors_dir) / "split_manifest.json").read_text())
    group = {**{str(pid): "train" for pid in split["train_problem_ids"]},
             **{str(pid): "calibration" for pid in split["calibration_problem_ids"]},
             **{str(pid): "test" for pid in split["test_problem_ids"]}}
    unknown = {str(row["problem_id"]) for row in manifest_rows} - group.keys()
    if unknown:
        raise SystemExit(f"judge manifest has {len(unknown)} problems outside the split")
    indices = {name: np.array([i for i, example_id in enumerate(ids)
                               if group.get(str(manifest[example_id]["problem_id"])) == name], dtype=int)
               for name in ("train", "calibration", "test")}
    train = indices["train"]
    if len(train) < 2:
        raise SystemExit("judge training split has fewer than two attempts")
    mean = X[train].mean(axis=0)
    scale = X[train].std(axis=0) + 1e-6
    components = min(a.pca, len(train) - 1, X.shape[1])
    Z = PCA(n_components=components, random_state=0).fit(
        (X[train] - mean) / scale
    ).transform((X - mean) / scale).astype(np.float32)
    labels = np.array([bool(manifest[example_id]["correct"]) for example_id in ids], dtype=int)
    classifier = LogisticRegression(C=a.C, max_iter=3000).fit(Z[train], labels[train])
    scores = classifier.predict_proba(Z)[:, 1]
    print(f"{len(ids)} independent attempt scores; PCA {Z.shape[1]}")
    for name in ("train", "calibration", "test"):
        rows = indices[name]
        if not len(rows):
            continue
        auc = roc_auc_score(labels[rows], scores[rows]) if len(np.unique(labels[rows])) == 2 else float("nan")
        print(f"  {name:<11} n={len(rows)} AUC={auc:.3f} "
              f"mean_pred={scores[rows].mean():.3f} actual={labels[rows].mean():.3f}")
    print("  fixed mode: independent; test metrics are diagnostic only")
    with open(a.out, "w") as output:
        for example_id, score in zip(ids, scores):
            output.write(json.dumps({"example_id": example_id, "p_correct": float(score),
                                     "judge_mode": "independent"}) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

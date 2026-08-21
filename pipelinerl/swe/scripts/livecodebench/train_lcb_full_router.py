#!/usr/bin/env python3
"""Train a three-action LCB direct-routing baseline from scout evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from pipelinerl.swe.scripts.offline_router.abstention_features import build_abstention_input


def _read(path: Path) -> list[dict]:
    with open(path) as in_f:
        return [json.loads(line) for line in in_f if line.strip()]


def _text(row: dict, input_only: bool, include_feedback: bool) -> str:
    return build_abstention_input(
        str(row["problem_statement"]),
        str(row.get("thinking_text", "")),
        str(row.get("patch_text", "")),
        include_thinking=False,
        input_only=input_only,
        test_feedback_text=str(row.get("test_feedback", "")) if include_feedback else "",
        include_test_feedback=include_feedback,
    )


def _fit(texts: list[str], targets: np.ndarray) -> Pipeline:
    estimator = (
        DummyClassifier(strategy="prior")
        if len(set(targets.tolist())) < 2
        else LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    )
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000)),
            ("model", estimator),
        ]
    )
    pipeline.fit(texts, targets)
    return pipeline


def _positive_probability(model: Pipeline, texts: list[str]) -> np.ndarray:
    classes = model.named_steps["model"].classes_
    probabilities = model.predict_proba(texts)
    if 1 not in classes:
        return np.zeros(len(texts), dtype=float)
    return probabilities[:, list(classes).index(1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-only", action="store_true")
    parser.add_argument("--include-test-feedback", action="store_true")
    args = parser.parse_args()

    data_dir, output_dir = Path(args.router_data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, eval_rows = _read(data_dir / "router_train.jsonl"), _read(data_dir / "router_eval.jsonl")
    route_labels = train[0]["route_labels"]
    if any(row["route_labels"] != route_labels for row in [*train, *eval_rows]):
        raise ValueError("Route labels are inconsistent")
    train_texts = [_text(row, args.input_only, args.include_test_feedback) for row in train]
    eval_texts = [_text(row, args.input_only, args.include_test_feedback) for row in eval_rows]

    predictions = np.zeros((len(eval_rows), len(route_labels)), dtype=float)
    aucs: dict[str, float | None] = {}
    for index, label in enumerate(route_labels):
        targets = np.array([int(row["route_successes"][index]) for row in train])
        model = _fit(train_texts, targets)
        joblib.dump(model, output_dir / f"{label}_model.joblib")
        predictions[:, index] = _positive_probability(model, eval_texts)
        actual = np.array([int(row["route_successes"][index]) for row in eval_rows])
        aucs[label] = (
            float(roc_auc_score(actual, predictions[:, index]))
            if len(set(actual.tolist())) == 2
            else None
        )

    with open(output_dir / "eval_predictions.jsonl", "w") as out_f:
        for index, row in enumerate(eval_rows):
            out_f.write(
                json.dumps(
                    {
                        "problem_id": row["problem_id"],
                        "route_labels": route_labels,
                        "p_successes": predictions[index].tolist(),
                        "route_successes": row["route_successes"],
                        "route_public_successes": row["route_public_successes"],
                        "route_prompt_tokens": row["route_prompt_tokens"],
                        "route_completion_tokens": row["route_completion_tokens"],
                    }
                )
                + "\n"
            )
    with open(output_dir / "summary.json", "w") as out_f:
        json.dump(
            {
                "router": "tfidf_logistic_per_route",
                "input_only": args.input_only,
                "include_test_feedback": args.include_test_feedback,
                "route_labels": route_labels,
                "n_train": len(train),
                "n_eval": len(eval_rows),
                "per_route_eval_auc": aucs,
            },
            out_f,
            indent=2,
            sort_keys=True,
        )
    print(json.dumps({"n_train": len(train), "n_eval": len(eval_rows), "auc": aucs}), flush=True)


if __name__ == "__main__":
    main()

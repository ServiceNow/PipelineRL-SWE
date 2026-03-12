#!/usr/bin/env python
import argparse
import csv
import json
import logging
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import matplotlib.pyplot as plt  # type: ignore

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

from pipelinerl.finetune.value_model import AutoModelForCausalLMWithValueHead
from pipelinerl.swe.scripts.new.router_trace_utils import load_router_traces

logger = logging.getLogger(__name__)


EXT_LANGUAGE_MAP = {
    ".py": "Python",
    ".pyi": "Python",
    ".ipynb": "Python",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".kt": "Kotlin",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".cpp": "C++",
    ".cc": "C++",
    ".c": "C",
    ".cs": "C#",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".scala": "Scala",
}


def _pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _select_best_split(traces: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return traces
    return [trace for trace in traces if trace.get("split") == split]


def _split_within_repo(
    traces: list[dict[str, Any]],
    train_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_fraction <= 0.0 or train_fraction >= 1.0:
        raise ValueError(f"train_fraction must be in (0,1), got {train_fraction}")
    by_repo: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        repo = str(trace.get("repo") or "unknown")
        by_repo.setdefault(repo, []).append(trace)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    eval_: list[dict[str, Any]] = []
    for repo_traces in by_repo.values():
        local = list(repo_traces)
        rng.shuffle(local)
        n = len(local)
        if n == 1:
            train.extend(local)
            continue
        n_train = int(round(n * train_fraction))
        n_train = max(1, min(n - 1, n_train))
        train.extend(local[:n_train])
        eval_.extend(local[n_train:])
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def _infer_language(trace: dict[str, Any]) -> str:
    for key in ("language", "lang", "repo_language"):
        value = trace.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    dataset = str(trace.get("dataset") or "").lower()
    if "smith_py" in dataset or dataset.endswith("_py"):
        return "Python"
    if "smith_go" in dataset or dataset.endswith("_go"):
        return "Go"
    if "smith_rs" in dataset or dataset.endswith("_rs"):
        return "Rust"
    if "smith_java" in dataset or dataset.endswith("_java"):
        return "Java"

    files = trace.get("files_for_repair")
    if isinstance(files, list) and files:
        ext_counts = Counter()
        for item in files:
            if not isinstance(item, str):
                continue
            suffix = Path(item).suffix.lower()
            if suffix:
                ext_counts[EXT_LANGUAGE_MAP.get(suffix, "Other/Unknown")] += 1
        if ext_counts:
            return ext_counts.most_common(1)[0][0]
    return "Unknown"


def _encode_prompt_last_embedding(
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: Any,
    device: torch.device,
    prompt_text: str,
    output_text: str,
) -> np.ndarray | None:
    if not prompt_text:
        return None
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt_text + (output_text or ""), add_special_tokens=False, return_tensors="pt").input_ids
    prompt_len = int(prompt_ids.shape[1])
    full_len = int(full_ids.shape[1])
    if prompt_len <= 0 or full_len <= prompt_len:
        return None

    input_ids = full_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        return None
    last_hidden = hidden_states[-1]
    prompt_last_idx = prompt_len - 1
    if prompt_last_idx >= last_hidden.shape[1]:
        return None
    return last_hidden[0, prompt_last_idx].detach().float().cpu().numpy()


def _encode_prompt_last_embeddings(
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: Any,
    device: torch.device,
    prompt_text: str,
    output_text: str,
    hidden_indices: list[int],
) -> dict[int, np.ndarray] | None:
    if not prompt_text:
        return None
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt_text + (output_text or ""), add_special_tokens=False, return_tensors="pt").input_ids
    prompt_len = int(prompt_ids.shape[1])
    full_len = int(full_ids.shape[1])
    if prompt_len <= 0 or full_len <= prompt_len:
        return None

    input_ids = full_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        return None
    prompt_last_idx = prompt_len - 1
    if prompt_last_idx < 0 or prompt_last_idx >= hidden_states[-1].shape[1]:
        return None

    embeddings: dict[int, np.ndarray] = {}
    for hidden_idx in hidden_indices:
        if hidden_idx < 0 or hidden_idx >= len(hidden_states):
            return None
        vec = hidden_states[hidden_idx][0, prompt_last_idx].detach().float().cpu().numpy()
        embeddings[hidden_idx] = vec
    return embeddings


def _parse_probe_hidden_indices(
    probe_layers: str,
    num_hidden_layers: int,
    include_embedding_layer: bool,
) -> list[int]:
    total_hidden_states = num_hidden_layers + 1  # includes embedding state at index 0
    text = (probe_layers or "last").strip().lower()
    if text == "last":
        return [num_hidden_layers]
    if text == "all":
        start = 0 if include_embedding_layer else 1
        return list(range(start, total_hidden_states))

    indices: list[int] = []
    for chunk in probe_layers.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        idx = int(chunk)
        if idx < 0:
            idx = total_hidden_states + idx
        if idx < 0 or idx >= total_hidden_states:
            raise ValueError(
                f"Hidden index out of range: {chunk}. Valid range is [0,{total_hidden_states-1}] "
                "or negative python-style indexing."
            )
        indices.append(idx)
    if not indices:
        raise ValueError(f"Failed to parse --probe-layers={probe_layers}")
    # Stable unique order.
    unique: list[int] = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


def _hidden_index_to_label(hidden_idx: int) -> str:
    if hidden_idx == 0:
        return "embedding"
    return f"layer_{hidden_idx - 1:02d}"


def _fit_ridge_multiclass(
    x_train: np.ndarray,
    y_train_idx: np.ndarray,
    n_classes: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_train = np.zeros((x_train.shape[0], n_classes), dtype=np.float32)
    y_train[np.arange(x_train.shape[0]), y_train_idx] = 1.0

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    x_norm = (x_train - x_mean) / x_std
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_center = y_train - y_mean

    n, d = x_norm.shape
    if n <= d:
        k = x_norm @ x_norm.T
        k_reg = k + alpha * np.eye(n, dtype=x_norm.dtype)
        dual = np.linalg.solve(k_reg, y_center)
        weights = x_norm.T @ dual
    else:
        gram = x_norm.T @ x_norm
        gram_reg = gram + alpha * np.eye(d, dtype=x_norm.dtype)
        weights = np.linalg.solve(gram_reg, x_norm.T @ y_center)
    bias = y_mean - (x_mean / x_std) @ weights
    return weights, bias.reshape(-1), x_mean.reshape(-1), x_std.reshape(-1)


def _predict_scores(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> np.ndarray:
    x_norm = (x - x_mean[None, :]) / x_std[None, :]
    return x_norm @ weights + bias[None, :]


def _plot_per_class_accuracy(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not MATPLOTLIB_AVAILABLE or not rows:
        return
    labels = [row["language"] for row in rows]
    values = [float(row["accuracy"]) for row in rows]
    plt.figure(figsize=(max(8, 0.6 * len(labels)), 4.8))
    plt.bar(labels, values, color="tab:blue")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Per-language probe accuracy")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_confusion(confusion: np.ndarray, labels: list[str], out_path: Path) -> None:
    if not MATPLOTLIB_AVAILABLE or confusion.size == 0:
        return
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, np.maximum(row_sums, 1), where=row_sums > 0)
    plt.figure(figsize=(max(8, 0.65 * len(labels)), max(6, 0.55 * len(labels))))
    image = plt.imshow(normalized, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(image, label="Row-normalized fraction")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted language")
    plt.ylabel("True language")
    plt.title("Language probe confusion matrix")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_accuracy_by_layer(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not MATPLOTLIB_AVAILABLE or not rows:
        return
    labels = [row["layer_label"] for row in rows]
    values = [float(row["accuracy_eval"]) for row in rows]
    baseline = [float(row["majority_baseline_accuracy_eval"]) for row in rows]
    plt.figure(figsize=(max(10, 0.45 * len(labels)), 4.8))
    plt.plot(range(len(labels)), values, marker="o", linewidth=1.8, label="Probe accuracy")
    plt.plot(range(len(labels)), baseline, marker="x", linewidth=1.2, linestyle="--", label="Majority baseline")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Language probe accuracy by hidden layer")
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear language probe on prompt-last trunk embeddings.")
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train-split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--eval-split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--split-mode", default="existing", choices=["existing", "within-repo"])
    parser.add_argument("--within-repo-source-split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--within-repo-train-fraction", type=float, default=0.8)
    parser.add_argument("--within-repo-seed", type=int, default=0)
    parser.add_argument("--all-model-versions", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-eval-records", type=int, default=None)
    parser.add_argument("--include-unknown-language", action="store_true")
    parser.add_argument("--min-train-class-count", type=int, default=10)
    parser.add_argument(
        "--probe-layers",
        default="last",
        help=(
            "Which hidden states to probe: 'last', 'all', or comma-separated hidden-state indices "
            "(index 0 is embedding state, index N is final layer output)."
        ),
    )
    parser.add_argument(
        "--include-embedding-layer",
        action="store_true",
        help="When --probe-layers=all, also include hidden index 0 (embedding state).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.split_mode == "within-repo":
        traces = load_router_traces(
            input_globs=args.input_glob,
            split=None,
            latest_model_only=not args.all_model_versions,
            dedupe_by_problem=not args.keep_duplicates,
        )
        if not traces:
            raise ValueError("No traces found")
        source = _select_best_split(traces, args.within_repo_source_split)
        train_traces, eval_traces = _split_within_repo(
            source,
            train_fraction=float(args.within_repo_train_fraction),
            seed=int(args.within_repo_seed),
        )
    else:
        train_split = None if args.train_split == "all" else args.train_split
        eval_split = None if args.eval_split == "all" else args.eval_split
        train_traces = load_router_traces(
            input_globs=args.input_glob,
            split=train_split,
            latest_model_only=not args.all_model_versions,
            dedupe_by_problem=not args.keep_duplicates,
        )
        eval_traces = load_router_traces(
            input_globs=args.input_glob,
            split=eval_split,
            latest_model_only=not args.all_model_versions,
            dedupe_by_problem=not args.keep_duplicates,
        )

    if args.max_train_records is not None:
        train_traces = train_traces[: args.max_train_records]
    if args.max_eval_records is not None:
        eval_traces = eval_traces[: args.max_eval_records]
    if not train_traces or not eval_traces:
        raise ValueError(f"Empty split after filtering: train={len(train_traces)} eval={len(eval_traces)}")

    device = _pick_device(args.device)
    logger.info("Loading trunk model from %s on %s", args.model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_path)
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if num_hidden_layers <= 0:
        raise ValueError(f"Invalid num_hidden_layers={num_hidden_layers} in model config")
    hidden_indices = _parse_probe_hidden_indices(
        probe_layers=args.probe_layers,
        num_hidden_layers=num_hidden_layers,
        include_embedding_layer=bool(args.include_embedding_layer),
    )
    logger.info(
        "Probing hidden indices=%s (labels=%s)",
        hidden_indices,
        [_hidden_index_to_label(idx) for idx in hidden_indices],
    )

    def build_xy(traces_in: list[dict[str, Any]], tag: str) -> tuple[dict[int, np.ndarray], list[str], int]:
        xs_by_layer: dict[int, list[np.ndarray]] = {idx: [] for idx in hidden_indices}
        ys: list[str] = []
        skipped = 0
        for trace in tqdm(traces_in, desc=f"Embedding {tag}", unit="trace"):
            policy = trace.get("policy") or {}
            prompt_text = policy.get("prompt_text")
            output_text = policy.get("output_text") or ""
            if not isinstance(prompt_text, str):
                skipped += 1
                continue
            language = _infer_language(trace)
            if language == "Unknown" and not args.include_unknown_language:
                skipped += 1
                continue
            embeddings = _encode_prompt_last_embeddings(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt_text=prompt_text,
                output_text=output_text,
                hidden_indices=hidden_indices,
            )
            if embeddings is None:
                skipped += 1
                continue
            for hidden_idx in hidden_indices:
                xs_by_layer[hidden_idx].append(embeddings[hidden_idx].astype(np.float32))
            ys.append(language)
        if not ys:
            raise ValueError(f"No valid examples for {tag}")
        return {idx: np.stack(xs_by_layer[idx]) for idx in hidden_indices}, ys, skipped

    x_train_by_layer, y_train_labels, train_skipped = build_xy(train_traces, "train")
    x_eval_by_layer, y_eval_labels, eval_skipped = build_xy(eval_traces, "eval")

    train_counts = Counter(y_train_labels)
    allowed_languages = {lang for lang, cnt in train_counts.items() if cnt >= args.min_train_class_count}
    if len(allowed_languages) < 2:
        raise ValueError(
            f"Need at least 2 classes after min-train-class-count filtering; got {len(allowed_languages)}. "
            f"Counts={dict(train_counts)}"
        )

    train_keep = [i for i, lang in enumerate(y_train_labels) if lang in allowed_languages]
    eval_keep = [i for i, lang in enumerate(y_eval_labels) if lang in allowed_languages]
    if not train_keep or not eval_keep:
        raise ValueError(
            f"Empty keep set after class filtering. train_keep={len(train_keep)} eval_keep={len(eval_keep)} "
            f"allowed_languages={sorted(allowed_languages)}"
        )

    y_train_labels = [y_train_labels[i] for i in train_keep]
    y_eval_labels = [y_eval_labels[i] for i in eval_keep]
    x_train_by_layer = {idx: x_train_by_layer[idx][train_keep] for idx in hidden_indices}
    x_eval_by_layer = {idx: x_eval_by_layer[idx][eval_keep] for idx in hidden_indices}
    train_counts_filtered = Counter(y_train_labels)

    language_order = sorted(allowed_languages)
    lang_to_idx = {lang: idx for idx, lang in enumerate(language_order)}
    y_train_idx = np.asarray([lang_to_idx[lang] for lang in y_train_labels], dtype=np.int64)
    y_eval_idx = np.asarray([lang_to_idx[lang] for lang in y_eval_labels], dtype=np.int64)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_count_rows = [
        {
            "language": lang,
            "train_count": int(sum(1 for v in y_train_labels if v == lang)),
            "eval_count": int(sum(1 for v in y_eval_labels if v == lang)),
        }
        for lang in language_order
    ]
    _write_csv(output_dir / "label_counts.csv", label_count_rows, ["language", "train_count", "eval_count"])

    per_layer_rows: list[dict[str, Any]] = []
    multi_layer = len(hidden_indices) > 1

    for hidden_idx in hidden_indices:
        layer_label = _hidden_index_to_label(hidden_idx)
        layer_output_dir = output_dir / "layers" / layer_label if multi_layer else output_dir
        layer_output_dir.mkdir(parents=True, exist_ok=True)

        x_train = x_train_by_layer[hidden_idx]
        x_eval = x_eval_by_layer[hidden_idx]
        weights, bias, x_mean, x_std = _fit_ridge_multiclass(
            x_train=x_train,
            y_train_idx=y_train_idx,
            n_classes=len(language_order),
            alpha=float(args.alpha),
        )
        eval_scores = _predict_scores(x_eval, weights, bias, x_mean, x_std)
        eval_pred_idx = np.argmax(eval_scores, axis=1)
        eval_acc = float(np.mean(eval_pred_idx == y_eval_idx))
        majority_class = train_counts_filtered.most_common(1)[0][0]
        majority_idx = lang_to_idx[majority_class]
        majority_acc = float(np.mean(y_eval_idx == majority_idx))

        confusion = np.zeros((len(language_order), len(language_order)), dtype=np.int64)
        for t, p in zip(y_eval_idx, eval_pred_idx):
            confusion[t, p] += 1

        per_class_rows: list[dict[str, Any]] = []
        for lang, idx in lang_to_idx.items():
            support = int(np.sum(y_eval_idx == idx))
            correct = int(confusion[idx, idx])
            accuracy = correct / support if support else 0.0
            predicted = int(np.sum(confusion[:, idx]))
            precision = correct / predicted if predicted else 0.0
            recall = accuracy
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            per_class_rows.append(
                {
                    "language": lang,
                    "support_eval": support,
                    "correct": correct,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
        per_class_rows = sorted(per_class_rows, key=lambda row: row["support_eval"], reverse=True)

        _write_csv(
            layer_output_dir / "per_class_metrics.csv",
            per_class_rows,
            ["language", "support_eval", "correct", "accuracy", "precision", "recall", "f1"],
        )

        confusion_rows = []
        for true_idx, true_lang in enumerate(language_order):
            for pred_idx, pred_lang in enumerate(language_order):
                confusion_rows.append(
                    {
                        "true_language": true_lang,
                        "pred_language": pred_lang,
                        "count": int(confusion[true_idx, pred_idx]),
                    }
                )
        _write_csv(
            layer_output_dir / "confusion_matrix_long.csv",
            confusion_rows,
            ["true_language", "pred_language", "count"],
        )

        np.savez(
            layer_output_dir / "language_probe_weights.npz",
            weights=weights,
            bias=bias,
            x_mean=x_mean,
            x_std=x_std,
            languages=np.asarray(language_order),
        )

        _plot_per_class_accuracy(per_class_rows, layer_output_dir / "per_class_accuracy.png")
        _plot_confusion(confusion, language_order, layer_output_dir / "confusion_matrix.png")

        row = {
            "hidden_index": hidden_idx,
            "layer_label": layer_label,
            "transformer_layer_index": (hidden_idx - 1) if hidden_idx > 0 else -1,
            "accuracy_eval": eval_acc,
            "majority_baseline_accuracy_eval": majority_acc,
            "majority_class_train": majority_class,
            "n_train_used": int(x_train.shape[0]),
            "n_eval_used": int(x_eval.shape[0]),
            "embedding_dim": int(x_train.shape[1]),
        }
        per_layer_rows.append(row)
        logger.info(
            "Layer %s (hidden_idx=%d): eval_accuracy=%.4f majority_baseline=%.4f",
            layer_label,
            hidden_idx,
            eval_acc,
            majority_acc,
        )

    per_layer_rows = sorted(per_layer_rows, key=lambda row: row["hidden_index"])
    _write_csv(
        output_dir / "per_layer_metrics.csv",
        per_layer_rows,
        [
            "hidden_index",
            "layer_label",
            "transformer_layer_index",
            "accuracy_eval",
            "majority_baseline_accuracy_eval",
            "majority_class_train",
            "n_train_used",
            "n_eval_used",
            "embedding_dim",
        ],
    )
    if multi_layer:
        _plot_accuracy_by_layer(per_layer_rows, output_dir / "accuracy_by_layer.png")

    best_row = max(per_layer_rows, key=lambda row: row["accuracy_eval"])
    summary = {
        "model_path": args.model_path,
        "split_mode": args.split_mode,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "alpha": float(args.alpha),
        "probe_layers": args.probe_layers,
        "include_embedding_layer": bool(args.include_embedding_layer),
        "hidden_indices": hidden_indices,
        "n_train_input": len(train_traces),
        "n_eval_input": len(eval_traces),
        "n_train_used": int(per_layer_rows[0]["n_train_used"]),
        "n_eval_used": int(per_layer_rows[0]["n_eval_used"]),
        "train_skipped_before_class_filter": train_skipped,
        "eval_skipped_before_class_filter": eval_skipped,
        "languages": language_order,
        "best_layer": best_row,
        "all_layers": per_layer_rows,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info(
        "Done. best_layer=%s acc=%.4f baseline=%.4f classes=%d output=%s",
        best_row["layer_label"],
        best_row["accuracy_eval"],
        best_row["majority_baseline_accuracy_eval"],
        len(language_order),
        output_dir,
    )


if __name__ == "__main__":
    main()

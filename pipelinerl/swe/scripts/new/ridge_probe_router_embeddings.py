#!/usr/bin/env python
import argparse
import csv
import json
import logging
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
from pipelinerl.swe.scripts.new.router_trace_utils import (
    dedupe_latest_by_problem,
    extract_reward_vector,
    extract_route_labels,
    load_router_traces,
)

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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size == 0 or y.size == 0 or x.shape != y.shape:
        return None
    x_center = x - x.mean()
    y_center = y - y.mean()
    denom = float(np.sqrt(np.sum(x_center**2) * np.sum(y_center**2)))
    if denom <= 1e-12:
        return None
    return float(np.sum(x_center * y_center) / denom)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    pos = 0
    while pos < len(order):
        start = pos
        current = values[order[pos]]
        while pos + 1 < len(order) and abs(values[order[pos + 1]] - current) <= 1e-12:
            pos += 1
        end = pos
        avg_rank = (start + end) / 2.0 + 1.0
        ranks[order[start : end + 1]] = avg_rank
        pos += 1
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size == 0 or y.size == 0 or x.shape != y.shape:
        return None
    return _pearson_corr(_average_ranks(x), _average_ranks(y))


def _roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if labels.size == 0 or scores.size == 0 or labels.shape != scores.shape:
        return None
    positives = int(np.sum(labels == 1))
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(scores, dtype=np.float64)
    pos = 0
    while pos < len(order):
        start = pos
        current = scores[order[pos]]
        while pos + 1 < len(order) and abs(scores[order[pos + 1]] - current) <= 1e-12:
            pos += 1
        end = pos
        avg_rank = (start + end) / 2.0 + 1.0
        ranks[order[start : end + 1]] = avg_rank
        pos += 1
    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    u_stat = rank_sum_pos - positives * (positives + 1) / 2.0
    return float(u_stat / (positives * negatives))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.size == 0 or y_pred.size == 0 or y_true.shape != y_pred.shape:
        return None
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot <= 1e-12:
        return None
    return float(1.0 - (ss_res / ss_tot))


def _select_best_split(traces: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return traces
    return [trace for trace in traces if trace.get("split") == split]


def _as_model_version(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _choose_joint_model_version(
    traces: list[dict[str, Any]],
    train_split: str | None,
    eval_split: str | None,
) -> int | None:
    versions = sorted({_as_model_version(trace.get("model_version")) for trace in traces if trace.get("model_version") is not None})
    versions = [version for version in versions if version >= 0]
    if not versions:
        return None

    train_counts_all = {version: 0 for version in versions}
    eval_counts_all = {version: 0 for version in versions}
    for trace in traces:
        version = _as_model_version(trace.get("model_version"))
        if version not in train_counts_all:
            continue
        split = trace.get("split")
        if train_split is None or split == train_split:
            train_counts_all[version] += 1
        if eval_split is None or split == eval_split:
            eval_counts_all[version] += 1

    if eval_split is not None:
        eval_versions = [version for version in versions if eval_counts_all[version] > 0]
        recent_versions = (eval_versions or versions)[-10:]
    else:
        recent_versions = versions[-10:]

    train_counts = {version: train_counts_all[version] for version in recent_versions}
    eval_counts = {version: eval_counts_all[version] for version in recent_versions}

    target_eval_count = max(eval_counts.values())
    candidates = [version for version in recent_versions if eval_counts[version] == target_eval_count]
    if not candidates:
        return recent_versions[-1]

    latest_candidate = max(candidates)
    latest_train_count = train_counts[latest_candidate]
    max_train_count = max(train_counts[version] for version in candidates)
    if latest_train_count < max_train_count:
        best_candidates = [version for version in candidates if train_counts[version] == max_train_count]
        chosen = max(best_candidates)
        logger.warning(
            "Latest model_version=%s has %s train traces, but recent full-eval versions have up to %s train traces; using model_version=%s.",
            latest_candidate,
            latest_train_count,
            max_train_count,
            chosen,
        )
        return chosen
    return latest_candidate


def _choose_eval_model_version(
    traces: list[dict[str, Any]],
    eval_split: str | None,
) -> int | None:
    versions = sorted({_as_model_version(trace.get("model_version")) for trace in traces if trace.get("model_version") is not None})
    versions = [version for version in versions if version >= 0]
    if not versions:
        return None

    eval_counts = {version: 0 for version in versions}
    for trace in traces:
        version = _as_model_version(trace.get("model_version"))
        if version not in eval_counts:
            continue
        split = trace.get("split")
        if eval_split is None or split == eval_split:
            eval_counts[version] += 1

    eval_versions = [version for version in versions if eval_counts[version] > 0]
    if not eval_versions:
        return None

    recent_versions = eval_versions[-10:]
    target_eval_count = max(eval_counts[version] for version in recent_versions)
    candidates = [version for version in recent_versions if eval_counts[version] == target_eval_count]
    return max(candidates) if candidates else recent_versions[-1]


def _select_train_window_up_to_eval(
    traces: list[dict[str, Any]],
    train_split: str | None,
    eval_version: int,
    target_train_traces: int,
    max_versions_back: int | None,
) -> tuple[list[dict[str, Any]], list[int]]:
    versions = sorted(
        {
            _as_model_version(trace.get("model_version"))
            for trace in traces
            if trace.get("model_version") is not None and _as_model_version(trace.get("model_version")) >= 0
        }
    )
    eligible_versions = [version for version in versions if version <= eval_version]
    if max_versions_back is not None and max_versions_back > 0:
        eligible_versions = eligible_versions[-max_versions_back:]
    eligible_versions = list(reversed(eligible_versions))

    selected_versions: list[int] = []
    train_traces: list[dict[str, Any]] = []
    for version in eligible_versions:
        version_traces = [
            trace
            for trace in traces
            if _as_model_version(trace.get("model_version")) == version
            and (train_split is None or trace.get("split") == train_split)
        ]
        if not version_traces:
            continue
        selected_versions.append(version)
        train_traces.extend(version_traces)
        if len(train_traces) >= target_train_traces:
            break

    return train_traces, list(reversed(selected_versions))


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

    for repo, repo_traces in by_repo.items():
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
    logger.info(
        "Within-repo split built: repos=%d train=%d eval=%d train_fraction=%.3f seed=%d",
        len(by_repo),
        len(train),
        len(eval_),
        train_fraction,
        seed,
    )
    return train, eval_


def _encode_embeddings(
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: Any,
    device: torch.device,
    prompt_text: str,
    output_text: str,
    hidden_indices: list[int],
    pooling: str,
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
    completion_last_idx = full_len - 1
    if prompt_last_idx < 0 or prompt_last_idx >= hidden_states[-1].shape[1]:
        return None
    if completion_last_idx < 0 or completion_last_idx >= hidden_states[-1].shape[1]:
        return None

    embeddings: dict[int, np.ndarray] = {}
    for hidden_idx in hidden_indices:
        if hidden_idx < 0 or hidden_idx >= len(hidden_states):
            return None
        hidden = hidden_states[hidden_idx][0]
        if pooling == "prompt_last":
            vec = hidden[prompt_last_idx]
        elif pooling == "prompt_mean":
            vec = hidden[:prompt_len].mean(dim=0)
        elif pooling == "completion_last":
            vec = hidden[completion_last_idx]
        else:
            raise ValueError(f"Unknown pooling mode: {pooling}")
        vec = vec.detach().float().cpu().numpy()
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


def _build_language_onehot_features(
    train_languages: list[str],
    eval_languages: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    labels = sorted(set(train_languages))
    if not labels:
        return (
            np.zeros((len(train_languages), 0), dtype=np.float32),
            np.zeros((len(eval_languages), 0), dtype=np.float32),
            [],
            len(eval_languages),
        )
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    train_feat = np.zeros((len(train_languages), len(labels)), dtype=np.float32)
    eval_feat = np.zeros((len(eval_languages), len(labels)), dtype=np.float32)
    for row_idx, language in enumerate(train_languages):
        train_feat[row_idx, label_to_idx[language]] = 1.0

    unseen_eval = 0
    for row_idx, language in enumerate(eval_languages):
        idx = label_to_idx.get(language)
        if idx is None:
            unseen_eval += 1
            continue
        eval_feat[row_idx, idx] = 1.0
    return train_feat, eval_feat, labels, unseen_eval


def _fit_ridge_multi_target(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _predict_ridge(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> np.ndarray:
    x_norm = (x - x_mean[None, :]) / x_std[None, :]
    return x_norm @ weights + bias[None, :]


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def _find_policy_vs_gpt_pair(pair_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in pair_rows:
        left = str(row["left_label"]).lower()
        right = str(row["right_label"]).lower()
        labels = [left, right]
        if ("policy" in labels[0] or "policy" in labels[1]) and any("gpt" in label for label in labels):
            return row
    return None


def _pair_label(row: dict[str, Any]) -> str:
    return f"{row.get('left_label')} vs {row.get('right_label')}"


def _compute_pairwise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    route_labels: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    target_dim = int(y_true.shape[1])
    pair_rows: list[dict[str, Any]] = []
    for i in range(target_dim):
        for j in range(i + 1, target_dim):
            delta_true = y_true[:, i] - y_true[:, j]
            delta_pred = y_pred[:, i] - y_pred[:, j]
            labels = (delta_true > 0).astype(np.int64)

            sign_true = np.sign(delta_true)
            sign_pred = np.sign(delta_pred)
            ranking_acc = float(np.mean(sign_true == sign_pred))
            pearson = _pearson_corr(delta_pred, delta_true)
            spearman = _spearman_corr(delta_pred, delta_true)
            mae_delta = float(np.mean(np.abs(delta_pred - delta_true)))
            roc_auc = _roc_auc_binary(labels, delta_pred)

            pair_rows.append(
                {
                    "left_idx": i,
                    "right_idx": j,
                    "left_label": route_labels[i],
                    "right_label": route_labels[j],
                    "n_eval": int(delta_true.shape[0]),
                    "pearson_delta": pearson,
                    "spearman_delta": spearman,
                    "delta_mae": mae_delta,
                    "ranking_accuracy_sign": ranking_acc,
                    "roc_auc": roc_auc,
                }
            )
    return pair_rows, _find_policy_vs_gpt_pair(pair_rows)


def _compute_per_route_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    route_labels: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_dim = int(y_true.shape[1])
    for idx in range(target_dim):
        true_vals = y_true[:, idx]
        pred_vals = y_pred[:, idx]
        mse = float(np.mean((pred_vals - true_vals) ** 2))
        mae = float(np.mean(np.abs(pred_vals - true_vals)))
        rows.append(
            {
                "route_idx": idx,
                "route_label": route_labels[idx],
                "n_eval": int(true_vals.shape[0]),
                "mean_true": float(np.mean(true_vals)),
                "mean_pred": float(np.mean(pred_vals)),
                "std_true": float(np.std(true_vals)),
                "std_pred": float(np.std(pred_vals)),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "mae": mae,
                "pearson": _pearson_corr(pred_vals, true_vals),
                "spearman": _spearman_corr(pred_vals, true_vals),
                "r2": _r2_score(true_vals, pred_vals),
            }
        )
    return rows


def _plot_pairwise_metrics_by_layer(per_layer_pair_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not MATPLOTLIB_AVAILABLE or not per_layer_pair_rows:
        return

    layers = sorted({int(row["hidden_index"]) for row in per_layer_pair_rows})
    layer_pos = {idx: pos for pos, idx in enumerate(layers)}
    layer_labels = [_hidden_index_to_label(idx) for idx in layers]
    pair_keys = sorted({_pair_label(row) for row in per_layer_pair_rows})

    fig, axes = plt.subplots(1, 3, figsize=(max(15, 0.55 * len(layer_labels)), 5.6), sharex=True)
    metric_specs = [
        ("pearson_delta", "Pearson delta", "Pearson"),
        ("spearman_delta", "Spearman delta", "Spearman"),
        ("roc_auc", "ROC-AUC", "ROC-AUC"),
    ]
    for ax, (metric_key, metric_name, title) in zip(axes, metric_specs):
        for pair_key in pair_keys:
            series = [float("nan")] * len(layers)
            for row in per_layer_pair_rows:
                if _pair_label(row) != pair_key:
                    continue
                metric_val = row.get(metric_key)
                if metric_val is None:
                    continue
                series[layer_pos[int(row["hidden_index"])]] = float(metric_val)
            ax.plot(range(len(layers)), series, marker="o", linewidth=1.6, label=pair_key)

        if metric_key == "roc_auc":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.set_ylim(0.0, 1.0)
        else:
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.set_ylim(-1.0, 1.0)
        ax.set_title(title)
        ax.set_ylabel(metric_name)
        ax.grid(alpha=0.2)
        ax.set_xticks(range(len(layer_labels)))
        ax.set_xticklabels(layer_labels, rotation=30, ha="right")

    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Routing probe pairwise metrics by hidden layer")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_policy_vs_gpt_by_layer(per_layer_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not MATPLOTLIB_AVAILABLE or not per_layer_rows:
        return
    valid = [row for row in per_layer_rows if row.get("policy_vs_gpt_pearson_delta") is not None]
    if not valid:
        return

    x = list(range(len(valid)))
    labels = [str(row["layer_label"]) for row in valid]
    pearson = [_safe_float(row.get("policy_vs_gpt_pearson_delta"), default=float("nan")) for row in valid]
    spearman = [_safe_float(row.get("policy_vs_gpt_spearman_delta"), default=float("nan")) for row in valid]
    roc_auc = [_safe_float(row.get("policy_vs_gpt_roc_auc"), default=float("nan")) for row in valid]

    plt.figure(figsize=(max(12, 0.35 * len(labels)), 5.5))
    plt.plot(x, pearson, marker="o", linewidth=1.8, label="Policy-vs-GPT Pearson")
    plt.plot(x, spearman, marker="o", linewidth=1.8, label="Policy-vs-GPT Spearman")
    plt.plot(x, roc_auc, marker="o", linewidth=1.8, label="Policy-vs-GPT ROC-AUC")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.8, label="0.5 reference")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylim(-1.0, 1.0)
    plt.ylabel("Metric value")
    plt.title("Policy-vs-GPT routing probe metrics by hidden layer")
    plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/eval ridge regression probe on frozen trunk prompt-last embeddings.")
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train-split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--eval-split", default="test", choices=["train", "test", "all"])
    parser.add_argument(
        "--split-mode",
        default="existing",
        choices=["existing", "within-repo"],
        help="existing: use trace split field; within-repo: create train/eval split inside each repo.",
    )
    parser.add_argument(
        "--within-repo-source-split",
        default="train",
        choices=["train", "test", "all"],
        help="Pool of traces used before within-repo split.",
    )
    parser.add_argument("--within-repo-train-fraction", type=float, default=0.8)
    parser.add_argument("--within-repo-seed", type=int, default=0)
    parser.add_argument("--all-model-versions", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-eval-records", type=int, default=None)
    parser.add_argument("--save-predictions-jsonl", action="store_true")
    parser.add_argument(
        "--target-train-traces",
        type=int,
        default=1000,
        help="When using version-coupled selection, walk backward from eval version until at least this many train traces are collected.",
    )
    parser.add_argument(
        "--max-train-versions-back",
        type=int,
        default=None,
        help="Optional cap on how many model versions back the train window may extend.",
    )
    parser.add_argument(
        "--append-language-onehot",
        action="store_true",
        help="Append language one-hot features to trunk embeddings before ridge fitting.",
    )
    parser.add_argument(
        "--pooling",
        default="prompt_last",
        choices=["prompt_last", "prompt_mean", "completion_last"],
        help="Embedding pooling over prompt tokens before ridge probe.",
    )
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
        if not source:
            raise ValueError(f"No traces available for within-repo source split={args.within_repo_source_split}")
        train_traces, eval_traces = _split_within_repo(
            source,
            train_fraction=float(args.within_repo_train_fraction),
            seed=int(args.within_repo_seed),
        )
        traces_for_metadata = traces
    else:
        train_split = None if args.train_split == "all" else args.train_split
        eval_split = None if args.eval_split == "all" else args.eval_split
        if args.all_model_versions:
            train_traces = load_router_traces(
                input_globs=args.input_glob,
                split=train_split,
                latest_model_only=False,
                dedupe_by_problem=not args.keep_duplicates,
            )
            eval_traces = load_router_traces(
                input_globs=args.input_glob,
                split=eval_split,
                latest_model_only=False,
                dedupe_by_problem=not args.keep_duplicates,
            )
            traces_for_metadata = train_traces + eval_traces
        else:
            all_traces = load_router_traces(
                input_globs=args.input_glob,
                split=None,
                latest_model_only=False,
                dedupe_by_problem=False,
            )
            chosen_eval_version = _choose_eval_model_version(
                traces=all_traces,
                eval_split=eval_split,
            )
            if chosen_eval_version is None:
                raise ValueError("Could not determine an eval model_version for probe data selection")
            eval_traces = [
                trace
                for trace in all_traces
                if _as_model_version(trace.get("model_version")) == chosen_eval_version
                and (eval_split is None or trace.get("split") == eval_split)
            ]
            train_traces, train_versions_used = _select_train_window_up_to_eval(
                traces=all_traces,
                train_split=train_split,
                eval_version=chosen_eval_version,
                target_train_traces=int(args.target_train_traces),
                max_versions_back=args.max_train_versions_back,
            )
            if not args.keep_duplicates:
                train_traces = dedupe_latest_by_problem(train_traces)
                eval_traces = dedupe_latest_by_problem(eval_traces)
            traces_for_metadata = train_traces + eval_traces
            logger.info(
                "Using eval model_version=%s and train window versions=%s for probe selection: train=%d eval=%d",
                chosen_eval_version,
                train_versions_used,
                len(train_traces),
                len(eval_traces),
            )
    if args.max_train_records is not None:
        train_traces = train_traces[: args.max_train_records]
    if args.max_eval_records is not None:
        eval_traces = eval_traces[: args.max_eval_records]
    if not train_traces or not eval_traces:
        raise ValueError(f"Empty split after filtering: train={len(train_traces)} eval={len(eval_traces)}")

    n_experts = max((len(trace.get("experts") or []) for trace in traces_for_metadata), default=0)
    route_labels = extract_route_labels(traces_for_metadata, n_experts=n_experts)
    target_dim = len(route_labels)
    logger.info(
        "Loaded traces total=%d train=%d eval=%d target_dim=%d",
        len(traces_for_metadata),
        len(train_traces),
        len(eval_traces),
        target_dim,
    )
    logger.info("Routes: %s", route_labels)

    device = _pick_device(args.device)
    logger.info("Loading frozen trunk from %s on %s", args.model_path, device)
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
        "Probing hidden indices=%s (labels=%s), pooling=%s",
        hidden_indices,
        [_hidden_index_to_label(idx) for idx in hidden_indices],
        args.pooling,
    )

    def build_xy(
        traces_in: list[dict[str, Any]],
        tag: str,
    ) -> tuple[dict[int, np.ndarray], np.ndarray, list[dict[str, Any]], list[str]]:
        xs_by_layer: dict[int, list[np.ndarray]] = {idx: [] for idx in hidden_indices}
        ys: list[np.ndarray] = []
        kept: list[dict[str, Any]] = []
        languages: list[str] = []
        skipped = 0
        for trace in tqdm(traces_in, desc=f"Embedding {tag}", unit="trace"):
            policy = trace.get("policy") or {}
            prompt_text = policy.get("prompt_text")
            output_text = policy.get("output_text") or ""
            if not isinstance(prompt_text, str):
                skipped += 1
                continue
            embeddings = _encode_embeddings(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt_text=prompt_text,
                output_text=output_text,
                hidden_indices=hidden_indices,
                pooling=args.pooling,
            )
            if embeddings is None:
                skipped += 1
                continue
            rewards = extract_reward_vector(trace)
            if len(rewards) < target_dim:
                skipped += 1
                continue
            for hidden_idx in hidden_indices:
                xs_by_layer[hidden_idx].append(embeddings[hidden_idx].astype(np.float32))
            ys.append(np.asarray(rewards[:target_dim], dtype=np.float32))
            kept.append(trace)
            languages.append(_infer_language(trace))
        if not ys:
            raise ValueError(f"No valid examples for {tag}")
        logger.info("%s examples kept=%d skipped=%d", tag, len(ys), skipped)
        return {idx: np.stack(xs_by_layer[idx]) for idx in hidden_indices}, np.stack(ys), kept, languages

    x_train_by_layer, y_train, _kept_train, train_languages = build_xy(train_traces, "train")
    x_eval_by_layer, y_eval, kept_eval, eval_languages = build_xy(eval_traces, "eval")
    logger.info(
        "Embedding dims by layer: %s",
        {idx: x_train_by_layer[idx].shape for idx in hidden_indices},
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    language_onehot_train = np.zeros((y_train.shape[0], 0), dtype=np.float32)
    language_onehot_eval = np.zeros((y_eval.shape[0], 0), dtype=np.float32)
    language_feature_labels: list[str] = []
    language_eval_unseen = 0
    if args.append_language_onehot:
        (
            language_onehot_train,
            language_onehot_eval,
            language_feature_labels,
            language_eval_unseen,
        ) = _build_language_onehot_features(train_languages, eval_languages)
        logger.info(
            "Language one-hot enabled: dim=%d unseen_eval=%d labels=%s",
            language_onehot_train.shape[1],
            language_eval_unseen,
            language_feature_labels,
        )
        train_counts = Counter(train_languages)
        eval_counts = Counter(eval_languages)
        _write_csv(
            output_dir / "language_feature_categories.csv",
            [
                {
                    "language": label,
                    "feature_index": idx,
                    "train_count": int(train_counts.get(label, 0)),
                    "eval_count": int(eval_counts.get(label, 0)),
                }
                for idx, label in enumerate(language_feature_labels)
            ],
            ["language", "feature_index", "train_count", "eval_count"],
        )

    pairwise_headers = [
        "left_idx",
        "right_idx",
        "left_label",
        "right_label",
        "n_eval",
        "pearson_delta",
        "spearman_delta",
        "delta_mae",
        "ranking_accuracy_sign",
        "roc_auc",
    ]
    per_route_headers = [
        "route_idx",
        "route_label",
        "n_eval",
        "mean_true",
        "mean_pred",
        "std_true",
        "std_pred",
        "mse",
        "rmse",
        "mae",
        "pearson",
        "spearman",
        "r2",
    ]
    multi_layer = len(hidden_indices) > 1
    per_layer_rows: list[dict[str, Any]] = []
    per_layer_pair_rows: list[dict[str, Any]] = []
    per_layer_route_rows: list[dict[str, Any]] = []

    for hidden_idx in hidden_indices:
        layer_label = _hidden_index_to_label(hidden_idx)
        layer_output_dir = output_dir / "layers" / layer_label if multi_layer else output_dir
        layer_output_dir.mkdir(parents=True, exist_ok=True)

        base_x_train = x_train_by_layer[hidden_idx]
        base_x_eval = x_eval_by_layer[hidden_idx]
        x_train = base_x_train
        x_eval = base_x_eval
        if args.append_language_onehot and language_onehot_train.shape[1] > 0:
            x_train = np.concatenate([x_train, language_onehot_train], axis=1)
            x_eval = np.concatenate([x_eval, language_onehot_eval], axis=1)
        weights, bias, x_mean, x_std = _fit_ridge_multi_target(x_train, y_train, alpha=float(args.alpha))
        y_hat_eval = _predict_ridge(x_eval, weights, bias, x_mean, x_std)
        pair_rows, policy_vs_gpt = _compute_pairwise_metrics(y_eval, y_hat_eval, route_labels)
        route_rows = _compute_per_route_metrics(y_eval, y_hat_eval, route_labels)
        for pair_row in pair_rows:
            enriched = dict(pair_row)
            enriched["hidden_index"] = hidden_idx
            enriched["layer_label"] = layer_label
            enriched["transformer_layer_index"] = (hidden_idx - 1) if hidden_idx > 0 else -1
            per_layer_pair_rows.append(enriched)
        for route_row in route_rows:
            enriched = dict(route_row)
            enriched["hidden_index"] = hidden_idx
            enriched["layer_label"] = layer_label
            enriched["transformer_layer_index"] = (hidden_idx - 1) if hidden_idx > 0 else -1
            per_layer_route_rows.append(enriched)

        _write_csv(layer_output_dir / "pairwise_metrics.csv", pair_rows, pairwise_headers)
        _write_csv(layer_output_dir / "route_metrics.csv", route_rows, per_route_headers)
        np.savez(
            layer_output_dir / "ridge_probe_weights.npz",
            weights=weights,
            bias=bias,
            x_mean=x_mean,
            x_std=x_std,
            route_labels=np.asarray(route_labels),
            hidden_index=np.asarray([hidden_idx]),
            layer_label=np.asarray([layer_label]),
        )

        if args.save_predictions_jsonl:
            pred_path = layer_output_dir / "eval_predictions.jsonl"
            with pred_path.open("w") as sink:
                for idx, trace in enumerate(kept_eval):
                    row = {
                        "problem_id": trace.get("problem_id"),
                        "split": trace.get("split"),
                        "model_version": trace.get("model_version"),
                        "routes": route_labels,
                        "true_rewards": y_eval[idx].tolist(),
                        "pred_rewards": y_hat_eval[idx].tolist(),
                    }
                    sink.write(json.dumps(row) + "\n")
            logger.info("Wrote eval predictions to %s", pred_path)

        selection_score = None
        if policy_vs_gpt and policy_vs_gpt.get("pearson_delta") is not None:
            selection_score = float(policy_vs_gpt["pearson_delta"])
        elif pair_rows:
            valid_pearsons = [float(row["pearson_delta"]) for row in pair_rows if row.get("pearson_delta") is not None]
            if valid_pearsons:
                selection_score = float(sum(valid_pearsons) / len(valid_pearsons))

        row = {
            "hidden_index": hidden_idx,
            "layer_label": layer_label,
            "transformer_layer_index": (hidden_idx - 1) if hidden_idx > 0 else -1,
            "n_train": int(x_train.shape[0]),
            "n_eval": int(x_eval.shape[0]),
            "embedding_dim": int(base_x_train.shape[1]),
            "language_onehot_dim": int(language_onehot_train.shape[1]),
            "feature_dim_total": int(x_train.shape[1]),
            "n_pairs": len(pair_rows),
            "policy_vs_gpt_pearson_delta": policy_vs_gpt.get("pearson_delta") if policy_vs_gpt else None,
            "policy_vs_gpt_spearman_delta": policy_vs_gpt.get("spearman_delta") if policy_vs_gpt else None,
            "policy_vs_gpt_roc_auc": policy_vs_gpt.get("roc_auc") if policy_vs_gpt else None,
            "selection_score": selection_score,
        }
        per_layer_rows.append(row)

        layer_summary = {
            "model_path": args.model_path,
            "alpha": float(args.alpha),
            "split_mode": args.split_mode,
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "within_repo_source_split": args.within_repo_source_split,
            "within_repo_train_fraction": float(args.within_repo_train_fraction),
            "within_repo_seed": int(args.within_repo_seed),
            "probe_layers": args.probe_layers,
            "include_embedding_layer": bool(args.include_embedding_layer),
            "pooling": args.pooling,
            "append_language_onehot": bool(args.append_language_onehot),
            "language_feature_labels": language_feature_labels,
            "language_eval_unseen_count": int(language_eval_unseen),
            "hidden_index": hidden_idx,
            "layer_label": layer_label,
            "n_train": int(x_train.shape[0]),
            "n_eval": int(x_eval.shape[0]),
            "embedding_dim": int(base_x_train.shape[1]),
            "language_onehot_dim": int(language_onehot_train.shape[1]),
            "feature_dim_total": int(x_train.shape[1]),
            "target_dim": target_dim,
            "routes": route_labels,
            "policy_vs_gpt_pair_metrics": policy_vs_gpt,
            "per_route_metrics": route_rows,
            "notes": (
                "Ridge probe trained on frozen trunk embeddings with "
                f"pooling={args.pooling}, append_language_onehot={bool(args.append_language_onehot)}."
            ),
        }
        with (layer_output_dir / "summary.json").open("w") as handle:
            json.dump(layer_summary, handle, indent=2)

        logger.info(
            "Layer %s (hidden_idx=%d): pairwise_metrics=%d policy_vs_gpt_pearson=%s",
            layer_label,
            hidden_idx,
            len(pair_rows),
            f"{row['policy_vs_gpt_pearson_delta']:.4f}" if row["policy_vs_gpt_pearson_delta"] is not None else "n/a",
        )

    per_layer_rows = sorted(per_layer_rows, key=lambda item: int(item["hidden_index"]))
    per_layer_pair_rows = sorted(
        per_layer_pair_rows,
        key=lambda item: (int(item["hidden_index"]), int(item["left_idx"]), int(item["right_idx"])),
    )
    if multi_layer:
        _write_csv(
            output_dir / "per_layer_metrics.csv",
            per_layer_rows,
            [
                "hidden_index",
                "layer_label",
                "transformer_layer_index",
                "n_train",
                "n_eval",
                "embedding_dim",
                "language_onehot_dim",
                "feature_dim_total",
                "n_pairs",
                "policy_vs_gpt_pearson_delta",
                "policy_vs_gpt_spearman_delta",
                "policy_vs_gpt_roc_auc",
                "selection_score",
            ],
        )
        _write_csv(
            output_dir / "per_layer_pairwise_metrics.csv",
            per_layer_pair_rows,
            [
                "hidden_index",
                "layer_label",
                "transformer_layer_index",
                "left_idx",
                "right_idx",
                "left_label",
                "right_label",
                "n_eval",
                "pearson_delta",
                "spearman_delta",
                "delta_mae",
                "ranking_accuracy_sign",
                "roc_auc",
            ],
        )
        _write_csv(
            output_dir / "per_layer_route_metrics.csv",
            per_layer_route_rows,
            [
                "hidden_index",
                "layer_label",
                "transformer_layer_index",
                "route_idx",
                "route_label",
                "n_eval",
                "mean_true",
                "mean_pred",
                "std_true",
                "std_pred",
                "mse",
                "rmse",
                "mae",
                "pearson",
                "spearman",
                "r2",
            ],
        )
        _plot_pairwise_metrics_by_layer(per_layer_pair_rows, output_dir / "metrics_by_layer.png")
        _plot_policy_vs_gpt_by_layer(per_layer_rows, output_dir / "policy_vs_gpt_by_layer.png")
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib is not available; skipping layer-metric plots.")

    best_layer = max(
        per_layer_rows,
        key=lambda row: (_safe_float(row.get("selection_score"), default=-1e9), -int(row["hidden_index"])),
    )
    summary = {
        "model_path": args.model_path,
        "alpha": float(args.alpha),
        "split_mode": args.split_mode,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "within_repo_source_split": args.within_repo_source_split,
        "within_repo_train_fraction": float(args.within_repo_train_fraction),
        "within_repo_seed": int(args.within_repo_seed),
        "probe_layers": args.probe_layers,
        "include_embedding_layer": bool(args.include_embedding_layer),
        "pooling": args.pooling,
        "append_language_onehot": bool(args.append_language_onehot),
        "language_onehot_dim": int(language_onehot_train.shape[1]),
        "language_feature_labels": language_feature_labels,
        "language_eval_unseen_count": int(language_eval_unseen),
        "hidden_indices": hidden_indices,
        "n_train": int(y_train.shape[0]),
        "n_eval": int(y_eval.shape[0]),
        "target_dim": target_dim,
        "routes": route_labels,
        "best_layer": best_layer,
        "all_layers": per_layer_rows,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    if len(hidden_indices) == 1:
        logger.info("Done. pairwise metrics: %s", output_dir / "pairwise_metrics.csv")
    else:
        logger.info("Done. per-layer metrics: %s", output_dir / "per_layer_metrics.csv")


if __name__ == "__main__":
    main()

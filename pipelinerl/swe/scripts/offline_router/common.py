import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


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


@dataclass(frozen=True)
class RouteSpec:
    label: str
    model_name: str
    base_url: str
    parameters: dict[str, Any]
    api_key: str | None = None


def problem_id_from_item(item: dict[str, Any]) -> str:
    for key in ("problem_id", "issue_id", "instance_id", "id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    repo = str(item.get("repo") or "").strip()
    base_commit = str(item.get("base_commit") or "").strip()
    if repo and base_commit:
        return f"{repo}@{base_commit}"
    raise ValueError("Missing problem identifier (problem_id/issue_id/instance_id/id)")


def route_label_for_expert(expert_rank: int, model_name: str) -> str:
    return f"expert_{expert_rank}:{model_name}"


def infer_language_from_problem(problem: dict[str, Any]) -> str:
    for key in ("language", "lang", "repo_language"):
        value = problem.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    dataset = str(problem.get("dataset") or "").lower()
    if "smith_py" in dataset or dataset.endswith("_py"):
        return "Python"
    if "smith_go" in dataset or dataset.endswith("_go"):
        return "Go"
    if "smith_rs" in dataset or dataset.endswith("_rs"):
        return "Rust"
    if "smith_java" in dataset or dataset.endswith("_java"):
        return "Java"

    file_contents = problem.get("file_contents") or {}
    ext_counts: dict[str, int] = {}
    if isinstance(file_contents, dict):
        for path in file_contents:
            suffix = Path(str(path)).suffix.lower()
            label = EXT_LANGUAGE_MAP.get(suffix)
            if label:
                ext_counts[label] = ext_counts.get(label, 0) + 1
    if ext_counts:
        return max(ext_counts.items(), key=lambda item: item[1])[0]
    return "Unknown"


def render_prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return "".join(f"{msg['role']}: {msg['content']}\n" for msg in messages)


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, subvalue in value.items():
            if "api_key" in str(key).lower():
                sanitized[key] = "***"
            else:
                sanitized[key] = sanitize_for_json(subvalue)
        return sanitized
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def tokenize_prompt_completion(
    tokenizer: Any,
    prompt_text: str,
    output_text: str,
    max_seq_length: int | None,
) -> dict[str, Any] | None:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt_text + (output_text or ""), add_special_tokens=False).input_ids
    if not prompt_ids or not full_ids or len(full_ids) <= len(prompt_ids):
        return None

    prompt_len = len(prompt_ids)
    if max_seq_length is not None and max_seq_length > 0 and len(full_ids) > max_seq_length:
        drop = len(full_ids) - max_seq_length
        full_ids = full_ids[drop:]
        prompt_len = max(0, prompt_len - drop)

    if not full_ids or len(full_ids) <= prompt_len:
        return None

    return {
        "input_ids": full_ids,
        "completion_last_index": len(full_ids) - 1,
        "prompt_len": prompt_len,
    }


def configure_router_training_mode(model: torch.nn.Module, mode: str) -> list[str]:
    if mode not in {"frozen_backbone", "full_backbone"}:
        raise ValueError(f"Unsupported training mode: {mode}")

    for param in model.parameters():
        param.requires_grad = False

    for param in model.value_head.parameters():
        param.requires_grad = False

    trainable_prefixes: list[str] = []
    for param in model.performance_value_head.parameters():
        param.requires_grad = True
    trainable_prefixes.append("performance_value_head")

    if mode == "full_backbone":
        for param in model.pretrained_model.parameters():
            param.requires_grad = True
        trainable_prefixes.append("pretrained_model")

    return trainable_prefixes


def count_parameters(model: torch.nn.Module, trainable_only: bool) -> int:
    total = 0
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        # Under DeepSpeed ZeRO-3, local shards can report a tiny placeholder
        # numel(); ds_numel preserves the original full parameter size.
        total += int(getattr(param, "ds_numel", param.numel()))
    return total


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


def compute_per_route_metrics(
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


def compute_pairwise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    route_labels: list[str],
) -> list[dict[str, Any]]:
    target_dim = int(y_true.shape[1])
    rows: list[dict[str, Any]] = []
    for left_idx in range(target_dim):
        for right_idx in range(left_idx + 1, target_dim):
            delta_true = y_true[:, left_idx] - y_true[:, right_idx]
            delta_pred = y_pred[:, left_idx] - y_pred[:, right_idx]
            sign_true = np.sign(delta_true)
            sign_pred = np.sign(delta_pred)
            labels = (delta_true > 0).astype(np.int64)
            rows.append(
                {
                    "left_idx": left_idx,
                    "right_idx": right_idx,
                    "left_label": route_labels[left_idx],
                    "right_label": route_labels[right_idx],
                    "n_eval": int(delta_true.shape[0]),
                    "pearson_delta": _pearson_corr(delta_pred, delta_true),
                    "spearman_delta": _spearman_corr(delta_pred, delta_true),
                    "delta_mae": float(np.mean(np.abs(delta_pred - delta_true))),
                    "ranking_accuracy_sign": float(np.mean(sign_true == sign_pred)),
                    "roc_auc": _roc_auc_binary(labels, delta_pred),
                }
            )
    return rows


def csv_headers_for_route_metrics() -> list[str]:
    return [
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


def csv_headers_for_pairwise_metrics() -> list[str]:
    return [
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

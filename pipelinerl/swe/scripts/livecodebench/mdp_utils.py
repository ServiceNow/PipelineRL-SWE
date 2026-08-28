"""Shared invariants for the LiveCodeBench sequential-allocation experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SPLIT_SCHEMA_VERSION = 2
TENSOR_SCHEMA_VERSION = 3


def redact_sensitive_text(value: object) -> str:
    """Return an exception/message string with credentials and auth headers removed."""
    text = str(value)
    patterns = (
        (r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?)(bearer\s+)?[^\s,;>'\"]+", r"\1<REDACTED>"),
        (r"(?i)(api[-_ ]?key[\"']?\s*[:=]\s*[\"']?)[^\s,;>'\"]+", r"\1<REDACTED>"),
        (r"\bsk-or-v1-[A-Za-z0-9_-]+\b", "<REDACTED_OPENROUTER_KEY>"),
        (r"\bsk-[A-Za-z0-9_-]{16,}\b", "<REDACTED_API_KEY>"),
    )
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text


def build_split_manifest(
    problem_ids: Iterable[str],
    *,
    seed: int = 0,
    train_fraction: float = 0.5,
    calibration_fraction: float = 0.25,
) -> dict[str, Any]:
    """Create exact deterministic train/calibration/test problem splits."""
    ids = sorted({str(pid) for pid in problem_ids})
    if len(ids) < 3:
        raise ValueError("At least three problems are required for a three-way split")
    if train_fraction <= 0.0 or calibration_fraction <= 0.0:
        raise ValueError("train and calibration fractions must be positive")
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError("train_fraction + calibration_fraction must be below one")
    rng = np.random.default_rng(int(seed))
    shuffled = [ids[int(i)] for i in rng.permutation(len(ids))]
    n_train = min(max(int(round(len(ids) * train_fraction)), 1), len(ids) - 2)
    n_cal = min(max(int(round(len(ids) * calibration_fraction)), 1), len(ids) - n_train - 1)
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "method": "sorted_ids_numpy_permutation",
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "calibration_fraction": float(calibration_fraction),
        "train_problem_ids": sorted(shuffled[:n_train]),
        "calibration_problem_ids": sorted(shuffled[n_train:n_train + n_cal]),
        "test_problem_ids": sorted(shuffled[n_train + n_cal:]),
    }

def validate_split_manifest(manifest: dict[str, Any], problem_ids: Iterable[str]) -> None:
    expected = {str(pid) for pid in problem_ids}
    groups = {
        name: {str(pid) for pid in manifest.get(f"{name}_problem_ids", [])}
        for name in ("train", "calibration", "test")
    }
    if any(not values for values in groups.values()):
        raise ValueError("Split manifest must have non-empty train, calibration, and test sets")
    if any(groups[left] & groups[right] for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test"))):
        raise ValueError("Split manifest has overlapping problem IDs")
    observed = set().union(*groups.values())
    if observed != expected:
        missing = sorted(expected - observed)[:5]
        extra = sorted(observed - expected)[:5]
        raise ValueError(f"Split manifest does not match tensors: missing={missing}, extra={extra}")

def write_split_manifest(
    path: Path,
    problem_ids: Iterable[str],
    *,
    seed: int = 0,
    train_fraction: float = 0.5,
    calibration_fraction: float = 0.25,
) -> dict[str, Any]:
    manifest = build_split_manifest(
        problem_ids,
        seed=seed,
        train_fraction=train_fraction,
        calibration_fraction=calibration_fraction,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_source_temporal_split_manifest(
    problem_metadata: Iterable[dict[str, Any]],
    *,
    calibration_fraction_of_later: float = 0.5,
) -> dict[str, Any]:
    """Build a source-temporal train/calibration/test split without random re-splitting."""
    if not 0.0 < calibration_fraction_of_later < 1.0:
        raise ValueError("calibration_fraction_of_later must be in (0, 1)")
    rows = [dict(row) for row in problem_metadata]
    by_id = {str(row["problem_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Problem metadata has duplicate IDs")
    train_ids = sorted(pid for pid, row in by_id.items() if row.get("source_temporal_split") == "train")
    later_rows = [row for row in by_id.values() if row.get("source_temporal_split") == "eval"]
    unknown = [str(row["problem_id"]) for row in by_id.values() if row.get("source_temporal_split") not in {"train", "eval"}]
    if unknown:
        raise ValueError(f"Unknown source temporal split for examples: {unknown[:5]}")
    if not train_ids or len(later_rows) < 2:
        raise ValueError("Need non-empty earlier train and at least two later problems")
    if any(not str(row.get("contest_date") or "") for row in rows):
        raise ValueError("Source-temporal splitting requires contest_date for every problem")
    earliest_later = min(str(row["contest_date"]) for row in later_rows)
    latest_train = max(str(by_id[pid]["contest_date"]) for pid in train_ids)
    if latest_train >= earliest_later:
        raise ValueError(f"Source temporal ranges overlap: latest train={latest_train}, earliest later={earliest_later}")
    date_groups: dict[str, list[str]] = {}
    for row in later_rows:
        date_groups.setdefault(str(row["contest_date"]), []).append(str(row["problem_id"]))
    ordered_groups = [(date, sorted(ids)) for date, ids in sorted(date_groups.items())]
    if len(ordered_groups) < 2:
        raise ValueError("Need at least two distinct later contest dates")
    target = len(later_rows) * calibration_fraction_of_later
    cumulative = 0
    candidates: list[tuple[float, int]] = []
    for index, (_, ids) in enumerate(ordered_groups[:-1]):
        cumulative += len(ids)
        candidates.append((abs(cumulative - target), index))
    _, boundary = min(candidates)
    calibration_ids = sorted(pid for _, ids in ordered_groups[: boundary + 1] for pid in ids)
    test_ids = sorted(pid for _, ids in ordered_groups[boundary + 1 :] for pid in ids)
    manifest = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "method": "source_temporal_contiguous_later_partition",
        "train_source_split": "train",
        "later_source_split": "eval",
        "calibration_fraction_of_later": float(calibration_fraction_of_later),
        "train_end_date": latest_train,
        "calibration_end_date": ordered_groups[boundary][0],
        "test_start_date": ordered_groups[boundary + 1][0],
        "train_problem_ids": train_ids,
        "calibration_problem_ids": calibration_ids,
        "test_problem_ids": test_ids,
    }
    validate_split_manifest(manifest, by_id)
    return manifest


def write_source_temporal_split_manifest(
    path: Path,
    problem_metadata: Iterable[dict[str, Any]],
    *,
    calibration_fraction_of_later: float = 0.5,
) -> dict[str, Any]:
    manifest = build_source_temporal_split_manifest(
        problem_metadata, calibration_fraction_of_later=calibration_fraction_of_later
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

def load_split_manifest(path: Path, problem_ids: Iterable[str]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing canonical split manifest: {path}. Rebuild the tensor bundle with schema v3."
        )
    manifest = json.loads(path.read_text())
    validate_split_manifest(manifest, problem_ids)
    return manifest


def split_indices(
    manifest: dict[str, Any], problem_ids: Iterable[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = [str(pid) for pid in problem_ids]
    groups = [set(manifest[f"{name}_problem_ids"]) for name in ("train", "calibration", "test")]
    return tuple(
        np.array([i for i, pid in enumerate(ids) if pid in group], dtype=int)
        for group in groups
    )

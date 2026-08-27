"""Shared invariants for the LiveCodeBench sequential-allocation experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SPLIT_SCHEMA_VERSION = 1
TENSOR_SCHEMA_VERSION = 2


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

def load_split_manifest(path: Path, problem_ids: Iterable[str]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing canonical split manifest: {path}. Rebuild the tensor bundle with schema v2."
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

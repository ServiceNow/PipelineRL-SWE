"""Utility helpers for SWE rollout logging/metadata."""
from __future__ import annotations

from typing import Any, Dict

from pipelinerl.rollouts import TrainingText


def get_problem_id(problem: Dict[str, Any]) -> str:
    """Best-effort extraction of a stable problem identifier."""
    for key in ("id", "instance_id", "issue_id", "problem_id"):
        value = problem.get(key)
        if value:
            return str(value)
    return ""


def annotate_training_text(
    training_text: TrainingText | None,
    *,
    stage: str,
    problem: Dict[str, Any],
    llm: Any | None = None,
    source: str = "actor",
    extra: Dict[str, Any] | None = None,
) -> None:
    """Attach consistent metadata to a TrainingText instance."""
    if training_text is None:
        return

    metadata = dict(training_text.metadata or {})
    metadata.setdefault("problem_id", get_problem_id(problem))
    metadata.setdefault("dataset", problem.get("dataset"))
    metadata.setdefault("repo", problem.get("repo"))
    metadata["stage"] = stage
    metadata["source"] = source
    if llm is not None:
        metadata.setdefault("model_name", getattr(llm, "model_name", None))
    if extra:
        metadata.update(extra)

    training_text.metadata = metadata


__all__ = ["annotate_training_text", "get_problem_id"]

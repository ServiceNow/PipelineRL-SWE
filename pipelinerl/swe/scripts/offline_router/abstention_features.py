"""Canonical feature serialization for abstention predictor training and scoring."""

from __future__ import annotations

from typing import Any


TEST_FEEDBACK_FORMATS = ("full", "names_only", "count_only")


def build_test_feedback_text(traj: dict[str, Any], fmt: str) -> str:
    """Render stored scout test results in one of the supported ablation formats."""
    if fmt not in TEST_FEEDBACK_FORMATS:
        raise ValueError(
            f"Unknown test-feedback format {fmt!r}; choose from {TEST_FEEDBACK_FORMATS}"
        )

    failing = list(traj.get("_tf_failing") or [])
    passing = list(traj.get("_tf_passing") or [])
    resolved = bool(traj.get("_tf_resolved", False))
    patch_exists = bool(traj.get("_tf_patch_exists", True))
    n_fail, n_pass = len(failing), len(passing)
    observed = n_fail + n_pass
    total = int(traj.get("_tf_total") or observed)

    # Legacy augmented trajectories only stored the preformatted block.
    if not failing and not passing and not resolved:
        return str(traj.get("test_feedback") or "")

    if not patch_exists:
        return "Scout test execution: NO PATCH"

    if fmt == "count_only":
        if resolved:
            return f"Scout test execution: PASSED ({n_pass}/{total} tests)"
        if total == 0:
            return "Scout test execution: FAILED (no test output available)"
        return (
            f"Scout test execution: FAILED after {n_pass} passed tests "
            f"({total} tests in suite)"
        )

    if fmt == "names_only":
        all_names = failing + passing
        if not all_names:
            return "Scout test execution: no test names available"
        parts = [f"Scout test execution: {total} tests in suite"]
        parts.extend(str(name) for name in all_names[:200])
        overflow = observed - min(observed, 200)
        if overflow:
            parts.append(f"(+{overflow} more)")
        return "\n".join(parts)

    return str(traj.get("test_feedback") or "")


def has_test_feedback(traj: dict[str, Any]) -> bool:
    """Return whether a trajectory contains an actual test-execution feature."""
    return bool(
        str(traj.get("test_feedback") or "").strip()
        or traj.get("_tf_failing")
        or traj.get("_tf_passing")
        or traj.get("_tf_resolved")
        or traj.get("_tf_patch_exists") is False
    )


def build_abstention_input(
    problem_statement: str,
    thinking_text: str,
    patch_text: str,
    include_thinking: bool,
    *,
    input_only: bool = False,
    test_feedback_text: str = "",
    include_test_feedback: bool = False,
) -> str:
    """Build the exact predictor input used by both training and inference."""
    if input_only:
        return "\n".join([
            "Predict whether a strong model will successfully resolve this task.",
            "Use only the problem description.",
            "",
            "[Problem Statement]",
            problem_statement.strip(),
        ])

    parts = [
        "Predict whether a strong model will successfully resolve this software repair task.",
        "Use the problem description and the scout model's repair attempt.",
        "",
        "[Problem Statement]",
        problem_statement.strip(),
        "",
        "[Scout Repair Attempt]",
    ]
    if include_thinking and thinking_text:
        parts += ["<think>", thinking_text.strip(), "</think>"]
    parts.append(patch_text.strip())
    if include_test_feedback and test_feedback_text:
        parts += ["", "[Scout Test Execution Feedback]", test_feedback_text.strip()]
    return "\n".join(parts)

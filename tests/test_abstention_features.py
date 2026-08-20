from pipelinerl.swe.scripts.offline_router.abstention_features import (
    build_abstention_input,
    build_test_feedback_text,
    has_test_feedback,
)


def test_feedback_formats_use_raw_execution_results() -> None:
    trajectory = {
        "test_feedback": "Scout test execution: FAILED\nFAILED: case_0001",
        "_tf_failing": ["case_0001:wrong_answer"],
        "_tf_passing": ["case_0000"],
        "_tf_resolved": False,
        "_tf_patch_exists": True,
        "_tf_total": 10,
    }

    assert has_test_feedback(trajectory)
    assert build_test_feedback_text(trajectory, "full") == trajectory["test_feedback"]
    assert "after 1 passed tests (10 tests in suite)" in build_test_feedback_text(
        trajectory, "count_only"
    )
    names = build_test_feedback_text(trajectory, "names_only")
    assert "case_0001:wrong_answer" in names
    assert "case_0000" in names


def test_training_and_scoring_serializer_keeps_feedback() -> None:
    text = build_abstention_input(
        "problem",
        "reasoning",
        "patch",
        include_thinking=False,
        test_feedback_text="Scout test execution: PASSED (1/1 tests)",
        include_test_feedback=True,
    )

    assert "reasoning" not in text
    assert "patch" in text
    assert "[Scout Test Execution Feedback]" in text
    assert "PASSED (1/1 tests)" in text

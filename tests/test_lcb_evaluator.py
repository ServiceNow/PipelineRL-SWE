import json

import pytest

from pipelinerl.swe.scripts.livecodebench.collect_lcb_trajectories import (
    evaluate_code,
    is_evaluator_infrastructure_error,
    lcb_release_files,
)
from pipelinerl.swe.scripts.offline_router.train_cot_abstention_predictor import (
    load_unresolved_oracle_problem_ids,
)


pytest.importorskip("lcb_runner.evaluation.compute_code_generation_metrics")


@pytest.mark.parametrize(
    ("release", "expected"),
    [
        ("release_v1", ["test.jsonl"]),
        ("release_v3", ["test.jsonl", "test2.jsonl", "test3.jsonl"]),
        (
            "release_v6",
            [
                "test.jsonl",
                "test2.jsonl",
                "test3.jsonl",
                "test4.jsonl",
                "test5.jsonl",
                "test6.jsonl",
            ],
        ),
        ("v3", ["test3.jsonl"]),
        ("v2_v4", ["test2.jsonl", "test3.jsonl", "test4.jsonl"]),
    ],
)
def test_lcb_release_files_match_official_loader(
    release: str, expected: list[str]
) -> None:
    assert lcb_release_files(release) == expected


def test_lcb_release_files_reject_invalid_range() -> None:
    with pytest.raises(ValueError, match="Unsupported LiveCodeBench release"):
        lcb_release_files("v4_v2")


def test_evaluator_infrastructure_error_detection() -> None:
    assert is_evaluator_infrastructure_error(
        [], {"error_message": "RuntimeError('os.fork is unsafe')"}
    )
    assert is_evaluator_infrastructure_error(
        [], {"error_message": "anything"}, ["TEST_RUNNER_ERROR"]
    )
    assert is_evaluator_infrastructure_error(
        [], {"error_message": "IndexError('list index out of range')"}
    )
    assert not is_evaluator_infrastructure_error(
        [], {"error_message": "EmptyGeneration"}
    )
    assert not is_evaluator_infrastructure_error(
        [-2], {"error_message": "Wrong Answer"}
    )
    assert not is_evaluator_infrastructure_error(
        [-1], {"error_message": "global timeout"}
    )


def test_unresolved_oracle_calls_are_quarantined(tmp_path) -> None:
    results = tmp_path / "oracle_eval.jsonl"
    rows = [
        {
            "problem_id": "completed",
            "full_output": "print(1)",
            "eval_metadata": {"error_message": "Wrong Answer"},
        },
        {
            "problem_id": "empty_generation",
            "full_output": "",
            "eval_metadata": {"error_message": "EmptyGeneration"},
        },
        {
            "problem_id": "timed_out",
            "full_output": "",
            "eval_metadata": {"error_message": "TimeoutError()"},
        },
    ]
    results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    assert load_unresolved_oracle_problem_ids(str(results)) == {"timed_out"}


def _row(input_output: dict) -> dict:
    return {
        "platform": "test",
        "question_id": "sample",
        "_evaluation_sample": {"input_output": json.dumps(input_output)},
    }


def test_official_runner_grades_stdin_program() -> None:
    row = _row({"inputs": ["2 3\n"], "outputs": ["5\n"], "fn_name": None})
    report = evaluate_code("a, b = map(int, input().split())\nprint(a + b)", row)

    assert report["resolved"] is True
    assert report["result_codes"] == [True]


def test_official_runner_grades_function_call() -> None:
    row = _row({"inputs": ["2\n3"], "outputs": ["5"], "fn_name": "add"})
    code = "class Solution:\n    def add(self, a, b):\n        return a + b\n"
    report = evaluate_code(code, row)

    assert report["resolved"] is True
    assert report["result_codes"] == [True]


def test_global_timeout_without_metadata_uses_official_fallback(monkeypatch) -> None:
    from lcb_runner.evaluation import compute_code_generation_metrics

    def raise_missing_metadata(*args, **kwargs):
        raise IndexError("list index out of range")

    monkeypatch.setattr(
        compute_code_generation_metrics,
        "check_correctness",
        raise_missing_metadata,
    )
    row = _row(
        {"inputs": ["1\n", "2\n"], "outputs": ["1\n", "2\n"], "fn_name": None}
    )

    report = evaluate_code("print(input())", row)

    assert report["resolved"] is False
    assert report["result_codes"] == [-1, -1]
    assert report["failing"] == [
        "case_0000:global_timeout",
        "case_0001:global_timeout",
    ]
    assert report["metadata"]["error_message"] == "GlobalTimeout"


def test_public_feedback_is_separate_from_full_evaluation() -> None:
    row = _row({"inputs": ["2\n3", "10\n20"], "outputs": ["5", "30"], "fn_name": "add"})
    row["_public_evaluation_sample"] = {
        "input_output": json.dumps(
            {"inputs": ["2\n3"], "outputs": ["5"], "fn_name": "add"}
        )
    }
    code = "class Solution:\n    def add(self, a, b):\n        return 5\n"

    public_report = evaluate_code(code, row, sample_key="_public_evaluation_sample")
    full_report = evaluate_code(code, row)

    assert public_report["resolved"] is True
    assert full_report["resolved"] is False

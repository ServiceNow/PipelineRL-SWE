import json

import pytest

from pipelinerl.swe.scripts.livecodebench.collect_lcb_trajectories import (
    evaluate_code,
    lcb_release_files,
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

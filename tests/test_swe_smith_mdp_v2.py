import json

from pipelinerl.swe.scripts.livecodebench.mdp_utils import validate_split_manifest
from pipelinerl.swe.scripts.offline_router.swe_smith_mdp_builder import (
    MODEL_SPECS,
    execution_feedback,
    read_reports,
    write_external_holdout,
)


def test_primary_portfolio_matches_lcb_three_tiers() -> None:
    assert [row[0] for row in MODEL_SPECS] == ["scout", "oss20", "oss120"]


def test_real_report_feedback_contains_outcome_and_test_counts() -> None:
    report = {
        "resolved": False,
        "patch_exists": True,
        "tests_status": {
            "FAIL_TO_PASS": {"success": ["fixed"], "failure": ["still_bad"]},
            "PASS_TO_PASS": {"success": ["old_1", "old_2"], "failure": []},
        },
    }
    text = execution_feedback(report)
    assert "Sandbox execution: FAILED" in text
    assert "FAIL_TO_PASS: passed=1/2" in text
    assert "still_bad" in text
    assert "PASS_TO_PASS: passed=2/2" in text


def test_missing_reports_are_absent_not_negative_labels(tmp_path) -> None:
    run_dir = tmp_path / "run"
    reported = run_dir / "reported"
    reported.mkdir(parents=True)
    (reported / "report.json").write_text(json.dumps({"resolved": False}) + "\n")
    # A missing instance has no directory/report and therefore cannot appear as False.
    assert read_reports(run_dir) == {"reported": {"resolved": False}}


def test_external_holdout_remains_exact_and_disjoint(tmp_path) -> None:
    problem_ids = [f"p{i}" for i in range(12)]
    ids_path = tmp_path / "test_ids.txt"
    ids_path.write_text("p9\np10\np11\n")
    manifest = write_external_holdout(
        tmp_path / "split.json", problem_ids, ids_path, seed=7, cal_fraction=0.25
    )
    assert manifest["test_problem_ids"] == ["p10", "p11", "p9"]
    validate_split_manifest(manifest, problem_ids)

from pipelinerl.swe.scripts.livecodebench.build_mdp_tensors_v2 import _result_feedback


def test_full_execution_feedback_uses_official_result_names() -> None:
    feedback = _result_feedback(False, [True, -1, -2, -3, -4, -5])
    assert "passed=1/6" in feedback
    assert "global_timeout=1" in feedback
    assert "wrong_answer=1" in feedback
    assert "time_limit_exceeded=1" in feedback
    assert "runtime_error=1" in feedback
    assert "test_runner_error=1" in feedback

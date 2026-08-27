import json
import sys

import numpy as np
import pytest

from pipelinerl.swe.scripts.livecodebench.build_mdp_reachable_dataset import (
    _nothing_remaining_target,
    _remaining_draw_counts,
    _render_state,
    main as build_dataset,
)
from pipelinerl.swe.scripts.livecodebench.mdp_utils import (
    build_split_manifest,
    redact_sensitive_text,
    validate_split_manifest,
    write_split_manifest,
)
from pipelinerl.swe.scripts.livecodebench.replay_mdp_full_execution import (
    _paired_problem_bootstrap,
    _summarize_decisions,
    replay_adaptive,
    replay_fixed,
)


def test_split_manifest_is_deterministic_disjoint_and_complete() -> None:
    problem_ids = [f"p{i}" for i in range(20)]
    left = build_split_manifest(problem_ids, seed=7)
    right = build_split_manifest(reversed(problem_ids), seed=7)
    assert left == right
    validate_split_manifest(left, problem_ids)
    groups = [set(left[f"{name}_problem_ids"]) for name in ("train", "calibration", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert set().union(*groups) == set(problem_ids)


def test_sensitive_exception_text_is_redacted() -> None:
    message = "Authorization: Bearer secret-value api_key=another-secret sk-examplelongsecret123"
    redacted = redact_sensitive_text(message)
    assert "secret-value" not in redacted
    assert "another-secret" not in redacted
    assert "sk-examplelongsecret123" not in redacted


def _records(pid: str, slots: list[str], draws: int) -> dict:
    return {
        (pid, slot, draw): {
            "problem_id": pid,
            "model_slot": slot,
            "draw_index": draw,
            "code": f"{slot}-{draw}",
            "full_execution_feedback": "Full execution: FAILED",
        }
        for slot in slots
        for draw in range(draws)
    }


def test_policy_state_keeps_counts_and_only_latest_failed_attempt() -> None:
    attempts = [
        {
            "model_slot": "scout",
            "code": "OLD_SCOUT_CODE",
            "full_execution_feedback": "old failure",
        },
        {
            "model_slot": "oss20",
            "code": "LATEST_OSS20_CODE",
            "full_execution_feedback": "latest failure",
        },
    ]
    text = _render_state(
        "solve this problem", attempts, {"scout": 8, "oss20": 9, "oss120": 10}
    )
    assert "scout: failed=1, remaining=8" in text
    assert "oss20: failed=1, remaining=9" in text
    assert "oss120: failed=0, remaining=10" in text
    assert "latest_attempted_route: oss20" in text
    assert "LATEST_OSS20_CODE" in text
    assert "latest failure" in text
    assert "OLD_SCOUT_CODE" not in text
    assert "old failure" not in text
    assert text.index("[Execution state]") < text.index("[Latest verified failed attempt")


def test_remaining_counts_exclude_invalid_draws() -> None:
    valid = np.array([
        [True, False, True],
        [False, True, True],
        [True, True, True],
    ])
    orders = np.tile(np.arange(3), (3, 1))
    ptr = np.array([1, 0, 2])
    assert _remaining_draw_counts(
        valid, orders, ptr, ["scout", "oss20", "oss120"]
    ) == {"scout": 1, "oss20": 2, "oss120": 1}


def test_nothing_target_looks_beyond_the_next_draw() -> None:
    outcomes = np.array([
        [False, True, False],
        [False, False, False],
        [False, False, False],
    ])
    valid = np.ones_like(outcomes)
    orders = np.tile(np.arange(3), (3, 1))
    ptr = np.zeros(3, dtype=int)
    # All three next draws fail, but a later scout draw succeeds.
    assert _nothing_remaining_target(outcomes, valid, orders, ptr) == 0.0
    ptr[0] = 2
    assert _nothing_remaining_target(outcomes, valid, orders, ptr) == 1.0


def test_adaptive_replay_enters_router_only_after_scout_failure() -> None:
    slots = ["scout", "oss20", "oss120"]
    ordering = np.tile(np.arange(2), (3, 1))
    valid = np.ones((3, 2), dtype=bool)
    costs = np.ones((3, 2), dtype=float)

    scout_pass = np.zeros((3, 2), dtype=bool)
    scout_pass[0, 0] = True
    out = replay_adaptive(
        scout_pass, valid, costs, np.ones(3), ordering, 10, np.ones(3) * 0.5,
        2.0, slots, _records("p", slots, 2), "p", "problem",
    )
    assert out["correct"] is True
    assert out["entered_router"] is False
    assert out["attempts"] == 1

    scout_fail_then_expert_pass = np.zeros((3, 2), dtype=bool)
    scout_fail_then_expert_pass[1, 0] = True
    out = replay_adaptive(
        scout_fail_then_expert_pass, valid, costs, np.ones(3), ordering, 10,
        np.array([0.1, 0.9, 0.1]), 2.0, slots, _records("p", slots, 2), "p", "problem",
    )
    assert out["correct"] is True
    assert out["entered_router"] is True
    assert out["attempts"] == 2


def test_adaptive_replay_trace_records_probabilities_state_and_choice() -> None:
    slots = ["scout", "oss20", "oss120"]
    outcomes = np.zeros((3, 2), dtype=bool)
    outcomes[1, 0] = True
    scorer = lambda _: np.array([0.1, 0.9, 0.1, 0.2])
    result = replay_adaptive(
        outcomes,
        np.ones_like(outcomes),
        np.ones_like(outcomes, dtype=float),
        np.ones(3),
        np.tile(np.arange(2), (3, 1)),
        3.0,
        np.ones(3) * 0.5,
        2.0,
        slots,
        _records("p", slots, 2),
        "p",
        "problem",
        scorer=scorer,
        capture_trace=True,
    )
    assert result["correct"] is True
    assert result["route_attempt_counts"] == {"scout": 1, "oss20": 1, "oss120": 0}
    assert len(result["decision_trace"]) == 1
    decision = result["decision_trace"][0]
    assert decision["failure_counts"] == {"scout": 1, "oss20": 0, "oss120": 0}
    assert decision["remaining_valid_draws"] == {"scout": 1, "oss20": 2, "oss120": 2}
    assert decision["p_success_next"]["oss20"] == 0.9
    assert decision["p_any_remaining"] == 0.8
    assert decision["chosen_route"] == "oss20"
    assert decision["chosen_draw_index"] == 0
    assert decision["result"] == "pass"
    assert decision["state_key"]


def test_sequential_decay_applies_route_failure_prior_before_choice() -> None:
    slots = ["scout", "oss20", "oss120"]
    outcomes = np.zeros((3, 2), dtype=bool)
    outcomes[1, 0] = True
    result = replay_adaptive(
        outcomes,
        np.ones_like(outcomes),
        np.ones_like(outcomes, dtype=float),
        np.ones(3),
        np.tile(np.arange(2), (3, 1)),
        3.0,
        np.ones(3) * 0.5,
        2.0,
        slots,
        _records("p", slots, 2),
        "p",
        "problem",
        scorer=lambda _: np.array([0.6, 0.5, 0.1, 0.2]),
        capture_trace=True,
        apply_failure_decay=True,
    )
    decision = result["decision_trace"][0]
    assert decision["belief_source"] == "sequential_decay"
    assert np.isclose(decision["p_success_next"]["scout"], 0.4)
    assert np.isclose(decision["p_success_next"]["oss20"], 0.5)
    assert decision["chosen_route"] == "oss20"
    assert result["correct"] is True


def test_free_start_can_abstain_before_any_model_call() -> None:
    slots = ["scout", "oss20", "oss120"]
    result = replay_adaptive(
        np.zeros((3, 2), dtype=bool),
        np.ones((3, 2), dtype=bool),
        np.ones((3, 2), dtype=float),
        np.ones(3),
        np.tile(np.arange(2), (3, 1)),
        3.0,
        np.ones(3) * 0.5,
        2.0,
        slots,
        _records("p", slots, 2),
        "p",
        "problem",
        scorer=lambda _: np.array([0.01, 0.01, 0.01, 0.99]),
        tau_abstain=0.1,
        capture_trace=True,
        mandatory_scout=False,
    )
    assert result["abstained"] is True
    assert result["attempts"] == 0
    assert result["start_protocol"] == "free_start"
    assert result["decision_trace"][0]["failure_depth"] == 0
    assert result["decision_trace"][0]["result"] == "abstain"


def test_free_start_can_choose_an_expert_at_the_root() -> None:
    slots = ["scout", "oss20", "oss120"]
    outcomes = np.zeros((3, 2), dtype=bool)
    outcomes[1, 0] = True
    result = replay_adaptive(
        outcomes,
        np.ones_like(outcomes),
        np.ones_like(outcomes, dtype=float),
        np.ones(3),
        np.tile(np.arange(2), (3, 1)),
        3.0,
        np.ones(3) * 0.5,
        2.0,
        slots,
        _records("p", slots, 2),
        "p",
        "problem",
        scorer=lambda _: np.array([0.1, 0.9, 0.1, 0.1]),
        capture_trace=True,
        mandatory_scout=False,
    )
    assert result["correct"] is True
    assert result["attempts"] == 1
    assert result["route_attempt_counts"] == {"scout": 0, "oss20": 1, "oss120": 0}
    assert result["decision_trace"][0]["failure_depth"] == 0
    assert result["decision_trace"][0]["chosen_route"] == "oss20"


def test_marginal_value_stop_can_quit_immediately_after_mandatory_failure() -> None:
    slots = ["scout", "oss20", "oss120"]
    result = replay_adaptive(
        np.zeros((3, 2), dtype=bool),
        np.ones((3, 2), dtype=bool),
        np.ones((3, 2), dtype=float),
        np.ones(3),
        np.tile(np.arange(2), (3, 1)),
        3.0,
        np.ones(3) * 0.5,
        2.0,
        slots,
        _records("p", slots, 2),
        "p",
        "problem",
        scorer=lambda _: np.array([0.3, 0.2, 0.1, 0.9]),
        min_success_per_cost=0.31,
        capture_trace=True,
    )
    assert result["abstained"] is True
    assert result["attempts"] == 1
    assert result["decision_trace"][0]["failure_depth"] == 1
    assert result["decision_trace"][0]["abstain_reason"] == "marginal_value"


def _unequal_cost_replay(**kwargs):
    """Ratio ranks scout first (50 > 25 > 20); dollar value ranks oss120 first."""
    slots = ["scout", "oss20", "oss120"]
    return replay_adaptive(
        np.zeros((3, 2), dtype=bool),
        np.ones((3, 2), dtype=bool),
        np.ones((3, 2), dtype=float),
        np.array([0.001, 0.004, 0.03]),
        np.tile(np.arange(2), (3, 1)),
        10.0,
        np.array([0.05, 0.10, 0.60]),
        2.0,
        slots,
        _records("p", slots, 2),
        "p",
        "problem",
        scorer=lambda _: np.array([0.05, 0.10, 0.60, 0.5]),
        capture_trace=True,
        mandatory_scout=False,
        **kwargs,
    )


def test_ratio_selection_prefers_the_cheapest_route() -> None:
    result = _unequal_cost_replay()
    assert result["decision_trace"][0]["chosen_route"] == "scout"
    assert result["decision_trace"][0]["action_value"] is None


def test_value_selection_prefers_the_expensive_high_probability_route() -> None:
    result = _unequal_cost_replay(value_of_correct=1.0)
    assert result["decision_trace"][0]["chosen_route"] == "oss120"
    assert result["decision_trace"][0]["action_value"]["oss120"] > 0.0


def test_value_selection_ranking_is_not_scale_invariant() -> None:
    """The ratio rule ranks routes identically at every valuation; value does not."""
    low = _unequal_cost_replay(value_of_correct=0.04)
    high = _unequal_cost_replay(value_of_correct=1.0)
    assert low["decision_trace"][0]["chosen_route"] == "scout"
    assert high["decision_trace"][0]["chosen_route"] == "oss120"


def test_value_control_abstains_at_the_root_without_a_threshold() -> None:
    result = _unequal_cost_replay(value_of_correct=0.01)
    assert result["abstained"] is True
    assert result["attempts"] == 0
    assert result["decision_trace"][0]["failure_depth"] == 0
    assert result["decision_trace"][0]["abstain_reason"] == "non_positive_action_value"
    assert result["decision_trace"][0]["tau_abstain"] is None
    assert result["decision_trace"][0]["min_success_per_cost"] is None


def test_stopping_controls_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="at most one"):
        _unequal_cost_replay(value_of_correct=1.0, tau_abstain=0.5)


def test_decision_summary_and_problem_clustered_bootstrap() -> None:
    traced = [{
        "decision_trace": [{
            "p_success_next": {"scout": 0.1, "oss20": 0.8, "oss120": 0.2},
            "p_any_remaining": 0.9,
            "chosen_route": "oss20",
            "result": "pass",
        }]
    }]
    summary = _summarize_decisions(traced, ["scout", "oss20", "oss120"])
    assert summary["route_choice_counts"] == {"scout": 0, "oss20": 1, "oss120": 0}
    assert summary["route_pass_counts"]["oss20"] == 1
    assert summary["mean_predicted_any_remaining"] == 0.9

    reference = []
    candidate = []
    for pid in ("p0", "p1"):
        for ordering_index in range(2):
            reference.append({
                "problem_id": pid,
                "ordering_index": ordering_index,
                "correct": False,
                "realized_spend": 1.0,
                "attempts": 1,
            })
            candidate.append({
                "problem_id": pid,
                "ordering_index": ordering_index,
                "correct": pid == "p0",
                "realized_spend": 2.0 if pid == "p0" else 0.5,
                "attempts": 2 if pid == "p0" else 1,
            })
    comparison = _paired_problem_bootstrap(
        reference, candidate, samples=500, seed=7
    )
    assert comparison["n_problems"] == 2
    assert comparison["n_paired_episodes"] == 4
    assert comparison["correctness_delta"] == 0.5
    assert comparison["realized_cost_delta"] == 0.25
    assert comparison["attempts_delta"] == 0.5


def test_invalid_draws_are_skipped_without_cost() -> None:
    outcomes = np.array([[True, False]])
    valid = np.array([[False, True]])
    costs = np.array([[100.0, 2.0]])
    result = replay_fixed(outcomes, valid, costs, np.array([[0, 1]]), [0])
    assert result == {"correct": False, "realized_spend": 2.0, "attempts": 1}


def test_reachable_dataset_contains_only_failed_histories(tmp_path, monkeypatch) -> None:
    tensor_dir = tmp_path / "tensors"
    output_dir = tmp_path / "dataset"
    tensor_dir.mkdir()
    pids = ["p0", "p1", "p2", "p3", "p4", "p5"]
    slots = ["scout", "oss20", "oss120"]
    outcomes = np.zeros((len(pids), len(slots), 3), dtype=bool)
    outcomes[0, 0, :] = True  # Always resolved by the mandatory scout: no router state.
    outcomes[1:, 1, 0] = True
    valid = np.ones_like(outcomes)
    np.savez_compressed(
        tensor_dir / "tensors.npz",
        final_outcome=outcomes,
        execution_outcome=outcomes,
        valid=valid,
        problem_ids=np.array(pids),
        model_slots=np.array(slots),
    )
    with open(tensor_dir / "problems.jsonl", "w") as handle:
        for pid in pids:
            handle.write(json.dumps({"problem_id": pid, "problem_statement": f"problem {pid}"}) + "\n")
    with open(tensor_dir / "draw_records.jsonl", "w") as handle:
        for pi, pid in enumerate(pids):
            for mi, slot in enumerate(slots):
                for draw in range(3):
                    handle.write(json.dumps({
                        "problem_id": pid,
                        "model_slot": slot,
                        "draw_index": draw,
                        "code": f"{slot}-{draw}",
                        "full_execution_feedback": (
                            "Full execution: PASSED" if outcomes[pi, mi, draw]
                            else "Full execution: FAILED"
                        ),
                    }) + "\n")
    write_split_manifest(tensor_dir / "split_manifest.json", pids, seed=0)
    monkeypatch.setattr(sys, "argv", [
        "build_mdp_reachable_dataset.py",
        "--tensors-dir", str(tensor_dir),
        "--output-dir", str(output_dir),
        "--histories-per-problem", "4",
        "--max-failures", "4",
    ])
    build_dataset()
    rows = [
        json.loads(line)
        for split in ("train", "calibration", "test")
        for line in open(output_dir / f"{split}.jsonl")
    ]
    assert rows
    assert all(row["problem_id"] != "p0" for row in rows)
    assert all(row["failure_depth"] >= 1 for row in rows)
    for row in rows:
        assert row["attempt_history"][0]["model_slot"] == "scout"
        pi = pids.index(row["problem_id"])
        for attempt in row["attempt_history"]:
            mi = slots.index(attempt["model_slot"])
            assert not outcomes[pi, mi, attempt["draw_index"]]


    free_start_dir = tmp_path / "free_start_dataset"
    monkeypatch.setattr(sys, "argv", [
        "build_mdp_reachable_dataset.py",
        "--tensors-dir", str(tensor_dir),
        "--output-dir", str(free_start_dir),
        "--histories-per-problem", "4",
        "--max-failures", "4",
        "--start-protocol", "free_start",
    ])
    build_dataset()
    free_start_rows = [
        json.loads(line)
        for split in ("train", "calibration", "test")
        for line in open(free_start_dir / f"{split}.jsonl")
    ]
    roots = [row for row in free_start_rows if row["failure_depth"] == 0]
    assert {row["problem_id"] for row in roots} == set(pids)
    assert all(row["attempt_history"] == [] for row in roots)
    assert all("latest_attempted_route: none" in row["text"] for row in roots)

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
from pipelinerl.swe.scripts.livecodebench.structured_state import (
    STATE_FEATURE_NAMES,
    build_structured_state_features,
)
from pipelinerl.swe.scripts.livecodebench.mdp_utils import (
    build_split_manifest,
    redact_sensitive_text,
    validate_split_manifest,
    write_split_manifest,
)
from pipelinerl.swe.scripts.livecodebench.replay_mdp_full_execution import (
    _paired_problem_bootstrap,
    _solve_bellman_action_values,
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
    assert "total_failures: 2" in text



def test_structured_state_features_expose_normalized_counts_and_latest_route() -> None:
    attempts = [
        {"model_slot": "scout"},
        {"model_slot": "oss20"},
    ]
    values = build_structured_state_features(
        attempts,
        {"scout": 3, "oss20": 9, "oss120": 10},
        {"scout": 4, "oss20": 10, "oss120": 10},
        ["scout", "oss20", "oss120"],
    )
    assert len(values) == len(STATE_FEATURE_NAMES) == 11
    assert values == pytest.approx([
        1 / 4, 1 / 10, 0.0,
        3 / 4, 9 / 10, 1.0,
        2 / 24,
        0.0, 0.0, 1.0, 0.0,
    ])


def test_counts_last_layout_puts_execution_state_at_the_readout_token() -> None:
    """The encoder pools the final token, so the decay signal must end the prompt."""
    attempts = [{
        "model_slot": "oss20",
        "code": "LATEST_OSS20_CODE",
        "full_execution_feedback": "latest failure",
    }]
    remaining = {"scout": 8, "oss20": 9, "oss120": 10}
    counts_last = _render_state("solve this", attempts, remaining, "counts_last")
    problem_first = _render_state("solve this", attempts, remaining, "problem_first")

    assert counts_last.rstrip().endswith("total_failures: 1")
    assert counts_last.index("[Latest verified failed attempt") < counts_last.index("[Execution state]")
    assert problem_first.index("[Execution state]") < problem_first.index("[Latest verified failed attempt")
    assert problem_first.rstrip().endswith("LATEST_OSS20_CODE")
    # Same information either way; only the read-out position differs.
    for fragment in ("scout: failed=0, remaining=8", "LATEST_OSS20_CODE", "latest failure"):
        assert fragment in counts_last and fragment in problem_first


def test_unknown_state_layout_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown state layout"):
        _render_state("p", [], {"scout": 1}, "nonsense")


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


def test_ucb_bonus_shifts_choice_toward_the_less_sampled_route() -> None:
    """RoR's UCB arm: sqrt(2 ln(t+1)/(n_m+1))/c_m added to the ratio."""
    slots = ["scout", "oss20", "oss120"]
    kwargs = dict(
        content_prior=None, capture_trace=True, mandatory_scout=False,
        scorer=lambda _, **__: np.array([0.30, 0.28, 0.05, 0.5]),
    )
    common = (
        np.zeros((3, 4), dtype=bool), np.ones((3, 4), dtype=bool),
        np.ones((3, 4), dtype=float), np.array([0.001, 0.001, 0.03]),
        np.tile(np.arange(4), (3, 1)), 10.0, np.array([0.3, 0.28, 0.05]), 2.0,
        slots, _records("p", slots, 4), "p", "problem",
    )
    greedy = replay_adaptive(*common, **kwargs)
    ucb = replay_adaptive(*common, exploration_bonus=True, **kwargs)
    # Greedy commits to the highest ratio every time; UCB must explore more routes.
    assert len({d["chosen_route"] for d in greedy["decision_trace"]}) <= \
           len({d["chosen_route"] for d in ucb["decision_trace"]})


def test_decay_pseudo_count_is_independent_of_the_ror_pseudo_count() -> None:
    """The RoR s-sweep must not drag our decay variant along with it."""
    slots = ["scout", "oss20", "oss120"]
    def go(decay_s):
        return replay_adaptive(
            np.zeros((3, 3), dtype=bool), np.ones((3, 3), dtype=bool),
            np.ones((3, 3), dtype=float), np.ones(3), np.tile(np.arange(3), (3, 1)),
            10.0, np.ones(3) * 0.5, 2.0, slots, _records("p", slots, 3), "p", "problem",
            scorer=lambda _, **__: np.array([0.4, 0.3, 0.2, 0.5]),
            apply_failure_decay=True, decay_pseudo_count=decay_s,
            capture_trace=True, mandatory_scout=False,
        )
    weak = go(0.5)["decision_trace"]
    strong = go(50.0)["decision_trace"]
    # A large decay pseudo-count barely shrinks beliefs after one failure; a small one does.
    assert weak[1]["p_success_next"]["scout"] < strong[1]["p_success_next"]["scout"]


def test_greedy_is_unchanged_when_exploration_is_off() -> None:
    slots = ["scout", "oss20", "oss120"]
    common = (
        np.zeros((3, 3), dtype=bool), np.ones((3, 3), dtype=bool),
        np.ones((3, 3), dtype=float), np.ones(3), np.tile(np.arange(3), (3, 1)),
        10.0, np.ones(3) * 0.5, 2.0, slots, _records("p", slots, 3), "p", "problem",
    )
    a = replay_adaptive(*common, capture_trace=True, mandatory_scout=False)
    b = replay_adaptive(*common, exploration_bonus=False, capture_trace=True, mandatory_scout=False)
    assert [d["chosen_route"] for d in a["decision_trace"]] == \
           [d["chosen_route"] for d in b["decision_trace"]]


# --- Exact Bellman solve over the failure-count lattice ---------------------------


def _solve(**kwargs):
    """One route, p=0.5 flat, cost 0.1, R=1.0 -- small enough to verify by hand."""
    params = dict(
        pbar=np.array([0.5]),
        failures=np.zeros(1, dtype=int),
        remaining_by_route=[2],
        expected_costs=np.array([0.1]),
        spent_budget=0.0,
        budget=10.0,
        value_of_correct=1.0,
        decay_s=None,
        horizon=2,
    )
    params.update(kwargs)
    return _solve_bellman_action_values(**params)


def test_bellman_backward_induction_matches_hand_computation() -> None:
    # V(1) = 0.5*1 - 0.1 + 0.5*V(2) = 0.4, with V(2) = 0 by horizon truncation.
    # V(0) = 0.5*1 - 0.1 + 0.5*0.4 = 0.6.
    action_values, state_value = _solve(horizon=2)
    assert action_values[0] == pytest.approx(0.6)
    assert state_value == pytest.approx(0.6)


def test_bellman_horizon_one_is_the_myopic_action_value() -> None:
    action_values, state_value = _solve(horizon=1)
    assert action_values[0] == pytest.approx(0.5 * 1.0 - 0.1)
    assert state_value == pytest.approx(0.4)


def test_bellman_state_value_is_monotone_in_horizon() -> None:
    """Deeper lookahead can only add option value: abstaining is always worth 0."""
    values = [_solve(horizon=h)[1] for h in (1, 2, 3, 4, 5)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
    assert values[-1] > values[0]


def test_bellman_respects_remaining_draw_capacity() -> None:
    action_values, state_value = _solve(remaining_by_route=[0])
    assert action_values == {}
    assert state_value == 0.0


def test_bellman_excludes_routes_the_budget_cannot_afford() -> None:
    action_values, _ = _solve_bellman_action_values(
        pbar=np.array([0.5, 0.9]),
        failures=np.zeros(2, dtype=int),
        remaining_by_route=[2, 2],
        expected_costs=np.array([0.1, 5.0]),
        spent_budget=0.0,
        budget=1.0,
        value_of_correct=1.0,
        decay_s=None,
        horizon=3,
    )
    assert set(action_values) == {0}


def test_bellman_abstains_when_every_action_is_worth_less_than_nothing() -> None:
    action_values, state_value = _solve(value_of_correct=0.01)
    assert state_value == 0.0
    assert max(action_values.values()) <= 0.0


def test_bellman_decay_lowers_the_value_of_repeating_a_failed_route() -> None:
    flat = _solve(decay_s=None, horizon=3)[1]
    decayed = _solve(decay_s=1.0, horizon=3)[1]
    assert decayed < flat


def _bellman_replay(**kwargs):
    """Myopic grabs oss120 outright; the cheap scout preserves that option for later."""
    slots = ["scout", "oss20", "oss120"]
    return replay_adaptive(
        np.zeros((3, 3), dtype=bool),
        np.ones((3, 3), dtype=bool),
        np.ones((3, 3), dtype=float),
        np.array([0.0001, 0.05, 0.2]),
        np.tile(np.arange(3), (3, 1)),
        10.0,
        np.array([0.05, 0.02, 0.5]),
        1.0,
        slots,
        _records("p", slots, 3),
        "p",
        "problem",
        scorer=lambda _: np.array([0.05, 0.02, 0.5, 0.5]),
        apply_failure_decay=True,
        decay_pseudo_count=1.0,
        capture_trace=True,
        mandatory_scout=False,
        **kwargs,
    )


def test_bellman_horizon_one_reproduces_the_myopic_replay_exactly() -> None:
    myopic = _bellman_replay(value_of_correct=1.0)
    lookahead_one = _bellman_replay(value_of_correct=1.0, bellman_horizon=1)
    assert myopic["correct"] == lookahead_one["correct"]
    assert myopic["realized_spend"] == pytest.approx(lookahead_one["realized_spend"])
    assert myopic["route_attempt_counts"] == lookahead_one["route_attempt_counts"]
    for a, b in zip(myopic["decision_trace"], lookahead_one["decision_trace"]):
        assert a["chosen_route"] == b["chosen_route"]
        for slot, value in a["action_value"].items():
            assert value == pytest.approx(b["action_value"][slot])


def test_continuation_value_orders_the_cheap_route_before_the_expensive_expert() -> None:
    myopic = _bellman_replay(value_of_correct=1.0)
    lookahead = _bellman_replay(value_of_correct=1.0, bellman_horizon=2)
    # Myopic maximizes p*R - c and takes the expert immediately. The Bellman arm
    # sees that scout costs a hundredth of a cent and leaves oss120 undecayed.
    assert myopic["decision_trace"][0]["chosen_route"] == "oss120"
    assert lookahead["decision_trace"][0]["chosen_route"] == "scout"
    assert lookahead["decision_trace"][0]["state_value"] > \
           myopic["decision_trace"][0]["action_value"]["oss120"]


def test_bellman_trace_records_horizon_and_state_value() -> None:
    result = _bellman_replay(value_of_correct=1.0, bellman_horizon=4)
    decision = result["decision_trace"][0]
    assert decision["bellman_horizon"] == 4
    assert decision["state_value"] > 0.0
    assert result["decision_trace"][0]["value_of_correct"] == 1.0


def test_myopic_replay_records_no_state_value() -> None:
    decision = _bellman_replay(value_of_correct=1.0)["decision_trace"][0]
    assert decision["bellman_horizon"] is None
    assert decision["state_value"] is None


def test_bellman_requires_the_value_control_and_a_positive_horizon() -> None:
    with pytest.raises(ValueError, match="value-of-correct"):
        _bellman_replay(bellman_horizon=2)
    with pytest.raises(ValueError, match="at least 1"):
        _bellman_replay(value_of_correct=1.0, bellman_horizon=0)


# --- Future-outcome stopping oracle (diagnostic upper bound only) -----------------


def test_oracle_stopping_quits_immediately_when_no_success_remains() -> None:
    slots = ["scout", "oss20", "oss120"]
    result = replay_adaptive(
        np.zeros((3, 2), dtype=bool),
        np.ones((3, 2), dtype=bool),
        np.ones((3, 2), dtype=float),
        np.ones(3),
        np.tile(np.arange(2), (3, 1)),
        10.0,
        np.ones(3) * 0.9,
        2.0,
        slots,
        _records("p", slots, 2),
        "p",
        "problem",
        capture_trace=True,
        mandatory_scout=False,
        oracle_stopping=True,
    )
    assert result["correct"] is False
    assert result["abstained"] is True
    assert result["attempts"] == 0
    assert result["decision_trace"][0]["abstain_reason"] == \
           "oracle_no_remaining_success"
    assert result["decision_trace"][0]["oracle_has_remaining_success"] is False


def test_oracle_stopping_overrides_value_stop_until_a_stored_success() -> None:
    slots = ["scout", "oss20", "oss120"]
    outcomes = np.zeros((3, 2), dtype=bool)
    outcomes[0] = [False, True]
    common = (
        outcomes,
        np.ones_like(outcomes),
        np.ones((3, 2), dtype=float) * 0.1,
        np.ones(3) * 0.1,
        np.tile(np.arange(2), (3, 1)),
        10.0,
        np.array([0.01, 0.0, 0.0]),
        2.0,
        slots,
        _records("p", slots, 2),
        "p",
        "problem",
    )
    ordinary = replay_adaptive(
        *common, value_of_correct=1.0, capture_trace=True, mandatory_scout=False,
    )
    oracle = replay_adaptive(
        *common, value_of_correct=1.0, capture_trace=True, mandatory_scout=False,
        oracle_stopping=True,
    )


# --- Learned one-step transitions inside the Bellman solve -------------------------


def _transition_replay(successor_probs=None, **kwargs):
    """Scorer returns root beliefs normally, and `successor_probs` once counts move.

    The analytic form reuses the root prediction at every lattice node, so it can
    never see a scout failure as evidence against oss120. A learned transition can.
    """
    slots = ["scout", "oss20", "oss120"]
    root = np.array([0.30, 0.20, 0.60, 0.5])

    def scorer(text, features=None):
        if successor_probs is not None and "total_failures: 1" in text:
            return np.asarray(successor_probs, dtype=float)
        return root

    return replay_adaptive(
        np.zeros((3, 3), dtype=bool),
        np.ones((3, 3), dtype=bool),
        np.ones((3, 3), dtype=float),
        np.array([0.001, 0.004, 0.03]),
        np.tile(np.arange(3), (3, 1)),
        10.0,
        np.array([0.30, 0.20, 0.60]),
        2.0,
        slots,
        _records("p", slots, 3),
        "p",
        "problem",
        scorer=scorer,
        apply_failure_decay=True,
        decay_pseudo_count=2.0,
        value_of_correct=1.0,
        capture_trace=True,
        mandatory_scout=False,
        **kwargs,
    )


def test_learned_transitions_require_a_bellman_horizon() -> None:
    with pytest.raises(ValueError, match="require a Bellman horizon"):
        _transition_replay(learned_transitions=True)


def test_learned_transitions_are_inert_when_the_model_agrees_with_the_decay() -> None:
    """A successor prediction equal to the root reproduces the analytic arm exactly."""
    analytic = _transition_replay(bellman_horizon=2)
    learned = _transition_replay(
        successor_probs=[0.30, 0.20, 0.60, 0.5], bellman_horizon=2, learned_transitions=True
    )
    assert [d["chosen_route"] for d in analytic["decision_trace"]] == \
           [d["chosen_route"] for d in learned["decision_trace"]]
    assert analytic["realized_spend"] == pytest.approx(learned["realized_spend"])


def test_learned_transitions_propagate_failure_across_routes() -> None:
    """A pessimistic successor lowers the continuation value the analytic arm keeps high."""
    analytic = _transition_replay(bellman_horizon=2)
    pessimistic = _transition_replay(
        successor_probs=[0.01, 0.01, 0.02, 0.9], bellman_horizon=2, learned_transitions=True
    )
    assert pessimistic["decision_trace"][0]["state_value"] < \
           analytic["decision_trace"][0]["state_value"]


def test_learned_transitions_do_not_change_the_myopic_horizon() -> None:
    """H=1 never evaluates a successor, so the transition model cannot matter.

    Only the root decision is compared: from the second decision onward the real
    state itself has a failure, which the stub cannot distinguish from a
    hypothetical successor by rendered text alone.
    """
    analytic = _transition_replay(bellman_horizon=1)["decision_trace"][0]
    learned = _transition_replay(
        successor_probs=[0.01, 0.01, 0.02, 0.9], bellman_horizon=1, learned_transitions=True
    )["decision_trace"][0]
    for slot, value in analytic["action_value"].items():
        assert value == pytest.approx(learned["action_value"][slot])


def test_learned_transitions_never_read_the_unseen_draw_outcome() -> None:
    """The successor state must be renderable from counts the policy already has."""
    seen: list[str] = []
    slots = ["scout", "oss20", "oss120"]

    def scorer(text, features=None):
        seen.append(text)
        return np.array([0.30, 0.20, 0.60, 0.5])

    replay_adaptive(
        np.zeros((3, 3), dtype=bool), np.ones((3, 3), dtype=bool),
        np.ones((3, 3), dtype=float), np.array([0.001, 0.004, 0.03]),
        np.tile(np.arange(3), (3, 1)), 10.0, np.array([0.30, 0.20, 0.60]), 2.0,
        slots, _records("p", slots, 3), "p", "problem",
        scorer=scorer, apply_failure_decay=True, value_of_correct=1.0,
        bellman_horizon=2, learned_transitions=True, mandatory_scout=False,
    )
    # The first four queries are the root and its three successors, all issued
    # before any attempt exists. A successor built from a real future draw would
    # carry that draw's code; a successor built from observed state cannot.
    assert any("total_failures: 1" in text for text in seen[:4]), "no successor was queried"
    for text in seen[:4]:
        assert "scout-" not in text and "oss20-" not in text and "oss120-" not in text


# --- Rolling-origin fold selection -------------------------------------------------


def _fold_manifest(tmp_path):
    path = tmp_path / "folds.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "folds": [{
            "fold": 0,
            "train_problem_ids": ["a"],
            "calibration_problem_ids": ["b"],
            "test_problem_ids": ["c"],
        }],
    }))
    return path


def test_fold_manifest_and_index_must_be_given_together(tmp_path, monkeypatch) -> None:
    from pipelinerl.swe.scripts.livecodebench import build_mdp_tensors_v2

    monkeypatch.setattr(sys, "argv", [
        "build", "--collection-dir", str(tmp_path), "--source-collection-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "o"),
        "--fold-manifest", str(_fold_manifest(tmp_path)),
    ])
    with pytest.raises(ValueError, match="must be given together"):
        build_mdp_tensors_v2.main()


def test_fold_index_must_exist_in_the_manifest(tmp_path, monkeypatch) -> None:
    from pipelinerl.swe.scripts.livecodebench import build_mdp_tensors_v2

    monkeypatch.setattr(sys, "argv", [
        "build", "--collection-dir", str(tmp_path), "--source-collection-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "o"),
        "--fold-index", "0",
    ])
    with pytest.raises(ValueError, match="must be given together"):
        build_mdp_tensors_v2.main()


def test_fold_shaped_manifest_satisfies_the_split_validator() -> None:
    """A fold covers one date block, so the validator must see only those problems."""
    manifest = {
        "train_problem_ids": ["a", "b"],
        "calibration_problem_ids": ["c"],
        "test_problem_ids": ["d"],
    }
    validate_split_manifest(manifest, ["a", "b", "c", "d"])
    # Problems outside the fold must not be silently tolerated.
    with pytest.raises(ValueError, match="does not match tensors"):
        validate_split_manifest(manifest, ["a", "b", "c", "d", "e"])


def test_oracle_routing_picks_the_cheapest_route_that_actually_succeeds() -> None:
    """Diagnostic upper bound on route choice: stopping is left untouched."""
    slots = ["scout", "oss20", "oss120"]
    outcomes = np.zeros((3, 2), dtype=bool)
    outcomes[1, 0] = True  # only oss20's next draw succeeds
    common = (
        outcomes, np.ones((3, 2), dtype=bool), np.ones((3, 2), dtype=float),
        np.array([0.001, 0.004, 0.03]), np.tile(np.arange(2), (3, 1)), 10.0,
        np.array([0.05, 0.05, 0.90]), 2.0, slots, _records("p", slots, 2), "p", "problem",
    )
    kwargs = dict(value_of_correct=1.0, capture_trace=True, mandatory_scout=False)
    # The value rule prefers oss120 on belief; the oracle takes the one that works.
    assert replay_adaptive(*common, **kwargs)["decision_trace"][0]["chosen_route"] == "oss120"
    oracle = replay_adaptive(*common, oracle_routing=True, **kwargs)
    assert oracle["decision_trace"][0]["chosen_route"] == "oss20"
    assert oracle["decision_trace"][0]["oracle_routed"] is True
    assert oracle["correct"] is True


def test_oracle_routing_falls_back_to_the_policy_when_nothing_succeeds() -> None:
    slots = ["scout", "oss20", "oss120"]
    common = (
        np.zeros((3, 2), dtype=bool), np.ones((3, 2), dtype=bool),
        np.ones((3, 2), dtype=float), np.array([0.001, 0.004, 0.03]),
        np.tile(np.arange(2), (3, 1)), 10.0, np.array([0.05, 0.05, 0.90]), 2.0,
        slots, _records("p", slots, 2), "p", "problem",
    )
    kwargs = dict(value_of_correct=1.0, capture_trace=True, mandatory_scout=False)
    plain = replay_adaptive(*common, **kwargs)
    oracle = replay_adaptive(*common, oracle_routing=True, **kwargs)
    assert [d["chosen_route"] for d in plain["decision_trace"]] == \
           [d["chosen_route"] for d in oracle["decision_trace"]]
    assert all(d["oracle_routed"] is False for d in oracle["decision_trace"])


def test_oracle_routing_does_not_change_when_the_policy_abstains() -> None:
    """Routing help is worthless if the policy has already decided to quit."""
    slots = ["scout", "oss20", "oss120"]
    outcomes = np.zeros((3, 2), dtype=bool)
    outcomes[2, 0] = True
    common = (
        outcomes, np.ones((3, 2), dtype=bool), np.ones((3, 2), dtype=float),
        np.array([0.001, 0.004, 0.03]), np.tile(np.arange(2), (3, 1)), 10.0,
        np.array([0.05, 0.05, 0.90]), 2.0, slots, _records("p", slots, 2), "p", "problem",
    )
    # R tiny -> every action value is negative -> abstain at the root regardless.
    kwargs = dict(value_of_correct=1e-4, capture_trace=True, mandatory_scout=False)
    assert replay_adaptive(*common, **kwargs)["abstained"] is True
    assert replay_adaptive(*common, oracle_routing=True, **kwargs)["abstained"] is True


# --- Stopping on the learned q(s) --------------------------------------------------


def _q_replay(nothing_prob, value_of_correct=1.0, **kwargs):
    """Scorer's 4th output is the `nothing` head: P(no valid draw remains anywhere)."""
    slots = ["scout", "oss20", "oss120"]
    return replay_adaptive(
        np.zeros((3, 3), dtype=bool), np.ones((3, 3), dtype=bool),
        np.ones((3, 3), dtype=float), np.array([0.001, 0.004, 0.03]),
        np.tile(np.arange(3), (3, 1)), 10.0, np.array([0.30, 0.20, 0.60]), 2.0,
        slots, _records("p", slots, 3), "p", "problem",
        scorer=lambda _: np.array([0.30, 0.20, 0.60, nothing_prob]),
        apply_failure_decay=True, value_of_correct=value_of_correct,
        capture_trace=True, mandatory_scout=False, **kwargs,
    )


def test_q_stopping_quits_when_the_learned_head_says_nothing_remains() -> None:
    # nothing=0.95 -> q=0.05, below the threshold, so abstain at the root even
    # though the value rule would happily spend.
    assert _q_replay(0.05, q_abstain=0.10)["abstained"] is False
    doomed = _q_replay(0.95, q_abstain=0.10)
    assert doomed["abstained"] is True
    assert doomed["attempts"] == 0
    assert doomed["decision_trace"][0]["abstain_reason"] == "learned_q_below_threshold"
    assert doomed["decision_trace"][0]["learned_q"] == pytest.approx(0.05)


def test_q_stopping_replaces_rather_than_augments_the_value_stop() -> None:
    """Mirrors the oracle arm: exactly one stopping source is active at a time."""
    # R is tiny, so every action value is negative and the value rule would quit.
    # A confident q keeps the policy going, proving q_abstain took over stopping.
    kept_going = _q_replay(0.01, value_of_correct=1e-4, q_abstain=0.10)
    assert kept_going["abstained"] is False
    assert kept_going["attempts"] > 0


def test_q_stopping_requires_a_nothing_head() -> None:
    slots = ["scout", "oss20", "oss120"]
    with pytest.raises(ValueError, match="`nothing` head"):
        replay_adaptive(
            np.zeros((3, 3), dtype=bool), np.ones((3, 3), dtype=bool),
            np.ones((3, 3), dtype=float), np.ones(3), np.tile(np.arange(3), (3, 1)),
            10.0, np.ones(3) * 0.5, 2.0, slots, _records("p", slots, 3), "p", "problem",
            value_of_correct=1.0, q_abstain=0.1, mandatory_scout=False,
        )


# --- Factorized difficulty model ---------------------------------------------------


def test_factorized_probs_decay_monotonically_and_derive_nothing() -> None:
    import torch
    from pipelinerl.swe.scripts.livecodebench.train_mdp_reachable_policy import (
        factorized_probs,
    )

    # theta = sigmoid(0) = 0.5 per route; softplus(0)+1e-3 ~= 0.694 persistence.
    logits = torch.zeros(3, 6)
    counts = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0]])
    remaining = torch.tensor([[2.0, 0, 0], [2.0, 0, 0], [2.0, 0, 0]])
    probs = factorized_probs(logits, counts, remaining)

    # Decay is monotone in the route's own failure count, by construction.
    assert probs[0, 0] > probs[1, 0] > probs[2, 0]
    # Routes with no failures are unaffected by another route's count.
    assert probs[0, 1] == pytest.approx(probs[2, 1])
    # P(nothing remains) rises as the surviving route decays.
    assert probs[0, 3] < probs[2, 3]


def test_factorized_nothing_is_consistent_with_the_route_beliefs() -> None:
    """The derived head equals the product of per-draw survivals, not a free output."""
    import torch
    from pipelinerl.swe.scripts.livecodebench.train_mdp_reachable_policy import (
        factorized_probs,
    )

    logits = torch.tensor([[0.4, -0.2, 0.1, 0.3, 0.0, -0.1]])
    counts = torch.tensor([[1.0, 0.0, 2.0]])
    remaining = torch.tensor([[2.0, 1.0, 0.0]])
    probs = factorized_probs(logits, counts, remaining)

    theta = torch.sigmoid(logits[:, :3])
    persistence = torch.nn.functional.softplus(logits[:, 3:]) + 1e-3
    expected = 1.0
    for route, draws in enumerate([2, 1, 0]):
        for step in range(draws):
            p = theta[0, route] * persistence[0, route] / (
                persistence[0, route] + counts[0, route] + step
            )
            expected *= float(1.0 - p)
    assert float(probs[0, 3]) == pytest.approx(expected, rel=1e-5)


def test_factorized_training_requires_the_rebuilt_dataset() -> None:
    from pipelinerl.swe.scripts.livecodebench.train_mdp_reachable_policy import (
        PolicyDataset,
    )

    class _Tok:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    stale = [{"problem_id": "p", "failure_depth": 0, "text": "t", "targets": [0, 0, 0, 1]}]
    with pytest.raises(ValueError, match="rebuild the reachable dataset"):
        PolicyDataset(stale, _Tok(), 128, require_state_features=False, factorized=True)


def test_factorized_scorer_rejects_the_analytic_decay() -> None:
    """The guard must fire on the arm itself, so a mistake cannot pass silently."""
    slots = ["scout", "oss20", "oss120"]

    class _Factorized:
        uses_raw_counts = True

        def factorized_beliefs(self, statement, failures, remaining):
            return np.array([0.3, 0.2, 0.6, 0.4])

    with pytest.raises(ValueError, match="double-decays"):
        replay_adaptive(
            np.zeros((3, 3), dtype=bool), np.ones((3, 3), dtype=bool),
            np.ones((3, 3), dtype=float), np.array([0.001, 0.004, 0.03]),
            np.tile(np.arange(3), (3, 1)), 10.0, np.ones(3) * 0.5, 2.0,
            slots, _records("p", slots, 3), "p", "problem",
            scorer=_Factorized(), apply_failure_decay=True,
            value_of_correct=1.0, mandatory_scout=False,
        )

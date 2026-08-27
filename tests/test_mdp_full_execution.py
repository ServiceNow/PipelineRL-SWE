import json
import sys

import numpy as np

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

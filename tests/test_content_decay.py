"""A content prior with no decay cannot abstain, so `content` was never a fair arm.

The activation probe emits one theta vector per problem, before any draw. Held constant
with depth, p_m*R - c_m never falls as draws fail, the stop action never fires, and the
measured arm abstained 0.0% at every operating point above 55% -- it was competing with
the give-up mechanism structurally disabled while the LoRA arms kept theirs.

`content_decay` applies the same analytic s/(s+n) the count family already uses, which is
what separates belief QUALITY from the state-conditioning a re-queried encoder also buys.
"""

from __future__ import annotations

import numpy as np
from pipelinerl.swe.scripts.livecodebench.replay_mdp_full_execution import replay_adaptive

SLOTS = ["scout", "oss20", "oss120"]


def _episode(*, content_prior, apply_failure_decay, value_of_correct, seed=0):
    """One problem nothing can solve, so every route keeps failing and depth accumulates."""
    K = 6
    M = len(SLOTS)
    outcomes = np.zeros((M, K), dtype=bool)
    valid = np.ones((M, K), dtype=bool)
    realized = np.full((M, K), 0.001, dtype=float)
    expected = np.array([0.0004, 0.003, 0.02], dtype=float)
    orderings = np.tile(np.arange(K), (M, 1))
    return replay_adaptive(
        outcomes, valid, realized, expected, orderings,
        budget=1.0, prior=np.array([0.5, 0.55, 0.8]), pseudo_count=2.0, slots=SLOTS,
        records={("p", slot, k): {"model_slot": slot, "code": "print(1)",
                                  "full_execution_feedback": "failed"}
                 for slot in SLOTS for k in range(K)},
        problem_id="p", problem_statement="stmt",
        content_prior=np.asarray(content_prior, dtype=float),
        value_of_correct=value_of_correct,
        apply_failure_decay=apply_failure_decay,
        capture_trace=True,
        mandatory_scout=False,
    )


def test_undecayed_content_never_abstains_on_a_hopeless_problem():
    r = _episode(content_prior=[0.5, 0.55, 0.8], apply_failure_decay=False,
                 value_of_correct=0.05)
    assert not r["abstained"]
    assert r["attempts"] == 18, "beliefs are frozen, so it buys every available draw"


def test_decayed_content_gives_up():
    r = _episode(content_prior=[0.5, 0.55, 0.8], apply_failure_decay=True,
                 value_of_correct=0.05)
    assert r["abstained"]
    assert r["attempts"] < 18


def test_decay_labels_the_belief_source():
    r = _episode(content_prior=[0.5, 0.55, 0.8], apply_failure_decay=True,
                 value_of_correct=0.05)
    sources = {step.get("belief_source") for step in r["decision_trace"]}
    assert sources == {"content_decay"}


def test_decayed_beliefs_fall_monotonically_with_failures():
    r = _episode(content_prior=[0.5, 0.55, 0.8], apply_failure_decay=True,
                 value_of_correct=1.0)
    seen: dict[int, list[float]] = {}
    for step in r["decision_trace"]:
        probs = step.get("p_success_next")
        if not probs:
            continue
        for mi, slot in enumerate(SLOTS):
            seen.setdefault(mi, []).append(float(probs[slot]))
    assert seen, "trace must expose per-route beliefs"
    for mi, series in seen.items():
        assert all(b <= a + 1e-9 for a, b in zip(series, series[1:])), (
            f"route {mi} beliefs must not rise as draws fail: {series}"
        )


def test_undecayed_beliefs_are_constant():
    r = _episode(content_prior=[0.5, 0.55, 0.8], apply_failure_decay=False,
                 value_of_correct=1.0)
    series = [tuple(sorted(step["p_success_next"].items()))
              for step in r["decision_trace"] if step.get("p_success_next")]
    assert len(set(series)) == 1

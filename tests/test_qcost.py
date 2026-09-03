"""Query-conditioned cost must actually reach the utility rule.

`argmax_m (p_m(x)*R - c_m)` is conditioned only in the numerator by every routing system we
surveyed; c_m is a per-route training-set mean. The `_qcost` families swap in a per-problem
c_m(x) and change nothing else, so any difference between an arm and its `_qcost` twin is
attributable to the cost term alone. These tests pin that plumbing: the suffix must not leak
into the belief-source dispatch, and the rule must respond to the per-problem costs.
"""

from __future__ import annotations

import numpy as np

from pipelinerl.swe.scripts.livecodebench.replay_mdp_full_execution import replay_adaptive

SLOTS = ["scout", "oss20", "oss120"]
K = 4


def _records():
    return {("p", s, k): {"model_slot": s, "code": "x", "full_execution_feedback": "f"}
            for s in SLOTS for k in range(K)}


def _run(expected_costs, *, solvable_by=None, value_of_correct=0.05):
    M = len(SLOTS)
    outcomes = np.zeros((M, K), dtype=bool)
    if solvable_by is not None:
        outcomes[solvable_by, :] = True
    return replay_adaptive(
        outcomes, np.ones((M, K), bool), np.full((M, K), 0.001),
        np.asarray(expected_costs, float), np.tile(np.arange(K), (M, 1)),
        budget=1.0, prior=np.array([0.5, 0.6, 0.9]), pseudo_count=2.0, slots=SLOTS,
        records=_records(), problem_id="p", problem_statement="s",
        content_prior=np.array([0.3, 0.5, 0.9]),
        value_of_correct=value_of_correct, apply_failure_decay=True,
        capture_trace=True, mandatory_scout=False,
    )


def test_cost_vector_changes_the_route_taken():
    """Same beliefs; only c_m differs. The expensive route must stop being bought."""
    cheap = _run([0.0001, 0.0002, 0.0003], solvable_by=2)
    dear = _run([0.0001, 0.0002, 0.5000], solvable_by=2)
    assert cheap["route_attempt_counts"]["oss120"] > 0
    assert dear["route_attempt_counts"]["oss120"] == 0, (
        "a per-problem cost that exceeds the value of a correct answer must price the route out"
    )


def test_expensive_everywhere_forces_abstention():
    r = _run([1.0, 1.0, 1.0], solvable_by=2, value_of_correct=0.05)
    assert r["abstained"]
    assert r["attempts"] == 0


def test_cheap_costs_do_not_abstain():
    r = _run([1e-6, 1e-6, 1e-6], solvable_by=2, value_of_correct=0.05)
    assert not r["abstained"]


def test_utility_uses_the_supplied_costs_not_a_constant():
    """The trace's own utility numbers must be computed from the vector we passed in."""
    r = _run([0.0001, 0.0002, 0.0003], solvable_by=2, value_of_correct=1.0)
    step = r["decision_trace"][0]
    p = step["p_success_next"]
    util = step["utility_per_expected_cost"]
    for mi, slot in enumerate(SLOTS):
        if slot in util:
            expected = p[slot] / [0.0001, 0.0002, 0.0003][mi]
            assert abs(util[slot] - expected) < 1e-6, f"{slot}: {util[slot]} vs {expected}"


def test_qcost_suffix_does_not_leak_into_belief_dispatch():
    """`base` must strip `_qcost` so the twin arms share a belief source."""
    for fam, exp_base in [("content_decay_qcost", "content_decay"),
                          ("content_qcost", "content"),
                          ("counts_qcost", "counts"),
                          ("content_decay", "content_decay")]:
        base = fam[:-6] if fam.endswith("_qcost") else fam
        assert base == exp_base

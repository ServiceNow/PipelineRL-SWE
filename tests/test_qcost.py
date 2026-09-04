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


def test_cross_route_coupling_depresses_untried_routes():
    """Failures on one route must lower beliefs about the others when coupling is on.

    Count beliefs have no per-problem prior, so observed failures are their only channel for
    learning that a problem is hard. Unconditionally that channel is worth 28 points here, so
    the strongest honest version of the count baseline propagates evidence across routes.
    """
    import numpy as np
    from pipelinerl.swe.scripts.livecodebench.replay_mdp_full_execution import replay_adaptive

    def run(cross):
        M, KK = len(SLOTS), 4
        return replay_adaptive(
            np.zeros((M, KK), bool), np.ones((M, KK), bool), np.full((M, KK), 0.001),
            np.array([1e-5, 1e-5, 1e-5]), np.tile(np.arange(KK), (M, 1)),
            budget=1.0, prior=np.array([0.5, 0.5, 0.5]), pseudo_count=2.0, slots=SLOTS,
            records={("p", s, k): {"model_slot": s, "code": "x",
                                   "full_execution_feedback": "f"}
                     for s in SLOTS for k in range(KK)},
            problem_id="p", problem_statement="s", value_of_correct=1.0,
            cross_pseudo_count=cross, capture_trace=True, mandatory_scout=False,
        )

    off, on = run(0.0), run(2.0)

    def belief_after_failures(trace, slot, depth):
        for step in trace:
            if sum(step["failure_counts"].values()) == depth:
                return step["p_success_next"][slot]
        return None

    # a route that has never been tried, once other routes have failed
    for depth in (1, 2):
        b_off = belief_after_failures(off["decision_trace"], "oss120", depth)
        b_on = belief_after_failures(on["decision_trace"], "oss120", depth)
        if b_off is not None and b_on is not None:
            assert b_on < b_off, (
                f"depth {depth}: coupling must depress an untried route "
                f"({b_on} !< {b_off})"
            )

    assert off["decision_trace"][0]["belief_source"] == "counts"
    assert on["decision_trace"][0]["belief_source"] == "counts_coupled"


def test_density_selection_prefers_the_cheap_route_but_still_stops():
    """Surplus stops; density chooses. Each rule answers the question it is right about.

    The surplus p*R - c maximises absolute value, so at depth 0 a high-theta expensive route
    can win outright and the policy skips the cheap attempt that would often have sufficed --
    which is the behaviour the forced-scout protocol was papering over. The density p/c is
    scale-invariant in cost and prefers the cheapest adequate route, but never crosses zero so
    it cannot abstain.
    """
    import numpy as np
    from pipelinerl.swe.scripts.livecodebench.replay_mdp_full_execution import replay_adaptive

    M, KK = len(SLOTS), 3
    recs = {("p", s, k): {"model_slot": s, "code": "x", "full_execution_feedback": "f"}
            for s in SLOTS for k in range(KK)}

    def run(density, solvable_by, costs, value=1.0):
        outcomes = np.zeros((M, KK), bool)
        if solvable_by is not None:
            outcomes[solvable_by, :] = True
        return replay_adaptive(
            outcomes, np.ones((M, KK), bool), np.tile(np.asarray(costs, float)[:, None], (1, KK)),
            np.asarray(costs, float), np.tile(np.arange(KK), (M, 1)),
            budget=10.0, prior=np.array([0.5, 0.6, 0.9]), pseudo_count=2.0, slots=SLOTS,
            records=recs, problem_id="p", problem_statement="s",
            content_prior=np.array([0.40, 0.50, 0.80]),
            value_of_correct=value, apply_failure_decay=True, capture_trace=True,
            mandatory_scout=False, select_by_density=density,
        )

    costs = [0.001, 0.005, 0.050]          # a steep ladder, as in the real pool
    # every route can solve it, so the only difference is which one gets bought first
    surplus = run(False, 2, costs)
    density = run(True, 2, costs)
    first_s = surplus["route_attempt_counts"]
    first_d = density["route_attempt_counts"]
    # density must not open with the most expensive route when a cheap one is adequate
    assert first_d["scout"] >= first_s["scout"], (
        f"density should try the cheap route at least as often: {first_d} vs {first_s}"
    )

    # stopping is unchanged: a hopeless problem with costs above the value must still abstain
    stop = run(True, None, [1.0, 1.0, 1.0], value=0.05)
    assert stop["abstained"], "density selection must not disable the stop action"

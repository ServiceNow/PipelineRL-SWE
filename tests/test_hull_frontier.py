"""The randomised frontier must be the convex hull, not the raw grid.

The frontier metric asks for the cheapest way to reach an accuracy target. Because operating
points can be mixed per episode, a point sitting below the chord between two others is never
the cheapest way to reach anything, and reporting it is what made three targets swing with the
draw-ordering seed.
"""
from __future__ import annotations

from pipelinerl.swe.scripts.livecodebench.hull_frontier import (
    advantage, cost_at, hull, policy_family,
)


def _pts(family, pts):
    return [{"policy": family, "mean_realized_cost": c, "correctness": a} for c, a in pts]


def test_family_ignores_arm_suffixes():
    for p in ("counts", "counts_value", "counts_abstain", "counts_value_frozen"):
        assert policy_family(p) == "counts"
    assert policy_family("content_decay_qcost_value") == "content_decay_qcost"


def test_a_point_below_the_chord_is_dropped():
    """(2.0, 0.55) is worse than mixing (1.0, 0.50) with (3.0, 0.70)."""
    v = hull(_pts("f", [(1.0, 0.50), (2.0, 0.55), (3.0, 0.70)]), "f")
    assert v == [(1.0, 0.50), (3.0, 0.70)]


def test_a_point_above_the_chord_is_kept():
    v = hull(_pts("f", [(1.0, 0.50), (2.0, 0.65), (3.0, 0.70)]), "f")
    assert v == [(1.0, 0.50), (2.0, 0.65), (3.0, 0.70)]


def test_dominated_points_are_dropped():
    """Costlier and no more accurate is never worth buying."""
    v = hull(_pts("f", [(1.0, 0.50), (2.0, 0.40), (3.0, 0.70)]), "f")
    assert v == [(1.0, 0.50), (3.0, 0.70)]


def test_cost_interpolates_between_vertices():
    """Halfway in accuracy is halfway in cost, because you flip a fair coin."""
    v = [(1.0, 0.50), (3.0, 0.70)]
    assert cost_at(v, 0.60) == 2.0
    assert cost_at(v, 0.50) == 1.0
    assert cost_at(v, 0.70) == 3.0


def test_unreachable_target_returns_none():
    assert cost_at([(1.0, 0.50), (3.0, 0.70)], 0.80) is None


def test_the_hull_removes_the_grid_artifact():
    """Same policy class, one grid missing a point near the target.

    On the raw grid the sparse family looks 33% worse at 60% accuracy, because its cheapest
    arm REACHING 60% is the 0.70 point. Mixing recovers the truth: the two families are
    identical, so the advantage is zero.
    """
    dense = _pts("counts", [(1.0, 0.50), (2.0, 0.60), (3.0, 0.70)])
    sparse = _pts("content_decay_qcost", [(1.0, 0.50), (3.0, 0.70)])
    assert advantage(dense + sparse, "counts", "content_decay_qcost", 0.60) == 0.0


def test_advantage_is_positive_when_we_are_genuinely_cheaper():
    base = _pts("counts", [(2.0, 0.50), (4.0, 0.70)])
    ours = _pts("content_decay_qcost", [(1.0, 0.50), (3.0, 0.70)])
    assert advantage(base + ours, "counts", "content_decay_qcost", 0.60) == 33.0 + 1.0 / 3.0

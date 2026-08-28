"""Machine-readable features for full-execution routing states.

The language-model prompt retains its human-readable execution summary.  This
module additionally exposes the same routing state as a small normalized
numeric vector, so a policy head need not infer arithmetic from text.
"""

from __future__ import annotations

from typing import Any


STATE_FEATURE_VERSION = "structured_v1"
STATE_FEATURE_NAMES = (
    "scout_failed_fraction",
    "oss20_failed_fraction",
    "oss120_failed_fraction",
    "scout_remaining_fraction",
    "oss20_remaining_fraction",
    "oss120_remaining_fraction",
    "total_failure_fraction",
    "latest_route_none",
    "latest_route_scout",
    "latest_route_oss20",
    "latest_route_oss120",
)


def build_structured_state_features(
    attempts: list[dict[str, Any]],
    remaining: dict[str, int],
    route_capacities: dict[str, int],
    slots: list[str],
) -> list[float]:
    """Return normalized counts plus a latest-route one-hot vector.

    Capacities are the number of valid pre-generated draws for this problem and
    route. They are known to the replay protocol at the beginning of an
    episode. All router states are failure-only, so ``failed + remaining`` is
    never affected by a hidden success.
    """
    if tuple(slots) != ("scout", "oss20", "oss120"):
        raise ValueError(f"Unexpected routing slots: {slots}")
    failed = {
        slot: sum(str(attempt["model_slot"]) == slot for attempt in attempts)
        for slot in slots
    }
    capacities = {slot: max(1, int(route_capacities[slot])) for slot in slots}
    values = [float(failed[slot]) / capacities[slot] for slot in slots]
    values.extend(float(remaining[slot]) / capacities[slot] for slot in slots)
    values.append(float(sum(failed.values())) / max(1, sum(capacities.values())))
    latest = str(attempts[-1]["model_slot"]) if attempts else "none"
    values.extend(float(latest == route) for route in ("none", *slots))
    if len(values) != len(STATE_FEATURE_NAMES):
        raise AssertionError("Structured feature dimension drifted")
    return values

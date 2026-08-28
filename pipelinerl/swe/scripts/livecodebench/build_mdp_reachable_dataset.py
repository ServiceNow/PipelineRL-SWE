#!/usr/bin/env python3
"""Build full-execution policy examples from reachable failure-only histories.

The start protocol is either scout-first or free-start. A passing attempt
terminates the trajectory, so no post-pass state can enter training or
evaluation. The policy text contains explicit per-route counts plus only the
latest failed attempt; the complete ordered history is retained separately as
provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pipelinerl.swe.scripts.livecodebench.mdp_utils import load_split_manifest
from pipelinerl.swe.scripts.livecodebench.structured_state import (
    STATE_FEATURE_NAMES,
    STATE_FEATURE_VERSION,
    build_structured_state_features,
)


HEADS = ["scout_next", "oss20_fresh", "oss120_fresh", "nothing"]
MAX_LATEST_ATTEMPT_CHARS = 8000


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _render_state(
    problem_statement: str,
    attempts: list[dict[str, Any]],
    remaining: dict[str, int],
    state_layout: str = "counts_last",
) -> str:
    """Render a policy state.

    `state_layout` controls where the execution-state block sits.

    The encoder pools `last_hidden_state[:, -1]`, so the read-out token is
    whatever ends the prompt. Under the original `problem_first` layout that was
    the last token of a truncated code block, leaving the failure counts hundreds
    to thousands of tokens upstream. The measured symptom was a model with strong
    problem discrimination (within-depth AUC 0.834 on oss120) and almost no depth
    sensitivity: predictions moved 0.180 -> 0.107 across failure depths 1-10 while
    the true scout hazard went 0.041 -> 0.000.

    `counts_last` puts the execution state immediately before the pooled token so
    the decay signal is adjacent to the read-out. Retained `problem_first` for
    reproducing the earlier artifacts.
    """
    if state_layout not in ("problem_first", "counts_last"):
        raise ValueError(f"Unknown state layout: {state_layout}")
    failure_counts = {
        slot: sum(attempt["model_slot"] == slot for attempt in attempts)
        for slot in remaining
    }
    latest = attempts[-1] if attempts else None
    execution_lines = [
        f"{slot}: failed={failure_counts[slot]}, remaining={count}"
        for slot, count in remaining.items()
    ]
    execution_lines.append(
        f"latest_attempted_route: {latest['model_slot'] if latest is not None else 'none'}"
    )
    execution_lines.append(f"total_failures: {len(attempts)}")
    execution_block = "[Execution state]\n" + "\n".join(execution_lines)
    attempt_block = (
        None if latest is None else
        f"[Latest verified failed attempt: {latest['model_slot']}]\n"
        f"[Full execution] {latest['full_execution_feedback']}\n"
        f"[Code]\n{latest['code'][:MAX_LATEST_ATTEMPT_CHARS]}"
    )
    parts = ["[Problem]\n" + problem_statement]
    if state_layout == "problem_first":
        parts.append(execution_block)
        if attempt_block is not None:
            parts.append(attempt_block)
    else:
        if attempt_block is not None:
            parts.append(attempt_block)
        parts.append(execution_block)
    return "\n\n".join(parts)


def _remaining_draw_counts(
    valid: np.ndarray,
    orders: np.ndarray,
    ptr: np.ndarray,
    slots: list[str],
) -> dict[str, int]:
    """Count only valid draws still available to each route."""
    return {
        slot: sum(
            bool(valid[mi, int(draw)])
            for draw in orders[mi, int(ptr[mi]):]
        )
        for mi, slot in enumerate(slots)
    }


def _nothing_remaining_target(
    outcomes: np.ndarray,
    valid: np.ndarray,
    orders: np.ndarray,
    ptr: np.ndarray,
) -> float:
    """Return one iff every still-available valid draw is unsuccessful."""
    for mi in range(outcomes.shape[0]):
        for position in range(int(ptr[mi]), orders.shape[1]):
            draw = int(orders[mi, position])
            if valid[mi, draw] and outcomes[mi, draw]:
                return 0.0
    return 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensors-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--histories-per-problem", type=int, default=10)
    parser.add_argument("--max-failures", type=int, default=10)
    parser.add_argument(
        "--start-protocol", choices=["scout_first", "free_start"], default="scout_first"
    )
    parser.add_argument(
        "--state-layout", choices=["problem_first", "counts_last"], default="counts_last",
        help="Where the execution-state block sits relative to the pooled read-out token",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tensors_dir = Path(args.tensors_dir)
    tensor_data = np.load(tensors_dir / "tensors.npz", allow_pickle=True)
    if "final_outcome" not in tensor_data:
        raise ValueError("Reachable-state construction requires a schema-v2 tensor bundle")
    outcomes = tensor_data["final_outcome"].astype(bool)
    valid = tensor_data["valid"].astype(bool)
    slots = [str(value) for value in tensor_data["model_slots"]]
    pids = [str(value) for value in tensor_data["problem_ids"]]
    if slots != ["scout", "oss20", "oss120"]:
        raise ValueError(f"Unexpected model slots: {slots}")

    problems = {row["problem_id"]: row for row in _read_jsonl(tensors_dir / "problems.jsonl")}
    records = {
        (row["problem_id"], row["model_slot"], int(row["draw_index"])): row
        for row in _read_jsonl(tensors_dir / "draw_records.jsonl")
    }
    manifest = load_split_manifest(tensors_dir / "split_manifest.json", pids)
    split_for = {
        pid: split
        for split in ("train", "calibration", "test")
        for pid in manifest[f"{split}_problem_ids"]
    }

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    terminated_on_pass = 0
    skipped_initial_pass = 0
    K = outcomes.shape[2]

    for pi, pid in enumerate(pids):
        statement = str(problems[pid].get("problem_statement") or "").strip()
        if not statement:
            raise ValueError(f"Missing problem statement for {pid}")
        for history_index in range(args.histories_per_problem):
            orders = np.array([rng.permutation(K) for _ in slots])
            ptr = np.zeros(len(slots), dtype=int)
            attempts: list[dict[str, Any]] = []

            if args.start_protocol == "scout_first":
                # A mandatory scout success resolves before any decision state exists.
                while ptr[0] < K and not valid[pi, 0, int(orders[0, ptr[0]])]:
                    ptr[0] += 1
                if ptr[0] >= K:
                    continue
                scout_draw = int(orders[0, ptr[0]])
                ptr[0] += 1
                if outcomes[pi, 0, scout_draw]:
                    skipped_initial_pass += 1
                    continue
                attempts.append(records[(pid, "scout", scout_draw)])

            while len(attempts) <= args.max_failures:
                next_draws: list[int | None] = []
                targets: list[float] = []
                for mi in range(len(slots)):
                    while ptr[mi] < K and not valid[pi, mi, int(orders[mi, ptr[mi]])]:
                        ptr[mi] += 1
                    if ptr[mi] >= K:
                        next_draws.append(None)
                        targets.append(0.0)
                    else:
                        draw = int(orders[mi, ptr[mi]])
                        next_draws.append(draw)
                        targets.append(float(outcomes[pi, mi, draw]))
                available = [mi for mi, draw in enumerate(next_draws) if draw is not None]
                if not available:
                    break
                remaining = _remaining_draw_counts(valid[pi], orders, ptr, slots)
                route_capacities = {
                    slot: int(valid[pi, mi].sum()) for mi, slot in enumerate(slots)
                }
                rows.append({
                    "problem_id": pid,
                    "split": split_for[pid],
                    "history_index": history_index,
                    "failure_depth": len(attempts),
                    "text": _render_state(statement, attempts, remaining, args.state_layout),
                    "state_features": build_structured_state_features(
                        attempts, remaining, route_capacities, slots
                    ),
                    "targets": targets + [_nothing_remaining_target(
                        outcomes[pi], valid[pi], orders, ptr
                    )],
                    "target_draw_indices": next_draws,
                    "attempt_history": [
                        {"model_slot": row["model_slot"], "draw_index": row["draw_index"]}
                        for row in attempts
                    ],
                })

                # Random behavior policy supplies diverse reachable histories.
                chosen = int(rng.choice(available))
                chosen_draw = int(next_draws[chosen])
                ptr[chosen] += 1
                if outcomes[pi, chosen, chosen_draw]:
                    terminated_on_pass += 1
                    break
                attempts.append(records[(pid, slots[chosen], chosen_draw)])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {}
    problem_counts: dict[str, int] = {}
    for split in ("train", "calibration", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        split_counts[split] = len(split_rows)
        problem_counts[split] = len({row["problem_id"] for row in split_rows})
        with open(output_dir / f"{split}.jsonl", "w") as handle:
            for row in split_rows:
                handle.write(json.dumps(row) + "\n")
    summary = {
        "protocol": f"{args.start_protocol}_full_execution_failure_region",
        "state_representation": "problem_counts_latest_failed_attempt",
        "state_layout": args.state_layout,
        "structured_state_feature_version": STATE_FEATURE_VERSION,
        "structured_state_feature_names": STATE_FEATURE_NAMES,
        "nothing_target": "no_successful_valid_draw_remains_in_any_route",
        "heads": HEADS,
        "histories_per_problem": int(args.histories_per_problem),
        "max_failures": int(args.max_failures),
        "seed": int(args.seed),
        "examples": split_counts,
        "problems_represented": problem_counts,
        "initial_scout_pass_trajectories_excluded": skipped_initial_pass,
        "later_pass_terminations": terminated_on_pass,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

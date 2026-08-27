#!/usr/bin/env python3
"""Replay resample/escalate/abstain policies with full execution.

The start protocol is either scout-first or free-start. Free-start exposes
{abstain, scout, oss20, oss120} at the root and after every failed attempt. A
verified pass terminates immediately. Decisions use expected model cost
estimated on the train split; reports charge realized prompt+completion tokens.

The replay also writes per-episode decision traces, aggregate route-choice
diagnostics, and paired confidence intervals clustered by problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pipelinerl.swe.scripts.livecodebench.build_mdp_reachable_dataset import (
    _remaining_draw_counts,
    _render_state,
)
from pipelinerl.swe.scripts.livecodebench.mdp_utils import load_split_manifest, split_indices


USD_PER_M_TOKENS = {"scout": 0.278, "oss20": 1.299, "oss120": 11.13}
COST_WEIGHTS = {"scout": 1.0, "oss20": 5.0, "oss120": 30.0}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _state_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class TorchSequentialScorer:
    """Load a saved Qwen embedding policy and score replay states on demand."""

    def __init__(self, model_dir: Path) -> None:
        import torch
        import torch.nn.functional as F
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.F = F
        config = json.loads((model_dir / "train_config.json").read_text())
        model_name = str(config["model_name"])
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if torch.cuda.is_available():
            kwargs["attn_implementation"] = config.get("attn_implementation", "flash_attention_2")
        base = AutoModel.from_pretrained(model_name, **kwargs)
        adapter_dir = model_dir / "encoder" / "reward_adapter"
        self.encoder = PeftModel.from_pretrained(base, str(adapter_dir), adapter_name="reward_adapter")
        head_state = torch.load(model_dir / "reward_head.pt", map_location="cpu", weights_only=True)
        hidden_size = int(self.encoder.config.hidden_size)
        first_weight = head_state.get("1.weight")
        if first_weight is None:
            raise ValueError("Unsupported reward-head layout: missing 1.weight")
        mlp_hidden = int(first_weight.shape[0])
        target_dim = int(head_state["4.weight"].shape[0])
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(0.0),
            torch.nn.Linear(hidden_size, mlp_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(mlp_hidden, target_dim),
        )
        self.head.load_state_dict(head_state)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"), padding_side="left")
        self.max_length = int(config.get("max_seq_length", 8192))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device).eval()
        self.head.to(self.device).eval()
        self.cache: dict[str, np.ndarray] = {}

    def __call__(self, text: str) -> np.ndarray:
        key = _state_key(text)
        if key in self.cache:
            return self.cache[key]
        torch = self.torch
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = outputs.last_hidden_state[:, -1]
            pooled = self.F.normalize(pooled.float(), p=2, dim=1)
            probs = torch.sigmoid(self.head(pooled)).cpu().numpy()[0]
        self.cache[key] = probs
        return probs


def _next_valid_draw(order: np.ndarray, ptr: int, valid: np.ndarray) -> tuple[int | None, int]:
    while ptr < len(order) and not valid[int(order[ptr])]:
        ptr += 1
    return (int(order[ptr]), ptr) if ptr < len(order) else (None, ptr)


def replay_adaptive(
    outcomes: np.ndarray,
    valid: np.ndarray,
    realized_costs: np.ndarray,
    expected_costs: np.ndarray,
    orderings: np.ndarray,
    budget: float,
    prior: np.ndarray,
    pseudo_count: float,
    slots: list[str],
    records: dict[tuple[str, str, int], dict[str, Any]],
    problem_id: str,
    problem_statement: str,
    *,
    content_prior: np.ndarray | None = None,
    scorer: Callable[[str], np.ndarray] | None = None,
    tau_abstain: float | None = None,
    min_success_per_cost: float | None = None,
    value_of_correct: float | None = None,
    capture_trace: bool = False,
    apply_failure_decay: bool = False,
    mandatory_scout: bool = True,
) -> dict[str, Any]:
    controls = (tau_abstain, min_success_per_cost, value_of_correct)
    if sum(control is not None for control in controls) > 1:
        raise ValueError(
            "Choose at most one of probability-threshold, marginal-value, or value-of-correct control"
        )
    ptr = np.zeros(len(slots), dtype=int)
    failures = np.zeros(len(slots), dtype=int)
    spent_budget = 0.0
    realized_spend = 0.0
    attempts: list[dict[str, Any]] = []
    route_attempt_counts = {slot: 0 for slot in slots}
    decision_trace: list[dict[str, Any]] = []
    entered_router = not mandatory_scout

    def finish(correct: bool, abstained: bool) -> dict[str, Any]:
        result = {
            "correct": correct,
            "entered_router": entered_router,
            "abstained": abstained,
            "start_protocol": "scout_first" if mandatory_scout else "free_start",
            "budget_spend": spent_budget,
            "realized_spend": realized_spend,
            "attempts": int(sum(route_attempt_counts.values())),
            "route_attempt_counts": route_attempt_counts.copy(),
        }
        if capture_trace:
            result["decision_trace"] = decision_trace
        return result

    def attempt(mi: int) -> tuple[bool | None, int | None]:
        nonlocal spent_budget, realized_spend
        draw, new_ptr = _next_valid_draw(orderings[mi], int(ptr[mi]), valid[mi])
        ptr[mi] = new_ptr
        if draw is None or spent_budget + expected_costs[mi] > budget:
            return None, draw
        ptr[mi] += 1
        spent_budget += float(expected_costs[mi])
        realized_spend += float(realized_costs[mi, draw])
        route_attempt_counts[slots[mi]] += 1
        if outcomes[mi, draw]:
            return True, draw
        failures[mi] += 1
        attempts.append(records[(problem_id, slots[mi], draw)])
        return False, draw

    if mandatory_scout:
        # Scout-first ablation: a verified pass resolves before any policy decision.
        first, _ = attempt(0)
        if first is True:
            return finish(True, False)
        if first is None:
            return finish(False, False)
        entered_router = True

    while True:
        available: list[int] = []
        for mi in range(len(slots)):
            draw, new_ptr = _next_valid_draw(orderings[mi], int(ptr[mi]), valid[mi])
            ptr[mi] = new_ptr
            if draw is not None and spent_budget + expected_costs[mi] <= budget:
                available.append(mi)
        if not available:
            break

        remaining = _remaining_draw_counts(valid, orderings, ptr, slots)
        state_text = (
            _render_state(problem_statement, attempts, remaining)
            if scorer is not None or capture_trace else None
        )
        if scorer is not None:
            if state_text is None:
                raise AssertionError("Sequential scoring requires rendered state text")
            all_probs = np.asarray(scorer(state_text), dtype=float)
            p_each = all_probs[: len(slots)]
            if apply_failure_decay:
                p_each = p_each * pseudo_count / (pseudo_count + failures)
                p_any = 1.0 - float(np.prod(1.0 - p_each))
                belief_source = "sequential_decay"
            else:
                p_any = (
                    1.0 - float(all_probs[len(slots)])
                    if len(all_probs) > len(slots)
                    else 1.0 - float(np.prod(1.0 - p_each))
                )
                belief_source = "sequential"
        elif content_prior is not None:
            p_each = np.asarray(content_prior, dtype=float)
            p_any = 1.0 - float(np.prod(1.0 - p_each))
            belief_source = "content"
        else:
            p_each = np.asarray([
                pseudo_count * prior[mi] / (pseudo_count + failures[mi])
                for mi in range(len(slots))
            ])
            p_any = 1.0 - float(np.prod(1.0 - p_each))
            belief_source = "counts"

        ratios = {mi: float(p_each[mi] / expected_costs[mi]) for mi in available}
        # The ratio is scale-invariant, so it ranks routes identically at every
        # valuation and always prefers the cheapest. The dollar difference does not.
        action_values = (
            {
                mi: float(p_each[mi] * value_of_correct - expected_costs[mi])
                for mi in available
            }
            if value_of_correct is not None
            else None
        )
        selection = ratios if action_values is None else action_values

        decision: dict[str, Any] | None = None
        if capture_trace:
            decision = {
                "failure_depth": int(failures.sum()),
                "state_key": _state_key(state_text) if state_text is not None else None,
                "belief_source": belief_source,
                "failure_counts": {
                    slot: int(failures[mi]) for mi, slot in enumerate(slots)
                },
                "remaining_valid_draws": remaining,
                "available_routes": [slots[mi] for mi in available],
                "spent_budget_before": float(spent_budget),
                "remaining_budget_before": float(budget - spent_budget),
                "p_success_next": {
                    slot: float(p_each[mi]) for mi, slot in enumerate(slots)
                },
                "p_any_remaining": float(p_any),
                "utility_per_expected_cost": {
                    slots[mi]: ratios[mi] for mi in available
                },
                "action_value": (
                    None if action_values is None
                    else {slots[mi]: action_values[mi] for mi in available}
                ),
                "tau_abstain": tau_abstain,
                "min_success_per_cost": min_success_per_cost,
                "value_of_correct": value_of_correct,
            }

        probability_stop = tau_abstain is not None and p_any <= tau_abstain
        marginal_stop = (
            min_success_per_cost is not None
            and max(ratios.values()) <= min_success_per_cost
        )
        # Under value control, stopping needs no separate threshold: abstaining is
        # the zero-value action, so quit when no route has positive dollar value.
        value_stop = action_values is not None and max(action_values.values()) <= 0.0
        if probability_stop or marginal_stop or value_stop:
            if decision is not None:
                decision.update({
                    "chosen_route": None,
                    "chosen_draw_index": None,
                    "result": "abstain",
                    "abstain_reason": (
                        "probability_threshold" if probability_stop
                        else "marginal_value" if marginal_stop
                        else "non_positive_action_value"
                    ),
                })
                decision_trace.append(decision)
            return finish(False, True)

        chosen = max(available, key=selection.__getitem__)
        result, chosen_draw = attempt(chosen)
        if decision is not None:
            decision.update({
                "chosen_route": slots[chosen],
                "chosen_draw_index": chosen_draw,
                "result": (
                    "pass" if result is True
                    else "fail" if result is False
                    else "unavailable"
                ),
            })
            decision_trace.append(decision)
        if result is True:
            return finish(True, False)

    return finish(False, False)


def replay_fixed(
    outcomes: np.ndarray,
    valid: np.ndarray,
    realized_costs: np.ndarray,
    orderings: np.ndarray,
    plan: list[int],
) -> dict[str, Any]:
    ptr = np.zeros(outcomes.shape[0], dtype=int)
    spend = 0.0
    attempts = 0
    for mi in plan:
        draw, new_ptr = _next_valid_draw(orderings[mi], int(ptr[mi]), valid[mi])
        ptr[mi] = new_ptr
        if draw is None:
            continue
        ptr[mi] += 1
        attempts += 1
        spend += float(realized_costs[mi, draw])
        if outcomes[mi, draw]:
            return {"correct": True, "realized_spend": spend, "attempts": attempts}
    return {"correct": False, "realized_spend": spend, "attempts": attempts}


def _aggregate(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    entered = [row for row in outputs if row.get("entered_router")]
    return {
        "correctness": float(np.mean([row["correct"] for row in outputs])),
        "mean_realized_cost": float(np.mean([row["realized_spend"] for row in outputs])),
        "mean_attempts": float(np.mean([row["attempts"] for row in outputs])),
        "router_entry_rate": float(np.mean([bool(row.get("entered_router")) for row in outputs])),
        "conditional_correctness_after_scout_failure": (
            float(np.mean([row["correct"] for row in entered])) if entered else None
        ),
        "abstention_rate_after_scout_failure": (
            float(np.mean([bool(row.get("abstained")) for row in entered])) if entered else None
        ),
        "overall_abstention_rate": float(
            np.mean([bool(row.get("abstained")) for row in outputs])
        ),
        "n_episodes": len(outputs),
    }


def _summarize_decisions(
    outputs: list[dict[str, Any]], slots: list[str]
) -> dict[str, Any]:
    decisions = [
        decision
        for output in outputs
        for decision in output.get("decision_trace", [])
    ]
    choice_counts = {slot: 0 for slot in slots}
    pass_counts = {slot: 0 for slot in slots}
    probability_values = {slot: [] for slot in slots}
    p_any_values: list[float] = []
    for decision in decisions:
        for slot in slots:
            probability_values[slot].append(float(decision["p_success_next"][slot]))
        p_any_values.append(float(decision["p_any_remaining"]))
        chosen = decision.get("chosen_route")
        if chosen in choice_counts:
            choice_counts[chosen] += 1
            if decision.get("result") == "pass":
                pass_counts[chosen] += 1
    return {
        "n_decisions": len(decisions),
        "route_choice_counts": choice_counts,
        "route_pass_counts": pass_counts,
        "abstain_decisions": sum(
            decision.get("result") == "abstain" for decision in decisions
        ),
        "mean_predicted_next_success": {
            slot: float(np.mean(values)) if values else None
            for slot, values in probability_values.items()
        },
        "mean_predicted_any_remaining": (
            float(np.mean(p_any_values)) if p_any_values else None
        ),
    }


def _paired_problem_bootstrap(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Cluster-bootstrap paired episode deltas by problem, preserving orderings."""
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    reference_by_key = {
        (str(row["problem_id"]), int(row["ordering_index"])): row for row in reference
    }
    candidate_by_key = {
        (str(row["problem_id"]), int(row["ordering_index"])): row for row in candidate
    }
    if reference_by_key.keys() != candidate_by_key.keys():
        raise ValueError("Paired policies must contain identical problem/ordering keys")

    metrics = {
        "correctness": "correct",
        "realized_cost": "realized_spend",
        "attempts": "attempts",
    }
    problem_ids = sorted({key[0] for key in reference_by_key})
    deltas: dict[str, np.ndarray] = {}
    for label, field in metrics.items():
        per_problem = []
        for pid in problem_ids:
            keys = sorted(key for key in reference_by_key if key[0] == pid)
            per_problem.append(float(np.mean([
                float(candidate_by_key[key][field]) - float(reference_by_key[key][field])
                for key in keys
            ])))
        deltas[label] = np.asarray(per_problem, dtype=float)

    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, len(problem_ids), size=(samples, len(problem_ids)))
    result: dict[str, Any] = {
        "n_problems": len(problem_ids),
        "n_paired_episodes": len(reference_by_key),
        "bootstrap_samples": samples,
    }
    for label, values in deltas.items():
        draws = values[sampled_indices].mean(axis=1)
        result[f"{label}_delta"] = float(values.mean())
        result[f"{label}_delta_ci95"] = [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ]
        result[f"probability_{label}_delta_gt_zero"] = float(np.mean(draws > 0.0))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensors-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-orderings", type=int, default=5)
    parser.add_argument("--pseudo-count", type=float, default=2.0)
    parser.add_argument("--cost-mode", choices=["usd", "weights"], default="usd")
    parser.add_argument("--execution-cost-usd", type=float, default=0.0)
    parser.add_argument("--content-preds")
    parser.add_argument("--sequential-model-dir")
    parser.add_argument("--retain-calibration-correctness", type=float, default=0.95)
    parser.add_argument(
        "--start-protocol", choices=["scout_first", "free_start"], default="scout_first"
    )
    parser.add_argument(
        "--include-marginal-value-stop",
        action="store_true",
        help=(
            "Also run the _value_stop variant. Superseded: a single threshold on max_m p/cost "
            "cannot both stop cheap re-rolls and permit escalation (see analysis README), and the "
            "value sweep replaces it. Kept only for reproducing that negative result."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    if args.execution_cost_usd < 0.0:
        raise ValueError("--execution-cost-usd must be non-negative")
    if args.execution_cost_usd and args.cost_mode != "usd":
        raise ValueError("--execution-cost-usd is only defined with --cost-mode usd")

    tensor_dir = Path(args.tensors_dir)
    data = np.load(tensor_dir / "tensors.npz", allow_pickle=True)
    if "execution_outcome" not in data:
        raise ValueError("Full-execution replay requires a schema-v2 tensor bundle")
    outcomes = data["execution_outcome"].astype(bool)
    valid = data["valid"].astype(bool)
    pids = [str(value) for value in data["problem_ids"]]
    slots = [str(value) for value in data["model_slots"]]
    manifest = load_split_manifest(tensor_dir / "split_manifest.json", pids)
    train_idx, cal_idx, test_idx = split_indices(manifest, pids)
    problems = {row["problem_id"]: row for row in _read_jsonl(tensor_dir / "problems.jsonl")}
    records = {
        (row["problem_id"], row["model_slot"], int(row["draw_index"])): row
        for row in _read_jsonl(tensor_dir / "draw_records.jsonl")
    }

    if args.cost_mode == "usd":
        total_tokens = data["prompt_tokens"].astype(float) + data["completion_tokens"].astype(float)
        realized_costs = np.stack([
            total_tokens[:, mi, :] * USD_PER_M_TOKENS[slot] / 1_000_000.0
            for mi, slot in enumerate(slots)
        ], axis=1)
    else:
        realized_costs = np.zeros_like(outcomes, dtype=float)
        for mi, slot in enumerate(slots):
            realized_costs[:, mi, :] = COST_WEIGHTS[slot]
    if args.execution_cost_usd:
        realized_costs += valid.astype(float) * float(args.execution_cost_usd)
    expected_costs = np.array([
        float(realized_costs[train_idx, mi][valid[train_idx, mi]].mean())
        for mi in range(len(slots))
    ])
    priors = np.array([
        float(outcomes[train_idx, mi][valid[train_idx, mi]].mean())
        for mi in range(len(slots))
    ])

    content: dict[str, np.ndarray] = {}
    if args.content_preds:
        for row in _read_jsonl(Path(args.content_preds)):
            content[str(row["problem_id"])] = np.asarray(row["p_successes"][: len(slots)], dtype=float)
    scorer = TorchSequentialScorer(Path(args.sequential_model_dir)) if args.sequential_model_dir else None

    rng = np.random.default_rng(args.seed)
    P, M, K = outcomes.shape
    orderings = np.array([
        [[rng.permutation(K) for _ in range(M)] for _ in range(args.num_orderings)]
        for _ in range(P)
    ])
    max_cost = float(expected_costs.max())
    # Full-exhaustion spend: every draw of every route. The value sweep can reach it, so the
    # budget-swept RoR cell must be able to as well or the high-cost comparison is unfair to it.
    # The original 12 points are preserved so earlier recorded numbers stay comparable.
    exhaustion_cost = float((expected_costs * K).sum())
    budgets = sorted(set(
        [float(x) for x in np.linspace(expected_costs[0], 3.0 * max_cost, 12)]
        + [float(x) for x in np.linspace(3.0 * max_cost, exhaustion_cost, 6)[1:]]
    ))
    tau_grid = [float(x) for x in np.linspace(0.05, 0.95, 19)]
    initial_max_ratio = float(np.max(priors / expected_costs))
    marginal_grid = [
        float(x)
        for x in np.geomspace(initial_max_ratio / 1000.0, initial_max_ratio * 1.5, 25)
    ]
    # Value control sweeps one scalar: the dollar value of a correct answer. It is the
    # Lagrange multiplier on correctness, so it needs no calibration-selected threshold.
    breakeven = expected_costs / priors
    value_grid = [
        float(x)
        for x in np.geomspace(0.5 * float(breakeven.min()), 200.0 * float(breakeven.max()), 24)
    ]
    # A cap that never binds, so the value frontier is generated by R alone.
    unconstrained_budget = float((expected_costs * K).sum()) + 1.0

    def run(
        indices: np.ndarray,
        budget: float,
        family: str,
        tau: float | None,
        min_success_per_cost: float | None = None,
        value_of_correct: float | None = None,
        *,
        capture_trace: bool = False,
    ) -> list[dict[str, Any]]:
        outputs = []
        for pi in indices:
            pid = pids[int(pi)]
            for oi in range(args.num_orderings):
                result = replay_adaptive(
                    outcomes[int(pi)], valid[int(pi)], realized_costs[int(pi)], expected_costs,
                    orderings[int(pi), oi], budget, priors, args.pseudo_count, slots,
                    records, pid, str(problems[pid]["problem_statement"]),
                    content_prior=content.get(pid) if family == "content" else None,
                    scorer=scorer if family in ("sequential", "sequential_decay") else None,
                    tau_abstain=tau,
                    min_success_per_cost=min_success_per_cost,
                    value_of_correct=value_of_correct,
                    capture_trace=capture_trace,
                    apply_failure_decay=family == "sequential_decay",
                    mandatory_scout=args.start_protocol == "scout_first",
                )
                result["problem_id"] = pid
                result["ordering_index"] = oi
                outputs.append(result)
        return outputs

    families = (
        ["counts"]
        + (["content"] if content else [])
        + (["sequential", "sequential_decay"] if scorer else [])
    )
    rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    action_summaries: list[dict[str, Any]] = []
    adaptive_outputs: dict[tuple[float, str], list[dict[str, Any]]] = {}
    # Budget-swept families reproduce the RoR baseline faithfully: count-based beliefs,
    # greedy allocation, budget as the knob. Kept on by default for that reason.
    for budget in budgets:
        for family in families:
            base_cal = _aggregate(run(cal_idx, budget, family, None))
            candidates = []
            for tau in tau_grid:
                metrics = _aggregate(run(cal_idx, budget, family, tau))
                if metrics["correctness"] >= args.retain_calibration_correctness * base_cal["correctness"]:
                    candidates.append((metrics["mean_realized_cost"], tau))
            tau_star = min(candidates)[1] if candidates else None
            marginal_star = None
            if args.include_marginal_value_stop:
                marginal_candidates = []
                for threshold in marginal_grid:
                    metrics = _aggregate(run(cal_idx, budget, family, None, threshold))
                    if metrics["correctness"] >= args.retain_calibration_correctness * base_cal["correctness"]:
                        marginal_candidates.append(
                            (metrics["mean_realized_cost"], -threshold, threshold)
                        )
                marginal_star = min(marginal_candidates)[2] if marginal_candidates else None
            # "" is the RoR-faithful cell (no give-up arm); "_abstain" is our give-up extension.
            variants = [("", None, None), ("_abstain", tau_star, None)]
            if args.include_marginal_value_stop:
                variants.append(("_value_stop", None, marginal_star))
            for suffix, tau, marginal_threshold in variants:
                policy = family + suffix
                outputs = run(
                    test_idx, budget, family, tau, marginal_threshold, capture_trace=True
                )
                adaptive_outputs[(budget, policy)] = outputs
                metrics = _aggregate(outputs)
                rows.append({
                    "policy": policy,
                    "budget": budget,
                    "tau": tau,
                    "min_success_per_cost": marginal_threshold,
                    "value_of_correct": None,
                    **metrics,
                })
                action_summaries.append({
                    "policy": policy,
                    "budget": budget,
                    "tau": tau,
                    "min_success_per_cost": marginal_threshold,
                    "value_of_correct": None,
                    **_summarize_decisions(outputs, slots),
                })
                episode_rows.extend({
                    "policy": policy,
                    "budget": budget,
                    "tau": tau,
                    "min_success_per_cost": marginal_threshold,
                    "value_of_correct": None,
                    **output,
                } for output in outputs)

    # Value-controlled frontier. One scalar R drives escalation and stopping jointly,
    # so there is no budget grid and no calibration-selected abstention threshold.
    value_outputs: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for family in families:
        for value_of_correct in value_grid:
            policy = family + "_value"
            outputs = run(
                test_idx, unconstrained_budget, family, None, None, value_of_correct,
                capture_trace=True,
            )
            value_outputs[(value_of_correct, policy)] = outputs
            metrics = _aggregate(outputs)
            rows.append({
                "policy": policy,
                "budget": None,
                "tau": None,
                "min_success_per_cost": None,
                "value_of_correct": value_of_correct,
                **metrics,
            })
            action_summaries.append({
                "policy": policy,
                "budget": None,
                "tau": None,
                "min_success_per_cost": None,
                "value_of_correct": value_of_correct,
                **_summarize_decisions(outputs, slots),
            })
            episode_rows.extend({
                "policy": policy,
                "budget": None,
                "tau": None,
                "min_success_per_cost": None,
                "value_of_correct": value_of_correct,
                **output,
            } for output in outputs)

    fixed_plans = {
        "single_scout": [0],
        "single_oss20": [1],
        "single_oss120": [2],
        "scout_then_oss20": [0, 1],
        "scout_then_oss120": [0, 2],
        "single_pass_cascade": [0, 1, 2],
        f"best_of_{K}_scout": [0] * K,
        f"best_of_{K}_oss120": [2] * K,
    }
    fixed_outputs: dict[str, list[dict[str, Any]]] = {}
    for name, plan in fixed_plans.items():
        outputs = []
        for pi in test_idx:
            pid = pids[int(pi)]
            for oi in range(args.num_orderings):
                result = replay_fixed(
                    outcomes[int(pi)], valid[int(pi)], realized_costs[int(pi)],
                    orderings[int(pi), oi], plan,
                )
                result["problem_id"] = pid
                result["ordering_index"] = oi
                outputs.append(result)
        fixed_outputs[name] = outputs
        metrics = {
            "correctness": float(np.mean([row["correct"] for row in outputs])),
            "mean_realized_cost": float(np.mean([row["realized_spend"] for row in outputs])),
            "mean_attempts": float(np.mean([row["attempts"] for row in outputs])),
            "n_episodes": len(outputs),
        }
        rows.append({
            "policy": name,
            "budget": None,
            "tau": None,
            "min_success_per_cost": None,
            "value_of_correct": None,
            **metrics,
        })
        episode_rows.extend({
            "policy": name,
            "budget": None,
            "tau": None,
            "min_success_per_cost": None,
            "value_of_correct": None,
            **output,
        } for output in outputs)

    paired_comparisons: list[dict[str, Any]] = []
    if scorer is not None:
        candidate_families = ("sequential", "sequential_decay")
        for fi, candidate_family in enumerate(candidate_families):
            for bi, budget in enumerate(budgets):
                suffixes = ("", "_abstain") + (
                    ("_value_stop",) if args.include_marginal_value_stop else ()
                )
                for suffix in suffixes:
                    reference_policy = "counts" + suffix
                    candidate_policy = candidate_family + suffix
                    paired_comparisons.append({
                        "reference_policy": reference_policy,
                        "candidate_policy": candidate_policy,
                        "budget": budget,
                        **_paired_problem_bootstrap(
                            adaptive_outputs[(budget, reference_policy)],
                            adaptive_outputs[(budget, candidate_policy)],
                            samples=args.bootstrap_samples,
                            seed=args.seed + 100_000 * fi + 1000 * bi + (1 if suffix else 0),
                        ),
                    })

        for fi, candidate_family in enumerate(candidate_families):
            for vi, value_of_correct in enumerate(value_grid):
                paired_comparisons.append({
                    "reference_policy": "counts_value",
                    "candidate_policy": candidate_family + "_value",
                    "budget": None,
                    "value_of_correct": value_of_correct,
                    **_paired_problem_bootstrap(
                        value_outputs[(value_of_correct, "counts_value")],
                        value_outputs[(value_of_correct, candidate_family + "_value")],
                        samples=args.bootstrap_samples,
                        seed=args.seed + 500_000 + 100_000 * fi + 1000 * vi,
                    ),
                })

        cascade_outputs = fixed_outputs["single_pass_cascade"]
        cascade_cost = float(np.mean([row["realized_spend"] for row in cascade_outputs]))
        for fi, candidate_family in enumerate(candidate_families):
            closest_budget = min(
                budgets,
                key=lambda budget: abs(
                    _aggregate(adaptive_outputs[(budget, candidate_family)])[
                        "mean_realized_cost"
                    ] - cascade_cost
                ),
            )
            paired_comparisons.append({
                "reference_policy": "single_pass_cascade",
                "candidate_policy": candidate_family,
                "budget": closest_budget,
                "matching": "nearest_mean_realized_cost",
                **_paired_problem_bootstrap(
                    cascade_outputs,
                    adaptive_outputs[(closest_budget, candidate_family)],
                    samples=args.bootstrap_samples,
                    seed=args.seed + 999_999 + fi,
                ),
            })

    diagnostics = {
        "schema_version": 1,
        "bootstrap_unit": "problem",
        "bootstrap_samples": args.bootstrap_samples,
        "action_summaries": action_summaries,
        "paired_problem_bootstrap": paired_comparisons,
    }

    output = {
        "schema_version": 2,
        "protocol": f"{args.start_protocol}_full_execution_mdp",
        "cost_mode": args.cost_mode,
        "execution_cost_usd_per_attempt": float(args.execution_cost_usd),
        "expected_decision_costs": dict(zip(slots, expected_costs.tolist())),
        "train_priors": dict(zip(slots, priors.tolist())),
        "split_counts": {"train": len(train_idx), "calibration": len(cal_idx), "test": len(test_idx)},
        "results": rows,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "replay_results.json").write_text(json.dumps(output, indent=2) + "\n")
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    with open(out_dir / "episode_traces.jsonl", "w") as handle:
        for row in episode_rows:
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {out_dir / 'replay_results.json'}")
    print(f"wrote {out_dir / 'diagnostics.json'}")
    print(f"wrote {out_dir / 'episode_traces.jsonl'}")



if __name__ == "__main__":
    main()

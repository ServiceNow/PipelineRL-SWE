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
import itertools
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pipelinerl.swe.scripts.livecodebench.build_mdp_reachable_dataset import (
    _remaining_draw_counts,
    _render_state,
)
from pipelinerl.swe.scripts.livecodebench.mdp_utils import load_split_manifest, split_indices
from pipelinerl.swe.scripts.livecodebench.structured_state import (
    STATE_FEATURE_NAMES,
    STATE_FEATURE_VERSION,
    build_structured_state_features,
)


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
        # A factorized model reads the problem statement alone and emits a difficulty
        # and a learned persistence per route; every depth is then derived, so the
        # replay must hand it raw counts instead of rendered state text.
        self.factorized = bool(config.get("factorized", False))
        self.uses_raw_counts = self.factorized
        self.state_feature_mode = str(config.get("state_feature_mode", "text_only"))
        self.uses_structured_state_features = self.state_feature_mode == STATE_FEATURE_VERSION
        self.state_feature_dim = int(config.get("state_feature_dim", 0))
        self.state_feature_hidden_size = int(config.get("state_feature_hidden_size", 0))
        if self.state_feature_mode not in {"text_only", STATE_FEATURE_VERSION}:
            raise ValueError(f"Unsupported state feature mode: {self.state_feature_mode}")
        if self.state_feature_mode == STATE_FEATURE_VERSION and (
            self.state_feature_dim != len(STATE_FEATURE_NAMES)
            or self.state_feature_hidden_size < 1
        ):
            raise ValueError("Structured model has incompatible state-feature metadata")
        model_name = str(config["model_name"])
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if torch.cuda.is_available():
            kwargs["attn_implementation"] = config.get("attn_implementation", "flash_attention_2")
        base = AutoModel.from_pretrained(model_name, **kwargs)
        # A frozen-encoder probe has no adapter to load: the encoder is the unmodified
        # base model, so only the head was trained and only the head was saved.
        if bool(config.get("frozen_encoder", False)):
            self.encoder = base
        else:
            adapter_dir = model_dir / "encoder" / "reward_adapter"
            self.encoder = PeftModel.from_pretrained(
                base, str(adapter_dir), adapter_name="reward_adapter"
            )
        head_state = torch.load(model_dir / "reward_head.pt", map_location="cpu", weights_only=True)
        hidden_size = int(self.encoder.config.hidden_size)
        first_weight = head_state.get("1.weight")
        if first_weight is None:
            raise ValueError("Unsupported reward-head layout: missing 1.weight")
        mlp_hidden = int(first_weight.shape[0])
        target_dim = int(head_state["4.weight"].shape[0])
        head_input_size = hidden_size
        if self.state_feature_mode == STATE_FEATURE_VERSION:
            head_input_size += self.state_feature_hidden_size
            self.state_feature_encoder = torch.nn.Sequential(
                torch.nn.Linear(self.state_feature_dim, self.state_feature_hidden_size),
                torch.nn.GELU(),
            )
            self.state_feature_encoder.load_state_dict(
                torch.load(
                    model_dir / "state_feature_encoder.pt", map_location="cpu", weights_only=True
                )
            )
        else:
            self.state_feature_encoder = None
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(0.0),
            torch.nn.Linear(head_input_size, mlp_hidden),
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
        if self.state_feature_encoder is not None:
            self.state_feature_encoder.to(self.device).eval()
        self.cache: dict[str, np.ndarray] = {}

    def __call__(self, text: str, state_features: list[float] | None = None) -> np.ndarray:
        if self.state_feature_mode == STATE_FEATURE_VERSION:
            if state_features is None or len(state_features) != self.state_feature_dim:
                raise ValueError("Structured scorer requires the configured state feature vector")
            feature_array = np.asarray(state_features, dtype=np.float32)
            key = _state_key(text + "\0" + feature_array.tobytes().hex())
        else:
            feature_array = None
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
            if feature_array is not None:
                if self.state_feature_encoder is None:
                    raise AssertionError("Missing structured state encoder")
                numeric = torch.from_numpy(feature_array).to(self.device).reshape(1, -1)
                numeric = self.state_feature_encoder(numeric.to(dtype=pooled.dtype))
                pooled = torch.cat([pooled, numeric], dim=1)
            raw = self.head(pooled)
            probs = (
                raw.float().cpu().numpy()[0] if self.factorized
                else torch.sigmoid(raw).cpu().numpy()[0]
            )
        self.cache[key] = probs
        return probs

    def factorized_beliefs(
        self, problem_statement: str, failures: np.ndarray, remaining: np.ndarray
    ) -> np.ndarray:
        """p_m(n) per route plus the derived P(nothing remains).

        The six emitted numbers depend only on the problem, so the encoder runs once
        per problem and every failure depth is then closed-form. That also makes the
        Bellman lattice exact at any horizon from a single forward pass, rather than
        needing a query per successor.
        """
        latent = self(problem_statement)
        theta = 1.0 / (1.0 + np.exp(-latent[:3]))
        persistence = np.log1p(np.exp(np.clip(latent[3:6], -30, 30))) + 1e-3
        counts = np.asarray(failures, dtype=float)
        p_next = theta * persistence / (persistence + counts)
        log_survive = 0.0
        for step in range(int(np.max(remaining)) if len(remaining) else 0):
            p_step = theta * persistence / (persistence + counts + step)
            active = (np.asarray(remaining, dtype=float) > step).astype(float)
            log_survive += float(
                (np.log1p(-np.clip(p_step, 1e-7, 1 - 1e-7)) * active).sum()
            )
        return np.clip(
            np.concatenate([p_next, [np.exp(log_survive)]]), 1e-7, 1 - 1e-7
        )


def _load_calibrator(path: Path) -> Callable[[np.ndarray, int], np.ndarray]:
    """Per-head, per-failure-depth recalibration fit by fit_depth_calibration.py.

    Heads are indexed positionally, matching the model's output columns. Depths
    outside the fitted range clamp to the nearest fitted depth, which is the
    conservative choice: beyond the deepest observed bucket the hazards are flat
    at (or near) zero anyway.
    """
    spec = json.loads(path.read_text())
    heads = list(spec["heads"])
    table = spec["calibration"]
    by_index = [
        {int(depth): entry for depth, entry in table[head].items()}
        for head in heads
    ]

    def calibrate(probs: np.ndarray, depth: int) -> np.ndarray:
        out = np.asarray(probs, dtype=float).copy()
        for hi in range(min(len(out), len(by_index))):
            buckets = by_index[hi]
            if not buckets:
                continue
            key = depth if depth in buckets else min(buckets, key=lambda d: abs(d - depth))
            entry = buckets[key]
            if entry["mode"] == "constant":
                out[hi] = float(entry["value"])
            else:
                x = float(np.log(np.clip(out[hi], 1e-6, 1 - 1e-6) / (1 - np.clip(out[hi], 1e-6, 1 - 1e-6))))
                out[hi] = float(1.0 / (1.0 + np.exp(-(entry["a"] * x + entry["b"]))))
        return out

    return calibrate


def _next_valid_draw(order: np.ndarray, ptr: int, valid: np.ndarray) -> tuple[int | None, int]:
    while ptr < len(order) and not valid[int(order[ptr])]:
        ptr += 1
    return (int(order[ptr]), ptr) if ptr < len(order) else (None, ptr)


def _has_remaining_success(
    outcomes: np.ndarray,
    valid: np.ndarray,
    orderings: np.ndarray,
    ptr: np.ndarray,
) -> bool:
    """Whether any still-unseen valid draw succeeds (diagnostic oracle only)."""
    for mi in range(outcomes.shape[0]):
        for position in range(int(ptr[mi]), len(orderings[mi])):
            draw = int(orderings[mi, position])
            if valid[mi, draw] and outcomes[mi, draw]:
                return True
    return False


def _solve_bellman_action_values(
    pbar: np.ndarray,
    failures: np.ndarray,
    remaining_by_route: list[int],
    expected_costs: np.ndarray,
    spent_budget: float,
    budget: float,
    value_of_correct: float,
    decay_s: float | None,
    horizon: int,
    raw_belief_at: Callable[[tuple[int, ...]], np.ndarray] | None = None,
) -> tuple[dict[int, float], float]:
    """Exact backward induction over the reachable failure-count lattice.

        V(k) = max( 0, max_m [ p_m(k)*R - c_m + (1 - p_m(k)) * V(k + e_m) ] )

    The myopic rule this replaces drops the `(1 - p) V(k + e_m)` term, so it
    cannot tell "scout is cheap and oss120 is still in reserve" from "this is the
    last affordable shot", and it does not know draws are finite.

    Nothing here is learned. Draws are exchangeable given the problem, so the
    per-route failure counts are a sufficient statistic and this lattice is the
    entire state space -- at most C(h+3,3) nodes. The transition model is fully
    determined by the beliefs the scorer already emits, so the Bellman equation
    is solved rather than approximated: no bootstrapping, no value regression,
    and no hindsight bias from an oracle target.

    Future beliefs are extrapolated analytically as `pbar_m * s / (s + n_m)`,
    the same Beta-Bernoulli posterior the myopic decay arm applies. That keeps
    the belief model identical between the two arms so the comparison isolates
    the continuation value. `decay_s=None` holds beliefs constant with depth.

    Both budget and capacity feasibility are deterministic functions of `k`
    (costs are per-route constants), so neither needs to enter the state.

    Returns (action value per route at the current state, state value). The
    state value is 0 exactly when abstaining is optimal. `horizon=1` truncates
    every successor to 0 and therefore reproduces the myopic rule exactly.
    """
    n_routes = len(pbar)
    caps = [min(int(remaining_by_route[m]), horizon) for m in range(n_routes)]
    values: dict[tuple[int, ...], float] = {}
    root_action_values: dict[int, float] = {}
    zero = tuple(0 for _ in range(n_routes))

    def raw_belief(offsets: tuple[int, ...]) -> np.ndarray:
        """Undecayed per-route beliefs at a lattice node.

        Analytic default reuses the root prediction at every node, which asserts
        that failing route m says nothing about route m'. That is the assumption
        `raw_belief_at` replaces: the model is queried at the successor and can
        lower every route at once, because a hard problem is hard for everyone.
        The decay below is applied identically either way, so swapping this
        function isolates the transition model and nothing else.
        """
        if raw_belief_at is None:
            return pbar
        supplied = raw_belief_at(offsets)
        return pbar if supplied is None else supplied

    # Descending on every axis, so every successor k + e_m is already solved.
    for offsets in itertools.product(*[range(cap, -1, -1) for cap in caps]):
        depth = sum(offsets)
        if depth >= horizon:
            values[offsets] = 0.0
            continue
        spent = spent_budget + sum(
            offsets[m] * float(expected_costs[m]) for m in range(n_routes)
        )
        action_values: dict[int, float] = {}
        best = 0.0  # abstaining is always available and worth exactly zero
        node_raw = raw_belief(offsets)
        for m in range(n_routes):
            if offsets[m] >= caps[m]:
                continue  # route exhausted its remaining valid draws
            if spent + float(expected_costs[m]) > budget + 1e-12:
                continue
            if decay_s is None:
                p = float(node_raw[m])
            else:
                p = float(node_raw[m]) * decay_s / (decay_s + float(failures[m] + offsets[m]))
            successor = list(offsets)
            successor[m] += 1
            q = (
                p * value_of_correct
                - float(expected_costs[m])
                + (1.0 - p) * values[tuple(successor)]
            )
            action_values[m] = q
            best = max(best, q)
        values[offsets] = best
        if offsets == zero:
            root_action_values = action_values
    return root_action_values, values[zero]


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
    scorer: Callable[..., np.ndarray] | None = None,
    calibrator: Callable[[np.ndarray, int], np.ndarray] | None = None,
    state_layout: str = "counts_last",
    exploration_bonus: bool = False,
    decay_pseudo_count: float | None = None,
    tau_abstain: float | None = None,
    min_success_per_cost: float | None = None,
    value_of_correct: float | None = None,
    bellman_horizon: int | None = None,
    learned_transitions: bool = False,
    oracle_routing: bool = False,
    q_abstain: float | None = None,
    capture_trace: bool = False,
    apply_failure_decay: bool = False,
    mandatory_scout: bool = True,
    oracle_stopping: bool = False,
) -> dict[str, Any]:
    if bellman_horizon is not None:
        if value_of_correct is None:
            raise ValueError("Bellman lookahead requires the value-of-correct control")
        if bellman_horizon < 1:
            raise ValueError("--bellman-horizon must be at least 1 (1 == myopic)")
    if learned_transitions and bellman_horizon is None:
        raise ValueError("Learned transitions require a Bellman horizon")
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

        learned_q: float | None = None
        remaining = _remaining_draw_counts(valid, orderings, ptr, slots)
        route_capacities = {slot: int(valid[mi].sum()) for mi, slot in enumerate(slots)}
        latest_attempt = attempts[-1] if attempts else None
        state_features = build_structured_state_features(
            attempts, remaining, route_capacities, slots
        )
        state_text = (
            _render_state(problem_statement, attempts, remaining, state_layout)
            if scorer is not None or capture_trace else None
        )
        if scorer is not None:
            if getattr(scorer, "uses_raw_counts", False):
                raw_probs = scorer.factorized_beliefs(
                    problem_statement,
                    failures,
                    np.array([remaining[slot] for slot in slots], dtype=float),
                )
            else:
                if state_text is None:
                    raise AssertionError("Sequential scoring requires rendered state text")
                raw_probs = (
                    scorer(state_text, state_features)
                    if getattr(scorer, "uses_structured_state_features", False)
                    else scorer(state_text)
                )
            all_probs = np.asarray(raw_probs, dtype=float)
            if calibrator is not None:
                all_probs = calibrator(all_probs, int(failures.sum()))
            p_each = all_probs[: len(slots)]
            # The `nothing` head predicts that NO valid draw remains anywhere, so
            # 1 - it is q(s) = P(some rescue exists). Kept whatever the family: the
            # decay branch below overwrites p_any with an independence product and
            # would otherwise discard the only head that answers the stopping
            # question directly.
            if len(all_probs) > len(slots):
                learned_q = 1.0 - float(all_probs[len(slots)])
            # The undecayed prediction is the lattice prior: the Bellman solve
            # re-applies the decay itself at each future depth.
            bellman_pbar, bellman_decay_s = p_each, None
            if apply_failure_decay and getattr(scorer, "uses_raw_counts", False):
                # A factorized scorer already returns theta_m * s_m/(s_m + n_m) with a
                # LEARNED s_m. Multiplying by the hand-set s/(s+n) again compounds the
                # very constant the model was built to replace, and silently makes the
                # arm untestable rather than failing.
                raise ValueError(
                    "apply_failure_decay double-decays a factorized scorer: its beliefs are "
                    "already count-conditioned. Use the `sequential` family for these models."
                )
            if apply_failure_decay:
                decay_s = pseudo_count if decay_pseudo_count is None else decay_pseudo_count
                bellman_decay_s = decay_s
                p_each = p_each * decay_s / (decay_s + failures)
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
            bellman_pbar, bellman_decay_s = p_each, None
            p_any = 1.0 - float(np.prod(1.0 - p_each))
            belief_source = "content"
        else:
            p_each = np.asarray([
                pseudo_count * prior[mi] / (pseudo_count + failures[mi])
                for mi in range(len(slots))
            ])
            bellman_pbar, bellman_decay_s = np.asarray(prior, dtype=float), pseudo_count
            p_any = 1.0 - float(np.prod(1.0 - p_each))
            belief_source = "counts"

        ratios = {mi: float(p_each[mi] / expected_costs[mi]) for mi in available}
        if exploration_bonus:
            t = int(failures.sum())
            ratios = {
                mi: ratios[mi] + float(
                    np.sqrt(2.0 * np.log(t + 1) / (failures[mi] + 1)) / expected_costs[mi]
                )
                for mi in available
            }
        # The ratio is scale-invariant, so it ranks routes identically at every
        # valuation and always prefers the cheapest. The dollar difference does not.
        state_value: float | None = None
        if value_of_correct is None:
            action_values = None
        elif bellman_horizon is None:
            action_values = {
                mi: float(p_each[mi] * value_of_correct - expected_costs[mi])
                for mi in available
            }
        else:
            raw_belief_at = None
            if learned_transitions and scorer is not None:
                def raw_belief_at(offsets: tuple[int, ...]) -> np.ndarray | None:
                    """Query the model one failure ahead instead of extrapolating.

                    Only depth 1 is queried: for horizon 2 that is every node the
                    recursion touches, so H=2 becomes the exact optimum under the
                    model's own beliefs with no analytic extrapolation at all.
                    Deeper nodes fall back to the analytic form.

                    The successor's counts are known exactly at decision time, but
                    the content of an attempt not yet made is not, so the latest
                    observed attempt's text is carried forward unchanged. Using the
                    real future draw's code here would leak an outcome the policy
                    cannot see in deployment.
                    """
                    if sum(offsets) != 1:
                        return None
                    mi = next(i for i, k in enumerate(offsets) if k == 1)
                    if latest_attempt is None:
                        hypothetical = {
                            "model_slot": slots[mi],
                            "code": "",
                            "full_execution_feedback": "Full execution: FAILED",
                        }
                    else:
                        hypothetical = dict(latest_attempt)
                        hypothetical["model_slot"] = slots[mi]
                    next_attempts = attempts + [hypothetical]
                    next_remaining = dict(remaining)
                    next_remaining[slots[mi]] = max(0, next_remaining[slots[mi]] - 1)
                    text = _render_state(
                        problem_statement, next_attempts, next_remaining, state_layout
                    )
                    features = build_structured_state_features(
                        next_attempts, next_remaining, route_capacities, slots
                    )
                    probs = np.asarray(
                        scorer(text, features)
                        if getattr(scorer, "uses_structured_state_features", False)
                        else scorer(text),
                        dtype=float,
                    )
                    if calibrator is not None:
                        probs = calibrator(probs, int(failures.sum()) + 1)
                    return probs[: len(slots)]

            root_action_values, state_value = _solve_bellman_action_values(
                bellman_pbar,
                failures,
                [int(remaining[slot]) for slot in slots],
                expected_costs,
                spent_budget,
                budget,
                value_of_correct,
                bellman_decay_s,
                bellman_horizon,
                raw_belief_at=raw_belief_at,
            )
            action_values = {
                mi: root_action_values[mi] for mi in available if mi in root_action_values
            }
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
                "structured_state_features": state_features,
                "available_routes": [slots[mi] for mi in available],
                "spent_budget_before": float(spent_budget),
                "remaining_budget_before": float(budget - spent_budget),
                "p_success_next": {
                    slot: float(p_each[mi]) for mi, slot in enumerate(slots)
                },
                "p_any_remaining": float(p_any),
                "learned_q": learned_q,
                "q_abstain": q_abstain,
                "utility_per_expected_cost": {
                    slots[mi]: ratios[mi] for mi in available
                },
                "action_value": (
                    None if action_values is None
                    else {slots[mi]: value for mi, value in action_values.items()}
                ),
                "bellman_horizon": bellman_horizon,
                "state_value": state_value,
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
        value_stop = action_values is not None and (
            not action_values or max(action_values.values()) <= 0.0
        )
        # Diagnostic upper bound: replace only the stopping decision with perfect
        # knowledge of the stored future outcomes. Route scores, rankings, costs,
        # capacities, R, and Bellman horizon remain untouched. This deliberately
        # leaks future outcomes and must never be reported as a deployable policy.
        oracle_has_remaining_success = (
            _has_remaining_success(outcomes, valid, orderings, ptr)
            if oracle_stopping else None
        )
        oracle_stop = oracle_stopping and not oracle_has_remaining_success
        # Replaces stopping only, exactly as the oracle arm does, so the two are
        # directly comparable: how much of the oracle's headroom does the belief
        # we already have actually recover? Routing is left to the value rule.
        if q_abstain is not None and learned_q is None:
            raise ValueError("q-based stopping requires a scorer with a `nothing` head")
        q_stop = q_abstain is not None and learned_q <= q_abstain
        regular_stop = (
            not oracle_stopping and q_abstain is None
            and (probability_stop or marginal_stop or value_stop)
        )
        if oracle_stop or q_stop or regular_stop:
            if decision is not None:
                decision.update({
                    "chosen_route": None,
                    "chosen_draw_index": None,
                    "result": "abstain",
                    "abstain_reason": (
                        "oracle_no_remaining_success" if oracle_stop
                        else "learned_q_below_threshold" if q_stop
                        else "probability_threshold" if probability_stop
                        else "marginal_value" if marginal_stop
                        else "non_positive_action_value"
                    ),
                    "oracle_has_remaining_success": oracle_has_remaining_success,
                })
                decision_trace.append(decision)
            return finish(False, True)

        oracle_route = None
        if oracle_routing:
            # Diagnostic upper bound on ROUTE CHOICE only. Among routes the policy
            # could take, prefer one whose next draw actually succeeds, cheapest
            # first. Stopping is untouched, so pairing this against the same policy
            # without it isolates how much is lost by picking the wrong route.
            succeeding = []
            for mi in available:
                draw, _ = _next_valid_draw(orderings[mi], int(ptr[mi]), valid[mi])
                if draw is not None and outcomes[mi, draw]:
                    succeeding.append(mi)
            if succeeding:
                oracle_route = min(succeeding, key=lambda mi: float(expected_costs[mi]))
        chosen = (
            oracle_route if oracle_route is not None
            else max(selection, key=selection.__getitem__)
        )
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
            if oracle_stopping:
                decision["oracle_has_remaining_success"] = oracle_has_remaining_success
            if oracle_routing:
                decision["oracle_routed"] = oracle_route is not None
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
    parser.add_argument(
        "--pseudo-count-grid", default="",
        help=(
            "Comma-separated s values for RoR's belief update (s*p_bar+w)/(s+n). s sets how fast "
            "failures pull the estimate off the offline prior, i.e. how quickly RoR escalates. "
            "RoR reports insensitivity over an order of magnitude on their pool; verify it on ours "
            "and report the baseline at its BEST s rather than inheriting the claim."
        ),
    )
    parser.add_argument(
        "--decay-pseudo-count", type=float, default=None,
        help="Pseudo-count for our sequential_decay variant. Defaults to --pseudo-count; set "
             "separately so the RoR sweep and our decay do not move in lockstep.",
    )
    parser.add_argument(
        "--ucb", action="store_true",
        help="Also run RoR's UCB variant (exploration bonus sqrt(2 ln(t+1)/(n_m+1))/c_m).",
    )
    parser.add_argument(
        "--retention-grid", default="0.95,0.98,0.99,1.0",
        help=(
            "Correctness-retention targets for freezing R on CALIBRATION. For each target the "
            "cheapest R meeting it on calibration is frozen, then reported once on test. Fixes the "
            "post-hoc selection of R=0.1863 off the test frontier."
        ),
    )
    parser.add_argument(
        "--bellman-horizons", default="",
        help=(
            "Comma-separated lookahead depths for the exact Bellman solve over the failure-count "
            "lattice, e.g. '1,2,4,8'. Each adds <family>_bellman_h<H>_value arms alongside the "
            "myopic <family>_value arms. H=1 truncates every successor to zero and so reproduces "
            "the myopic rule exactly; larger H credits the continuation value the myopic rule "
            "drops. Empty disables the arms entirely."
        ),
    )
    parser.add_argument(
        "--learned-transition-horizons", default="",
        help=(
            "Comma-separated Bellman horizons to ALSO run with learned one-step transitions, "
            "querying the scorer at each depth-1 successor instead of extrapolating the root "
            "prediction analytically. At H=2 that covers every node the recursion touches, so "
            "the arm is the exact optimum under the model's own beliefs. Adds "
            "<family>_bellman_h<H>_learned_value arms. Requires --sequential-model-dir."
        ),
    )
    parser.add_argument(
        "--q-stop-family",
        choices=["counts", "content", "sequential", "sequential_decay"],
        help=(
            "Replace stopping for this family's frozen-R policy with a threshold on the "
            "learned q(s) = 1 - P(nothing remains), sweeping --q-stop-grid. Routing is left to "
            "the value rule, exactly as in the oracle arm, so the two are directly comparable: "
            "this measures how much of the oracle's stopping headroom the beliefs we already "
            "have can actually recover. Unlike the oracle arm this is deployable."
        ),
    )
    parser.add_argument("--q-stop-horizon", type=int, default=None)
    parser.add_argument(
        "--q-stop-grid", default="0.02,0.05,0.08,0.12,0.16,0.20,0.25,0.30,0.40,0.50",
        help="Abstain when learned q(s) <= threshold.",
    )
    parser.add_argument(
        "--oracle-stopping-family",
        choices=["counts", "content", "sequential", "sequential_decay"],
        help=(
            "Diagnostic only: for this value-policy family, replace stopping with perfect "
            "knowledge of whether any stored future draw succeeds. Routing is unchanged."
        ),
    )
    parser.add_argument(
        "--oracle-stopping-horizon",
        type=int,
        default=None,
        help=(
            "Bellman horizon whose calibration-frozen policy receives oracle stopping. "
            "Omit for the myopic value policy."
        ),
    )
    parser.add_argument("--cost-mode", choices=["usd", "weights"], default="usd")
    parser.add_argument("--execution-cost-usd", type=float, default=0.0)
    parser.add_argument("--content-preds")
    parser.add_argument("--sequential-model-dir")
    parser.add_argument(
        "--state-layout", choices=["problem_first", "counts_last"], default="counts_last",
        help="Must match the layout the sequential model was trained on",
    )
    parser.add_argument(
        "--calibration-map",
        help=(
            "JSON from fit_depth_calibration.py. Applies per-route, per-failure-depth "
            "recalibration to the learned heads before any decision is taken."
        ),
    )
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
    if args.q_stop_horizon is not None and args.q_stop_family is None:
        raise ValueError("--q-stop-horizon requires --q-stop-family")
    if args.oracle_stopping_horizon is not None and args.oracle_stopping_family is None:
        raise ValueError("--oracle-stopping-horizon requires --oracle-stopping-family")
    if args.oracle_stopping_horizon is not None and args.oracle_stopping_horizon < 1:
        raise ValueError("--oracle-stopping-horizon must be at least 1")

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
    calibrator = _load_calibrator(Path(args.calibration_map)) if args.calibration_map else None

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
        pseudo_count: float | None = None,
        exploration_bonus: bool = False,
        bellman_horizon: int | None = None,
        learned_transitions: bool = False,
        oracle_stopping: bool = False,
        oracle_routing: bool = False,
        q_abstain: float | None = None,
    ) -> list[dict[str, Any]]:
        outputs = []
        for pi in indices:
            pid = pids[int(pi)]
            for oi in range(args.num_orderings):
                result = replay_adaptive(
                    outcomes[int(pi)], valid[int(pi)], realized_costs[int(pi)], expected_costs,
                    orderings[int(pi), oi], budget, priors,
                    args.pseudo_count if pseudo_count is None else pseudo_count, slots,
                    records, pid, str(problems[pid]["problem_statement"]),
                    content_prior=content.get(pid) if family == "content" else None,
                    scorer=scorer if family in ("sequential", "sequential_decay") else None,
                    calibrator=calibrator,
                    state_layout=args.state_layout,
                    tau_abstain=tau,
                    min_success_per_cost=min_success_per_cost,
                    value_of_correct=value_of_correct,
                    bellman_horizon=bellman_horizon,
                    learned_transitions=learned_transitions,
                    capture_trace=capture_trace,
                    apply_failure_decay=family == "sequential_decay",
                    decay_pseudo_count=args.decay_pseudo_count,
                    exploration_bonus=exploration_bonus,
                    mandatory_scout=args.start_protocol == "scout_first",
                    oracle_stopping=oracle_stopping,
                    oracle_routing=oracle_routing,
                    q_abstain=q_abstain,
                )
                result["problem_id"] = pid
                result["ordering_index"] = oi
                outputs.append(result)
        return outputs

    # A factorized scorer already returns theta_m * s_m/(s_m + n_m) with a learned
    # s_m, so the sequential_decay family would apply the hand-set analytic decay on
    # top of it. That arm is not meaningful for such models, so it is dropped rather
    # than allowed to raise and abort the whole replay.
    factorized_scorer = bool(getattr(scorer, "uses_raw_counts", False))
    scorer_families = (
        ["sequential"] if factorized_scorer else ["sequential", "sequential_decay"]
    )
    families = (
        ["counts"]
        + (["content"] if content else [])
        + (scorer_families if scorer else [])
    )
    if args.oracle_stopping_family and args.oracle_stopping_family not in families:
        raise ValueError(
            f"Oracle family {args.oracle_stopping_family!r} is unavailable; "
            "provide its required predictions/model"
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

    # --- Baseline fairness arms ---------------------------------------------------
    # (a) Pseudo-count sensitivity for RoR's belief update. s controls how quickly
    #     observed failures pull the estimate off the offline prior, i.e. how fast the
    #     baseline escalates. We must report RoR at its best s, not at our default.
    # (b) RoR ships a greedy AND a UCB variant; comparing only against greedy leaves an
    #     unverified assumption that greedy is their stronger arm.
    pseudo_grid = [float(x) for x in args.pseudo_count_grid.split(",") if x.strip()]
    for s_value in pseudo_grid:
        for budget in budgets:
            outputs = run(test_idx, budget, "counts", None, pseudo_count=s_value,
                          capture_trace=False)
            rows.append({
                "policy": f"counts_s{s_value:g}", "budget": budget, "tau": None,
                "min_success_per_cost": None, "value_of_correct": None,
                "pseudo_count": s_value, **_aggregate(outputs),
            })
    if args.ucb:
        for budget in budgets:
            outputs = run(test_idx, budget, "counts", None, exploration_bonus=True,
                          capture_trace=False)
            rows.append({
                "policy": "counts_ucb", "budget": budget, "tau": None,
                "min_success_per_cost": None, "value_of_correct": None,
                **_aggregate(outputs),
            })

    # --- R frozen on calibration ----------------------------------------------------
    # The headline R was previously chosen by reading the TEST frontier for the cheapest
    # point at ceiling accuracy. That is post-hoc selection. Here each retention target
    # picks its R on calibration only; the test split is then touched once per target.
    retention_grid = [float(x) for x in args.retention_grid.split(",") if x.strip()]
    # horizon None is the myopic rule; each explicit horizon is an exact Bellman solve
    # over the same beliefs, so the pair isolates the continuation value.
    bellman_horizons = [int(x) for x in args.bellman_horizons.split(",") if x.strip()]
    learned_horizons = [
        int(x) for x in args.learned_transition_horizons.split(",") if x.strip()
    ]
    if learned_horizons and scorer is None:
        raise ValueError("--learned-transition-horizons requires --sequential-model-dir")
    # Learned transitions only alter beliefs the SCORER produces, so they are
    # meaningless for the count-based families.
    learned_families = [f for f in families if f in ("sequential", "sequential_decay")]
    value_arms: list[tuple[str, int | None, bool]] = (
        [(family, None, False) for family in families]
        + [(family, h, False) for family in families for h in bellman_horizons]
        + [(family, h, True) for family in learned_families for h in learned_horizons]
    )
    oracle_arm = (
        (args.oracle_stopping_family, args.oracle_stopping_horizon)
        if args.oracle_stopping_family else None
    )
    if oracle_arm is not None and (*oracle_arm, False) not in value_arms:
        value_arms.append((*oracle_arm, False))
    q_stop_arm = (
        (args.q_stop_family, args.q_stop_horizon) if args.q_stop_family else None
    )
    if q_stop_arm is not None and (*q_stop_arm, False) not in value_arms:
        value_arms.append((*q_stop_arm, False))
    q_stop_grid = [float(x) for x in args.q_stop_grid.split(",") if x.strip()]

    def _value_policy(family: str, horizon: int | None, learned: bool = False) -> str:
        if horizon is None:
            return f"{family}_value"
        return f"{family}_bellman_h{horizon}{'_learned' if learned else ''}_value"

    frozen_rows: list[dict[str, Any]] = []
    oracle_frozen_pairs: list[dict[str, Any]] = []
    for family, horizon, learned in value_arms:
        cal_by_r = {
            r: _aggregate(run(cal_idx, unconstrained_budget, family, None, None, r,
                              bellman_horizon=horizon, learned_transitions=learned))
            for r in value_grid
        }
        cal_ceiling = max(m["correctness"] for m in cal_by_r.values())
        for target in retention_grid:
            ok = [(m["mean_realized_cost"], r) for r, m in cal_by_r.items()
                  if m["correctness"] >= target * cal_ceiling]
            if not ok:
                continue
            r_star = min(ok)[1]
            test_outputs = run(test_idx, unconstrained_budget, family, None, None,
                               r_star, bellman_horizon=horizon,
                               learned_transitions=learned)
            test_metrics = _aggregate(test_outputs)
            policy = f"{_value_policy(family, horizon, learned)}_frozen"
            frozen_rows.append({
                "policy": policy,
                "budget": None, "tau": None, "min_success_per_cost": None,
                "value_of_correct": r_star,
                "bellman_horizon": horizon,
                "learned_transitions": learned,
                "selection": "calibration_only",
                "retention_target": target,
                "calibration_correctness": cal_by_r[r_star]["correctness"],
                "calibration_ceiling": cal_ceiling,
                **test_metrics,
            })
            if q_stop_arm == (family, horizon) and not learned:
                for threshold in q_stop_grid:
                    q_outputs = run(
                        test_idx, unconstrained_budget, family, None, None, r_star,
                        bellman_horizon=horizon, q_abstain=threshold, capture_trace=True,
                    )
                    q_policy = f"{policy}_qstop{threshold:g}"
                    frozen_rows.append({
                        "policy": q_policy,
                        "budget": None, "tau": None, "min_success_per_cost": None,
                        "value_of_correct": r_star,
                        "bellman_horizon": horizon,
                        "q_abstain": threshold,
                        "selection": "calibration_only_R_with_swept_q_threshold",
                        "retention_target": target,
                        **_aggregate(q_outputs),
                    })
                    episode_rows.extend({
                        "policy": q_policy, "budget": None, "tau": None,
                        "min_success_per_cost": None, "value_of_correct": r_star,
                        "bellman_horizon": horizon, "q_abstain": threshold,
                        **output,
                    } for output in q_outputs)

            # Full headroom decomposition: which of stopping, routing, or the
            # portfolio itself limits us. Stopping and routing are run separately
            # so their contributions are attributable, then jointly to expose any
            # interaction; "both" is also the pool ceiling this policy could reach.
            oracle_variants = (
                (("_oracle_stop", True, False),
                 ("_oracle_route", False, True),
                 ("_oracle_both", True, True))
                if oracle_arm == (family, horizon) and not learned else ()
            )
            for suffix, ora_stop, ora_route in oracle_variants:
                oracle_outputs = run(
                    test_idx, unconstrained_budget, family, None, None, r_star,
                    bellman_horizon=horizon, oracle_stopping=ora_stop,
                    oracle_routing=ora_route, capture_trace=True,
                )
                oracle_policy = f"{policy}{suffix}"
                frozen_rows.append({
                    "policy": oracle_policy,
                    "budget": None, "tau": None, "min_success_per_cost": None,
                    "value_of_correct": r_star,
                    "bellman_horizon": horizon,
                    "selection": "calibration_only_R_with_test_outcome_oracle",
                    "retention_target": target,
                    "calibration_correctness": cal_by_r[r_star]["correctness"],
                    "calibration_ceiling": cal_ceiling,
                    "diagnostic_only": True,
                    "oracle_information": (
                        "any_remaining_stored_draw_succeeds" if ora_stop and not ora_route
                        else "cheapest_route_whose_next_draw_succeeds" if ora_route and not ora_stop
                        else "both_remaining_success_and_succeeding_route"
                    ),
                    "oracle_stopping": ora_stop,
                    "oracle_routing": ora_route,
                    **_aggregate(oracle_outputs),
                })
                episode_rows.extend({
                    "policy": oracle_policy,
                    "budget": None, "tau": None, "min_success_per_cost": None,
                    "value_of_correct": r_star,
                    "bellman_horizon": horizon,
                    "diagnostic_only": True,
                    **output,
                } for output in oracle_outputs)
                oracle_frozen_pairs.append({
                    "reference_policy": policy,
                    "candidate_policy": oracle_policy,
                    "retention_target": target,
                    "value_of_correct": r_star,
                    "reference_outputs": test_outputs,
                    "candidate_outputs": oracle_outputs,
                })
    rows.extend(frozen_rows)

    # Value-controlled frontier. One scalar R drives escalation and stopping jointly,
    # so there is no budget grid and no calibration-selected abstention threshold.
    value_outputs: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for family, horizon, learned in value_arms:
        for value_of_correct in value_grid:
            policy = _value_policy(family, horizon, learned)
            outputs = run(
                test_idx, unconstrained_budget, family, None, None, value_of_correct,
                capture_trace=True, bellman_horizon=horizon,
                learned_transitions=learned,
            )
            value_outputs[(value_of_correct, policy)] = outputs
            metrics = _aggregate(outputs)
            rows.append({
                "policy": policy,
                "budget": None,
                "tau": None,
                "min_success_per_cost": None,
                "value_of_correct": value_of_correct,
                "bellman_horizon": horizon,
                **metrics,
            })
            action_summaries.append({
                "policy": policy,
                "budget": None,
                "tau": None,
                "min_success_per_cost": None,
                "value_of_correct": value_of_correct,
                "bellman_horizon": horizon,
                **_summarize_decisions(outputs, slots),
            })
            episode_rows.extend({
                "policy": policy,
                "budget": None,
                "tau": None,
                "min_success_per_cost": None,
                "value_of_correct": value_of_correct,
                "bellman_horizon": horizon,
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
    for oi, comparison in enumerate(oracle_frozen_pairs):
        paired_comparisons.append({
            "reference_policy": comparison["reference_policy"],
            "candidate_policy": comparison["candidate_policy"],
            "budget": None,
            "value_of_correct": comparison["value_of_correct"],
            "retention_target": comparison["retention_target"],
            "diagnostic_only": True,
            **_paired_problem_bootstrap(
                comparison["reference_outputs"],
                comparison["candidate_outputs"],
                samples=args.bootstrap_samples,
                seed=args.seed + 1_500_000 + oi,
            ),
        })
    # Bellman vs myopic at matched R and matched beliefs: the only difference is
    # whether the continuation value is credited, so this isolates it directly.
    # Not gated on the scorer -- the counts family is the cleanest read on the
    # continuation value alone, with no learned prior in the way.
    for fi, (family, horizon, learned) in enumerate(value_arms):
        if horizon is None:
            continue
        # Learned-transition arms pair against the SAME horizon with analytic
        # transitions, so the delta is the transition model and nothing else.
        reference_policy = (
            _value_policy(family, horizon, False) if learned else _value_policy(family, None)
        )
        candidate_policy = _value_policy(family, horizon, learned)
        for vi, value_of_correct in enumerate(value_grid):
            paired_comparisons.append({
                "reference_policy": reference_policy,
                "candidate_policy": candidate_policy,
                "budget": None,
                "value_of_correct": value_of_correct,
                "bellman_horizon": horizon,
                "learned_transitions": learned,
                **_paired_problem_bootstrap(
                    value_outputs[(value_of_correct, reference_policy)],
                    value_outputs[(value_of_correct, candidate_policy)],
                    samples=args.bootstrap_samples,
                    seed=args.seed + 900_000 + 10_000 * fi + 100 * vi,
                ),
            })
    if scorer is not None:
        candidate_families = tuple(f for f in ("sequential", "sequential_decay")
                                   if f in families)
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
        "calibration_map": args.calibration_map,
        "state_layout": args.state_layout,
        "bellman_horizons": bellman_horizons,
        "learned_transition_horizons": learned_horizons,
        "q_stop_arm": ({"family": args.q_stop_family, "bellman_horizon": args.q_stop_horizon}
                      if args.q_stop_family else None),
        "oracle_stopping_arm": (
            None if oracle_arm is None else {
                "family": oracle_arm[0],
                "bellman_horizon": oracle_arm[1],
                "diagnostic_only": True,
            }
        ),
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

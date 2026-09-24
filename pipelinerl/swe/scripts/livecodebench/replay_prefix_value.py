#!/usr/bin/env python3
"""Train and replay a no-verifier router conditioned on observed judge-score prefixes.

The regressor predicts the change in final-answer correctness from drawing each
route once. Training labels may inspect the stored next draw, but features contain
only information available before that draw. A one-shot route is always drawn
first; subsequent draws require positive predicted marginal value.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from pipelinerl.swe.scripts.livecodebench.mdp_utils import load_split_manifest, split_indices


FEATURE_VERSION = "judge_score_prefix_v1"
ROUTE_WEIGHTS = {"oss20lo": .45, "oss20md": .15, "dsv4f": .25,
                 "oss120md": .10, "oss120hi": .05}


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def next_draw(valid: np.ndarray, ordering: np.ndarray, ptr: np.ndarray,
              route: int) -> int | None:
    while ptr[route] < len(ordering[route]) and not valid[route, ordering[route, ptr[route]]]:
        ptr[route] += 1
    if ptr[route] == len(ordering[route]):
        return None
    return int(ordering[route, ptr[route]])


def prefix_features(prior: np.ndarray, costs: np.ndarray, route: int,
                    counts: np.ndarray, sums: np.ndarray, maxima: np.ndarray,
                    held_score: float, held_route: int, last_score: float,
                    last_route: int) -> np.ndarray:
    """Encode only the problem's pre-draw predictions and already observed scores."""
    n_routes = len(prior)
    eye = np.eye(n_routes, dtype=np.float32)
    zeros = np.zeros(n_routes, dtype=np.float32)
    return np.r_[prior, costs * 100.0, counts / 12.0,
                 sums / np.maximum(counts, 1.0), maxima, held_score,
                 eye[held_route] if held_route >= 0 else zeros,
                 counts.sum() / 20.0, last_score,
                 eye[last_route] if last_route >= 0 else zeros,
                 eye[route], prior[route], costs[route] * 100.0].astype(np.float32)


def training_rows(indices: np.ndarray, orders: np.ndarray, valid: np.ndarray,
                  outcomes: np.ndarray, judge: np.ndarray, priors: np.ndarray,
                  costs: np.ndarray, slots: list[str], max_steps: int,
                  seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 17)
    route_weights = np.array([ROUTE_WEIGHTS.get(s, 1.0) for s in slots])
    features: list[np.ndarray] = []
    labels: list[int] = []
    n_routes = len(slots)
    for pi in indices:
        for oi in range(orders.shape[1]):
            ptr = np.zeros(n_routes, dtype=int)
            counts = np.zeros(n_routes)
            sums = np.zeros(n_routes)
            maxima = np.zeros(n_routes)
            held_score, held_true, held_route = 0.0, False, -1
            last_score, last_route = 0.0, -1
            for _ in range(max_steps):
                available = [mi for mi in range(n_routes)
                             if next_draw(valid[pi], orders[pi, oi], ptr, mi) is not None]
                if not available:
                    break
                for mi in available:
                    draw = next_draw(valid[pi], orders[pi, oi], ptr, mi)
                    # The next draw's label is a supervised target, never a state feature.
                    gain = (int(outcomes[pi, mi, draw]) - int(held_true)
                            if judge[pi, mi, draw] > held_score else 0)
                    features.append(prefix_features(
                        priors[pi], costs[pi], mi, counts, sums, maxima,
                        held_score, held_route, last_score, last_route))
                    labels.append(gain)
                if oi % 2:
                    weights = route_weights[available]
                    mi = int(rng.choice(available, p=weights / weights.sum()))
                else:
                    mi = int(rng.choice(available))
                draw = next_draw(valid[pi], orders[pi, oi], ptr, mi)
                ptr[mi] += 1
                score = float(judge[pi, mi, draw])
                counts[mi] += 1
                sums[mi] += score
                maxima[mi] = max(maxima[mi], score)
                last_score, last_route = score, mi
                if score > held_score:
                    held_score = score
                    held_true = bool(outcomes[pi, mi, draw])
                    held_route = mi
    return np.stack(features), np.asarray(labels, dtype=float)


def replay_one(problem: int, ordering: int, value: float, *, adaptive: bool,
               model: HistGradientBoostingRegressor, orders: np.ndarray,
               valid: np.ndarray, outcomes: np.ndarray, judge: np.ndarray,
               priors: np.ndarray, expected_costs: np.ndarray,
               realized_costs: np.ndarray, max_steps: int,
               judge_cost: float = 0.0) -> tuple[float, float]:
    """judge_cost (USD) is the prefill charged for scoring each drawn attempt. Only the adaptive
    arm scores attempts -- its continue/stop decision after every draw depends on the score --
    so one-shot never pays it."""
    n_routes = priors.shape[1]
    ptr = np.zeros(n_routes, dtype=int)
    counts = np.zeros(n_routes)
    sums = np.zeros(n_routes)
    maxima = np.zeros(n_routes)
    held_score, held_true, held_route = 0.0, False, -1
    last_score, last_route = 0.0, -1
    spent = 0.0
    for step in range(max_steps if adaptive else 1):
        available = [mi for mi in range(n_routes)
                     if next_draw(valid[problem], orders[problem, ordering], ptr, mi) is not None]
        if not available:
            break
        if step == 0:
            # Both arms make exactly the same initial one-shot routing decision.
            route = max(available, key=lambda mi:
                        priors[problem, mi] * value - expected_costs[problem, mi])
        else:
            features = np.stack([prefix_features(
                priors[problem], expected_costs[problem], mi, counts, sums, maxima,
                held_score, held_route, last_score, last_route) for mi in available])
            net_values = (model.predict(features) * value
                          - expected_costs[problem, available] - judge_cost)
            if net_values.max() <= 0.0:
                break
            route = available[int(np.argmax(net_values))]
        draw = next_draw(valid[problem], orders[problem, ordering], ptr, route)
        ptr[route] += 1
        score = float(judge[problem, route, draw])
        spent += float(realized_costs[problem, route, draw]) + (judge_cost if adaptive else 0.0)
        counts[route] += 1
        sums[route] += score
        maxima[route] = max(maxima[route], score)
        last_score, last_route = score, route
        if score > held_score:
            held_score = score
            held_true = bool(outcomes[problem, route, draw])
            held_route = route
    return float(held_true), spent * 100.0  # cents


def run_split(indices: np.ndarray, value: float, *, adaptive: bool,
              model: HistGradientBoostingRegressor, orders: np.ndarray,
              valid: np.ndarray, outcomes: np.ndarray, judge: np.ndarray,
              priors: np.ndarray, expected_costs: np.ndarray,
              realized_costs: np.ndarray, max_steps: int,
              judge_cost: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    accuracy = np.zeros(len(indices))
    cost = np.zeros(len(indices))
    for row, pi in enumerate(indices):
        for ordering in range(orders.shape[1]):
            correct, spent = replay_one(
                int(pi), ordering, value, adaptive=adaptive, model=model,
                orders=orders, valid=valid, outcomes=outcomes, judge=judge,
                priors=priors, expected_costs=expected_costs,
                realized_costs=realized_costs, max_steps=max_steps, judge_cost=judge_cost)
            accuracy[row] += correct / orders.shape[1]
            cost[row] += spent / orders.shape[1]
    return accuracy, cost


def calibration_mix(points: list[dict], budget: float) -> tuple[int, int, float]:
    """Pick at most two value settings, with an optional zero-call mixture."""
    candidates = [(-1, 0.0, 0.0)] + [
        (i, row["cal_cost_cents"], row["cal_accuracy"])
        for i, row in enumerate(points)]
    best = (-np.inf, -1, -1, 0.0)
    for i, cost_i, accuracy_i in candidates:
        if cost_i > budget:
            continue
        for j, cost_j, accuracy_j in candidates[1:]:
            if cost_j < cost_i:
                continue
            weight_j = (1.0 if cost_j <= budget else
                        (budget - cost_i) / (cost_j - cost_i))
            accuracy = (1.0 - weight_j) * accuracy_i + weight_j * accuracy_j
            if accuracy > best[0]:
                best = (accuracy, i, j, weight_j)
    return int(best[1]), int(best[2]), float(best[3])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensors-dir", required=True)
    parser.add_argument("--content-preds", required=True)
    parser.add_argument("--judge-preds", required=True)
    parser.add_argument("--cost-preds", help="Our learned per-problem cost head; omit for train mean cost")
    parser.add_argument("--prices", required=True, help="route=USD-per-million-total-tokens,...")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-orderings", type=int, default=5)
    parser.add_argument("--train-max-steps", type=int, default=20)
    parser.add_argument("--replay-max-steps", type=int, default=35)
    parser.add_argument("--value-grid-points", type=int, default=24)
    parser.add_argument("--budget-cents", default="0.02,0.05,0.1,0.2,0.3,0.5")
    parser.add_argument("--baseline-cost-rule", choices=["train_mean", "paper"], default="train_mean",
                        help="cost estimate when --cost-preds is omitted. 'paper' is arXiv "
                             "2603.20895 exactly: this query's input tokens + the route's median "
                             "TRAIN output tokens, priced")
    parser.add_argument("--judge-cost-cents", type=float, default=0.0,
                        help="judge prefill charged per attempt the adaptive arm scores")
    parser.add_argument("--routes", help="comma-separated subset of rungs to keep (default: all)")
    args = parser.parse_args()

    tensor_dir = Path(args.tensors_dir)
    with np.load(tensor_dir / "tensors.npz", allow_pickle=True) as data:
        ids = [str(value) for value in data["problem_ids"]]
        slots = [str(value) for value in data["model_slots"]]
        valid = data["valid"].astype(bool)
        outcomes = data["execution_outcome"].astype(bool)
        prompt_tokens = data["prompt_tokens"].astype(float)
        completion_tokens = data["completion_tokens"].astype(float)
    all_slots = slots
    keep = list(range(len(slots)))
    if args.routes:
        wanted = args.routes.split(",")
        if unknown := set(wanted) - set(slots):
            raise ValueError(f"--routes names unknown rungs: {sorted(unknown)}")
        keep = [mi for mi, slot in enumerate(slots) if slot in wanted]
        slots = [all_slots[mi] for mi in keep]
        valid, outcomes = valid[:, keep], outcomes[:, keep]
        prompt_tokens, completion_tokens = prompt_tokens[:, keep], completion_tokens[:, keep]
    train, calibration, test = split_indices(
        load_split_manifest(tensor_dir / "split_manifest.json", ids), ids)
    prices = dict(item.split("=", 1) for item in args.prices.split(","))
    if set(prices) != set(slots):
        raise ValueError(f"--prices must name exactly these routes: {slots}")
    realized = np.stack([
        (prompt_tokens[:, mi] + completion_tokens[:, mi]) * float(prices[slot]) / 1e6
        for mi, slot in enumerate(slots)], axis=1)
    global_costs = np.array([
        realized[train, mi][valid[train, mi]].mean() for mi in range(len(slots))])
    expected = np.broadcast_to(global_costs, (len(ids), len(slots))).copy()
    if args.baseline_cost_rule == "paper":
        # Input length is known before generation; the prompt is identical across draws.
        with np.errstate(invalid="ignore"):
            input_tokens = np.nanmean(np.where(valid, prompt_tokens, np.nan), axis=2)
        for mi, slot in enumerate(slots):
            median_out = np.median(completion_tokens[train, mi][valid[train, mi]])
            fallback = np.nanmean(input_tokens[train, mi])
            expected[:, mi] = (np.where(np.isfinite(input_tokens[:, mi]), input_tokens[:, mi], fallback)
                               + median_out) * float(prices[slot]) / 1e6
    if args.cost_preds:
        rows = {str(row["problem_id"]): row for row in read_jsonl(Path(args.cost_preds))}
        if set(rows) != set(ids):
            raise ValueError("learned cost predictions must cover exactly the tensor problems")
        expected = np.stack([rows[pid]["expected_costs"][:len(all_slots)]
                             for pid in ids]).astype(float)[:, keep]
    content_rows = {str(row["problem_id"]): row for row in read_jsonl(Path(args.content_preds))}
    if set(content_rows) != set(ids):
        raise ValueError("content predictions must cover exactly the tensor problems")
    priors = np.stack([content_rows[pid]["p_successes"][:len(all_slots)]
                       for pid in ids]).astype(float)[:, keep]
    if np.any(expected <= 0) or not np.isfinite(expected).all():
        raise ValueError("expected costs must be finite and positive")

    judge = np.full(valid.shape, np.nan)
    row_by_id = {pid: i for i, pid in enumerate(ids)}
    route_by_name = {slot: i for i, slot in enumerate(slots)}
    pattern = re.compile("^(" + "|".join(re.escape(s) for s in
                                         sorted(all_slots, key=len, reverse=True)) + r")(\d+)$")
    for row in read_jsonl(Path(args.judge_preds)):
        if row.get("judge_mode") != "independent":
            raise ValueError("judge predictions must be independent and causal")
        pid, sep, suffix = str(row["example_id"]).partition("||")
        match = pattern.match(suffix) if sep else None
        if pid not in row_by_id or match is None:
            raise ValueError(f"unrecognized judge attempt ID: {row['example_id']}")
        if match.group(1) not in route_by_name:
            continue
        pi, mi, draw = row_by_id[pid], route_by_name[match.group(1)], int(match.group(2))
        if draw >= valid.shape[2] or not valid[pi, mi, draw] or np.isfinite(judge[pi, mi, draw]):
            raise ValueError(f"duplicate or invalid judge attempt: {row['example_id']}")
        judge[pi, mi, draw] = float(row["p_correct"])
    if not np.isfinite(judge[valid]).all():
        raise ValueError("judge predictions must cover every valid draw")

    judge_cost = args.judge_cost_cents / 100.0  # USD, like realized/expected costs
    rng = np.random.default_rng(args.seed)
    n_problems, n_routes, n_draws = valid.shape
    orders = np.array([[[rng.permutation(n_draws) for _ in range(n_routes)]
                        for _ in range(args.num_orderings)] for _ in range(n_problems)])
    features, labels = training_rows(
        train, orders, valid, outcomes, judge, priors, expected,
        slots, args.train_max_steps, args.seed)
    model = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=31, min_samples_leaf=120,
        learning_rate=.06, l2_regularization=1.0, random_state=args.seed)
    model.fit(features, labels)
    print(f"fit {len(labels)} train prefix/actions: "
          f"{(labels > 0).mean():.3%} improvements, {(labels < 0).mean():.3%} harms",
          flush=True)

    train_priors = np.array([
        outcomes[train, mi][valid[train, mi]].mean() for mi in range(n_routes)])
    break_even = global_costs / train_priors
    values = np.geomspace(.5 * break_even.min(), 200.0 * break_even.max(),
                          args.value_grid_points)
    cal_grid: dict[str, list[dict]] = {"prefix": [], "one_shot": []}
    for arm, adaptive in (("prefix", True), ("one_shot", False)):
        for i, value in enumerate(values):
            acc, cost = run_split(
                calibration, float(value), adaptive=adaptive, model=model,
                orders=orders, valid=valid, outcomes=outcomes, judge=judge,
                priors=priors, expected_costs=expected, realized_costs=realized,
                max_steps=args.replay_max_steps, judge_cost=judge_cost)
            cal_grid[arm].append({"value": float(value),
                                  "cal_accuracy": float(acc.mean()),
                                  "cal_cost_cents": float(cost.mean())})
            print(f"{arm} calibration {i + 1}/{len(values)}: "
                  f"{acc.mean():.3%} at {cost.mean():.4f} cents", flush=True)

    test_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    results = []
    for budget in [float(value) for value in args.budget_cents.split(",")]:
        for arm, adaptive in (("prefix", True), ("one_shot", False)):
            left, right, weight = calibration_mix(cal_grid[arm], budget)
            def test_at(index: int) -> tuple[np.ndarray, np.ndarray]:
                if index < 0:
                    return np.zeros(len(test)), np.zeros(len(test))
                key = (arm, index)
                if key not in test_cache:
                    test_cache[key] = run_split(
                        test, float(values[index]), adaptive=adaptive, model=model,
                        orders=orders, valid=valid, outcomes=outcomes, judge=judge,
                        priors=priors, expected_costs=expected, realized_costs=realized,
                        max_steps=args.replay_max_steps, judge_cost=judge_cost)
                return test_cache[key]
            left_acc, left_cost = test_at(left)
            right_acc, right_cost = test_at(right)
            accuracy = (1.0 - weight) * left_acc + weight * right_acc
            cost = (1.0 - weight) * left_cost + weight * right_cost
            results.append({"arm": arm, "calibration_budget_cents": budget,
                            "left_value": None if left < 0 else float(values[left]),
                            "right_value": float(values[right]),
                            "right_mix_weight": weight,
                            "test_accuracy": float(accuracy.mean()),
                            "test_cost_cents": float(cost.mean()),
                            "test_accuracy_by_problem": accuracy.tolist(),
                            "test_cost_cents_by_problem": cost.tolist()})
            print(f"{arm} budget {budget:.3f}: test {accuracy.mean():.3%} "
                  f"at {cost.mean():.4f} cents", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"feature_version": FEATURE_VERSION, "slots": slots, "model": model},
                output_dir / "prefix_value_model.joblib")
    (output_dir / "results.json").write_text(json.dumps({
        "feature_version": FEATURE_VERSION,
        "split_counts": {"train": len(train), "calibration": len(calibration), "test": len(test)},
        "routes": slots,
        "judge_cost_cents": args.judge_cost_cents,
        "cost_source": "learned" if args.cost_preds else args.baseline_cost_rule,
        "num_orderings": args.num_orderings,
        "train_rows": len(labels),
        "calibration_grid": cal_grid,
        "results": results,
        "test_problem_ids": [ids[int(i)] for i in test],
    }, indent=2) + "\n")
    print(f"wrote {output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    # Predictions are batches of only a few routes. Many OpenMP workers make
    # these tiny calls dramatically slower without changing the result.
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api="openmp"):
        main()

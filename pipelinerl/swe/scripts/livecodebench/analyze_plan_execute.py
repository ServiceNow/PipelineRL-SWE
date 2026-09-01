#!/usr/bin/env python3
"""Is "large plans, small executes" a new rung on the ladder or a new dimension?

Reads the composite route's execution rows (from collect_lcb_plan_execute.py) and the
existing (problem x model x draw) tensor, and answers, on the shared problems:

  1. solve rate per draw, pass@k, and the hazard P(draw j succeeds | first j-1 failed);
  2. c_inf against oss120: union(composite, oss120) - max(composite, oss120) at matched k,
     plus the count of problems ONLY the composite solves (off-ladder successes);
  3. cost: plan tokens at the planner's price plus executor tokens at the executor's price,
     against one oss120 call, and cost per solved problem for the stop-on-pass policies
     {oss120 x1, composite x k, scout x k}.

Prices default to the project's recorded USD/M-token rates (replay_mdp_baseline.py). Pass
--price-* to re-derive at current list prices; the qualitative reading should not depend
on it, and if it does that is a finding.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

RECORDED_USD_PER_M = {"scout": 0.278, "oss20": 1.299, "oss120": 11.13}


def read_rows(path: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[str(r["problem_id"])] = r
    return rows


def load_composite(exec_dir: Path, route_label: str, split: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict]:
    files = sorted(glob.glob(str(exec_dir / f"{route_label}_{split}_d*.jsonl")),
                   key=lambda p: int(p.rsplit("_d", 1)[1].split(".")[0]))
    if not files:
        raise FileNotFoundError(f"no {route_label}_{split}_d*.jsonl under {exec_dir}")
    draws = [read_rows(f) for f in files]
    pids = sorted(set.intersection(*[set(d) for d in draws]))
    K = len(draws)
    outcome = np.zeros((len(pids), K), bool)
    exec_tok = np.zeros((len(pids), K), float)
    for k, d in enumerate(draws):
        for i, pid in enumerate(pids):
            outcome[i, k] = bool(d[pid].get("resolved"))
            exec_tok[i, k] = float(d[pid].get("completion_tokens") or 0) + float(d[pid].get("prompt_tokens") or 0)
    plan_tok = np.array([
        float(draws[0][pid].get("plan_prompt_tokens") or 0) + float(draws[0][pid].get("plan_completion_tokens") or 0)
        for pid in pids
    ])
    plan_meta = {
        "plan_route_label": draws[0][pids[0]].get("plan_route_label"),
        "plan_model": draws[0][pids[0]].get("plan_model"),
        "mean_plan_words": float(np.mean([draws[0][p].get("plan_words") or 0 for p in pids])),
        "mean_plan_completion_tokens": float(np.mean([draws[0][p].get("plan_completion_tokens") or 0 for p in pids])),
        "mean_plan_prompt_tokens": float(np.mean([draws[0][p].get("plan_prompt_tokens") or 0 for p in pids])),
        "n_draw_files": K,
    }
    return pids, outcome, exec_tok, plan_tok, plan_meta


def pass_at_k(outcome: np.ndarray) -> np.ndarray:
    """Fraction solved within the first k draws, k = 1..K."""
    return np.cumsum(outcome, axis=1).astype(bool).mean(axis=0)


def hazard(outcome: np.ndarray) -> list[float]:
    """P(draw j succeeds | draws < j all failed)."""
    out = []
    alive = np.ones(outcome.shape[0], bool)
    for j in range(outcome.shape[1]):
        n = alive.sum()
        out.append(float(outcome[alive, j].mean()) if n else float("nan"))
        alive &= ~outcome[:, j]
    return out


def c_inf(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """union - best single, within k draws each, in percentage points."""
    sa = a[:, :k].any(axis=1)
    sb = b[:, :k].any(axis=1)
    return 100.0 * ((sa | sb).mean() - max(sa.mean(), sb.mean()))


def stop_on_pass_cost(outcome: np.ndarray, per_draw_cost: np.ndarray, fixed_cost: np.ndarray, k: int) -> tuple[float, float]:
    """Mean cost and solve rate of: pay fixed_cost, then draw up to k times, stopping on a pass."""
    total = fixed_cost.astype(float).copy()
    alive = np.ones(outcome.shape[0], bool)
    solved = np.zeros(outcome.shape[0], bool)
    for j in range(min(k, outcome.shape[1])):
        total[alive] += per_draw_cost[alive, j]
        solved |= alive & outcome[:, j]
        alive &= ~outcome[:, j]
    return float(total.mean()), float(solved.mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tensors-dir", required=True, help="dir with tensors.npz + problems.jsonl")
    p.add_argument("--exec-dir", required=True)
    p.add_argument("--route-label", required=True, help="composite route label, e.g. plan120_scout")
    p.add_argument("--split", default="eval", choices=["train", "eval", "all"])
    p.add_argument("--executor-slot", default="scout", help="tensor slot whose price the executor pays")
    p.add_argument("--planner-slot", default="oss120", help="tensor slot whose price the planner pays")
    p.add_argument("--reference-slot", default="oss120", help="tensor slot the composite is compared against")
    p.add_argument("--price-scout", type=float, default=RECORDED_USD_PER_M["scout"])
    p.add_argument("--price-oss20", type=float, default=RECORDED_USD_PER_M["oss20"])
    p.add_argument("--price-oss120", type=float, default=RECORDED_USD_PER_M["oss120"])
    p.add_argument("--out", default="")
    args = p.parse_args()

    prices = {"scout": args.price_scout, "oss20": args.price_oss20, "oss120": args.price_oss120}
    t = np.load(Path(args.tensors_dir) / "tensors.npz", allow_pickle=True)
    problems = [json.loads(l) for l in open(Path(args.tensors_dir) / "problems.jsonl") if l.strip()]
    split_of = {pr["problem_id"]: pr.get("source_temporal_split") for pr in problems}
    slots = list(t["model_slots"])
    tensor_pids = list(t["problem_ids"])
    idx_of = {pid: i for i, pid in enumerate(tensor_pids)}

    splits = ["train", "eval"] if args.split == "all" else [args.split]
    pids_all, out_all, tok_all, plan_all, meta = [], [], [], [], None
    for s in splits:
        pids, outcome, exec_tok, plan_tok, meta = load_composite(Path(args.exec_dir), args.route_label, s)
        keep = [i for i, pid in enumerate(pids) if pid in idx_of and split_of.get(pid, s) == s]
        pids_all += [pids[i] for i in keep]
        out_all.append(outcome[keep]); tok_all.append(exec_tok[keep]); plan_all.append(plan_tok[keep])
    outcome = np.concatenate(out_all); exec_tok = np.concatenate(tok_all); plan_tok = np.concatenate(plan_all)
    rows = np.array([idx_of[pid] for pid in pids_all])
    K = outcome.shape[1]

    ref = slots.index(args.reference_slot)
    exe = slots.index(args.executor_slot)
    final = t["final_outcome"][rows] & t["valid"][rows]
    ref_out = final[:, ref, :]
    exe_out = final[:, exe, :]
    ref_tok = (t["prompt_tokens"] + t["completion_tokens"])[rows, ref, :]
    exe_tok_alone = (t["prompt_tokens"] + t["completion_tokens"])[rows, exe, :]

    report: dict = {"n_problems": len(pids_all), "split": args.split, "composite": args.route_label,
                    "plan": meta, "prices_usd_per_m": prices}

    report["solve_rate_per_draw"] = outcome.mean(axis=0).round(4).tolist()
    report["pass_at_k"] = {
        "composite": pass_at_k(outcome).round(4).tolist(),
        args.executor_slot: pass_at_k(exe_out).round(4).tolist(),
        args.reference_slot: pass_at_k(ref_out).round(4).tolist(),
    }
    report["hazard"] = {"composite": hazard(outcome), args.executor_slot: hazard(exe_out),
                        args.reference_slot: hazard(ref_out)}

    ks = [k for k in (1, 2, 3, 4, 6, 10) if k <= min(K, ref_out.shape[1])]
    report["c_inf_vs_reference_pt"] = {str(k): round(c_inf(outcome, ref_out, k), 2) for k in ks}
    report["c_inf_executor_alone_vs_reference_pt"] = {str(k): round(c_inf(exe_out, ref_out, k), 2) for k in ks}
    comp_any = outcome.any(axis=1); ref_any = ref_out.any(axis=1); exe_any = exe_out.any(axis=1)
    report["off_ladder"] = {
        "composite_only_vs_reference_all_draws": int((comp_any & ~ref_any).sum()),
        "composite_only_vs_reference_and_executor": int((comp_any & ~ref_any & ~exe_any).sum()),
        "reference_only": int((ref_any & ~comp_any).sum()),
        "composite_solves_that_executor_alone_missed": int((comp_any & ~exe_any).sum()),
    }

    # Cost. Composite pays the plan once at the planner's price, then executor draws.
    plan_usd = plan_tok * prices[args.planner_slot] / 1e6
    exec_usd = exec_tok * prices[args.executor_slot] / 1e6
    ref_usd = ref_tok * prices[args.reference_slot] / 1e6
    exe_alone_usd = exe_tok_alone * prices[args.executor_slot] / 1e6
    zero = np.zeros(len(pids_all))
    report["mean_tokens"] = {
        "plan_total": float(plan_tok.mean()),
        "reference_single_call_total": float(ref_tok[:, 0].mean()),
        "plan_over_reference_call": float(plan_tok.mean() / max(1e-9, ref_tok[:, 0].mean())),
        "executor_draw_total": float(exec_tok.mean()),
    }
    policies = {}
    for k in ks:
        c, s = stop_on_pass_cost(outcome, exec_usd, plan_usd, k)
        policies[f"composite_x{k}"] = {"cost": c, "solve": s, "cost_per_solved": c / max(s, 1e-9)}
        c, s = stop_on_pass_cost(exe_out, exe_alone_usd, zero, k)
        policies[f"{args.executor_slot}_x{k}"] = {"cost": c, "solve": s, "cost_per_solved": c / max(s, 1e-9)}
        c, s = stop_on_pass_cost(ref_out, ref_usd, zero, k)
        policies[f"{args.reference_slot}_x{k}"] = {"cost": c, "solve": s, "cost_per_solved": c / max(s, 1e-9)}
    report["stop_on_pass_policies"] = policies

    # Verdict heuristics, stated as numbers not adjectives.
    print(f"\n== {args.route_label} on {len(pids_all)} {args.split} problems  (plan: {meta['plan_route_label']} / {meta['plan_model']}) ==")
    print(f"plan: {meta['mean_plan_words']:.0f} words, {meta['mean_plan_completion_tokens']:.0f} completion tok "
          f"(+{meta['mean_plan_prompt_tokens']:.0f} prompt); one {args.reference_slot} call: {ref_tok[:,0].mean():.0f} tok "
          f"-> plan is {100*report['mean_tokens']['plan_over_reference_call']:.0f}% of a reference call")
    print("pass@k        " + "  ".join(f"k={k}" for k in range(1, K + 1)))
    for name, v in report["pass_at_k"].items():
        print(f"  {name:12s}" + "  ".join(f"{100*x:5.1f}" for x in v[:K]))
    print("hazard        " + "  ".join(f"{100*x:5.1f}" for x in report["hazard"]["composite"]))
    print("c_inf vs " + args.reference_slot + " (pt): composite " + str(report["c_inf_vs_reference_pt"])
          + " | executor alone " + str(report["c_inf_executor_alone_vs_reference_pt"]))
    print("off-ladder: " + json.dumps(report["off_ladder"]))
    print(f"{'policy':16s} {'solve':>7s} {'$/problem':>10s} {'$/solved':>10s}")
    for name, v in policies.items():
        print(f"{name:16s} {100*v['solve']:6.1f}% {v['cost']:10.5f} {v['cost_per_solved']:10.5f}")

    out = Path(args.out) if args.out else Path(args.exec_dir) / f"analysis_{args.route_label}_{args.split}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

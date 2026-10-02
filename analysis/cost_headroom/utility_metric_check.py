"""Does a per-problem utility metric give tighter intervals than savings-at-matched-accuracy?

Metric: realized net utility at a fixed value per correct answer V (cents), U(V) = mean over problems of [V * correct - spend],
where each arm routes with argmax_m V p_m(x) - c_m(x) (its own cost estimate) and pays realized market spend. Two arms at the
same V give a paired per-problem difference, so no frontier, hull, accuracy band or cost ratio is involved.
V is fixed from CALIBRATION only: the range where the median-pricing arm's calibration accuracy moves from 5% to 95% of its span;
we report three V values at 25/50/75% of that range (log scale) and the average over a log-uniform grid across it.
Data: fresh MMLU-Pro (6,500) and Omni-MATH (1,000) with Codex's arrays, strata and weights (evaluate_expanded_calibrated_policies.py),
and the original test sets of LCB / Omni / MMLU-Pro. Comparisons: learned cost vs median-length pricing (same success predictions);
on the original test sets also cost-from-success vs learned, and Intern-Decision success vs ours (our costs).
Usage: python utility_metric_check.py
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import POOLS, read_predictions
from decompose import MK, R

VALUES = np.geomspace(1e-5, 100, 400)
rng0 = 20261004


def v_range(p, cost, q, sel):
    """calibration-only V range where accuracy actually moves (median-pricing arm)"""
    acc = np.array([q[sel][np.arange(len(sel)), (V * p[sel] - cost[sel]).argmax(1)].mean() for V in VALUES])
    lo, hi = acc.min(), acc.max()
    a = VALUES[np.argmax(acc >= lo + .05 * (hi - lo))]; b = VALUES[np.argmax(acc >= lo + .95 * (hi - lo))]
    return a, b


def per_problem_utility(p, cost, q, paid, V, rows):
    m = (V * p[rows] - cost[rows]).argmax(1); r = np.arange(len(rows))
    return V * q[rows][r, m] - paid[rows][r, m], q[rows][r, m], paid[rows][r, m]


def compare(tag, pA, cA, pB, cB, q, paid, cal, test, mean_fn, draws):
    """A minus B (A = the arm we report as 'ours' unless stated); V range from calibration with arm B's predictions"""
    a, b = v_range(pB, cB, q, cal)
    grid = np.geomspace(a, b, 25); pts = {"V25": np.exp(np.log(a) + .25 * np.log(b / a)), "V50": np.sqrt(a * b), "V75": np.exp(np.log(a) + .75 * np.log(b / a))}
    out = {"V_range_cents": [float(a), float(b)]}
    for name, V in pts.items():
        uA, qA, sA = per_problem_utility(pA, cA, q, paid, V, test); uB, qB, sB = per_problem_utility(pB, cB, q, paid, V, test)
        d = uA - uB; boot = [mean_fn(d, s) for s in draws]
        ref = mean_fn(sB)                                             # express the utility gain relative to arm B's spend
        out[name] = dict(V_cents=float(V), dU_cents=mean_fn(d), ci=np.percentile(boot, [2.5, 97.5]).tolist(),
                         dU_pct_of_spend=100 * mean_fn(d) / ref, ci_pct=(100 * np.percentile(boot, [2.5, 97.5]) / ref).tolist(),
                         acc=[mean_fn(qA), mean_fn(qB)], spend_cents=[mean_fn(sA), ref])
    D = np.mean([per_problem_utility(pA, cA, q, paid, V, test)[0] - per_problem_utility(pB, cB, q, paid, V, test)[0] for V in grid], 0)
    ref = np.mean([mean_fn(per_problem_utility(pB, cB, q, paid, V, test)[2]) for V in grid])
    boot = [mean_fn(D, s) for s in draws]
    out["avg_over_range"] = dict(dU_cents=mean_fn(D), ci=np.percentile(boot, [2.5, 97.5]).tolist(), dU_pct_of_spend=100 * mean_fn(D) / ref,
                                 ci_pct=(100 * np.percentile(boot, [2.5, 97.5]) / ref).tolist())
    rel = lambda x: (x["ci_pct"][1] - x["ci_pct"][0]) / 2 / abs(x["dU_pct_of_spend"]) if x["dU_pct_of_spend"] else np.inf
    print(f"  {tag}: V range {a:.4g}-{b:.4g} cents/correct")
    for k in ("V25", "V50", "V75", "avg_over_range"):
        x = out[k]; extra = f"  acc {x['acc'][0]*100:.1f} vs {x['acc'][1]*100:.1f}%" if "acc" in x else ""
        print(f"     {k:<15} utility gain = {x['dU_pct_of_spend']:+6.1f}% of baseline spend [{x['ci_pct'][0]:+6.1f}, {x['ci_pct'][1]:+6.1f}]"
              f"  (CI half-width / |effect| = {rel(x):.2f}){extra}")
    return out


res = {}
# ---------------- fresh sets (Codex's exact arrays) ----------------
for ds in ("mmlupro", "omni500"):
    label = "MMLU-Pro" if ds == "mmlupro" else "Omni"; name, cost_file = POOLS[label]; old = R / name
    folder = R / "expanded_eval_20261001" / ds; t = np.load(folder / "tensors.npz", allow_pickle=True)
    ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"])); index = {p: i for i, p in enumerate(ids)}
    split = json.loads((old / "split_manifest.json").read_text())
    tr, ca = [np.asarray([index[str(p)] for p in split[k + "_problem_ids"]]) for k in ("train", "calibration")]
    n_old = len(np.load(old / "tensors.npz", allow_pickle=True)["problem_ids"]); fresh = np.arange(n_old, len(ids))
    valid = t["valid"].astype(bool); counts = valid.sum(2)
    q = np.where(valid, t["final_outcome"], 0).sum(2) / counts; lengths = np.where(valid, t["completion_tokens"], 0).sum(2) / counts
    inputs = np.where(valid, t["prompt_tokens"], 0).sum(2) / counts
    pin = np.array([MK[s][0] for s in slots]) / 1e6; pout = np.array([MK[s][1] for s in slots]) / 1e6; paid = (inputs * pin + lengths * pout) * 100
    median = np.asarray([np.median(t["completion_tokens"][tr, j][valid[tr, j]]) for j in range(len(slots))])
    learned = read_predictions(folder / "paper_cost_preds.jsonl", ids, "expected_costs", len(slots)) * 100
    learned[:n_old] = read_predictions(old / cost_file, ids[:n_old], "expected_costs", len(slots)) * 100
    med = (inputs * pin + median * pout) * 100
    p = read_predictions(folder / "success_preds.jsonl", ids, "p_successes", len(slots))
    p[:n_old] = read_predictions(old / "content_preds.jsonl", ids[:n_old], "p_successes", len(slots))
    problems = [json.loads(l) for l in (folder / "problems.jsonl").read_text().splitlines()]
    strata = np.asarray([str(x.get("subject", "")) if ds == "mmlupro" else str(round(float(x.get("difficulty", 0)))) for x in problems[n_old:]])
    weights = json.loads((folder / "expansion_manifest.json").read_text())["stratum_weights"]
    groups = {s: np.flatnonzero(strata == s) for s in weights}
    rng = np.random.default_rng(rng0); draws = [{s: ix[rng.integers(0, len(ix), len(ix))] for s, ix in groups.items()} for _ in range(2000)]
    mean_fn = lambda x, sample=None: sum(float(weights[s]) * float(x[(sample or groups)[s]].mean()) for s in groups)
    print(f"\n===== FRESH {label} (n={len(fresh)}): learned cost vs median-length pricing, same success predictions")
    res[f"fresh_{label}"] = compare("learned - median", p, learned, p, med, q, paid, ca, fresh, mean_fn, draws)

# ---------------- original test sets ----------------
ORIG = [("LCB", "pool_v2_tensors_5rung", "cost_preds_probe.jsonl", "lcb"), ("Omni", "omni500_tensors", "cost_preds_probe_thinking.jsonl", "omni"),
        ("MMLU-Pro", "mmlupro_tensors", "cost_preds_probe_instruct.jsonl", "mmlu_pro")]
for label, name, cf, intern_dir in ORIG:
    D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
    ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"])); index = {p: i for i, p in enumerate(ids)}; M = len(slots)
    valid = t["valid"].astype(bool); counts = np.maximum(valid.sum(2), 1)
    q = np.where(valid, t["final_outcome"], 0).sum(2) / counts; lengths = np.where(valid, t["completion_tokens"], 0).sum(2) / counts
    inputs = np.where(valid, t["prompt_tokens"], 0).sum(2) / counts
    pin = np.array([MK[s][0] for s in slots]) / 1e6; pout = np.array([MK[s][1] for s in slots]) / 1e6; paid = (inputs * pin + lengths * pout) * 100
    sp = json.loads((D / "split_manifest.json").read_text()); tr, ca, te = [np.asarray([index[str(p)] for p in sp[k + "_problem_ids"]]) for k in ("train", "calibration", "test")]
    median = np.asarray([np.median(t["completion_tokens"][tr, j][valid[tr, j]]) for j in range(M)]); med = (inputs * pin + median * pout) * 100
    learned = read_predictions(D / cf, ids, "expected_costs", M) * 100; fromsucc = read_predictions(D / "cost_preds_fromsuccess.jsonl", ids, "expected_costs", M) * 100
    p = read_predictions(D / "content_preds.jsonl", ids, "p_successes", M)
    rng = np.random.default_rng(rng0); draws = [rng.integers(0, len(te), len(te)) for _ in range(2000)]
    pos = {i: j for j, i in enumerate(te)}
    mean_fn = lambda x, sample=None: float(x.mean() if sample is None else x[sample].mean())
    print(f"\n===== ORIGINAL TEST {label} (n={len(te)})")
    r = {"learned - median": compare("learned - median", p, learned, p, med, q, paid, ca, te, mean_fn, draws),
         "learned - cost-from-success": compare("learned - cost-from-success", p, learned, p, fromsucc, q, paid, ca, te, mean_fn, draws)}
    z = np.load(R / "intern_finetune_20261001" / intern_dir / "test_predictions.npz", allow_pickle=True)
    pI = p.copy(); zi = [index[str(x)] for x in z["problem_ids"]]; pI[zi] = z["p_successes"]
    if set(zi) == set(te.tolist()):
        r["Intern - ours (our costs)"] = compare("Intern success - ours (our costs)", pI, learned, p, learned, q, paid, ca, te, mean_fn, draws)
    res[f"original_{label}"] = r
json.dump(res, open(Path(__file__).parent / "utility_metric_check.json", "w"), indent=1, default=float)

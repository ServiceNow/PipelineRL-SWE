"""Fresh-set routing frontiers at BILLED prices for Figure 1(b) (NEW_PATH 4.A.42/4.A.43/4.A.46 protocol).

Same inputs and arithmetic as analysis/cost_headroom/fresh_baselines.py and deploy_matched.py: frozen success predictions shared by
both arms, realized cost = billed usage_cost, predictions priced at effective billed $/token (billed.py), unweighted fresh problems,
V grid geomspace(1e-7, 1, 400). For each arm we store the deterministic V sweep and its convex hull (the frontier the matched-accuracy
metric integrates over); for the figure's connectors we store our cost and the median rule's cost at a few accuracies inside the
interior 90% of the shared band, with the cost saved. Calibration-selected deployable policies (Table 2 targets) are stored as points.
Output: data/fresh_billed_curves.json.  Usage: python make_billed_curves.py
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis" / "cost_headroom"))
from carrot_compare import POOLS, read_predictions  # noqa: E402
from decompose import MK, R, hull, cost_at  # noqa: E402
from billed import RATE  # noqa: E402

VALUES = np.geomspace(1e-7, 1, 400)
TARGETS = {"mmlupro": (0.65, 0.75, 0.85), "omni500": (0.65, 0.70, 0.75), "lcb": ()}
rate_of = lambda s: RATE["oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")]


def pick(p, c, q, paid, target):
    ch = (VALUES[:, None, None] * p[None] - c[None]).argmax(2); r = np.arange(len(p))[None]
    acc, sp = q[r, ch].mean(1), paid[r, ch].mean(1); ok = np.flatnonzero(acc >= target - 1e-12)
    return None if not len(ok) else int(ok[np.argmin(sp[ok])])


out = {"protocol": __doc__.strip().splitlines()[0], "datasets": {}}
for ds in ("lcb", "mmlupro", "omni500"):
    if ds == "lcb":                       # LCB: its own temporal TEST split; realized tokens x billed rates (no per-call billing there)
        label, name, cost_file = "LCB", "pool_v2_tensors_5rung", "cost_preds_probe.jsonl"; old = F = R / name
    else:
        label = "MMLU-Pro" if ds == "mmlupro" else "Omni"; name, cost_file = POOLS[label]; old = R / name
        F = R / "expanded_eval_20261001" / ds
    t = np.load(F / "tensors.npz", allow_pickle=True)
    ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"])); idx = {p: i for i, p in enumerate(ids)}
    sp = json.loads((old / "split_manifest.json").read_text())
    tr, ca = [np.array([idx[str(p)] for p in sp[k + "_problem_ids"]]) for k in ("train", "calibration")]
    n_old = len(np.load(old / "tensors.npz", allow_pickle=True)["problem_ids"]); fresh = np.arange(n_old, len(ids))
    v = t["valid"].astype(bool); cnt = np.maximum(v.sum(2), 1)
    q = np.where(v, t["final_outcome"], 0).sum(2) / cnt; L = np.where(v, t["completion_tokens"], 0).sum(2) / cnt
    I = np.where(v, t["prompt_tokens"], 0).sum(2) / cnt
    rates = np.array([rate_of(s) for s in slots]); paid = I * rates[:, 0] + L * rates[:, 1]
    if ds == "lcb":
        fresh = np.array([idx[str(x)] for x in sp["test_problem_ids"]])
    for f in ([] if ds == "lcb" else glob.glob(f"{R}/math_expand_20261001/{ds}/*_d0.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("finish_reason") != "error" and r.get("usage_cost") is not None and r["problem_id"] in idx and r["route_label"] in slots:
                paid[idx[r["problem_id"]], slots.index(r["route_label"])] = r["usage_cost"]
    if ds == "lcb":
        learned = read_predictions(old / cost_file, ids, "expected_costs", len(slots)); p = read_predictions(old / "content_preds.jsonl", ids, "p_successes", len(slots))
    else:
        learned = read_predictions(F / "paper_cost_preds.jsonl", ids, "expected_costs", len(slots))
        learned[:n_old] = read_predictions(old / cost_file, ids[:n_old], "expected_costs", len(slots))
        p = read_predictions(F / "success_preds.jsonl", ids, "p_successes", len(slots))
        p[:n_old] = read_predictions(old / "content_preds.jsonl", ids[:n_old], "p_successes", len(slots))
    asg = np.array([[MK[s][0] / 1e6, MK[s][1] / 1e6] for s in slots]); tok = np.maximum((learned - I * asg[:, 0]) / asg[:, 1], 1)
    med = np.array([np.median(t["completion_tokens"][tr, k][v[tr, k]]) for k in range(len(slots))])
    cost = {"learned": I * rates[:, 0] + tok * rates[:, 1], "median": I * rates[:, 0] + med[None] * rates[:, 1]}
    fr = fresh[(v[fresh].sum(2) > 0).all(1)]; rr = np.arange(len(fr))[None]
    res = {"n": int(len(fr)), "curves": {}, "selected": {}}
    H = {}
    for arm, c in cost.items():
        ch = (VALUES[:, None, None] * p[fr][None] - c[fr][None]).argmax(2)
        acc, spend = q[fr][rr, ch].mean(1), paid[fr][rr, ch].mean(1)
        H[arm] = hull(list(zip(spend.tolist(), acc.tolist())))
        res["curves"][arm] = dict(accuracy=acc.tolist(), mean_cost_usd=spend.tolist(),
                                  hull_cost_usd=[h[0] for h in H[arm]], hull_accuracy=[h[1] for h in H[arm]])
        sel = {}
        for target in TARGETS[ds]:
            j = pick(p[ca], c[ca], q[ca], paid[ca], target)
            if j is not None:
                sel[str(target)] = dict(accuracy=float(acc[j]), mean_cost_usd=float(spend[j]))
        res["selected"][arm] = sel
    lo, hi = max(H["learned"][0][1], H["median"][0][1]), min(H["learned"][-1][1], H["median"][-1][1])
    band = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    res["band"] = [float(lo), float(hi)]
    res["saved_band_mean"] = 1 - float(np.exp(np.nanmean(np.log([cost_at(H["learned"], x) / cost_at(H["median"], x) for x in band]))))
    res["connectors"] = []
    for x in band[[1, 5, 10]]:
        a, b = cost_at(H["learned"], x), cost_at(H["median"], x)
        res["connectors"].append(dict(accuracy=float(x), ours_cost_usd=float(a), median_cost_usd=float(b), saved=float(1 - a / b)))
    print(f"{label}: n={len(fr)} band {lo:.3f}-{hi:.3f} saved {res['saved_band_mean']*100:.1f}%  connectors "
          + ", ".join(f"{c['accuracy']*100:.1f}%: {c['saved']*100:.0f}%" for c in res["connectors"]))
    out["datasets"][ds] = res
(HERE / "data" / f"fresh_billed_curves{os.environ.get('RESULT_TAG', '')}.json").write_text(json.dumps(out, indent=1) + "\n")

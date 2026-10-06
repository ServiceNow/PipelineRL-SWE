"""Deployable policies at matched accuracy (NEW_PATH 4.A.46): for each calibration-selected policy (cheapest V reaching the target on
ORIGINAL calibration, billed prices), report raw savings vs the median policy AND savings vs the median-pricing frontier evaluated at
OUR achieved fresh accuracy; plus our efficiency loss vs our own fresh frontier. Paired problem bootstrap (300). Unweighted.
Usage: python deploy_matched.py
"""
import json, os, glob, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import POOLS, read_predictions
from decompose import MK, R, hull, cost_at
from billed import RATE                                       # same fit as provider_routing.billed_rates, without running that analysis
VALUES = np.geomspace(1e-7, 1, 400)                          # same grid as billed_reprice.py (Table 2)
rate_of = lambda s: RATE["oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")]


def pick(p, c, q, paid, target):
    ch = (VALUES[:, None, None] * p[None] - c[None]).argmax(2); r = np.arange(len(p))[None]
    acc, sp = q[r, ch].mean(1), paid[r, ch].mean(1); ok = np.flatnonzero(acc >= target - 1e-12)
    return None if not len(ok) else float(VALUES[ok[np.argmin(sp[ok])]])


out = {}
for ds, targets in (("mmlupro", (0.65, 0.75, 0.85)), ("omni500", (0.65, 0.70, 0.75))):
    label = "MMLU-Pro" if ds == "mmlupro" else "Omni"; name, cost_file = POOLS[label]; old = R / name
    F = R / "expanded_eval_20261001" / ds; t = np.load(F / "tensors.npz", allow_pickle=True)
    ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"])); idx = {p: i for i, p in enumerate(ids)}
    sp = json.loads((old / "split_manifest.json").read_text()); tr, ca = [np.array([idx[str(p)] for p in sp[k + "_problem_ids"]]) for k in ("train", "calibration")]
    n_old = len(np.load(old / "tensors.npz", allow_pickle=True)["problem_ids"]); fresh = np.arange(n_old, len(ids))
    v = t["valid"].astype(bool); cnt = np.maximum(v.sum(2), 1)
    q = np.where(v, t["final_outcome"], 0).sum(2) / cnt; L = np.where(v, t["completion_tokens"], 0).sum(2) / cnt; I = np.where(v, t["prompt_tokens"], 0).sum(2) / cnt
    rates = np.array([rate_of(s) for s in slots]); paid = I * rates[:, 0] + L * rates[:, 1]
    for f in glob.glob(f"{R}/math_expand_20261001/{ds}/*_d0.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("finish_reason") != "error" and r.get("usage_cost") is not None and r["problem_id"] in idx and r["route_label"] in slots:
                paid[idx[r["problem_id"]], slots.index(r["route_label"])] = r["usage_cost"]
    learned = read_predictions(F / "paper_cost_preds.jsonl", ids, "expected_costs", len(slots)); learned[:n_old] = read_predictions(old / cost_file, ids[:n_old], "expected_costs", len(slots))
    p = read_predictions(F / "success_preds.jsonl", ids, "p_successes", len(slots)); p[:n_old] = read_predictions(old / "content_preds.jsonl", ids[:n_old], "p_successes", len(slots))
    asg = np.array([[MK[s][0] / 1e6, MK[s][1] / 1e6] for s in slots]); tok = np.maximum((learned - I * asg[:, 0]) / asg[:, 1], 1)
    med = np.array([np.median(t["completion_tokens"][tr, k][v[tr, k]]) for k in range(len(slots))])
    cL = I * rates[:, 0] + tok * rates[:, 1]; cM = I * rates[:, 0] + med[None] * rates[:, 1]
    fr = fresh[(v[fresh].sum(2) > 0).all(1)]
    # per-problem choices for every V, for both arms (fresh)
    chL = (VALUES[:, None, None] * p[fr][None] - cL[fr][None]).argmax(2); chM = (VALUES[:, None, None] * p[fr][None] - cM[fr][None]).argmax(2)
    r = np.arange(len(fr))[None]; QL, PL, QM, PM = q[fr][r, chL], paid[fr][r, chL], q[fr][r, chM], paid[fr][r, chM]   # [nV, n]

    def stats(ii, VL, VM):
        HM = hull(list(zip(PM[:, ii].mean(1), QM[:, ii].mean(1)))); HL = hull(list(zip(PL[:, ii].mean(1), QL[:, ii].mean(1))))
        jL, jM = int(np.argmin(abs(VALUES - VL))), int(np.argmin(abs(VALUES - VM)))
        aL, sL, sM = QL[jL, ii].mean(), PL[jL, ii].mean(), PM[jM, ii].mean()
        cm = cost_at(HM, aL); cl = cost_at(HL, aL)
        return 1 - sL / sM, (1 - sL / cm) if np.isfinite(cm) else np.nan, (sL / cl - 1) if np.isfinite(cl) else np.nan
    rng = np.random.default_rng(0); BS = [rng.integers(0, len(fr), len(fr)) for _ in range(300)]; res = {}
    for target in targets:
        VL, VM = pick(p[ca], cL[ca], q[ca], paid[ca], target), pick(p[ca], cM[ca], q[ca], paid[ca], target)
        raw, adj, eff = stats(np.arange(len(fr)), VL, VM); bs = np.array([stats(b, VL, VM) for b in BS])
        res[target] = dict(raw=raw, matched=adj, matched_ci=list(np.nanpercentile(bs[:, 1], [2.5, 97.5])), own_loss=eff)
        print(f"{label} {target:.2f}: raw {raw*100:+.1f}%  at matched accuracy {adj*100:+.1f}% [{np.nanpercentile(bs[:,1],2.5)*100:+.1f}, {np.nanpercentile(bs[:,1],97.5)*100:+.1f}]  own-frontier loss {eff*100:+.1f}%", flush=True)
    out[label] = res
json.dump(out, open(Path(__file__).parent / f"deploy_matched{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)

"""Intern-Decision (fine-tuned LoRA, adapters selected on original calibration) vs our prefill success readout on FRESH problems
(NEW_PATH 4.A.45). Same cost estimates for both (our cost readout, and median-length as a control), billed prices; prediction
quality (per-draw log-loss, AUC by route) and routing (cost saved by Intern at matched accuracy over the shared band); paired
problem bootstrap (300). Usage: python intern_fresh_eval.py
"""
import glob, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import POOLS, read_predictions
from decompose import MK, R, hull, cost_at
from provider_routing import RATE

VALUES = np.geomspace(1e-7, 1, 300)
rate_of = lambda s: RATE["oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")]
out = {}
for ds, pool in (("omni500", "Omni"), ("mmlupro", "MMLU-Pro")):
    f = R / "intern_finetune_20261001" / pool.lower().replace("-", "_") / "fresh_predictions.npz"
    if not f.exists():
        print(f"{pool}: fresh Intern predictions not ready"); continue
    name, cost_file = POOLS[pool]; old = R / name; F = R / "expanded_eval_20261001" / ds
    t = np.load(F / "tensors.npz", allow_pickle=True); ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"]))
    idx = {p: i for i, p in enumerate(ids)}; M = len(slots)
    sp = json.loads((old / "split_manifest.json").read_text()); tr = np.array([idx[str(p)] for p in sp["train_problem_ids"]])
    v = t["valid"].astype(bool); cnt = np.maximum(v.sum(2), 1); q = np.where(v, t["final_outcome"], 0).sum(2) / cnt
    L = np.where(v, t["completion_tokens"], 0).sum(2) / cnt; I = np.where(v, t["prompt_tokens"], 0).sum(2) / cnt
    rates = np.array([rate_of(s) for s in slots]); paid = I * rates[:, 0] + L * rates[:, 1]
    for g in glob.glob(f"{R}/math_expand_20261001/{ds}/*_d0.jsonl"):
        for l in open(g):
            r = json.loads(l)
            if r.get("finish_reason") != "error" and r.get("usage_cost") is not None and r["problem_id"] in idx and r["route_label"] in slots:
                paid[idx[r["problem_id"]], slots.index(r["route_label"])] = r["usage_cost"]
    z = np.load(f, allow_pickle=True); assert list(map(str, z["model_slots"])) == slots
    fr = np.array([idx[str(p)] for p in z["problem_ids"]]); keep = (v[fr].sum(2) > 0).all(1); fr = fr[keep]
    PI = np.clip(z["p_successes"][keep], 1e-4, 1 - 1e-4)
    PO = np.clip(read_predictions(F / "success_preds.jsonl", ids, "p_successes", M)[fr], 1e-4, 1 - 1e-4)
    asg = np.array([[MK[s][0] / 1e6, MK[s][1] / 1e6] for s in slots])
    learned = read_predictions(F / "paper_cost_preds.jsonl", ids, "expected_costs", M)
    tok = np.maximum((learned - I * asg[:, 0]) / asg[:, 1], 1)
    med = np.array([np.median(t["completion_tokens"][tr, k][v[tr, k]]) for k in range(M)])
    costs = {"our costs": (I * rates[:, 0] + tok * rates[:, 1])[fr], "median costs": (I * rates[:, 0] + med[None] * rates[:, 1])[fr]}
    Q, PD = q[fr], paid[fr]
    ll = lambda P: -(Q * np.log(P) + (1 - Q) * np.log(1 - P)).mean(1)
    d = ll(PO) - ll(PI); rng = np.random.default_rng(0); BS = [rng.integers(0, len(fr), len(fr)) for _ in range(300)]
    print(f"\n===== {pool} fresh n={len(fr)}: log-loss ours {ll(PO).mean():.4f} vs Intern {ll(PI).mean():.4f}; ours - Intern {d.mean():+.4f} "
          f"[{np.percentile([d[b].mean() for b in BS], 2.5):+.4f}, {np.percentile([d[b].mean() for b in BS], 97.5):+.4f}] (positive = Intern better)")
    print("   AUC ours   " + " ".join(f"{s} {roc_auc_score(Q[:, k] > .5, PO[:, k]):.3f}" for k, s in enumerate(slots)))
    print("   AUC Intern " + " ".join(f"{s} {roc_auc_score(Q[:, k] > .5, PI[:, k]):.3f}" for k, s in enumerate(slots)))

    def front(P, C, ii):
        pts = []
        for V in VALUES:
            m = (V * P[ii] - C[ii]).argmax(1); pts.append((PD[ii][np.arange(len(ii)), m].mean(), Q[ii][np.arange(len(ii)), m].mean()))
        return hull(pts)

    def saved(ii, C):
        Ha, Hb = front(PI, C, ii), front(PO, C, ii); lo, hi = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1])
        T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
        return 1 - float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T]))))
    res = {"logloss_ours_minus_intern": float(d.mean())}
    for tag, C in costs.items():
        g = saved(np.arange(len(fr)), C); gb = [saved(b, C) for b in BS]
        res[tag] = [g, *np.percentile(gb, [2.5, 97.5])]
        print(f"   routing, {tag}: Intern success saves {g*100:+.1f}% [{np.percentile(gb,2.5)*100:+.1f}, {np.percentile(gb,97.5)*100:+.1f}] vs ours at matched accuracy")
    out[pool] = res
json.dump(out, open(Path(__file__).parent / "intern_fresh_eval.json", "w"), indent=1, default=float)

"""Re-price the fresh-set comparison (learned cost vs median-length pricing) with BILLED costs (NEW_PATH 4.A.41).
Realized cost of a fresh call = its usage_cost (math_expand_20261001 rows). Predicted costs and calibration/training costs use each
model's effective billed $/M (least squares over the fresh usage_cost; provider_routing.billed_rates) instead of the list prices we
assumed (oss20 .018/.09, dsv4f .047/.094, oss120 .15/.60). The learned head's token predictions are unchanged (backed out of its
cost predictions at the assumed prices). Deterministic policies chosen on ORIGINAL calibration (cheapest single V reaching each
target), applied once to fresh problems; plus the test-frontier savings over the shared accuracy band. Unweighted means; paired
problem bootstrap (2,000). Same comparison at the assumed prices is printed alongside for reference.
Usage: python billed_reprice.py
"""
import glob, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import POOLS, read_predictions
from decompose import MK, R, hull, cost_at
from provider_routing import RATE

VALUES = np.geomspace(1e-7, 1, 400)                          # $ per correct answer
rate_of = lambda s: RATE["oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")]


def pick(p, c, q, paid, target):
    ch = (VALUES[:, None, None] * p[None] - c[None]).argmax(2); r = np.arange(len(p))[None]
    acc, sp = q[r, ch].mean(1), paid[r, ch].mean(1); ok = np.flatnonzero(acc >= target - 1e-12)
    return None if not len(ok) else float(VALUES[ok[np.argmin(sp[ok])]])


out = {}
for ds in ("mmlupro", "omni500"):
    label = "MMLU-Pro" if ds == "mmlupro" else "Omni"; name, cost_file = POOLS[label]; old = R / name
    F = R / "expanded_eval_20261001" / ds; t = np.load(F / "tensors.npz", allow_pickle=True)
    ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"])); idx = {p: i for i, p in enumerate(ids)}
    sp = json.loads((old / "split_manifest.json").read_text()); tr, ca = [np.array([idx[str(p)] for p in sp[k + "_problem_ids"]]) for k in ("train", "calibration")]
    n_old = len(np.load(old / "tensors.npz", allow_pickle=True)["problem_ids"]); fresh = np.arange(n_old, len(ids))
    v = t["valid"].astype(bool); cnt = np.maximum(v.sum(2), 1)
    q = np.where(v, t["final_outcome"], 0).sum(2) / cnt; L = np.where(v, t["completion_tokens"], 0).sum(2) / cnt; I = np.where(v, t["prompt_tokens"], 0).sum(2) / cnt
    usage = np.full(q.shape, np.nan)
    for f in glob.glob(f"{R}/math_expand_20261001/{ds}/*_d0.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("finish_reason") != "error" and r.get("usage_cost") is not None and r["problem_id"] in idx and r["route_label"] in slots:
                usage[idx[r["problem_id"]], slots.index(r["route_label"])] = r["usage_cost"]
    learned = read_predictions(F / "paper_cost_preds.jsonl", ids, "expected_costs", len(slots))
    learned[:n_old] = read_predictions(old / cost_file, ids[:n_old], "expected_costs", len(slots))
    p = read_predictions(F / "success_preds.jsonl", ids, "p_successes", len(slots))
    p[:n_old] = read_predictions(old / "content_preds.jsonl", ids[:n_old], "p_successes", len(slots))
    tok = np.maximum((learned - I * np.array([MK[s][0] for s in slots]) / 1e6) / (np.array([MK[s][1] for s in slots]) / 1e6), 1)
    med = np.array([np.median(t["completion_tokens"][tr, k][v[tr, k]]) for k in range(len(slots))])
    for tag, rates in (("ASSUMED list prices", np.array([[MK[s][0] / 1e6, MK[s][1] / 1e6] for s in slots])),
                       ("BILLED prices", np.array([rate_of(s) for s in slots]))):
        paid = I * rates[:, 0] + L * rates[:, 1]
        if tag.startswith("BILLED"):
            paid = np.where(np.isfinite(usage), usage, paid)            # fresh calls: actual billed cost
        cL = I * rates[:, 0] + tok * rates[:, 1]; cM = I * rates[:, 0] + med[None] * rates[:, 1]
        fr = fresh[np.isfinite(paid[fresh]).all(1)]
        rng = np.random.default_rng(0); BS = [rng.integers(0, len(fr), len(fr)) for _ in range(2000)]
        k = np.arange(len(fr)); res = {}
        print(f"\n{label} fresh (n={len(fr)}), {tag}: mean billed-or-priced cost per call by route " +
              ", ".join(f"{s} {paid[fr, j].mean()*1e3:.3f}m$" for j, s in enumerate(slots)))
        for target in (0.65, 0.70, 0.75, 0.80, 0.85):
            VL, VM = pick(p[ca], cL[ca], q[ca], paid[ca], target), pick(p[ca], cM[ca], q[ca], paid[ca], target)
            if VL is None or VM is None:
                continue
            mL = (VL * p[fr] - cL[fr]).argmax(1); mM = (VM * p[fr] - cM[fr]).argmax(1)
            aL, sL, aM, sM = q[fr][k, mL], paid[fr][k, mL], q[fr][k, mM], paid[fr][k, mM]
            bs = np.array([(1 - sL[b].mean() / sM[b].mean(), (aL[b] - aM[b]).mean()) for b in BS])
            res[target] = dict(savings=1 - sL.mean() / sM.mean(), savings_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                               dacc=(aL - aM).mean(), dacc_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist())
            print(f"   target {target:.2f}: savings {res[target]['savings']*100:+5.1f}% [{bs[:,0].min()*0+np.percentile(bs[:,0],2.5)*100:+.1f}, {np.percentile(bs[:,0],97.5)*100:+.1f}]"
                  f"   acc diff {(aL-aM).mean()*100:+.2f} pt [{np.percentile(bs[:,1],2.5)*100:+.2f}, {np.percentile(bs[:,1],97.5)*100:+.2f}]")
        def front(c_, e):
            pts = []
            for V in VALUES:
                m = (V * p[e] - c_[e]).argmax(1); pts.append((paid[e][np.arange(len(e)), m].mean(), q[e][np.arange(len(e)), m].mean()))
            return hull(pts)
        def g(e):
            HL, HM = front(cL, e), front(cM, e); lo, hi = max(HL[0][1], HM[0][1]), min(HL[-1][1], HM[-1][1]); T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
            return 1 - float(np.exp(np.nanmean(np.log([cost_at(HL, x) / cost_at(HM, x) for x in T]))))
        gb = [g(fr[b]) for b in BS[:300]]
        res["frontier"] = dict(savings=g(fr), ci=np.percentile(gb, [2.5, 97.5]).tolist())
        print(f"   test-frontier savings (shared band): {g(fr)*100:+.1f}% [{np.percentile(gb,2.5)*100:+.1f}, {np.percentile(gb,97.5)*100:+.1f}]")
        out[f"{label}|{tag}"] = res
json.dump(out, open(Path(__file__).parent / "billed_reprice.json", "w"), indent=1, default=float)

"""Headroom and route table for the 4-pager (NEW_PATH 4.A.60), pinned deepseek-v4-flash, billed prices.
Same frontier arithmetic as fresh_baselines.py (V grid geomspace(1e-7, 1, 300), interior 90% band, 12 targets, paired bootstrap 300):
  ours     cost saved by our cost readouts vs training-median pricing (= Table 1 row 1)
  oracle   cost saved by PERFECT per-query cost knowledge (each problem priced at its realized output) vs median, same success
           predictions = the headroom; capture = ours / oracle
Route table: per route on each test set, accuracy and mean billed cost per call.
Test sets: LCB temporal test split (341; realized tokens x billed rates), Omni-MATH (1,000) and MMLU-Pro (6,500) test problems (billed).
Usage: REASON_ROOT=.../reason_pinned python headroom_routes.py
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import POOLS, read_predictions
from decompose import MK, R, hull, cost_at
from billed import RATE

VALUES = np.geomspace(1e-7, 1, 300)
rate_of = lambda s: RATE["oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")]
out = {}
for ds in ("lcb", "omni500", "mmlupro"):
    if ds == "lcb":
        label, name, cost_file = "LCB", "pool_v2_tensors_5rung", "cost_preds_probe.jsonl"; old = F = R / name
    else:
        label = "MMLU-Pro" if ds == "mmlupro" else "Omni"; name, cost_file = POOLS[label]; old = R / name; F = R / "expanded_eval_20261001" / ds
    t = np.load(F / "tensors.npz", allow_pickle=True)
    ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"])); idx = {p: i for i, p in enumerate(ids)}; M = len(slots)
    sp = json.loads((old / "split_manifest.json").read_text()); tr = np.array([idx[str(p)] for p in sp["train_problem_ids"]])
    n_old = len(np.load(old / "tensors.npz", allow_pickle=True)["problem_ids"])
    ev = np.array([idx[str(p)] for p in sp["test_problem_ids"]]) if ds == "lcb" else np.arange(n_old, len(ids))
    v = t["valid"].astype(bool); cnt = np.maximum(v.sum(2), 1)
    q = np.where(v, t["final_outcome"], 0).sum(2) / cnt; L = np.where(v, t["completion_tokens"], 0).sum(2) / cnt; I = np.where(v, t["prompt_tokens"], 0).sum(2) / cnt
    rates = np.array([rate_of(s) for s in slots]); paid = I * rates[:, 0] + L * rates[:, 1]
    for f in ([] if ds == "lcb" else glob.glob(f"{R}/math_expand_20261001/{ds}/*_d0.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("finish_reason") != "error" and r.get("usage_cost") is not None and r["problem_id"] in idx and r["route_label"] in slots:
                paid[idx[r["problem_id"]], slots.index(r["route_label"])] = r["usage_cost"]
    ev = ev[(v[ev].sum(2) > 0).all(1)]
    if ds == "lcb":
        learned = read_predictions(old / cost_file, ids, "expected_costs", M); P = read_predictions(old / "content_preds.jsonl", ids, "p_successes", M)
    else:
        learned = read_predictions(F / "paper_cost_preds.jsonl", ids, "expected_costs", M); learned[:n_old] = read_predictions(old / cost_file, ids[:n_old], "expected_costs", M)
        P = read_predictions(F / "success_preds.jsonl", ids, "p_successes", M); P[:n_old] = read_predictions(old / "content_preds.jsonl", ids[:n_old], "p_successes", M)
    P = np.clip(P, 1e-4, 1 - 1e-4)
    asg = np.array([[MK[s][0] / 1e6, MK[s][1] / 1e6] for s in slots]); tok = np.maximum((learned - I * asg[:, 0]) / asg[:, 1], 1)
    med = np.array([np.median(t["completion_tokens"][tr, k][v[tr, k]]) for k in range(M)])
    C = {"ours": I * rates[:, 0] + tok * rates[:, 1], "median": I * rates[:, 0] + med[None] * rates[:, 1], "oracle": paid}

    def front(c, ii):
        pts = []
        for V in VALUES:
            m = (V * P[ii] - c[ii]).argmax(1); k = np.arange(len(ii)); pts.append((paid[ii][k, m].mean(), q[ii][k, m].mean()))
        return hull(pts)

    def saved(a, b, ii):
        Ha, Hb = front(C[a], ii), front(C[b], ii); lo, hi = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1])
        T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
        return 1 - float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T]))))
    rng = np.random.default_rng(0); BS = [ev[rng.integers(0, len(ev), len(ev))] for _ in range(300)]
    res = {"n": int(len(ev))}
    for a in ("ours", "oracle"):
        g = saved(a, "median", ev); b = [saved(a, "median", bb) for bb in BS]
        res[a] = [g, *np.percentile(b, [2.5, 97.5]).tolist()]
    cap = [saved("ours", "median", bb) / max(saved("oracle", "median", bb), 1e-9) for bb in BS[:100]]
    res["capture"] = [res["ours"][0] / res["oracle"][0], *np.percentile(cap, [2.5, 97.5]).tolist()]
    res["routes"] = {s: dict(acc=float(q[ev, k].mean()), cost_per_call_musd=float(paid[ev, k].mean() * 1e3), mean_out=float(L[ev, k].mean())) for k, s in enumerate(slots)}
    print(f"===== {label} test n={len(ev)}: ours vs median {res['ours'][0]*100:+.1f}% [{res['ours'][1]*100:+.1f}, {res['ours'][2]*100:+.1f}]  "
          f"oracle (headroom) {res['oracle'][0]*100:+.1f}% [{res['oracle'][1]*100:+.1f}, {res['oracle'][2]*100:+.1f}]  capture {res['capture'][0]:.2f} "
          f"[{res['capture'][1]:.2f}, {res['capture'][2]:.2f}]", flush=True)
    for s, r_ in res["routes"].items():
        print(f"    {s:<9} acc {r_['acc']*100:5.1f}%  cost/call {r_['cost_per_call_musd']:.3f} m$  mean out {r_['mean_out']:.0f}")
    out[label] = res
json.dump(out, open(Path(__file__).parent / f"headroom_routes{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)

"""Layer-wise readouts (NEW_PATH 4.A.62): do the success and cost readouts draw on the same layers? Pinned deepseek-v4-flash.
For each of the 8 stored layers (Qwen3-4B layers 9..36) and each pooling (mean, last token), refit on TRAIN only:
  success  per-route logistic regression on all valid draws (binomial weights), C chosen on CALIBRATION by AUC (grid 1e-4..1e-2,
           scaled by width as activation_content_preds.py) + Platt on calibration -> test AUC (mean over routes)
  cost     per-route RidgeCV(1e1..1e7) on log mean output length, smearing + level match -> test log-length R2 (mean over routes)
  routing  cost saved at matched accuracy vs median pricing when ONLY the cost readout comes from this layer (success = the paper's
           all-layer readout), and when ONLY the success readout does (cost = the paper's all-layer readout); billed prices
  shared   correlation, over test problems, between the layer's mean success logit and its mean predicted log length
Reference row: the paper's all-layer (rich) readouts. Usage: python layerwise.py LCB|Omni|MMLU-Pro
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import POOLS, read_predictions
from decompose import MK, R, hull, cost_at
from billed import RATE

VALUES = np.geomspace(1e-7, 1, 300); CGRID = (1e-4, 1e-3, 1e-2)
rate_of = lambda s: RATE["oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")]
POOL = sys.argv[1]
if POOL == "LCB":
    name, cost_file = "pool_v2_tensors_5rung", "cost_preds_probe.jsonl"; old = F = R / name
    feat = Path("/mnt/llmd/results/exps/aristides/reason/pv2_scout_prefill_1756715297/scout.npz")
else:
    name, cost_file = POOLS[POOL]; old = R / name; ds = "mmlupro" if POOL == "MMLU-Pro" else "omni500"; F = R / "expanded_eval_20261001" / ds
    feat = F / "prefill_combined.npz"
t = np.load(F / "tensors.npz", allow_pickle=True)
ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"])); idx = {p: i for i, p in enumerate(ids)}; M = len(slots)
sp = json.loads((old / "split_manifest.json").read_text()); tr, ca = [np.array([idx[str(p)] for p in sp[k + "_problem_ids"]]) for k in ("train", "calibration")]
n_old = len(np.load(old / "tensors.npz", allow_pickle=True)["problem_ids"])
ev = np.array([idx[str(p)] for p in sp["test_problem_ids"]]) if POOL == "LCB" else np.arange(n_old, len(ids))
v = t["valid"].astype(bool); ok = t["final_outcome"] & v; cnt = np.maximum(v.sum(2), 1)
q = np.where(v, t["final_outcome"], 0).sum(2) / cnt; L = np.maximum(np.where(v, t["completion_tokens"], 0).sum(2) / cnt, 1)
I = np.where(v, t["prompt_tokens"], 0).sum(2) / cnt; Y = np.log(L)
rates = np.array([rate_of(s) for s in slots]); paid = I * rates[:, 0] + L * rates[:, 1]
if POOL != "LCB":
    for f in glob.glob(f"{R}/math_expand_20261001/{ds}/*_d0.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("finish_reason") != "error" and r.get("usage_cost") is not None and r["problem_id"] in idx and r["route_label"] in slots:
                paid[idx[r["problem_id"]], slots.index(r["route_label"])] = r["usage_cost"]
ev = ev[(v[ev].sum(2) > 0).all(1)]
# paper (all-layer) readouts
if POOL == "LCB":
    learned = read_predictions(old / cost_file, ids, "expected_costs", M); Pp = read_predictions(old / "content_preds.jsonl", ids, "p_successes", M)
else:
    learned = read_predictions(F / "paper_cost_preds.jsonl", ids, "expected_costs", M); learned[:n_old] = read_predictions(old / cost_file, ids[:n_old], "expected_costs", M)
    Pp = read_predictions(F / "success_preds.jsonl", ids, "p_successes", M); Pp[:n_old] = read_predictions(old / "content_preds.jsonl", ids[:n_old], "p_successes", M)
Pp = np.clip(Pp, 1e-4, 1 - 1e-4)
asg = np.array([[MK[s][0] / 1e6, MK[s][1] / 1e6] for s in slots]); tok_p = np.maximum((learned - I * asg[:, 0]) / asg[:, 1], 1)
med = np.array([np.median(t["completion_tokens"][tr, k][v[tr, k]]) for k in range(M)])
Cp = I * rates[:, 0] + tok_p * rates[:, 1]; Cm = I * rates[:, 0] + med[None] * rates[:, 1]


def front(P, C, ii):
    pts = []
    for V in VALUES:
        m = (V * P[ii] - C[ii]).argmax(1); k = np.arange(len(ii)); pts.append((paid[ii][k, m].mean(), q[ii][k, m].mean()))
    return hull(pts)


def saved(a, b, ii):
    Ha, Hb = front(*a, ii), front(*b, ii); lo, hi = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1])
    T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    return 1 - float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T]))))


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.mean() < 1 else np.nan


def readouts(X):
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X); scale = max(1, X.shape[1] // 2560)
    P = np.zeros((len(ids), M)); tok = np.zeros((len(ids), M)); aucs, r2s = [], []
    for k in range(M):
        succ = np.array([ok[i, k, v[i, k]].sum() for i in range(len(ids))], float); tot = v[:, k].sum(1).astype(float)
        y0 = ok[:, k, 0].astype(int); rate = succ / np.maximum(tot, 1)
        Xb = np.vstack([Xs[tr], Xs[tr]]); yb = np.r_[np.ones(len(tr)), np.zeros(len(tr))]; wb = np.r_[rate[tr] * tot[tr], (1 - rate[tr]) * tot[tr]]; kw = wb > 1e-9
        best = None
        for c in CGRID:
            m = LogisticRegression(max_iter=1000, C=c / scale).fit(Xb[kw], yb[kw], sample_weight=wb[kw]); a = auc(y0[ca], m.decision_function(Xs[ca]))
            if best is None or a > best[0]:
                best = (a, m)
        lo = best[1].decision_function(Xs)
        pl = LogisticRegression(max_iter=1000, C=1e6).fit(lo[ca].reshape(-1, 1), y0[ca]); P[:, k] = pl.predict_proba(lo.reshape(-1, 1))[:, 1]
        aucs.append(auc((q[ev, k] > .5).astype(int), P[ev, k]))
        rm = RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(Xs[tr], Y[tr, k]); yh = rm.predict(Xs)
        sm = np.mean(np.exp(Y[tr, k] - yh[tr])); tk = np.exp(yh) * sm; tok[:, k] = tk * L[tr, k].mean() / tk[tr].mean()
        r2s.append(1 - ((Y[ev, k] - np.log(tok[ev, k])) ** 2).sum() / ((Y[ev, k] - Y[ev, k].mean()) ** 2).sum())
    P = np.clip(P, 1e-4, 1 - 1e-4); C = I * rates[:, 0] + tok * rates[:, 1]
    shared = float(np.corrcoef(np.log(P[ev] / (1 - P[ev])).mean(1), np.log(tok[ev]).mean(1))[0, 1])
    return dict(auc=float(np.nanmean(aucs)), r2=float(np.mean(r2s)), cost_routing=saved((Pp, C), (Pp, Cm), ev),
                success_routing=saved((P, Cp), (Pp, Cp), ev), shared_corr=shared)


z = np.load(feat, allow_pickle=True); zid = {str(p): i for i, p in enumerate(z["problem_ids"])}; rows = [zid[p] for p in ids]
layers = list(map(int, z["layers"])) if "layers" in z.files else list(range(z["mean"].shape[1]))
out = {"pool": POOL, "n_eval": int(len(ev)), "layers": layers, "paper": dict(cost_routing=saved((Pp, Cp), (Pp, Cm), ev)), "by_layer": {}}
print(f"===== {POOL}: test n={len(ev)}; paper all-layer readouts save {out['paper']['cost_routing']*100:+.1f}% vs median", flush=True)
for pool_kind in ("mean", "last"):
    A = z[pool_kind]
    for j, Lyr in enumerate(layers):
        r = readouts(A[rows, j, :].astype(np.float32)); out["by_layer"][f"{pool_kind}_{Lyr}"] = r
        print(f"  {pool_kind:<4} layer {Lyr:>2}: success AUC {r['auc']:.3f}  cost R2 {r['r2']:.3f}  | cost-from-layer saves {r['cost_routing']*100:+5.1f}% vs median"
              f"  success-from-layer vs paper success {r['success_routing']*100:+5.1f}%  | corr(success logit, log length) {r['shared_corr']:+.2f}", flush=True)
json.dump(out, open(Path(__file__).parent / f"layerwise_{POOL.replace('-', '').lower()}{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)

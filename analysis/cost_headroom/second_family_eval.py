"""Second model family on MMLU-Pro (NEW_PATH 4.A.48): Qwen3-32B and GLM-4.7-flash (one draw each) on the original train+cal problems
and the 2,000 pinned fresh problems. New routes get their own readouts from the SAME frozen 4B prefill features (PCA-256 fitted on
train; logistic success with CV-chosen C + Platt on calibration; RidgeCV log-length + smearing + train-mean level match), fitted on
original TRAIN only. Old routes keep their archived readouts. Billed prices (fresh usage_cost; predictions at each model's
effective billed rate).
Reported: (1) test log-length R2 and success AUC of the new readouts on fresh problems; (2) routing on fresh problems for two pools:
  all7       5 original routes + 2 new
  newpool    oss20lo, oss20md, qw32, glm47f  (no deepseek / gpt-oss-120b, so the new families carry the top end)
  arms: ours, median, mean, cost-from-success, difficulty bins (same success predictions); cost saved by ours vs each at matched
  accuracy, paired bootstrap (300). (3) onboarding: new routes' cost from k = 10 examples by one offset on the shared level (mean
  predicted log length of the old routes), vs its full readout.
Usage: python second_family_eval.py
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import read_predictions
from decompose import MK, R, hull, cost_at
from billed import RATE

VALUES = np.geomspace(1e-7, 1, 300); NEW = ["qw32", "glm47f"]
F = R / "expanded_eval_20261001" / "mmlupro"; old = R / "mmlupro_tensors"; SF = R / "second_family_20261002" / "mmlupro"
t = np.load(F / "tensors.npz", allow_pickle=True); ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"]))
idx = {p: i for i, p in enumerate(ids)}; n_old = len(np.load(old / "tensors.npz", allow_pickle=True)["problem_ids"])
sp = json.loads((old / "split_manifest.json").read_text()); tr, ca = [np.array([idx[str(p)] for p in sp[k + "_problem_ids"]]) for k in ("train", "calibration")]
v = t["valid"].astype(bool); cnt = np.maximum(v.sum(2), 1)
q = {s: np.where(v, t["final_outcome"], 0).sum(2)[:, k] / cnt[:, k] for k, s in enumerate(slots)}
L = {s: np.where(v, t["completion_tokens"], 0).sum(2)[:, k] / cnt[:, k] for k, s in enumerate(slots)}
I = {s: np.where(v, t["prompt_tokens"], 0).sum(2)[:, k] / cnt[:, k] for k, s in enumerate(slots)}
rate_of = lambda s: RATE["oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")]
paid = {s: I[s] * rate_of(s)[0] + L[s] * rate_of(s)[1] for s in slots}
for f in glob.glob(f"{R}/math_expand_20261001/mmlupro/*_d0.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r.get("finish_reason") != "error" and r.get("usage_cost") is not None and r["problem_id"] in idx and r["route_label"] in slots:
            paid[r["route_label"]][idx[r["problem_id"]]] = r["usage_cost"]
okall = np.ones(len(ids), bool)
for s in NEW:
    q[s], L[s], I[s], paid[s] = (np.full(len(ids), np.nan) for _ in range(4))
    for l in open(SF / f"{s}_d0.jsonl"):
        r = json.loads(l)
        if r.get("finish_reason") != "error" and r["problem_id"] in idx and r.get("completion_tokens"):
            i = idx[r["problem_id"]]; q[s][i] = float(r["resolved"]); L[s][i] = r["completion_tokens"]; I[s][i] = r["prompt_tokens"]; paid[s][i] = r["usage_cost"] or 0
    okall &= np.isfinite(q[s])
pilot = set(json.load(open(R / "provider_pilot_20261002" / "mmlupro" / "problem_ids.json")))
fr = np.array([idx[p] for p in pilot if p in idx and okall[idx[p]] and (v[idx[p]].sum(1) > 0).all()])
trn, can = tr[okall[tr]], ca[okall[ca]]
newrate = {s: np.linalg.lstsq(np.c_[I[s][fr], L[s][fr]], paid[s][fr], rcond=None)[0] for s in NEW}   # effective billed $/token
print(f"fresh eval n={len(fr)}; new-route train {len(trn)}, cal {len(can)}; billed out $/M: " + ", ".join(f"{s} {newrate[s][1]*1e6:.3f}" for s in NEW))
# features
z = np.load(F / "prefill_combined.npz", allow_pickle=True); zid = {str(p): i for i, p in enumerate(z["problem_ids"])}
X = np.concatenate([z[k].reshape(len(z[k]), -1) for k in ("mean", "last")], 1)[[zid[p] for p in ids]].astype(np.float32); del z
sc = StandardScaler().fit(X[trn]); Z = PCA(256, random_state=0).fit(sc.transform(X[trn])).transform(sc.transform(X)); del X
Z /= Z[trn].std(0) + 1e-6
Pold = read_predictions(F / "success_preds.jsonl", ids, "p_successes", len(slots)); Pold[:n_old] = read_predictions(old / "content_preds.jsonl", ids[:n_old], "p_successes", len(slots))
Cold = read_predictions(F / "paper_cost_preds.jsonl", ids, "expected_costs", len(slots)); Cold[:n_old] = read_predictions(old / "cost_preds_probe_instruct.jsonl", ids[:n_old], "expected_costs", len(slots))
p = {s: np.clip(Pold[:, k], 1e-4, 1 - 1e-4) for k, s in enumerate(slots)}
tok = {s: np.maximum((Cold[:, k] - I[s] * MK[s][0] / 1e6) / (MK[s][1] / 1e6), 1) for k, s in enumerate(slots)}
for s in NEW:
    lr = LogisticRegressionCV(Cs=np.geomspace(1e-4, 10, 12), cv=5, max_iter=3000).fit(Z[trn], q[s][trn].astype(int))
    raw = lr.decision_function(Z); pl = LogisticRegression().fit(raw[can][:, None], q[s][can].astype(int))
    p[s] = np.clip(pl.predict_proba(raw[:, None])[:, 1], 1e-4, 1 - 1e-4)
    y = np.log(np.maximum(L[s], 1)); m = RidgeCV(alphas=np.geomspace(1e-1, 1e6, 15)).fit(Z[trn], y[trn]); yh = m.predict(Z)
    o = np.exp(yh) * np.mean(np.exp(y[trn] - yh[trn])); tok[s] = o * L[s][trn].mean() / o[trn].mean()
    r2 = 1 - ((y[fr] - np.log(tok[s][fr])) ** 2).sum() / ((y[fr] - y[fr].mean()) ** 2).sum()
    print(f"  {s}: fresh log-length R2 {r2:.2f} (old routes on fresh: " + ", ".join(f"{o_} {1 - ((np.log(np.maximum(L[o_][fr],1)) - np.log(tok[o_][fr]))**2).sum() / ((np.log(np.maximum(L[o_][fr],1)) - np.log(np.maximum(L[o_][fr],1)).mean())**2).sum():.2f}" for o_ in slots)
          + f"); success AUC {roc_auc_score(q[s][fr] > .5, p[s][fr]):.3f}")
rate_any = lambda s: newrate[s] if s in NEW else rate_of(s)
routes_all = slots + NEW; Lg = {s: np.log(p[s] / (1 - p[s])) for s in routes_all}
med = {s: np.median(L[s][trn]) for s in routes_all}; meanL = {s: L[s][trn].mean() for s in routes_all}
def price(s, tk): return I[s] * rate_any(s)[0] + tk * rate_any(s)[1]
def from_success(routes, s):
    F_ = np.c_[np.stack([Lg[r] for r in routes], 1), np.stack([Lg[r] for r in routes], 1) ** 2]; y = np.log(np.maximum(L[s], 1))
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-3, 1e4, 15))).fit(F_[trn], y[trn]); yh = m.predict(F_)
    o = np.exp(yh) * np.mean(np.exp(y[trn] - yh[trn])); return o * L[s][trn].mean() / o[trn].mean()
def bins(routes, s, K=10):
    d = np.mean([Lg[r] for r in routes], 0); e = np.quantile(d[trn], np.linspace(0, 1, K + 1)[1:-1]); b = np.searchsorted(e, d)
    tab = np.array([L[s][trn][b[trn] == j].mean() if (b[trn] == j).any() else L[s][trn].mean() for j in range(K)]); return tab[b]
out = {}
rng = np.random.default_rng(0); BS = [fr[rng.integers(0, len(fr), len(fr))] for _ in range(300)]
for pool, routes in (("all7", routes_all), ("newpool", ["oss20lo", "oss20md", "qw32", "glm47f"])):
    PM = np.stack([p[s] for s in routes], 1); Q = np.stack([q[s] for s in routes], 1); PD = np.stack([paid[s] for s in routes], 1)
    arms = {"ours": np.stack([price(s, tok[s]) for s in routes], 1), "median": np.stack([price(s, med[s]) for s in routes], 1),
            "mean": np.stack([price(s, meanL[s]) for s in routes], 1),
            "cost-from-success": np.stack([price(s, from_success(routes, s)) for s in routes], 1),
            "difficulty bins": np.stack([price(s, bins(routes, s)) for s in routes], 1)}
    lvl = np.mean([np.log(tok[s]) for s in slots], 0)
    onb = []
    for s in routes:
        if s in NEW:
            kk = rng.choice(trn, 10, replace=False); off = np.log(L[s][kk].sum() / np.exp(lvl[kk]).sum()); onb.append(price(s, np.exp(lvl + off)))
        else:
            onb.append(price(s, tok[s]))
    arms["ours, new routes onboarded from 10"] = np.stack(onb, 1)

    def front(C, ii):
        pts = []
        for V in VALUES:
            m = (V * PM[ii] - C[ii]).argmax(1); pts.append((PD[ii][np.arange(len(ii)), m].mean(), Q[ii][np.arange(len(ii)), m].mean()))
        return hull(pts)
    def saved(a, b, ii):
        Ha, Hb = front(arms[a], ii), front(arms[b], ii); lo, hi = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1])
        T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
        return 1 - float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T]))))
    Hb = front(arms["ours"], fr)
    print(f"\n===== pool {pool} ({', '.join(routes)}): max acc {Hb[-1][1]*100:.1f}%; route acc " + ", ".join(f"{s} {q[s][fr].mean()*100:.1f}%" for s in routes))
    res = {}
    for a in arms:
        if a == "ours": continue
        g = saved("ours", a, fr); b = [saved("ours", a, bb) for bb in BS]
        res[a] = [g, *np.percentile(b, [2.5, 97.5])]
        print(f"  ours saves vs {a:<36} {g*100:+6.1f}% [{np.percentile(b,2.5)*100:+.1f}, {np.percentile(b,97.5)*100:+.1f}]", flush=True)
    out[pool] = res
json.dump(out, open(Path(__file__).parent / f"second_family_eval{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)

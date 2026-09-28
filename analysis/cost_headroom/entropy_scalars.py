"""Do prefill ENTROPY scalars add cost information beyond the probe? (DiffAdapt-style: reasoning entropy is U-shaped in
difficulty.) The 4B prefill files carry prompt_nll, next_entropy, next_max_logprob per problem; the cost heads never used
them. Per route: ridge on [plain-probe log-output prediction, scalars, scalars^2], fitted on TRAIN; test R2 per route and
on the shared LEVEL / between-route DIFFERENCES; writes cost_preds_probe_entropy.jsonl for decompose.py.
Usage: python entropy_scalars.py <pool> <prefill.npz>
"""
import json, sys, numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R

name, act = sys.argv[1], sys.argv[2]; PROBE = sys.argv[3] if len(sys.argv) > 3 else "cost_preds_probe.jsonl"
D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
v = t["valid"].astype(bool); ct = t["completion_tokens"].astype(float); pt = t["prompt_tokens"].astype(float); n = v.sum(2)
Y = np.log(np.maximum(np.where(n > 0, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan), 1)); Y[n == 0] = np.nan
inp = np.nanmean(np.where(v, pt, np.nan), 2)
sp = json.load(open(D / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]]); te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
z = np.load(act, allow_pickle=True); zi = {str(p): i for i, p in enumerate(z["problem_ids"])}
Sc = z["scalars"][[zi[p] for p in pids]].astype(float); Sc = np.c_[Sc, Sc ** 2]
lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:len(S)] for l in open(D / PROBE)}
LC = np.array([lc[p] for p in pids])
probe = np.stack([np.log(np.maximum((LC[:, m] - np.nan_to_num(inp[:, m]) * MK[s][0] / 1e6) / (MK[s][1] / 1e6), 1)) for m, s in enumerate(S)], 1)
r2 = lambda y, x: 1 - np.nansum((y - x) ** 2) / np.nansum((y - np.nanmean(y)) ** 2)
P = np.zeros_like(probe); C = np.zeros_like(probe)
for m, s in enumerate(S):
    ok = tr[np.isfinite(Y[tr, m])]; F = np.c_[probe[:, m], Sc]
    P[:, m] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-3, 1e3, 13))).fit(F[ok], Y[ok, m]).predict(F)
    o = np.exp(P[:, m]) * np.mean(np.exp(Y[ok, m] - P[ok, m])); o *= np.exp(Y[ok, m]).mean() / o[ok].mean()
    C[:, m] = (np.nan_to_num(inp[:, m], nan=np.nanmean(inp[:, m])) * MK[s][0] + o * MK[s][1]) / 1e6
with open(D / "cost_preds_probe_entropy.jsonl", "w") as f:
    for i, p in enumerate(pids):
        f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in C[i]]}) + "\n")
lv = lambda X: np.nanmean(X[te], 1)
def df(X):                                        # between-route differences NET of each route's train-average offset
    dd = X - np.nanmean(X, 1, keepdims=True); return (dd[te] - np.nanmean(dd[tr], 0)).ravel()
print(f"{name}: scalars alone -> mean log length R2 (in-sample, train) "
      f"{r2(np.nanmean(Y[tr],1), make_pipeline(StandardScaler(), RidgeCV()).fit(Sc[tr], np.nanmean(Y[tr],1)).predict(Sc[tr])):.2f}")
print(f"   per-route test R2  probe: " + " ".join(f"{x:+.2f}" for x in (r2(Y[te, m], probe[te, m]) for m in range(len(S))))
      + "   probe+entropy: " + " ".join(f"{x:+.2f}" for x in (r2(Y[te, m], P[te, m]) for m in range(len(S)))))
print(f"   LEVEL R2 {r2(lv(Y), lv(probe)):+.2f} -> {r2(lv(Y), lv(P)):+.2f};  DIFFERENCES R2 {r2(df(Y), df(probe)):+.2f} -> {r2(df(Y), df(P)):+.2f}")

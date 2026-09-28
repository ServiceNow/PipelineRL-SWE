"""Idea #1: teach the probe to read DIFFICULTY from free labels at scale, then use it for cost.
Length labels need every route to run; difficulty labels are free: 6362 CodeContests Codeforces problems (outside the
700-problem pool) carry an API rating. Fit ridge(rating ~ 4B Instruct prefill) on those, predict the pool's ratings,
and stack the predicted rating (cubic) with the plain probe prediction in the per-route cost head (fitted on the pool's
TRAIN split, as before). Compare: probe alone vs probe + predicted rating (+ the TRUE rating as an upper bound).
Reports readability of the rating on the pool (R2 of predicted vs true) before (700-problem CV) and after (6362 aux).
"""
import json, sys, numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R
from baseline_cost_heads import rich

A = R / "cc_aux_prefill" / "instruct.npz"
aux = [json.loads(l) for l in open(R / "cc_aux_rating_prompts.jsonl")]
za = np.load(A, allow_pickle=True); have = {str(p) for p in za["problem_ids"]}
aux = [a for a in aux if a["problem_id"] in have]
Xa = rich(A, [a["problem_id"] for a in aux]); ya = np.array([a["cf_rating"] for a in aux], float)
D = R / "cc_tensors"; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
tasks = {json.loads(l)["problem_id"]: json.loads(l) for l in open(R / "cc_pool" / "cc_tasks.jsonl")}
true_r = np.array([tasks.get(p, {}).get("cf_rating") or np.nan for p in pids], float)
Xp = rich(R / "cc_pool" / "scout_prefill.npz", pids)
alphas = np.geomspace(1e1, 1e7, 13)
ok = np.isfinite(true_r) & (true_r > 0)
cv_pool = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)), Xp[ok], true_r[ok], cv=5)
before = 1 - ((true_r[ok] - cv_pool) ** 2).sum() / ((true_r[ok] - true_r[ok].mean()) ** 2).sum()
reader = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(Xa, ya)
pred_r = reader.predict(Xp)
after = 1 - ((true_r[ok] - pred_r[ok]) ** 2).sum() / ((true_r[ok] - true_r[ok].mean()) ** 2).sum()
print(f"{len(aux)} auxiliary rated problems. Readability of the Codeforces rating on the 700-problem pool: "
      f"probe CV on the pool itself R2 {before:.2f} -> reader trained on the auxiliary set R2 {after:.2f}")

v = t["valid"].astype(bool); ct = t["completion_tokens"].astype(float); pt = t["prompt_tokens"].astype(float); n = v.sum(2)
Y = np.log(np.maximum(np.where(n > 0, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan), 1)); Y[n == 0] = np.nan
inp = np.nanmean(np.where(v, pt, np.nan), 2)
sp = json.load(open(D / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]]); te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:len(S)] for l in open(D / "cost_preds_probe.jsonl")}
LC = np.array([lc[p] for p in pids])


def cub(r):
    z = (r - np.nanmean(r[tr])) / np.nanstd(r[tr]); return np.c_[z, z ** 2, z ** 3]


tr_fill = np.where(np.isfinite(true_r) & (true_r > 0), true_r, np.nanmedian(true_r))
for tag, extra in (("rating_pred", cub(pred_r)), ("rating_true", cub(tr_fill))):
    C = np.zeros((len(pids), len(S))); r2s = []
    for m, s in enumerate(S):
        probe = np.log(np.maximum((LC[:, m] - np.nan_to_num(inp[:, m]) * MK[s][0] / 1e6) / (MK[s][1] / 1e6), 1))
        F = np.c_[probe, extra]; trm = tr[np.isfinite(Y[tr, m])]
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-3, 1e3, 13))).fit(F[trm], Y[trm, m])
        yh = mdl.predict(F); o = np.exp(yh) * np.mean(np.exp(Y[trm, m] - yh[trm])); o *= np.exp(Y[trm, m]).mean() / o[trm].mean()
        C[:, m] = (np.nan_to_num(inp[:, m], nan=np.nanmean(inp[:, m])) * MK[s][0] + o * MK[s][1]) / 1e6
        tt = te[np.isfinite(Y[te, m])]; r2s.append(1 - ((Y[tt, m] - yh[tt]) ** 2).sum() / ((Y[tt, m] - Y[tt, m].mean()) ** 2).sum())
    with open(D / f"cost_preds_probe_{tag}.jsonl", "w") as f:
        for i, p in enumerate(pids):
            f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in C[i]]}) + "\n")
    print(f"probe + {tag:<12} test log-output R2: " + "  ".join(f"{s} {x:+.2f}" for s, x in zip(S, r2s)))

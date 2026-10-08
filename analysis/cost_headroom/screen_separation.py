"""Free screen (NEW_PATH 4.A.65): does output length separate from difficulty on a candidate dataset, before paying for a full pool?
Screen data = one gpt-oss-20b-low draw per problem (math_pool/<ds>/oss20lo_d0.jsonl) + the Qwen3-4B Instruct prefill (<ds>_probe/instruct.npz).
5-fold CV (seed 0). Per outer fold:
  direct        RidgeCV(1e1..1e7) on standardized rich features -> log output length
  from-success  logistic success readout on the same features (C=1e-3 / width scale); training logits from an inner 5-fold CV so they
                are out-of-sample like the test logits; then a ridge of log length on [logit, logit^2]
Gap = CV R2(direct) - CV R2(from-success): large where length tracks something beyond difficulty. References with known answers:
Omni-MATH (full pool: difficulty-only pricing ties) and MMLU-Pro (it loses by 24 pt). Usage: python screen_separation.py
"""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
R = Path("/mnt/llmd/results/exps/aristides/reason")
DSETS = ["omni500", "mmlupro", "aime", "kk", "supergpqa", "bbeh"]


def rich(path, pids):
    z = np.load(path, allow_pickle=True); X = np.concatenate([z[k].reshape(len(z[k]), -1) for k in ("mean", "last")], 1)
    aid = {str(p): i for i, p in enumerate(z["problem_ids"])}; keep = [p for p in pids if p in aid]
    return keep, X[[aid[p] for p in keep]].astype(np.float32)


def r2(y, yh):
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def logit_model(X, y, C):
    return LogisticRegression(max_iter=2000, C=C).fit(X, y)


out = {}
for ds in DSETS:
    rows = {}
    for l in open(R / "math_pool" / ds / "oss20lo_d0.jsonl"):
        r = json.loads(l)
        if r.get("finish_reason") != "error" and r.get("completion_tokens"):
            rows[r["problem_id"]] = (float(bool(r["resolved"])), np.log(max(r["completion_tokens"], 1)))
    pids, X = rich(R / f"{ds}_probe" / "instruct.npz", list(rows))
    y = np.array([rows[p][0] for p in pids]); Y = np.array([rows[p][1] for p in pids]); C = 1e-3 / max(1, X.shape[1] // 2560)
    yd, yf = np.zeros(len(pids)), np.zeros(len(pids))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        yd[te] = RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(Xtr, Y[tr]).predict(Xte)
        lg_tr = np.zeros(len(tr))
        for a, b in KFold(5, shuffle=True, random_state=1).split(Xtr):
            lg_tr[b] = logit_model(Xtr[a], y[tr][a], C).decision_function(Xtr[b])
        lg_te = logit_model(Xtr, y[tr], C).decision_function(Xte)
        F_tr, F_te = np.c_[lg_tr, lg_tr ** 2], np.c_[lg_te, lg_te ** 2]
        yf[te] = RidgeCV(alphas=np.geomspace(1e-3, 1e3, 13)).fit(F_tr, Y[tr]).predict(F_te)
    res = dict(n=len(pids), acc=float(y.mean()), sd_log_len=float(Y.std()), r2_direct=float(r2(Y, yd)), r2_from_success=float(r2(Y, yf)))
    res["gap"] = res["r2_direct"] - res["r2_from_success"]; out[ds] = res
    print(f"{ds:<10} n={len(pids):4d} acc {y.mean():.2f} sd(log len) {Y.std():.2f} | length R2 direct {res['r2_direct']:.2f}  "
          f"from success {res['r2_from_success']:.2f}  gap {res['gap']:+.2f}", flush=True)
json.dump(out, open(Path(__file__).parent / "screen_separation.json", "w"), indent=1)

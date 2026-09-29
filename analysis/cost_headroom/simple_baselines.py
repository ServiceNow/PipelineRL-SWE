"""Two simple baselines for the cost head (writes cost files for decompose.py):
  meanconst     cost = the problem's input x in-price + the model's TRAIN-MEAN output x out-price (the paper rule uses the MEDIAN;
                output length is right-skewed, so the mean is the natural competing constant)
  fromsuccess   cost inferred from the SUCCESS head alone: per model, ridge of log mean output on the success head's predictions
                for all models (logits and squares), fitted on TRAIN. Our analysis says most capturable cost value is a shared
                difficulty level, which the success head already reads -- if this matches the dedicated cost probe, the
                contribution is "use difficulty for cost", not a separate cost predictor.
Usage: python simple_baselines.py <pool>
"""
import json, sys, numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R

name = sys.argv[1]; D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; M = len(S); pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
v = t["valid"].astype(bool); ct = t["completion_tokens"].astype(float); pt = t["prompt_tokens"].astype(float); n = v.sum(2)
outm = np.where(n > 0, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan); Y = np.log(np.maximum(outm, 1))
inp = np.nanmean(np.where(v, pt, np.nan), 2)
tr = np.array([pi[str(p)] for p in json.load(open(D / "split_manifest.json"))["train_problem_ids"]])
te = np.array([pi[str(p)] for p in json.load(open(D / "split_manifest.json"))["test_problem_ids"]])
_lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D / "content_preds.jsonl")}
P = np.clip(np.array([_lp[p] for p in pids]), 1e-4, 1 - 1e-4); Lg = np.log(P / (1 - P)); F = np.c_[Lg, Lg ** 2]
fill = lambda m: np.where(np.isfinite(inp[:, m]), inp[:, m], np.nanmean(inp[tr, m]))


def write(tag, C):
    with open(D / f"cost_preds_{tag}.jsonl", "w") as f:
        for i, p in enumerate(pids):
            f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in C[i]]}) + "\n")


Cm = np.stack([(fill(m) * MK[s][0] + np.nanmean(ct[tr, m][v[tr, m]]) * MK[s][1]) / 1e6 for m, s in enumerate(S)], 1)
write("meanconst", Cm)
Cs = np.zeros_like(Cm); r2 = []
for m, s in enumerate(S):
    ok = tr[np.isfinite(Y[tr, m])]
    yh = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-3, 1e4, 15))).fit(F[ok], Y[ok, m]).predict(F)
    o = np.exp(yh) * np.mean(np.exp(Y[ok, m] - yh[ok])); o *= np.exp(Y[ok, m]).mean() / o[ok].mean()
    Cs[:, m] = (fill(m) * MK[s][0] + o * MK[s][1]) / 1e6
    tt = te[np.isfinite(Y[te, m])]; r2.append(1 - ((Y[tt, m] - yh[tt]) ** 2).sum() / ((Y[tt, m] - Y[tt, m].mean()) ** 2).sum())
write("fromsuccess", Cs)
print(f"{name}: cost-from-success-head test log-output R2 " + " ".join(f"{s} {x:+.2f}" for s, x in zip(S, r2)))

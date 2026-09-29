"""What drives output length on MMLU-Pro beyond difficulty (why difficulty-only pricing loses ~19 pt there, 4.A.26)?
Hypothesis: the KIND of work -- computational subjects need long worked solutions even when easy; recall subjects are short
even when hard -- which a frozen prefill reads but a success/difficulty latent does not.
(1) Per route: test R2 of log mean output from (a) the success head's logits [difficulty], (b) subject one-hot, (c) cheap
    text features (numbers in the question, statement length), (d) (a)+(b)+(c), (e) the 4B cost probe.
(2) Difficulty -> length correlation overall vs WITHIN subject (Simpson-style reversal?).
(3) Routing: cost from success + subject (+ text features) through decompose.py vs the probe (cost files written here).
Usage: python mmlupro_length_driver.py
"""
import json, re, sys, numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R

D = R / "mmlupro_tensors"; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; M = len(S); pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
v = t["valid"].astype(bool); ok = (t["final_outcome"] & t["valid"]).astype(float); ct = t["completion_tokens"].astype(float)
pt = t["prompt_tokens"].astype(float); n = v.sum(2)
Y = np.log(np.maximum(np.where(n > 0, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan), 1)); inp = np.nanmean(np.where(v, pt, np.nan), 2)
Q = np.where(n > 0, (ok * v).sum(2) / np.maximum(n, 1), np.nan)
sp = json.load(open(D / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]]); te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
prob = {json.loads(l)["problem_id"]: json.loads(l) for l in open(R / "math_pool" / "mmlupro" / "problems.jsonl")}
subj = [prob[p]["subject"] for p in pids]; SUB = sorted(set(subj)); SJ = np.array([[s == k for k in SUB] for s in subj], float)
txt = [prob[p]["problem_statement"] for p in pids]
TF = np.array([[len(re.findall(r"\d+\.?\d*", x)), np.log1p(len(x)), float(bool(re.search(r"calculate|compute|find|how (many|much)|what is the value", x, re.I)))] for x in txt])
lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D / "content_preds.jsonl")}
P = np.clip(np.array([lp[p] for p in pids]), 1e-4, 1 - 1e-4); Lg = np.log(P / (1 - P)); DIF = np.c_[Lg, Lg ** 2]
lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(D / "cost_preds_probe_instruct.jsonl")}
LC = np.array([lc[p] for p in pids]); pin = np.array([MK[s][0] for s in S]) / 1e6; pout = np.array([MK[s][1] for s in S]) / 1e6
PROBE = np.log(np.maximum((LC - np.nan_to_num(inp) * pin) / pout, 1))
feats = {"difficulty (success logits)": DIF, "subject": SJ, "text features": TF, "difficulty+subject": np.c_[DIF, SJ],
         "difficulty+subject+text": np.c_[DIF, SJ, TF]}


def r2(y, yh):
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


print("(1) test R2 of log mean output tokens, per route")
print(f"   {'features':<30}" + "".join(f"{s:>10}" for s in S))
preds = {}
for k, X in feats.items():
    row = []; preds[k] = np.zeros_like(Y)
    for m in range(M):
        a = tr[np.isfinite(Y[tr, m])]; b = te[np.isfinite(Y[te, m])]
        f = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-3, 1e4, 15))).fit(X[a], Y[a, m])
        preds[k][:, m] = f.predict(X); row.append(r2(Y[b, m], preds[k][b, m]))
    print(f"   {k:<30}" + "".join(f"{x:>10.2f}" for x in row))
row = [r2(Y[te[np.isfinite(Y[te, m])], m], PROBE[te[np.isfinite(Y[te, m])], m]) for m in range(M)]
print(f"   {'4B cost probe':<30}" + "".join(f"{x:>10.2f}" for x in row))

print("\n(2) difficulty -> length: corr(problem solve rate over all routes, log length) per route, overall vs within subject")
sr = np.nanmean(Q, 1)
for m in range(M):
    ok_ = np.isfinite(Y[:, m]); overall = np.corrcoef(sr[ok_], Y[ok_, m])[0, 1]
    w = []
    for k in SUB:
        ii = ok_ & (np.array(subj) == k)
        if ii.sum() > 10:
            w.append((ii.sum(), np.corrcoef(sr[ii], Y[ii, m])[0, 1]))
    within = np.average([c for _, c in w], weights=[c for c, _ in w])
    print(f"   {S[m]:<9} overall {overall:+.2f}   within-subject (weighted) {within:+.2f}")
print("\n   per subject: mean solve rate, mean output tokens (oss120hi), n")
for k in SUB:
    ii = np.array(subj) == k
    print(f"   {k:<18} solve {np.nanmean(sr[ii]):.2f}   out {np.exp(np.nanmean(Y[ii, S.index('oss120hi')])):6.0f}   n {ii.sum()}")

# (3) write cost files for routing
for k in ("difficulty+subject", "difficulty+subject+text"):
    C = np.zeros_like(LC)
    for m in range(M):
        a = tr[np.isfinite(Y[tr, m])]; o = np.exp(preds[k][:, m]) * np.mean(np.exp(Y[a, m] - preds[k][a, m]))
        C[:, m] = np.nan_to_num(inp[:, m], nan=np.nanmean(inp[tr, m])) * pin[m] + o * pout[m]
    tag = k.replace("+", "_").replace(" ", "")
    with open(D / f"cost_preds_driver_{tag}.jsonl", "w") as f:
        for i, p in enumerate(pids):
            f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in C[i]]}) + "\n")
    print(f"wrote cost_preds_driver_{tag}.jsonl")

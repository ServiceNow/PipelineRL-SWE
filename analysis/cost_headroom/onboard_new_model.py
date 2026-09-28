"""Onboarding a NEW model into a cost-aware router from k labelled examples (exploratory, offline).
Our analysis says per-query log cost ~ shared level(x) + a per-model offset, and the model-specific remainder is not
predictable anyway. So a new model h should need only its OFFSET: predicted log output for h = mean over the OTHER routes
of their probe predictions (the shared level) + offset_h estimated from k labelled TRAIN problems of h.
For each held-out route h and k in {5, 10, 20, 50} (20 random draws of the k problems), routing gain vs the paper rule on
the test split, with every other route on its full plain-probe cost head and the SAME success head for all:
  full       h also on its full cost head (trained on all TRAIN problems)       (the ceiling)
  onboard    h = shared level + offset from k examples
  median_k   h = input + median of h's output over the same k examples       (the paper rule, with k examples)
Usage: python onboard_new_model.py <pool> [<probe cost file>]
"""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R, hull, cost_at

name = sys.argv[1]; CF = sys.argv[2] if len(sys.argv) > 2 else "cost_preds_probe.jsonl"
D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; M = len(S); pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
v = t["valid"].astype(bool); ok = (t["final_outcome"] & t["valid"]).astype(float); ct = t["completion_tokens"].astype(float)
pt = t["prompt_tokens"].astype(float); n = v.sum(2); avail = n > 0
inp = np.nan_to_num(np.nanmean(np.where(v, pt, np.nan), 2))
pin = np.array([MK[s][0] for s in S]) / 1e6 * 100; pout = np.array([MK[s][1] for s in S]) / 1e6 * 100
outm = np.where(avail, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan); Y = np.log(np.maximum(outm, 1))
Q = np.where(avail, (ok * v).sum(2) / np.maximum(n, 1), 0)
Cr = np.where(avail, inp * pin + outm * pout, 1e9)
sp = json.load(open(D / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]]); te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
P = np.array([json.loads(l)["p_successes"][:M] for l in open(D / "content_preds.jsonl")])
lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(D / CF)}
LC = np.array([lc[p] for p in pids]) * 100
MU = np.log(np.maximum((LC - inp * pin) / pout, 1.0))
VS = np.geomspace(1e-5, 100, 250)


def frontier(C, ii):
    pts = []
    for V in VS:
        m = np.where(avail[ii], P[ii] * V - C[ii], -np.inf).argmax(1); r = np.arange(len(ii))
        pts.append((np.mean(Cr[ii][r, m]), np.mean(Q[ii][r, m])))
    return hull(pts)


med = np.array([np.median(ct[tr, m][v[tr, m]]) for m in range(M)])
H0 = frontier(inp * pin + med[None] * pout, te)


def gain(C):
    H = frontier(C, te); lo, hi = max(H[0][1], H0[0][1]), min(H[-1][1], H0[-1][1])
    T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    return 1 - float(np.exp(np.nanmean(np.log([cost_at(H, x) / cost_at(H0, x) for x in T]))))


full = gain(LC)
print(f"{name}: all routes on their full cost heads: {full*100:.1f}% vs the paper rule")
rng = np.random.default_rng(0); rows = {}
for h in range(M):
    lvl = np.nanmean(np.delete(MU, h, 1), 1)                  # shared level from the OTHER routes' predictions
    trh = tr[np.isfinite(Y[tr, h])]
    for k in (5, 10, 20, 50):
        g_on, g_md = [], []
        for _ in range(20):
            kk = rng.choice(trh, k, replace=False)
            off = np.mean(Y[kk, h] - lvl[kk]); smear = np.mean(np.exp(Y[kk, h] - (lvl[kk] + off)))
            C = LC.copy(); C[:, h] = inp[:, h] * pin[h] + np.exp(lvl + off) * smear * pout[h]; g_on.append(gain(C))
            C = LC.copy(); C[:, h] = inp[:, h] * pin[h] + np.median(outm[kk, h]) * pout[h]; g_md.append(gain(C))
        rows[(S[h], k)] = (np.mean(g_on), np.mean(g_md))
print(f"{'held-out route':<14}" + "".join(f"   k={k}: onboard / median_k" for k in (5, 10, 20, 50)))
for h in S:
    print(f"{h:<14}" + "".join(f"   {rows[(h, k)][0]*100:6.1f}% / {rows[(h, k)][1]*100:5.1f}%" for k in (5, 10, 20, 50)))
avg = lambda k, j: np.mean([rows[(h, k)][j] for h in S])
print(f"{'mean':<14}" + "".join(f"   {avg(k,0)*100:6.1f}% / {avg(k,1)*100:5.1f}%" for k in (5, 10, 20, 50)) + f"      (full heads: {full*100:.1f}%)")

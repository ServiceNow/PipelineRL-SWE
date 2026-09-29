"""Onboard a NEW model fully -- cost AND success -- from k labelled problems (exploratory, offline; extends onboard_new_model.py).
Cost: shared level (other routes' probe log-output predictions) + one offset from the k problems (as before).
Success: logit p_h(x) = a + b * dbar(x), dbar = mean logit of the OTHER routes' predicted success (the shared difficulty);
(a, b) fitted on the k problems' draws (L2-regularised logistic). Arms, gain vs the paper rule at matched accuracy (test),
every other route on its full heads, mean over held-out routes and 20 draws of the k problems:
  full        the new model's own trained cost + success heads            (ceiling)
  onboard     cost AND success onboarded from k examples (1 + 2 parameters)
  naive_k     median cost from k + a constant success rate from k            (what you'd do with k examples and no model)
Also: predicted onboarding LOSS -- is (full - onboard) explained by the model's share of MODEL-SPECIFIC cost variance?
Usage: python onboard_full.py <pool> <probe cost file>
"""
import json, sys, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R, hull, cost_at

name, CF = sys.argv[1], sys.argv[2]
D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; M = len(S); pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
v = t["valid"].astype(bool); okd = (t["final_outcome"] & t["valid"]).astype(bool); ct = t["completion_tokens"].astype(float)
pt = t["prompt_tokens"].astype(float); n = v.sum(2); avail = n > 0
inp = np.nan_to_num(np.nanmean(np.where(v, pt, np.nan), 2))
pin = np.array([MK[s][0] for s in S]) / 1e6 * 100; pout = np.array([MK[s][1] for s in S]) / 1e6 * 100
outm = np.where(avail, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan); Y = np.log(np.maximum(outm, 1))
Q = np.where(avail, (okd & v).sum(2) / np.maximum(n, 1), 0); Cr = np.where(avail, inp * pin + outm * pout, 1e9)
sp = json.load(open(D / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]]); te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
_lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D / "content_preds.jsonl")}
P = np.clip(np.array([_lp[p] for p in pids]), 1e-4, 1 - 1e-4); LOGIT = np.log(P / (1 - P))
lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(D / CF)}
LC = np.array([lc[p] for p in pids]) * 100; MU = np.log(np.maximum((LC - inp * pin) / pout, 1.0))
VS = np.geomspace(1e-5, 100, 250); med = np.array([np.median(ct[tr, m][v[tr, m]]) for m in range(M)])
import os
DROP = [x for x in os.environ.get("DROP", "").split(",") if x]           # routes removed from the pool entirely (e.g. dsv4f)
if DROP:
    keep = [i for i, s in enumerate(S) if s not in DROP]; S = [S[i] for i in keep]; M = len(S)
    v, okd, ct, pt, n, avail, inp, outm, Y, Q, Cr, P, LOGIT, LC, MU = (x[:, keep] for x in (v, okd, ct, pt, n, avail, inp, outm, Y, Q, Cr, P, LOGIT, LC, MU))
    pin, pout, med = pin[keep], pout[keep], med[keep]
# held-out GROUPS: each single route, plus extra groups onboarded together from the same k problems (e.g. a model's two efforts)
GROUPS = [[h] for h in range(M)] + [[S.index(x) for x in g.split("+")] for g in os.environ.get("HOLDGROUPS", "").split(",") if g]


def frontier(Pm, C, ii):
    pts = []
    for V in VS:
        m = np.where(avail[ii], Pm[ii] * V - C[ii], -np.inf).argmax(1); r = np.arange(len(ii))
        pts.append((np.mean(Cr[ii][r, m]), np.mean(Q[ii][r, m])))
    return hull(pts)


H0 = frontier(P, inp * pin + med[None] * pout, te)


MAXACC = []                                           # max test accuracy of the last gain() call's frontier


def gain(Pm, C):
    H = frontier(Pm, C, te); MAXACC.append(H[-1][1]); lo, hi = max(H[0][1], H0[0][1]), min(H[-1][1], H0[-1][1])
    T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    return 1 - float(np.exp(np.nanmean(np.log([cost_at(H, x) / cost_at(H0, x) for x in T]))))


full = gain(P, LC); FULLACC = MAXACC[-1]; rng = np.random.default_rng(0); out = {}
without = {}
for G in GROUPS:                                      # the pool WITHOUT the group at all (made unavailable)
    Pw = P.copy(); Cw = LC.copy(); Pw[:, G] = 0.0; Cw[:, G] = 1e9
    without["+".join(S[h] for h in G)] = gain(Pw, Cw); out[("+".join(S[h] for h in G), "maxacc_without")] = MAXACC[-1]
L = np.log(np.maximum(outm, 1)); lvl_true = np.nanmean(L, 1); dd = L - lvl_true[:, None]
for G in GROUPS:
    gname = "+".join(S[h] for h in G)
    lvl = np.nanmean(np.delete(MU, G, 1), 1); dbar = np.nanmean(np.delete(LOGIT, G, 1), 1)
    trh = tr[avail[tr][:, G].all(1)]
    acc_k = []
    for k in (5, 10, 20, 50):
        g_on, g_nv = [], []
        for _ in range(20):
            kk = rng.choice(trh, k, replace=False)
            Pm = P.copy(); C = LC.copy(); Pn = P.copy(); Cn = LC.copy()
            for h in G:
                off = np.mean(Y[kk, h] - lvl[kk]); smear = np.mean(np.exp(Y[kk, h] - (lvl[kk] + off)))
                Xk = np.repeat(dbar[kk], n[kk, h]); yk = np.concatenate([okd[i, h][v[i, h]] for i in kk]).astype(int)
                C[:, h] = inp[:, h] * pin[h] + np.exp(lvl + off) * smear * pout[h]
                if len(set(yk)) == 2:
                    lr = LogisticRegression(C=1.0).fit(Xk[:, None], yk); Pm[:, h] = lr.predict_proba(dbar[:, None])[:, 1]
                else:
                    Pm[:, h] = np.clip(yk.mean(), 0.02, 0.98)
                Pn[:, h] = np.clip(yk.mean(), 0.02, 0.98); Cn[:, h] = inp[:, h] * pin[h] + np.nanmedian(outm[kk, h]) * pout[h]
            g_on.append(gain(Pm, C)); a_on = MAXACC[-1]; g_nv.append(gain(Pn, Cn)); acc_k.append((a_on, MAXACC[-1]))
        out[(gname, k)] = (np.mean(g_on), np.mean(g_nv)); out[(gname, f"maxacc_{k}")] = tuple(np.mean(acc_k, 0)); acc_k.clear()
    if len(G) == 1:
        h = G[0]; out[(gname, "spec_share")] = float(np.nanvar(dd[:, h]) / (np.nanvar(lvl_true) + np.nanvar(dd[:, h])))
GN = ["+".join(S[h] for h in G) for G in GROUPS]
print(f"{name}: all routes on full heads {full*100:.1f}%.  per held-out route: WITHOUT it | onboard (cost+success) / naive_k")
for h in GN:
    print(f"   {h:<17} without {without[h]*100:5.1f}% |" + "".join(f"  k={k}: {out[(h,k)][0]*100:5.1f}% / {out[(h,k)][1]*100:5.1f}%" for k in (5, 10, 50))
          + f"   (full {full*100:.1f}%)\n{'':20}max test accuracy: full {FULLACC*100:.1f} | without {out[(h,'maxacc_without')]*100:.1f} | "
          + "  ".join(f"k={k}: onboard {out[(h,f'maxacc_{k}')][0]*100:.1f} / naive {out[(h,f'maxacc_{k}')][1]*100:.1f}" for k in (5, 10, 50)))
mean = lambda k, j: np.mean([out[(h, k)][j] for h in S])     # single routes only
print("   mean     " + "".join(f"  k={k}: {mean(k,0)*100:5.1f}% / {mean(k,1)*100:5.1f}%" for k in (5, 10, 20, 50)))
json.dump({f"{a}|{b}": v for (a, b), v in out.items()} | {"full": full} | {f"without|{k}": v for k, v in without.items()}, open(f"analysis/cost_headroom/onboard_full_{name}{'_no_' + '_'.join(DROP) if DROP else ''}.json", "w"), indent=1)

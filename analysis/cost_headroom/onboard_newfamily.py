"""A NEW model family shows up: onboard GLM-4.7-flash, Nemotron-3-super and MiniMax-M2.5 into the MMLU-Pro router (5 routes, full
heads) from k labelled problems, and score routing on MMLU-Pro's TEST split (the new models ran once on test + 50 train problems).
New model h, from k of the 50 train problems:
  onboard  cost = shared level (mean over the 5 existing routes' probe log-output predictions) + offset from k; success = logistic in
           the shared difficulty (mean logit of the existing routes' success predictions) fitted on k
  naive    cost = input + median output of the k; success = the k problems' success rate (constant)
Reported on the test split, vs the paper rule of the ORIGINAL 5-route pool (the fixed reference): the 5-route router alone, and the
router with all three new models added (onboard / naive), over 20 draws of the k problems. Question: does a router with new models
added from a handful of examples beat the router without them -- and beat adding them naively?
"""
import json, sys, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R, hull, cost_at

NEW = {"glm47f": (0.061, 0.40), "nemo120": (0.08, 0.45), "mm25": (0.27, 1.08)}             # $/M in, out (OpenRouter 2026-09-28)
D = R / "mmlupro_tensors"; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; M = len(S); pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
v = t["valid"].astype(bool); okd = (t["final_outcome"] & t["valid"]).astype(bool); ct = t["completion_tokens"].astype(float); pt = t["prompt_tokens"].astype(float)
n = v.sum(2); avail = n > 0; inp = np.nan_to_num(np.nanmean(np.where(v, pt, np.nan), 2))
pin = np.array([MK[s][0] for s in S]) / 1e6 * 100; pout = np.array([MK[s][1] for s in S]) / 1e6 * 100
outm = np.where(avail, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan)
Q = np.where(avail, (okd & v).sum(2) / np.maximum(n, 1), 0); Cr = np.where(avail, inp * pin + outm * pout, 1e9)
_lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D / "content_preds.jsonl")}
P = np.clip(np.array([_lp[p] for p in pids]), 1e-4, 1 - 1e-4); DBAR = np.log(P / (1 - P)).mean(1)
lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(D / "cost_preds_probe_instruct.jsonl")}
LC = np.array([lc[p] for p in pids]) * 100; LVL = np.log(np.maximum((LC - inp * pin) / pout, 1.0)).mean(1)
sp = json.load(open(R / "mmlupro_newfam_split.json")); te = np.array([pi[p] for p in sp["test"]]); trk = np.array([pi[p] for p in sp["onboard_train"]])
tr = np.array([pi[str(p)] for p in json.load(open(D / "split_manifest.json"))["train_problem_ids"]])
med = np.array([np.median(ct[tr, m][v[tr, m]]) for m in range(M)])
new = {}
for h, (a_in, a_out) in NEW.items():
    rows = {json.loads(l)["problem_id"]: json.loads(l) for l in open(R / "math_pool_newfam" / "mmlupro" / f"{h}_d0.jsonl") if json.loads(l).get("finish_reason") != "error"}
    q = np.array([float(rows[p]["resolved"]) if p in rows else np.nan for p in pids])
    out = np.array([rows[p]["completion_tokens"] if p in rows else np.nan for p in pids], float)
    ip = np.array([rows[p]["prompt_tokens"] if p in rows else np.nan for p in pids], float)
    new[h] = dict(q=q, out=out, cost=(ip * a_in + out * a_out) / 1e6 * 100, pin=a_in / 1e6 * 100, pout=a_out / 1e6 * 100)
    ok = np.isfinite(q[te]); print(f"{h:<8} test coverage {ok.mean()*100:.0f}%  test acc {np.nanmean(q[te]):.3f}  mean cost {np.nanmean(new[h]['cost'][te]):.4f}c")
VS = np.geomspace(1e-5, 100, 250)


def frontier(Pm, C, Qm, Crm, ii):
    pts = []
    for V in VS:
        m = np.where(np.isfinite(Crm[ii]) & (Crm[ii] < 1e8), Pm[ii] * V - C[ii], -np.inf).argmax(1); r = np.arange(len(ii))
        pts.append((np.mean(Crm[ii][r, m]), np.mean(Qm[ii][r, m])))
    return hull(pts)


H0 = frontier(P, inp * pin + med[None] * pout, Q, Cr, te)          # paper rule, original 5 routes: the fixed reference


def gain(H):
    lo, hi = max(H[0][1], H0[0][1]), min(H[-1][1], H0[-1][1]); T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    return 1 - float(np.exp(np.nanmean(np.log([cost_at(H, x) / cost_at(H0, x) for x in T])))), (H[-1][1])


base = gain(frontier(P, LC, Q, Cr, te)); print(f"5-route router (full heads): {base[0]*100:.1f}% vs the 5-route paper rule; max test accuracy {base[1]*100:.1f}%")
rng = np.random.default_rng(0)
for k in (5, 10, 20, 50):
    res = {"onboard": [], "naive": []}
    for _ in range(20):
        kk = rng.choice(trk, k, replace=False); Pm, Cm, Qm, Crm = [P], [LC], [Q], [Cr]; Pn, Cn = [P], [LC]
        for h, d in new.items():
            ok = kk[np.isfinite(d["q"][kk])]
            off = np.mean(np.log(d["out"][ok]) - LVL[ok]); cost_on = inp.mean(1) * d["pin"] + np.exp(LVL + off) * d["pout"]
            y = d["q"][ok].astype(int)
            p_on = LogisticRegression(C=1.0).fit(DBAR[ok][:, None], y).predict_proba(DBAR[:, None])[:, 1] if len(set(y)) == 2 else np.full(len(pids), np.clip(y.mean(), .02, .98))
            Pm.append(p_on[:, None]); Cm.append(cost_on[:, None])
            Pn.append(np.full((len(pids), 1), np.clip(y.mean(), .02, .98))); Cn.append((inp.mean(1) * d["pin"] + np.median(d["out"][ok]) * d["pout"])[:, None])
            Qm.append(np.nan_to_num(d["q"])[:, None]); Crm.append(np.where(np.isfinite(d["cost"]), d["cost"], 1e9)[:, None])
        Qa, Cra = np.hstack(Qm), np.hstack(Crm)
        res["onboard"].append(gain(frontier(np.hstack(Pm), np.hstack(Cm), Qa, Cra, te)))
        res["naive"].append(gain(frontier(np.hstack(Pn), np.hstack(Cn), Qa, Cra, te)))
    print(f"k={k:>2}: + 3 new models  onboard {np.mean([g for g, _ in res['onboard']])*100:5.1f}% (max acc {np.mean([a for _, a in res['onboard']])*100:.1f}%)"
          f"   naive {np.mean([g for g, _ in res['naive']])*100:5.1f}% (max acc {np.mean([a for _, a in res['naive']])*100:.1f}%)")

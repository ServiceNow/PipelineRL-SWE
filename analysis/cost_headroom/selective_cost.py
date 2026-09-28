"""Use the learned cost only where it is confident (idea #5). Per (problem, route): a bootstrap ensemble of 20 plain
ridge heads on the frozen probe gives a log-output prediction mu and an uncertainty sigma. The cost fed to the rule
is a precision-weighted blend of the learned prediction and the paper rule's (input + median train output):
    log out = w * mu + (1 - w) * log(median output),   w = 1 / (1 + sigma^2 / tau^2)
tau -> 0 recovers the paper rule exactly (a no-harm fallback); tau -> inf the plain learned head. tau is chosen on the
CALIBRATION split by the frontier gain vs the paper rule (same metric as decompose.py), then applied once to test.
Writes cost_preds_selective.jsonl; decompose.py then reports the test gain with its CI.
Usage: python selective_cost.py <pool> <activations.npz>
"""
import json, sys, numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R, VS, hull, cost_at
from baseline_cost_heads import rich

name, act = sys.argv[1], sys.argv[2]
D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; M = len(S); pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
v = t["valid"].astype(bool); ok = (t["final_outcome"] & t["valid"]).astype(float)
pt, ct = t["prompt_tokens"].astype(float), t["completion_tokens"].astype(float); n = v.sum(2); avail = n > 0
Q = np.where(avail, (ok * v).sum(2) / np.maximum(n, 1), 0)
real = np.stack([(pt[:, m] * MK[s][0] + ct[:, m] * MK[s][1]) / 1e6 * 100 for m, s in enumerate(S)], 1)
Cr = np.where(avail, (real * v).sum(2) / np.maximum(n, 1), 1e9)
outm = np.where(avail, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan); Y = np.log(np.maximum(outm, 1))
inp = np.nanmean(np.where(v, pt, np.nan), 2)
sp = json.load(open(D / "split_manifest.json")); idx = {k: np.array([pi[str(p)] for p in sp[f"{k}_problem_ids"]]) for k in ("train", "calibration", "test")}
tr, cal, te = idx["train"], idx["calibration"], idx["test"]
P = np.array([[json.loads(l)["p_successes"][:M] for l in open(D / "content_preds.jsonl") if json.loads(l)["problem_id"] == p][0] for p in pids]) \
    if False else None
lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D / "content_preds.jsonl")}
P = np.array([lp[p] for p in pids])
X = rich(act, pids)
rng = np.random.default_rng(0); MU = np.zeros((len(pids), M)); SD = np.zeros((len(pids), M)); med = np.zeros(M)
for m in range(M):
    trm = tr[np.isfinite(Y[tr, m])]; med[m] = np.median(ct[trm, m][v[trm, m]])
    sc = StandardScaler().fit(X[trm]); Xs = sc.transform(X)
    alpha = RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(Xs[trm], Y[trm, m]).alpha_
    preds = []
    for b in range(20):
        bs = rng.choice(trm, len(trm), replace=True)
        preds.append(Ridge(alpha=alpha).fit(Xs[bs], Y[bs, m]).predict(Xs))
    preds = np.array(preds); MU[:, m] = preds.mean(0); SD[:, m] = preds.std(0)


def costs(tau):
    w = 1.0 / (1.0 + SD ** 2 / max(tau, 1e-9) ** 2)
    lo = w * MU + (1 - w) * np.log(med)[None]
    C = np.zeros_like(MU)
    for m, s in enumerate(S):
        trm = tr[np.isfinite(Y[tr, m])]
        o = np.exp(lo[:, m]); o *= np.nanmean(outm[trm, m]) / o[trm].mean()
        C[:, m] = (np.nan_to_num(inp[:, m], nan=np.nanmean(inp[:, m])) * MK[s][0] + o * MK[s][1]) / 1e6 * 100
    return C


def gain(C, ii, Cpaper):
    def frontier(Cx):
        U = np.where(avail[ii][None], P[ii][None] * VS[:, None, None] - Cx[ii][None], -np.inf); ch = U.argmax(2)
        a = np.take_along_axis(np.broadcast_to(Q[ii], (len(VS),) + Q[ii].shape), ch[..., None], 2)[..., 0].mean(1)
        c = np.take_along_axis(np.broadcast_to(Cr[ii], (len(VS),) + Cr[ii].shape), ch[..., None], 2)[..., 0].mean(1)
        return hull(list(zip(c, a)))
    H, H0 = frontier(C), frontier(Cpaper)
    lo_, hi_ = max(H[0][1], H0[0][1]), min(H[-1][1], H0[-1][1]); T = np.linspace(lo_ + .05 * (hi_ - lo_), hi_ - .05 * (hi_ - lo_), 12)
    r = np.array([cost_at(H, x) / cost_at(H0, x) for x in T]); return 1 - float(np.exp(np.nanmean(np.log(r))))


Cpaper = costs(1e-9)
grid = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0, 1e3]
cal_g = {tau: gain(costs(tau), cal, Cpaper) for tau in grid}
best = max(grid, key=lambda k: cal_g[k])
print(f"{name}: median ensemble sigma (log) per route {np.median(SD[te], 0).round(3)}")
print("calibration gain by tau: " + "  ".join(f"{k:g}:{g*100:+.1f}%" for k, g in cal_g.items()) + f"  -> chosen tau {best:g}")
C = costs(best) / 100
with open(D / "cost_preds_selective.jsonl", "w") as f:
    for i, p in enumerate(pids):
        f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in C[i]]}) + "\n")
Cfull = costs(1e3) / 100
with open(D / "cost_preds_selective_off.jsonl", "w") as f:           # tau = inf: the plain ensemble head (reference)
    for i, p in enumerate(pids):
        f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in Cfull[i]]}) + "\n")

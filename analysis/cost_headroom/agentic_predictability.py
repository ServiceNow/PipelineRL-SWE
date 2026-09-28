"""Is AGENTIC per-task cost predictable from the task statement? SWE-rebench July 2026 trajectories (111 tasks).
Target per (task, model): log mean cost over runs {0,1,2} (the fit runs). Features: Qwen3-4B Instruct prefill of the issue
text (rich readout). 10-fold cross-validated ridge per model -> out-of-fold predicted cost for every task (no task ever
predicts itself). Then the cross-fitted routing test of agentic_swe_rebench.py among the 7 open-weight models: success
from runs {0,1,2}, scored on the realised outcome and cost of runs {3,4}; arms = model median cost (paper rule), PREDICTED
per-task cost, and ORACLE per-task cost (runs 0-2 mean). 111 tasks: indicative only.
"""
import json, numpy as np, sys
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from decompose import hull, cost_at
from baseline_cost_heads import rich
import agentic_swe_rebench as A                            # builds C, S, parts, tasks, kind at import (prints its report)

OPEN = sorted(A.OPEN); M = [A.pi[p] for p in OPEN]
X = rich(Path("/mnt/llmd/results/exps/aristides/reason/swe_rebench_probe/instruct.npz"), A.tasks)
fit, ev = [0, 1, 2], [3, 4]
Cf = np.nanmean(A.C[M][:, :, fit], 2).T; Pf = np.nanmean(A.S[M][:, :, fit], 2).T
Ce = np.nanmean(A.C[M][:, :, ev], 2).T; Qe = np.nanmean(A.S[M][:, :, ev], 2).T
Y = np.log(Cf); pred = np.full_like(Y, np.nan); r2 = []
for m in range(len(M)):
    ok = np.isfinite(Y[:, m]); idx = np.where(ok)[0]
    for tr, te in KFold(10, shuffle=True, random_state=0).split(idx):
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e1, 1e7, 13))).fit(X[idx[tr]], Y[idx[tr], m])
        pred[idx[te], m] = mdl.predict(X[idx[te]])
    y, p = Y[ok, m], pred[ok, m]
    r2.append(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
print("\nout-of-fold log-cost R2 from the task statement (4B prefill), per open model: " +
      "  ".join(f"{o.split('/')[-1]} {x:+.2f}" for o, x in zip(OPEN, r2)) + f"   mean {np.mean(r2):+.2f}")
avail = np.isfinite(Cf) & np.isfinite(Ce) & np.isfinite(Pf)
const = np.nanmedian(Cf, 0)[None, :].repeat(len(A.tasks), 0)
Cpred = np.exp(pred) * (np.nanmean(Cf, 0) / np.nanmean(np.exp(pred), 0))[None, :]      # level-matched
VS = np.geomspace(1e-4, 1e4, 400)


def frontier(Cest, ii):
    pts = []
    for V in VS:
        U = np.where(avail[ii], Pf[ii] * V - Cest[ii], -np.inf); ch = U.argmax(1); r = np.arange(len(ii))
        pts.append((np.nanmean(Ce[ii][r, ch]), np.nanmean(Qe[ii][r, ch])))
    return hull(pts)


def gains(ii):
    Hc, Hp, Ho = (frontier(np.where(avail, X_, 1e9), ii) for X_ in (const, Cpred, Cf))
    lo = max(h[0][1] for h in (Hc, Hp, Ho)); hi = min(h[-1][1] for h in (Hc, Hp, Ho))
    T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 10)
    g = lambda H: 1 - float(np.exp(np.nanmean(np.log([cost_at(H, x) / cost_at(Hc, x) for x in T]))))
    return g(Ho), g(Hp)


h, gp = gains(np.arange(len(A.tasks)))
rng = np.random.default_rng(0); B = np.array([gains(rng.integers(0, len(A.tasks), len(A.tasks))) for _ in range(1000)])
print(f"routing among the 7 open-weight models, cross-fitted: HEADROOM (oracle per-task cost) {h*100:.1f}% "
      f"[{np.percentile(B[:,0],2.5)*100:.1f}, {np.percentile(B[:,0],97.5)*100:.1f}]; PREDICTED per-task cost {gp*100:.1f}% "
      f"[{np.percentile(B[:,1],2.5)*100:.1f}, {np.percentile(B[:,1],97.5)*100:.1f}]  (vs each model's median cost)")

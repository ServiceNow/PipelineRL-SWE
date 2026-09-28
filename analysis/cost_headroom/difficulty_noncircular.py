"""Non-circular test of "difficulty drives length but is not legible" (reply to the circularity concern on 4.A.10).

The 4.A.10 'empirical difficulty' used each problem's solve rate measured on the SAME draws whose length it explains.
Here difficulty is cross-fitted so no draw contributes to both sides:
  cross-model  difficulty = mean solve rate of the OTHER routes; target = log mean length of route m
  split-draw   difficulty = solve rate on route m's EVEN draws (all routes); target = log mean length on route m's ODD draws
               (pools with >= 2 draws per route)
For each: (i) LINK = 5-fold CV R2 of the target on a cubic in the difficulty measure; (ii) LEGIBILITY = 5-fold CV R2 of
the difficulty measure from the 4B prefill (plain ridge); (iii) the external-label link where a label exists.
Reading: link high & legibility low -> difficulty that is there but not legible; link low -> length is not difficulty-driven.
"""
import json, sys, numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from decompose import R
from why_predictable import cv_r2
from baseline_cost_heads import rich
from why_decompose import labels

POOLS = {"pool_v2_tensors_5rung": R / "pv2_scout_prefill_1756715297/scout.npz", "cc_tensors": R / "cc_pool/scout_prefill.npz",
         "taco_tensors_ha": R / "taco_activations_1788500841/scout.npz", "bcb_tensors_5r": R / "bcb_scout_prefill.npz",
         "omni500_tensors": R / "omni500_probe/instruct.npz"}


def legible(X, y):
    ok = np.isfinite(y)
    p = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e1, 1e7, 13))), X[ok], y[ok], cv=5)
    return 1 - ((y[ok] - p) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()


cub = lambda d: np.c_[d, d ** 2, d ** 3]
print("per pool, mean over routes (5-fold CV R2). LINK = difficulty -> log length; LEGIBILITY = prefill -> difficulty")
print(f"{'pool':<24}{'cross-model: link  legib':>28}{'split-draw: link  legib':>27}{'external label: link  legib':>31}")
for name, act in POOLS.items():
    D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True); pids = [str(p) for p in t["problem_ids"]]
    v = t["valid"].astype(bool); ok = (t["final_outcome"] & t["valid"]).astype(float); ct = t["completion_tokens"].astype(float)
    n = v.sum(2); M = v.shape[1]; K = v.shape[2]
    Q = np.where(n > 0, (ok * v).sum(2) / np.maximum(n, 1), np.nan)
    Lm = np.log(np.maximum(np.where(n > 0, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan), 1))
    X = rich(act, pids); L = labels(name, pids)
    cm_link, cm_leg, sd_link, sd_leg, lab_link = [], [], [], [], []
    for m in range(M):
        other = np.nanmean(np.delete(Q, m, 1), 1); a = np.isfinite(Lm[:, m]) & np.isfinite(other)
        cm_link.append(cv_r2(cub(other[a]), Lm[a, m])); cm_leg.append(legible(X, other))
        if K >= 2:
            ev, od = v[:, :, 0::2], v[:, :, 1::2]
            q_even = np.nanmean(np.where(ev.sum(2) > 0, (ok[:, :, 0::2] * ev).sum(2) / np.maximum(ev.sum(2), 1), np.nan), 1)
            no = od[:, m].sum(1); l_odd = np.log(np.maximum(np.where(no > 0, np.where(od[:, m], ct[:, m, 1::2], 0).sum(1) / np.maximum(no, 1), np.nan), 1))
            b = np.isfinite(q_even) & np.isfinite(l_odd)
            sd_link.append(cv_r2(cub(q_even[b]), l_odd[b])); sd_leg.append(legible(X, q_even))
        if L is not None:
            lab_link.append(cv_r2(L[a], Lm[a, m]))
    lab_leg = legible(X, L[:, 0]) if L is not None else np.nan
    f = lambda x: f"{np.mean(x):6.2f}" if len(x) else "   n/a"
    print(f"{name:<24}{f(cm_link):>16}{f(cm_leg):>8}{f(sd_link):>18}{f(sd_leg):>8}{f(lab_link):>21}{lab_leg:8.2f}")

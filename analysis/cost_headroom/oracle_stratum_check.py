"""Does predicted success carry length cues beyond the measured solve rate? (NEW_PATH 4.A.131; paper app:mechanism)
Paper split. Per candidate: Spearman of its predicted success logit with its log length on all test problems, and among test problems
that every OTHER candidate solved on every draw (oracle difficulty constant there). Then the best R2 any function of the oracle can
reach fitted on the test set itself (group means), vs length from predicted success fitted on training.
Usage: REASON_ROOT=.../reason_pinned python oracle_stratum_check.py MMLU-Pro LCB
"""
import os, sys
from pathlib import Path
import numpy as np
CG = str(Path(__file__).resolve().parent / "cost_generalization.py")
sys.path.insert(0, str(Path(CG).parent)); __file__ = CG
WANT = sys.argv[1:]; sys.argv = ["x", "fc", "ALL"]
exec(open(CG).read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
from scipy.stats import spearmanr
for P in WANT:
    d = load(P); ev = d["ev"]; Y = np.log(d["L"]); lg = np.log(d["P"] / (1 - d["P"])); q = d["q"]; M = d["M"]
    sa = q[ev].mean(1)
    print(f"\n{P}: test n={len(ev)}, distinct solve-rate values={len(np.unique(sa.round(6)))}, share solved by all={np.mean(sa == 1):.2f}, by none={np.mean(sa == 0):.2f}")
    for k in range(M):
        so = np.delete(q, k, 1).mean(1)[ev]          # oracle over the other candidates
        allok = ev[so == 1]
        rho_all = spearmanr(lg[ev, k], Y[ev, k])[0]; rho_or = spearmanr(so, Y[ev, k])[0]
        rho_in = spearmanr(lg[allok, k], Y[allok, k])[0] if len(allok) > 20 else np.nan
        print(f"  {d['slots'][k]:<9} rho(logit,len) {rho_all:+.2f}  rho(oracle,len) {rho_or:+.2f}  | among {len(allok)} solved by all others: rho(logit,len) {rho_in:+.2f}")
print("\n--- best possible R2 of ANY function of the oracle, fit on the test set itself (group means), vs length-from-predicted-success (fitted on train)")
for P in WANT:
    d = load(P); N = len(d["ids"]); ev = d["ev"]; Y = np.log(d["L"]); q = d["q"]; M = d["M"]
    fs = np.log(np.maximum(fit_predict("fromsuccess", view(d, d["tr"]), view(d, np.arange(N))), 1))
    r2 = lambda y, yh: 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    for k in range(M):
        so = np.delete(q, k, 1).mean(1)[ev].round(6); y = Y[ev, k]
        gm = {v: y[so == v].mean() for v in np.unique(so)}; eta = r2(y, np.array([gm[v] for v in so]))
        print(f"  {P:<9}{d['slots'][k]:<9} oracle (in-sample, any shape) {eta:.2f}   predicted success (out of sample) {r2(y, fs[ev, k]):.2f}")

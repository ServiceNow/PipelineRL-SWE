"""Partial trajectory READ BY THE 4B MODEL: issue + the first 10 agent steps + the question "how many more steps / how costly
will the rest be?", prefilled by Qwen3-4B Instruct; last-token rich readout -> ridge. Same targets and grouped-by-task CV
as agentic_partial_traj.py (hand-crafted features: own R2 ~0 at k=10, cross negative), so the two compare directly.
  own    per model, log final cost of the run (runs that finished within 10 steps excluded)
  cross  the scout's (MiMo-V2.5-Pro, run 0) prefill predicts each other model's log mean cost on the task
"""
import json, gzip, glob, numpy as np, collections, sys
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from baseline_cost_heads import rich
from agentic_partial_traj import rows, models, tasks, ti, SCOUT        # parsed runs with cost + hand-crafted features

R = Path("/mnt/llmd/results/exps/aristides/reason"); K = 10
key = lambda r: f"{r['model']}||{r['task']}||{r['run']}"
z = np.load(R / "swe_rebench_partial_probe" / "instruct.npz", allow_pickle=True); have = {str(p) for p in z["problem_ids"]}
rs_all = [r for r in rows if key(r) in have]
X_all = rich(R / "swe_rebench_partial_probe" / "instruct.npz", [key(r) for r in rs_all]); xi = {key(r): i for i, r in enumerate(rs_all)}


def cv_r2(X, y, g):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        pred[te] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e1, 1e7, 13))).fit(X[tr], y[tr]).predict(X[te])
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


own = []
for m in models:
    rs = [r for r in rs_all if r["model"] == m and r["F"][K][1]]
    if len(rs) < 60:
        continue
    own.append(cv_r2(X_all[[xi[key(r)] for r in rs]], np.log([r["cost"] for r in rs]), [ti[r["task"]] for r in rs]))
print(f"OWN run, 4B reads issue + first {K} steps: log final-cost R2 mean {np.mean(own):+.2f} (range {min(own):+.2f}..{max(own):+.2f}, {len(own)} models)"
      f"   [hand-crafted features at k={K}: -0.01]")
Y = collections.defaultdict(dict)
for r in rows:
    Y[r["model"]].setdefault(r["task"], []).append(r["cost"])
scout = {r["task"]: r for r in rs_all if r["model"] == SCOUT and r["run"] == 0}
cross = []
for m in models:
    if m == SCOUT:
        continue
    ts = [t for t in tasks if t in scout and t in Y[m]]
    cross.append(cv_r2(X_all[[xi[key(scout[t])] for t in ts]], np.log([np.mean(Y[m][t]) for t in ts]), [ti[t] for t in ts]))
print(f"CROSS-model, 4B reads the scout's first {K} steps: other models' log mean-cost R2 mean {np.mean(cross):+.2f} "
      f"(range {min(cross):+.2f}..{max(cross):+.2f})   [hand-crafted: -0.54; statement prefill: +0.01]")

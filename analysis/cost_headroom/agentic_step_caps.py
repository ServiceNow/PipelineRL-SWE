"""Per-task STEP CAPS for agents (exploratory; exact offline simulation). nebius/SWE-agent-trajectories, SWE-agent +
Llama-3.1-70B, ~20 runs per task; cum_cost[k] = cost proxy (cumulative context + 4x output chars) after agent step k+1.
Under a cap of c steps a run that finished within c steps is unchanged; a longer run is a FAILURE that cost cum_cost[c-1].
Caps:
  global     one step cap for every task (a max_steps setting), swept
  pertask    c(x) = the q-quantile of the task's PREDICTED step distribution: log steps ~ N(mu(x), s^2); mu(x) from the issue-text
             prefill (5-fold CV over tasks, out-of-fold), s from out-of-fold run-level residuals; q swept
  oracle     c(x) = q-quantile around the task's TRUE mean log steps (same s) -- upper bound on what prediction can give
Metric: the (mean cost, resolve rate) trade-off curve over all runs; cost at matched resolve rate vs the global cap; paired
bootstrap over tasks. Question: do predicted per-task caps beat a single max_steps?
"""
import json, numpy as np, collections, sys
from pathlib import Path
from scipy.stats import norm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from baseline_cost_heads import rich
from decompose import hull, cost_at

R = Path("/mnt/llmd/results/exps/aristides/reason"); MODEL = "swe-agent-llama-70b"
runs = collections.defaultdict(list)
for l in open(R / "nebius_swe_agent_traj" / "light.jsonl"):
    r = json.loads(l)
    if r["model"] == MODEL and r["steps"] > 0 and r["cum_cost"]:
        runs[r["instance_id"]].append((r["steps"], bool(r["resolved"]), np.array(r["cum_cost"], float)))
z = np.load(R / "nebius_issue_probe" / "instruct.npz", allow_pickle=True); have = {str(p) for p in z["problem_ids"]}
tasks = sorted(t for t in runs if t in have and len(runs[t]) >= 4)
X = rich(R / "nebius_issue_probe" / "instruct.npz", tasks)
y = np.array([np.mean([np.log(s) for s, _, _ in runs[t]]) for t in tasks])            # per-task mean log steps
mu = np.zeros(len(tasks))
for trn, tst in KFold(5, shuffle=True, random_state=0).split(X):
    mu[tst] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e1, 1e7, 13))).fit(X[trn], y[trn]).predict(X[tst])
resid = np.concatenate([[np.log(s) - mu[i] for s, _, _ in runs[t]] for i, t in enumerate(tasks)]); s_pred = resid.std()
resid_o = np.concatenate([[np.log(s) - y[i] for s, _, _ in runs[t]] for i, t in enumerate(tasks)]); s_orc = resid_o.std()
r2 = 1 - ((y - mu) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"{len(tasks)} tasks, {sum(len(runs[t]) for t in tasks)} runs; out-of-fold R2 of per-task mean log steps {r2:.2f}; "
      f"run-level sd around prediction {s_pred:.2f} (around the true task mean {s_orc:.2f}); resolve rate "
      f"{np.mean([r for t in tasks for _, r, _ in runs[t]]):.3f}")


def evaluate(caps, ii):              # caps: per-task step caps (array over tasks) -> (mean cost, resolve rate) over runs
    cost, res = [], []
    for i in ii:
        c = caps[i]
        for s, r, cum in runs[tasks[i]]:
            if s <= c:
                cost.append(cum[-1]); res.append(r)
            else:
                cost.append(cum[max(int(c), 1) - 1]); res.append(False)
    return np.mean(cost), np.mean(res)


def curve(kind, ii):
    pts = []
    if kind == "global":
        for c in [3, 5, 7, 10, 12, 15, 18, 22, 26, 30, 35, 40, 50, 60, 80, 1e9]:
            pts.append(evaluate(np.full(len(tasks), c), ii))
    else:
        m, sd = (mu, s_pred) if kind == "pertask" else (y, s_orc)
        for q in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.995]:
            pts.append(evaluate(np.ceil(np.exp(m + sd * norm.ppf(q))), ii))
        pts.append(evaluate(np.full(len(tasks), 1e9), ii))
    return hull(pts)


def saving(H, H0):
    lo, hi = max(H[0][1], H0[0][1]), min(H[-1][1], H0[-1][1]); T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 10)
    return 1 - float(np.exp(np.nanmean(np.log([cost_at(H, x) / cost_at(H0, x) for x in T])))), (lo, hi)


ALL = np.arange(len(tasks)); HG = curve("global", ALL)
full = evaluate(np.full(len(tasks), 1e9), ALL)
print(f"no cap: mean cost proxy {full[0]:.3g}, resolve {full[1]:.3f}")
rng = np.random.default_rng(0)
for kind in ("pertask", "oracle"):
    g, band = saving(curve(kind, ALL), HG)
    B = []
    for _ in range(100):
        ii = rng.integers(0, len(tasks), len(tasks)); B.append(saving(curve(kind, ii), curve("global", ii))[0])
    print(f"   {kind:<8} caps vs ONE global step cap: cost saved at matched resolve rate {g*100:5.1f}% "
          f"[{np.percentile(B, 2.5)*100:5.1f}, {np.percentile(B, 97.5)*100:5.1f}]  (resolve band {band[0]*100:.1f}-{band[1]*100:.1f}%)")

"""Is AGENTIC per-task cost predictable from the issue text when there are thousands of tasks? (the SWE-rebench test had
111 and gave R2 ~ 0). nebius/SWE-agent-trajectories, SWE-agent + Llama-3.1-70B: target = per-task mean log cost proxy
(cumulative context + 4x output chars) over its ~20 runs. Features: Qwen3-4B Instruct prefill of the issue (rich readout),
ridge. 5-fold CV over tasks, and a LEARNING CURVE (train on n tasks, test on a fixed held-out 20%), n = 100 ... max.
Also the reliability ceiling: split-half correlation of the per-task mean (runs split in two), Spearman-Brown corrected.
"""
import json, numpy as np, collections, sys
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from baseline_cost_heads import rich

R = Path("/mnt/llmd/results/exps/aristides/reason"); MODEL = "swe-agent-llama-70b"
rows = [json.loads(l) for l in open(R / "nebius_swe_agent_traj" / "light.jsonl")]
by = collections.defaultdict(list)
for r in rows:
    if r["model"] == MODEL and r["cost_proxy"] > 0:
        by[r["instance_id"]].append(np.log(r["cost_proxy"]))
z = np.load(R / "nebius_issue_probe" / "instruct.npz", allow_pickle=True); have = {str(p) for p in z["problem_ids"]}
tasks = sorted(t for t in by if t in have and len(by[t]) >= 4)
y = np.array([np.mean(by[t]) for t in tasks]); X = rich(R / "nebius_issue_probe" / "instruct.npz", tasks)
rng = np.random.default_rng(0)
a = np.array([np.mean(v[0::2]) for v in (np.array(by[t])[rng.permutation(len(by[t]))] for t in tasks)])
b = np.array([np.mean(v[1::2]) for v in (np.array(by[t])[rng.permutation(len(by[t]))] for t in tasks)])
rho = np.corrcoef(a, b)[0, 1]; ceiling = 2 * rho / (1 + rho)
print(f"{MODEL}: {len(tasks)} tasks (>= 4 runs); per-task mean log-cost sd {y.std():.2f}; reliability ceiling of the mean "
      f"(split-half, Spearman-Brown) {ceiling:.2f}")
mdl = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)))
pred = np.zeros(len(y))
for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
    pred[te] = mdl().fit(X[tr], y[tr]).predict(X[te])
print(f"5-fold CV R2 of per-task mean log cost from the issue text: {1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum():+.2f}")
perm = rng.permutation(len(y)); test = perm[: len(y) // 5]; pool = perm[len(y) // 5:]
for n in (100, 300, 1000, len(pool)):
    tr = pool[:n]; p = mdl().fit(X[tr], y[tr]).predict(X[test])
    print(f"   learning curve: train on {n:>5} tasks -> held-out R2 {1 - ((y[test] - p) ** 2).sum() / ((y[test] - y[test].mean()) ** 2).sum():+.2f}")

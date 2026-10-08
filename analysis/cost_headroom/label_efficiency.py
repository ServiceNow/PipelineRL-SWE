"""Label efficiency of the cost readout (NEW_PATH 4.A.67), pinned. Scenario: a prefill router already has success readouts (the paper's,
trained with success labels); how many LENGTH labels does it need to price queries? For n training problems (10..all, 5 seeds):
  dedicated     the paper's cost readout (ridge on the 40,960-dim prefill features), refit on the n problems
  from-success  ridge of log length on the success logits (+ squares) of the existing success readouts, fitted on the n problems
Saving vs median pricing at matched accuracy on the test set (median from the full training set, as the deployed rule would use).
Usage: REASON_ROOT=.../reason_pinned python label_efficiency.py LCB|Omni|MMLU-Pro
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / "tmlr_free_analyses.py").read().split("# ---------------------------------------------------------------- 1.")[0]
     .replace('POOL = sys.argv[1]', 'POOL = sys.argv[1]').replace('print(f"===== {POOL}', 'print(f"===== label efficiency {POOL}'))
X = rich(feat, ids); Y = np.log(L); lg = np.log(P / (1 - P)); FS = np.c_[lg, lg ** 2]


def tok_dedicated(train):
    sc = StandardScaler().fit(X[train]); Xs = sc.transform(X); tk = np.zeros((len(ids), M))
    for k in range(M):
        yh = RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(Xs[train], Y[train, k]).predict(Xs)
        e = np.exp(yh) * np.mean(np.exp(Y[train, k] - yh[train])); tk[:, k] = e * L[train, k].mean() / e[train].mean()
    return tk


def tok_from_success(train):
    tk = np.zeros((len(ids), M))
    for k in range(M):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-3, 1e4, 15))).fit(FS[train], Y[train, k]); yh = m.predict(FS)
        e = np.exp(yh) * np.mean(np.exp(Y[train, k] - yh[train])); tk[:, k] = e * L[train, k].mean() / e[train].mean()
    return tk


out = {"pool": POOL, "n_eval": int(len(ev)), "curves": {}}; rng = np.random.default_rng(0)
for n in [10, 20, 50, 100, 200, len(tr)]:
    reps = 1 if n == len(tr) else 5; row = {"dedicated": [], "from_success": []}
    for _ in range(reps):
        sub = tr if n == len(tr) else rng.choice(tr, n, replace=False)
        row["dedicated"].append(saved(costs(tok_dedicated(sub), rates), C_med, paid, ev))
        row["from_success"].append(saved(costs(tok_from_success(sub), rates), C_med, paid, ev))
    out["curves"][str(n)] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in row.items()}
    print(f"  n={n:4d}: dedicated {np.mean(row['dedicated'])*100:+5.1f}% (sd {np.std(row['dedicated'])*100:.1f})   "
          f"from-success {np.mean(row['from_success'])*100:+5.1f}% (sd {np.std(row['from_success'])*100:.1f})   [{reps} seeds]", flush=True)
json.dump(out, open(Path(__file__).parent / f"label_efficiency_{POOL.replace('-', '').lower()}{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1)

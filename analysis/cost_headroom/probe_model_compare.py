"""Which model's prefill best predicts the reasoning routes' output length? Same plain RidgeCV head and post-processing
as baseline_cost_heads.py 'probe'; only the activations change (Qwen3-4B-Instruct = the scout, Qwen3-4B-Thinking,
Qwen3-4B-Base, gpt-oss-20b, gpt-oss-120b). Writes <pool>/cost_preds_probe_<tag>.jsonl.
Usage: python probe_model_compare.py <pool> tag=path.npz [tag=path.npz ...]"""
import json, sys, numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R
from baseline_cost_heads import rich
name = sys.argv[1]; D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
v = t["valid"].astype(bool); ct = t["completion_tokens"].astype(float); pt = t["prompt_tokens"].astype(float)
n = v.sum(2); outm = np.where(n > 0, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan); inp = np.nanmean(np.where(v, pt, np.nan), 2)
sp = json.load(open(D / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]]); te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
for kv in sys.argv[2:]:
    tag, path = kv.split("="); X = rich(path, pids); C = np.zeros((len(pids), len(S))); r2 = []
    for m, s in enumerate(S):
        y = np.log(np.maximum(outm[:, m], 1)); a = np.isfinite(outm[:, m]); trm = tr[a[tr]]
        Xs = StandardScaler().fit(X[trm]).transform(X)
        yh = RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(Xs[trm], y[trm]).predict(Xs)
        o = np.exp(yh) * np.mean(np.exp(y[trm] - yh[trm])); o *= np.nanmean(outm[trm, m]) / o[trm].mean()
        C[:, m] = np.nan_to_num(inp[:, m], nan=np.nanmean(inp[:, m])) * MK[s][0] / 1e6 + o * MK[s][1] / 1e6
        tt = te[a[te]]; r2.append(1 - ((y[tt] - np.log(o[tt])) ** 2).sum() / ((y[tt] - y[tt].mean()) ** 2).sum())
    with open(D / f"cost_preds_probe_{tag}.jsonl", "w") as f:
        for i, p in enumerate(pids):
            f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in C[i]]}) + "\n")
    print(f"{name} probe={tag:<9} test log-output R2: " + "  ".join(f"{s} {x:+.2f}" for s, x in zip(S, r2)), flush=True)

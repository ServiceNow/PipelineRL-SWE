"""ZeroRouter-style per-query cost (arXiv 2601.06220, Eq. 10) as a baseline: a query's complexity score from the shared
difficulty latent is discretised into K bins (train quantiles); each model's predicted output length is its mean TRAIN output
in that bin. Complexity here = our shared difficulty (mean over routes of the prefill success head's logit) -- the same
information ZeroRouter's IRT latent carries, read from our prefill. Writes <pool>/cost_preds_zerorouter_k<K>.jsonl.
Usage: python zerorouter_cost.py <pool> [K]
"""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R

name = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; M = len(S); pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
v = t["valid"].astype(bool); ct = t["completion_tokens"].astype(float); pt = t["prompt_tokens"].astype(float); n = v.sum(2)
outm = np.where(n > 0, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan); inp = np.nanmean(np.where(v, pt, np.nan), 2)
tr = np.array([pi[str(p)] for p in json.load(open(D / "split_manifest.json"))["train_problem_ids"]])
lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D / "content_preds.jsonl")}
P = np.clip(np.array([lp[p] for p in pids]), 1e-4, 1 - 1e-4); s = np.log(P / (1 - P)).mean(1)
edges = np.quantile(s[tr], np.linspace(0, 1, K + 1)[1:-1]); b = np.searchsorted(edges, s)
C = np.zeros((len(pids), M))
for m, r in enumerate(S):
    mean_b = np.array([np.nanmean(outm[tr[b[tr] == k], m]) for k in range(K)])
    mean_b = np.where(np.isfinite(mean_b), mean_b, np.nanmean(outm[tr, m]))
    C[:, m] = (np.nan_to_num(inp[:, m], nan=np.nanmean(inp[tr, m])) * MK[r][0] + mean_b[b] * MK[r][1]) / 1e6
with open(D / f"cost_preds_zerorouter_k{K}.jsonl", "w") as f:
    for i, p in enumerate(pids):
        f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in C[i]]}) + "\n")
print(f"{name}: wrote cost_preds_zerorouter_k{K}.jsonl")

"""Head-to-head: the cost predictors the literature uses, on our pools, through the SAME decision rule and the same
headroom/capture decomposition. Is the capture threshold a property of the DATA or of our 4B probe?

Every method predicts log mean OUTPUT tokens per (problem, route) from information available before generation,
fitted on TRAIN, then the same post-processing (Duan smearing, level matched to the train mean), priced at market
input/output prices (input exact). Writes <pool>/cost_preds_<method>.jsonl for decompose.py.
  mixllm     MixLLM (2502.18482): a text embedding of the query -> per-model MLP + random forest + k-NN, averaged.
             Embedding: jina-embeddings-v2-base-code (a code-aware 137M encoder; MixLLM used a BERT-family one).
  gbm        2607.18253-style gradient boosting on hand-crafted prompt features (length, numbers, constraints,
             examples, topic keywords); sklearn HistGradientBoosting stands in for LightGBM.
  probe      OUR 4B prefill activations with the same simple ridge + post-processing (no calibration shrinkage,
             no target-space selection, no floor) -- the apples-to-apples version of our head.
  ownprefill TRAIL / EGTP-style: each route's OWN prefill activations (TACO only: gpt-oss-20b / 120b readouts).
Usage: python baseline_cost_heads.py <pool> <scout_activations.npz> [--own route=path.npz,...]
"""
import json, re, sys, numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R

KW = ["modulo", "10^9", "graph", "tree", "query", "queries", "maximum", "minimum", "string", "array", "matrix", "probability",
      "permutation", "subsequence", "prime", "shortest", "interactive", "dynamic", "bit", "xor", "sum", "count", "expected",
      "geometry", "polygon", "binary", "sort", "import", "pandas", "numpy", "plot", "regex", "http", "file"]


def text_features(s):
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)[:500]]
    return [len(s), np.log1p(len(s)), s.count("\n"), len(nums), np.log1p(max(nums) if nums else 0), s.count("$"),
            s.count("Example") + s.count("Sample") + s.count("Input:"), s.count("Constraints"), s.count("```"),
            len(re.findall(r"\\le|<=|≤", s))] + [s.lower().count(k) for k in KW]


def embed(texts, cache):
    if cache.exists():
        return np.load(cache)
    import torch
    from transformers import AutoModel
    assert torch.cuda.is_available(), "run on a GPU node (eai job), not the CPU box"
    m = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-code", trust_remote_code=True).eval().cuda()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            out.append(np.asarray(m.encode(texts[i:i + 64], max_length=1024, device="cuda")))
    E = np.concatenate(out); np.save(cache, E); return E


def rich(path, pids):
    z = np.load(path, allow_pickle=True)
    X = np.concatenate([z[k].reshape(len(z[k]), -1) for k in ("mean", "last") if k in z.files], 1)
    aid = {str(p): i for i, p in enumerate(z["problem_ids"])}
    return X[[aid[p] for p in pids]].astype(np.float32)


def main():
    name, act = sys.argv[1], sys.argv[2]
    own = {}
    if "--own" in sys.argv:
        own = dict(kv.split("=") for kv in sys.argv[sys.argv.index("--own") + 1].split(","))
    D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
    S = [str(s) for s in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
    v = t["valid"].astype(bool); ct = t["completion_tokens"].astype(float); pt = t["prompt_tokens"].astype(float)
    n = v.sum(2); outm = np.where(n > 0, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan)
    inp = np.nanmean(np.where(v, pt, np.nan), 2)
    meta = {str(json.loads(l)["problem_id"]): json.loads(l) for l in open(D / "problems.jsonl")}
    texts = [str(meta[p].get("problem_statement", "")) for p in pids]
    sp = json.load(open(D / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]])
    te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
    feats = {"mixllm": embed(texts, D / "emb_jina_code.npy"), "gbm": np.array([text_features(s) for s in texts], float),
             "probe": rich(act, pids)}
    if own:
        feats["ownprefill"] = {s: rich(own[s], pids) if s in own else feats["probe"] for s in S}
    report = {}
    for meth, F in feats.items():
        C = np.zeros((len(pids), len(S))); r2s = []
        for m, s in enumerate(S):
            X = F[s] if isinstance(F, dict) else F
            y = np.log(np.maximum(outm[:, m], 1.0)); a = np.isfinite(outm[:, m]); trm = tr[a[tr]]
            sc = StandardScaler().fit(X[trm]); Xs = sc.transform(X)
            if meth == "mixllm":
                models = [MLPRegressor((128,), alpha=1e-2, max_iter=500, early_stopping=True, random_state=0),
                          RandomForestRegressor(300, min_samples_leaf=3, n_jobs=16, random_state=0),
                          KNeighborsRegressor(15, weights="distance")]
                yh = np.mean([mdl.fit(Xs[trm], y[trm]).predict(Xs) for mdl in models], 0)
            elif meth == "gbm":
                yh = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=10,
                                                   random_state=0).fit(X[trm], y[trm]).predict(X)
            else:
                yh = RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(Xs[trm], y[trm]).predict(Xs)
            smear = np.mean(np.exp(y[trm] - yh[trm])); out_tok = np.exp(yh) * smear
            out_tok *= np.nanmean(outm[trm, m]) / out_tok[trm].mean()
            pin, pout = MK[s][0] / 1e6, MK[s][1] / 1e6
            C[:, m] = np.nan_to_num(inp[:, m], nan=np.nanmean(inp[:, m])) * pin + out_tok * pout
            tt = te[a[te]]; r2s.append(1 - ((y[tt] - np.log(out_tok[tt])) ** 2).sum() / ((y[tt] - y[tt].mean()) ** 2).sum())
        with open(D / f"cost_preds_{meth}.jsonl", "w") as f:
            for i, p in enumerate(pids):
                f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in C[i]]}) + "\n")
        report[meth] = dict(zip(S, map(float, r2s)))
        print(f"{name} {meth:<11} test log-output R2 per route: " + "  ".join(f"{s} {r:+.2f}" for s, r in zip(S, r2s)), flush=True)
    json.dump(report, open(D / "baseline_cost_heads_r2.json", "w"), indent=1)


if __name__ == "__main__":
    main()

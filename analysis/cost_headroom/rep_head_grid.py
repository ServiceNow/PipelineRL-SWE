"""Representation x head grid for cost prediction on the FRESH sets, billed prices (NEW_PATH 4.A.47).
Is our gain the representation (a 4B generative prefill) or the head (ridge), and does a same-size or larger EMBEDDING model match it?
Representations (all frozen; fitted on original TRAIN problems only):
  prefill4b   our Qwen3-4B prefill activations (mean+last, 8 layers; prefill_combined.npz) -> standardize -> PCA 256 (fitted on train)
  qwen3emb8b  Qwen3-Embedding-8B (last-token, normalized)    jina137m  jina-embeddings-v2-base-code    minilm  all-MiniLM-L12-v2
Heads (per route, target = log mean output tokens; then residual smearing + train-mean level match, as all our cost heads):
  ridge    RidgeCV on standardized features (ours)
  knn      CARROT-style cosine uniform kNN, k chosen by 5-fold train CV over powers of two
  mixllm   MixLLM-style mean of MLP(128) + random forest(300) + distance-weighted kNN(15)
Routing: same success predictions (our prefill success readout) for every cell; argmax V p - c; fresh test frontier.
Reported per cell: cost saved vs median-length pricing at matched accuracy, and cost saved by OUR ARCHIVED head (prefill4b+ridge)
relative to the cell, paired problem bootstrap (200). ENCODER ACCOUNTING: every arm pays the 4B prefill pass (the success readout
needs it); cells using another encoder also pay that encoder's pass. Encoder passes are priced per input token at
ENC_RATE (USD/M): prefill4b 0.03 (gpt-oss-20b's billed input rate: an upper bound for a 4B), qwen3emb8b 0.02, jina 0.005, minilm 0.002
(public embedding-API prices, order of magnitude). Results with and without encoder cost.
Usage: python rep_head_grid.py
"""
import glob, json, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import POOLS, read_predictions
from decompose import MK, R, hull, cost_at
from billed import RATE

VALUES = np.geomspace(1e-7, 1, 300)
ENC_RATE = {"prefill4b": 0.03, "qwen3emb8b": 0.02, "jina137m": 0.005, "minilm": 0.002}
rate_of = lambda s: RATE["oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")]


def head_predict(head, X, y, tr):
    if head == "ridge":
        sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
        return RidgeCV(alphas=np.geomspace(1e-1, 1e7, 17)).fit(Xs[tr], y[tr]).predict(Xs)
    if head == "knn":
        best = None
        for k in [2 ** i for i in range(1, 9) if 2 ** i <= (4 * len(tr)) // 5 - 1]:      # every CV fold must support k
            sc = []
            for a, b in KFold(5, shuffle=True, random_state=0).split(tr):
                m = KNeighborsRegressor(k, metric="cosine").fit(X[tr[a]], y[tr[a]]); sc.append(-np.mean((m.predict(X[tr[b]]) - y[tr[b]]) ** 2))
            if best is None or np.mean(sc) > best[0]:
                best = (np.mean(sc), k)
        return KNeighborsRegressor(best[1], metric="cosine").fit(X[tr], y[tr]).predict(X)
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    ms = [MLPRegressor((128,), alpha=1e-2, max_iter=500, early_stopping=True, random_state=0),
          RandomForestRegressor(300, min_samples_leaf=3, n_jobs=16, random_state=0), KNeighborsRegressor(15, weights="distance")]
    return np.mean([m.fit(Xs[tr], y[tr]).predict(Xs) for m in ms], 0)


out = {}
for ds in ("omni500", "mmlupro"):
    label = "MMLU-Pro" if ds == "mmlupro" else "Omni"; name, cost_file = POOLS[label]; old = R / name
    F = R / "expanded_eval_20261001" / ds; t = np.load(F / "tensors.npz", allow_pickle=True)
    ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"])); idx = {p: i for i, p in enumerate(ids)}; M = len(slots)
    sp = json.loads((old / "split_manifest.json").read_text()); tr = np.array([idx[str(p)] for p in sp["train_problem_ids"]])
    n_old = len(np.load(old / "tensors.npz", allow_pickle=True)["problem_ids"]); fresh = np.arange(n_old, len(ids))
    v = t["valid"].astype(bool); cnt = np.maximum(v.sum(2), 1); q = np.where(v, t["final_outcome"], 0).sum(2) / cnt
    L = np.where(v, t["completion_tokens"], 0).sum(2) / cnt; I = np.where(v, t["prompt_tokens"], 0).sum(2) / cnt
    rates = np.array([rate_of(s) for s in slots]); paid = I * rates[:, 0] + L * rates[:, 1]
    for f in glob.glob(f"{R}/math_expand_20261001/{ds}/*_d0.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("finish_reason") != "error" and r.get("usage_cost") is not None and r["problem_id"] in idx and r["route_label"] in slots:
                paid[idx[r["problem_id"]], slots.index(r["route_label"])] = r["usage_cost"]
    fr = fresh[(v[fresh].sum(2) > 0).all(1)]
    P = read_predictions(F / "success_preds.jsonl", ids, "p_successes", M)
    P[:n_old] = read_predictions(old / "content_preds.jsonl", ids[:n_old], "p_successes", M)
    learned = read_predictions(F / "paper_cost_preds.jsonl", ids, "expected_costs", M)
    learned[:n_old] = read_predictions(old / cost_file, ids[:n_old], "expected_costs", M)
    asg = np.array([[MK[s][0] / 1e6, MK[s][1] / 1e6] for s in slots])
    Y = np.log(np.maximum(L, 1))
    reps = {}
    z = np.load(F / "prefill_combined.npz", allow_pickle=True); zid = {str(p): i for i, p in enumerate(z["problem_ids"])}
    X = np.concatenate([z[k].reshape(len(z[k]), -1) for k in ("mean", "last")], 1)[[zid[p] for p in ids]].astype(np.float32); del z
    sc = StandardScaler().fit(X[tr]); reps["prefill4b"] = PCA(min(256, len(tr) - 1), random_state=0).fit(sc.transform(X[tr])).transform(sc.transform(X)); del X
    e = np.load(F / "text_embeddings.npz", allow_pickle=True); eid = {str(p): i for i, p in enumerate(e["problem_ids"])}; sel = [eid[p] for p in ids]
    reps.update(qwen3emb8b=e["qwen3emb8b"][sel], jina137m=e["jina"][sel], minilm=e["minilm"][sel])
    enc_in = I.mean(1)                                                              # encoder reads the prompt once per query
    med = np.array([np.median(t["completion_tokens"][tr, k][v[tr, k]]) for k in range(M)])
    costs = {"median": I * rates[:, 0] + med[None] * rates[:, 1], "ours (archived)": I * rates[:, 0] + np.maximum((learned - I * asg[:, 0]) / asg[:, 1], 1) * rates[:, 1]}
    enc_of = {"median": ["prefill4b"], "ours (archived)": ["prefill4b"]}
    for rep, Xr in reps.items():
        for head in ("ridge", "knn", "mixllm"):
            tok = np.zeros_like(L)
            for k in range(M):
                yh = head_predict(head, Xr, Y[:, k], tr)
                o = np.exp(yh) * np.mean(np.exp(Y[tr, k] - yh[tr])); tok[:, k] = o * L[tr, k].mean() / o[tr].mean()
            costs[f"{rep}+{head}"] = I * rates[:, 0] + tok * rates[:, 1]; enc_of[f"{rep}+{head}"] = sorted({"prefill4b", rep})
            print(f"  {label}: fitted {rep}+{head}", flush=True)
    k_ = np.arange(len(fr))

    def front(c, ii, enc):
        pts = []
        for V in VALUES:
            m = (V * P[ii] - c[ii]).argmax(1); pts.append((paid[ii][np.arange(len(ii)), m].mean() + enc[ii].mean(), q[ii][np.arange(len(ii)), m].mean()))
        return hull(pts)

    def saved(a, b, ii, with_enc):
        ea = sum(ENC_RATE[r] for r in enc_of[a]) * enc_in / 1e6 if with_enc else np.zeros(len(ids))
        eb = sum(ENC_RATE[r] for r in enc_of[b]) * enc_in / 1e6 if with_enc else np.zeros(len(ids))
        Ha, Hb = front(costs[a], ii, ea), front(costs[b], ii, eb); lo, hi = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1])
        T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
        return 1 - float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T]))))
    rng = np.random.default_rng(0); BS = [fr[rng.integers(0, len(fr), len(fr))] for _ in range(200)]
    res = {}
    print(f"\n===== {label} fresh n={len(fr)}, billed prices. Columns: vs median (no enc) | ours(archived) saves vs it (no enc) [CI] | same WITH encoder cost")
    for a in costs:
        if a == "median":
            continue
        vm = saved(a, "median", fr, False); vo = saved("ours (archived)", a, fr, False) if a != "ours (archived)" else 0.0
        vo_b = [saved("ours (archived)", a, b, False) for b in BS] if a != "ours (archived)" else [0.0]
        vm_e = saved(a, "median", fr, True); vo_e = saved("ours (archived)", a, fr, True) if a != "ours (archived)" else 0.0
        res[a] = dict(vs_median=vm, ours_vs=vo, ours_vs_ci=list(np.percentile(vo_b, [2.5, 97.5])), vs_median_enc=vm_e, ours_vs_enc=vo_e)
        print(f"  {a:<20} {vm*100:+6.1f}% | {vo*100:+6.1f}% [{np.percentile(vo_b,2.5)*100:+.1f}, {np.percentile(vo_b,97.5)*100:+.1f}] | enc: {vm_e*100:+6.1f}% / {vo_e*100:+6.1f}%", flush=True)
    out[label] = res
json.dump(out, open(Path(__file__).parent / "rep_head_grid.json", "w"), indent=1, default=float)

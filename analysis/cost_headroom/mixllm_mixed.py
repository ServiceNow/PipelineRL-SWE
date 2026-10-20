"""MixLLM (arXiv 2502.18482) cost estimation on the mixed-traffic pool, with each query's benchmark as its domain (NEW_PATH 4.A.119).
Their Sec. 3.3: a BERT encoder fine-tuned with L = L_intra + L_inter (Eqs. 1-3; domains = InsTag tags clustered by hand into 20
coarse domains), then per-LLM response-length regressors on the embedding, "a combination of MLP, RF, and KNN ... depending on the
LLM" (Sec. 4.1.4); cost = known prompt length x input price + predicted length x output price. Here:
  domains     the nine benchmarks (training labels only: at test time the encoder reads the query text alone)
  encoder     bert-base-uncased, mean-pooled; their loss on L2-normalized embeddings and learnable centers (initialized at the
              domain means of the pre-trained embeddings), dot products / TAU -- the paper gives no normalization, and unnormalized
              dot products make L_inter unbounded below; AdamW lr 2e-5, batch 32, EPOCHS epochs (not given in the paper), the epoch
              (0 = pre-trained) chosen by calibration log-length MSE of the regressors: generous to them
  regressor   per route, MLP / RF / KNN (the hyperparameters of our MixLLM-style arm) chosen by calibration log-length MSE
Arms (same success predictions in every cost-only row; mixed_pool.py's data, splits and test resamples, so its rows are matched):
  ours, median, median_by_ds   as mixed_pool.py
  mixllm_style    our MixLLM-style arm: frozen jina-embeddings-v2-base-code -> average of MLP, RF, kNN (cost_generalization)
  mixllm_bert0    the pipeline below on pre-trained BERT (no domain loss): ablation of their encoder training
  mixllm          their pipeline: fine-tuned BERT + per-route regressor
  mixllm_own      mixllm cost with MixLLM's own success model (per-route random forest on the embedding, Sec. 4.1.4)
Usage: (GPU) REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned_d$s OUT_DIR=... DRAW=s python mixllm_mixed.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
_mp = open(Path(__file__).parent / "mixed_pool.py").read()
exec(_mp[_mp.index("import json"):_mp.index('tk = {"ours"')])                                    # data, splits, success readouts
from transformers import AutoModel, AutoTokenizer
assert torch.cuda.is_available(), "GPU job"
EPOCHS = int(os.environ.get("EPOCHS", "3")); TAU = float(os.environ.get("TAU", "0.1")); torch.manual_seed(0)


def texts_of(pool):
    name, _, ds = POOLS[pool]; F = R / "expanded_eval_20261001" / ds if ds else R / name
    return [str(json.loads(l)["problem_statement"]) for l in open(F / "problems.jsonl")]


texts = [x for p in PL for x in texts_of(p)]; assert len(texts) == N
use = np.unique(np.concatenate([tr, ca, ev])); tok = AutoTokenizer.from_pretrained("bert-base-uncased")
enc = tok([texts[i] for i in use], truncation=True, max_length=512, padding="max_length", return_tensors="pt"); row = {int(i): j for j, i in enumerate(use)}
bert = AutoModel.from_pretrained("bert-base-uncased").cuda()


def embed(ii):
    bert.eval(); out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for j in range(0, len(ii), 128):
            r_ = [row[int(i)] for i in ii[j:j + 128]]; m_ = enc["attention_mask"][r_].cuda()
            h = bert(input_ids=enc["input_ids"][r_].cuda(), attention_mask=m_).last_hidden_state.float()
            out.append(((h * m_[..., None]).sum(1) / m_.sum(1, keepdim=True)).cpu().numpy())
    return np.concatenate(out)


REG = {"mlp": lambda: MLPRegressor(hidden_layer_sizes=(128,), alpha=1e-2, max_iter=500, early_stopping=True, random_state=0),
       "rf": lambda: RandomForestRegressor(300, min_samples_leaf=3, n_jobs=int(os.environ.get("NTHREADS", "8")), random_state=0),
       "knn": lambda: KNeighborsRegressor(15, weights="distance")}


def length_heads(Eall):
    """per route: the regressor with the lowest calibration log-length MSE; smeared + level-matched like every estimator"""
    sc = StandardScaler().fit(Eall[tr]); Z = sc.transform(Eall); Y = np.log(D["L"]); tk_ = np.zeros((N, M)); pick, mse = [], []
    for k in range(M):
        fits = {nm: f().fit(Z[tr], Y[tr, k]) for nm, f in REG.items()}; err = {nm: float(((m_.predict(Z[ca]) - Y[ca, k]) ** 2).mean()) for nm, m_ in fits.items()}
        nm = min(err, key=err.get); m_ = fits[nm]; pick.append(nm); mse.append(err[nm])
        ys, yt = m_.predict(Z[tr]), m_.predict(Z[use]); sm = np.mean(np.exp(Y[tr, k] - ys))
        tk_[use, k] = np.exp(yt) * sm * np.exp(Y[tr, k]).mean() / (np.exp(ys) * sm).mean()
    tk_[np.setdiff1d(np.arange(N), use)] = D["L"][tr].mean(0)
    return tk_, pick, float(np.mean(mse)), Z


E_all = np.zeros((N, bert.config.hidden_size), np.float32); E_all[use] = embed(use)
tk0, pick0, mse0, _ = length_heads(E_all); cands = [(mse0, 0, tk0, pick0, E_all.copy())]
print(f"  epoch 0 (pre-trained BERT): calibration log-length MSE {mse0:.4f}, regressors {pick0}", flush=True)
dom = pool_of; nD = len(PL); E0n = E_all[tr] / np.linalg.norm(E_all[tr], axis=1, keepdims=True)
mu = torch.nn.Parameter(torch.tensor(np.stack([E0n[dom[tr] == j].mean(0) for j in range(nD)]), dtype=torch.float32, device="cuda"))
opt = torch.optim.AdamW(list(bert.parameters()) + [mu], lr=2e-5); rng_e = np.random.default_rng(0)
for ep in range(1, EPOCHS + 1):
    bert.train(); order = rng_e.permutation(tr); tot = 0.
    for j in range(0, len(order), 32):
        s = order[j:j + 32]; r_ = [row[int(i)] for i in s]; m_ = enc["attention_mask"][r_].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = bert(input_ids=enc["input_ids"][r_].cuda(), attention_mask=m_).last_hidden_state.float()
        e = torch.nn.functional.normalize((h * m_[..., None]).sum(1) / m_.sum(1, keepdim=True), dim=1); c = torch.nn.functional.normalize(mu, dim=1)
        l_intra = torch.nn.functional.cross_entropy(e @ c.T / TAU, torch.tensor(dom[s], device="cuda"))           # Eq. 2
        G = c @ c.T / TAU; l_inter = torch.logsumexp(G.masked_fill(torch.eye(nD, dtype=torch.bool, device="cuda"), -1e9), 1).mean()  # Eq. 3
        loss = l_intra + l_inter; opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(s)
    E_all[use] = embed(use); tk_e, pick_e, mse_e, _ = length_heads(E_all); cands.append((mse_e, ep, tk_e, pick_e, E_all.copy()))
    print(f"  epoch {ep}: domain loss {tot / len(tr):.4f}, calibration log-length MSE {mse_e:.4f}, regressors {pick_e}", flush=True)
best = min(cands[1:], key=lambda c: c[0]); _, ep_best, tk_mix, pick_mix, E_best = best
# MixLLM's own success model: per-route random forest on the (chosen) embedding, regressing the success rate
Zb = StandardScaler().fit(E_best[tr]).transform(E_best); P_own = np.full((N, M), .5)
for k in range(M):
    rf = RandomForestRegressor(300, min_samples_leaf=3, n_jobs=int(os.environ.get("NTHREADS", "8")), random_state=0).fit(Zb[tr], D["q"][tr, k])
    P_own[use, k] = rf.predict(Zb[use])
P_own = np.clip(P_own, 1e-4, 1 - 1e-4)
mbd = np.zeros((N, M))
for j in range(len(PL)):
    tj, ij = tr[pool_of[tr] == j], pool_of == j; mbd[ij] = np.array([np.nanmedian(D["CT"][tj, k]) for k in range(M)])[None]
tk = {"ours": fit_predict("ours", view(D, tr), view(D, np.arange(N))), "median": np.repeat(D["med"][None], N, 0), "median_by_ds": mbd,
      "mixllm_style": fit_predict("mixllm", view(D, tr), view(D, np.arange(N))), "mixllm_bert0": tk0, "mixllm": tk_mix, "mixllm_own": tk_mix}
Ps = {a: D["P"] for a in tk}; Ps["mixllm_own"] = P_own
C = {a: cost_of(D, x) for a, x in tk.items()}
BS = [ev] + [np.concatenate([e[rng.integers(0, len(e), len(e))] for e in evs]) for _ in range(NBOOT)]
Hs = [{a: curve(D, Ps[a], C[a], ii) for a in C} for ii in BS]
Y = np.log(D["L"]); r2 = lambda t_: [float(1 - ((np.log(t_[ev, k]) - Y[ev, k]) ** 2).sum() / ((Y[ev, k] - Y[ev, k].mean()) ** 2).sum()) for k in range(M)]
res = {"draw": os.environ.get("DRAW"), "cap": CAP, "n": dict(train=len(tr), calibration=len(ca), test=len(ev)), "epoch": ep_best,
       "regressors": dict(mixllm=pick_mix, mixllm_bert0=pick0), "cal_mse": {str(c[1]): c[0] for c in cands}, "arms": {}}
for a in C:
    res["arms"][a] = {f"vs_{b}": [float(saved(H[a], H[b]) * 100) for H in Hs] for b in ("median", "median_by_ds")} | {"ours_vs_it": [float(saved(H["ours"], H[a]) * 100) for H in Hs]}
    res["arms"][a]["r2"] = r2(tk[a]); res["arms"][a]["auc"] = float(np.mean([auc(D["y0"][ev, k], Ps[a][ev, k]) for k in range(M)]))
pct = lambda v: f"{v[0]:+5.1f} [{np.nanpercentile(v[1:], 2.5):+.1f}, {np.nanpercentile(v[1:], 97.5):+.1f}]"
print(f"  chosen epoch {ep_best}, regressors {pick_mix}", flush=True)
for a, r in res["arms"].items():
    print(f"  {a:<13} vs median {pct(r['vs_median'])}  vs median_by_ds {pct(r['vs_median_by_ds'])}  ours vs it {pct(r['ours_vs_it'])}"
          f"  R2 {np.mean(r['r2']):.2f}  AUC {r['auc']:.3f}", flush=True)
json.dump(res, open(OUT / f"mixllm_mixed{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
print("DONE", flush=True)

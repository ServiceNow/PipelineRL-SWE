"""How much simpler is our readout? Wall-clock FIT time (CPU, same node, NTHREADS threads) and configurations searched, per method, on
one pool's training split (TMLR, C7). The 4B prefill forward pass is shared by every prefill method and excluded; the text encoders
that MixLLM-style (jina-embeddings-v2-base-code, 137M) and ZeroRouter (fine-tuned DistilBERT, 66M) need are an EXTRA pass per query
and are listed, not timed (they need a GPU; embeddings are cached here). Methods:
  ours success        paper recipe (activation_content_preds.py --rich --select-C): per route 9 C values by calibration AUC, binomial
                      refit, Platt on calibration
  ours cost           per route RidgeCV (13 alphas, closed-form GCV) on the same features
  prefill router      prefill_router_repro.py's pipeline: per target (6 layers x 2 poolings x 4 PCA sizes) by 5-fold CV logistic,
                      concatenated features, SharedTrunkNet 10 seeds with early stopping -> top 5 (+ our cost readout, not re-timed)
  zerorouter          stage-1 IRT (D in {1, 5}) + PCA(256) + ridge + bins (K in {5, 10, 20}); excludes its DistilBERT fine-tune
  mixllm              per route MLP + random forest + kNN on the cached embedding (cost only, as in our suite)
  gbm                 per route HistGradientBoosting on prompt features
Usage: REASON_ROOT=.../reason_pinned OUT_DIR=... python fit_timing.py LCB|MMLU-Pro
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
src = open(Path(__file__).parent / "prefill_router_repro.py").read()
head, rest = src.split("cache = {}", 1)
search, rest = ("cache = {}" + rest).split("def train_net", 1)
nets_src = "def train_net" + rest.split("yb_ev = ")[0]
exec(head)
import torch
torch.set_num_threads(int(os.environ.get("NTHREADS", "16")))
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from baseline_cost_heads import text_features
from zr_dimsweep import fit_stage1

T = {}
t0 = time.perf_counter(); exec(search); T["prefill router: layer/pooling/PCA search (5-fold CV)"] = time.perf_counter() - t0
t0 = time.perf_counter(); exec(nets_src); T["prefill router: SharedTrunkNet x10 seeds"] = time.perf_counter() - t0

X = rich(feat, ids); okd_ = (t["final_outcome"] & v); s_ = np.where(v, okd_, 0).sum(2).astype(float); n_ = v.sum(2).astype(float)
y0 = np.take_along_axis(np.asarray(t["final_outcome"]).astype(int), v.argmax(2)[..., None], 2)[..., 0]
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.decomposition import PCA


def _auc(y, sc_):
    r = np.argsort(np.argsort(sc_)) + 1; npos = y.sum(); return (r[y == 1].sum() - npos * (npos + 1) / 2) / max(npos * (len(y) - npos), 1)


t0 = time.perf_counter()
sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X); scale = max(1, X.shape[1] // 2560)
for k in range(M):
    best = (1e-3, -1)
    for cand in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
        a_ = _auc(y0[ca, k], LogisticRegression(max_iter=2000, C=cand / scale).fit(Xs[tr], y0[tr, k]).decision_function(Xs[ca]))
        best = (cand, a_) if a_ > best[1] else best
    w = np.r_[s_[tr, k], n_[tr, k] - s_[tr, k]]; kw = w > 0
    clf = LogisticRegression(max_iter=2000, C=best[0] / scale).fit(np.vstack([Xs[tr], Xs[tr]])[kw], np.r_[np.ones(len(tr)), np.zeros(len(tr))][kw], sample_weight=w[kw])
    LogisticRegression(max_iter=2000, C=1e6).fit(clf.decision_function(Xs[ca])[:, None], y0[ca, k])
T["ours: success readouts (C search + Platt)"] = time.perf_counter() - t0
t0 = time.perf_counter()
for k in range(M):
    RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(Xs[tr], np.log(L[tr, k]))
T["ours: cost readouts (RidgeCV)"] = time.perf_counter() - t0

t0 = time.perf_counter()
Z = PCA(256, random_state=0).fit(X[tr]).transform(X); Z /= Z[tr].std(0) + 1e-6
for D in (1, 5):
    la, bb, th = fit_stage1(s_[tr], n_[tr], D, 0); pr = RidgeCV(alphas=np.geomspace(1, 1e5, 11)).fit(Z[tr], np.c_[la, bb]).predict(Z)
    sv = (np.exp(pr[:, :D]) * pr[:, D:]).sum(1)
    for K in (5, 10, 20):
        e = np.quantile(sv[tr], np.linspace(0, 1, K + 1)[1:-1]); bn = np.searchsorted(e, sv)
        np.array([[L[tr][bn[tr] == j, k].mean() if (bn[tr] == j).any() else L[tr, k].mean() for j in range(K)] for k in range(M)])
T["zerorouter: IRT + PCA + ridge + bins (6 configs; excl. DistilBERT fine-tune)"] = time.perf_counter() - t0

if POOL in SINGLE:
    E = np.load(old / "emb_jina_code.npy")
else:
    e_ = np.load(F / "text_embeddings.npz", allow_pickle=True); eid = {str(p): i for i, p in enumerate(e_["problem_ids"])}; E = e_["jina"][[eid[p] for p in ids]]
Es = StandardScaler().fit(E[tr]).transform(E)
t0 = time.perf_counter()
for k in range(M):
    for m_ in (MLPRegressor(hidden_layer_sizes=(128,), alpha=1e-2, max_iter=500, early_stopping=True, random_state=0),
               RandomForestRegressor(300, min_samples_leaf=3, n_jobs=int(os.environ.get("NTHREADS", "16")), random_state=0), KNeighborsRegressor(15, weights="distance")):
        m_.fit(Es[tr], np.log(L[tr, k]))
T["mixllm: MLP + RF + kNN (cost; excl. embedding pass)"] = time.perf_counter() - t0
texts = [json.loads(l)["problem_statement"] for l in (F / "problems.jsonl").read_text().splitlines()]
TF = np.array([text_features(x) for x in texts], float)
t0 = time.perf_counter()
for k in range(M):
    HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=10, random_state=0).fit(TF[tr], np.log(L[tr, k]))
T["gbm: prompt features (cost)"] = time.perf_counter() - t0

CONFIGS = {"ours: success readouts (C search + Platt)": "9 C values per route, chosen on calibration",
           "ours: cost readouts (RidgeCV)": "13 ridge alphas, closed-form leave-one-out",
           "prefill router: layer/pooling/PCA search (5-fold CV)": f"{len(upper) * 2 * len([d_ for d_ in DGRID if d_ < len(tr)])} configs x 5 folds per route",
           "prefill router: SharedTrunkNet x10 seeds": "10 seeds; width, depth, dropout, lr, patience fixed by us (unstated in the paper)",
           "zerorouter: IRT + PCA + ridge + bins (6 configs; excl. DistilBERT fine-tune)": "D x K = 6; + 40-epoch DistilBERT fine-tune in the original",
           "mixllm: MLP + RF + kNN (cost; excl. embedding pass)": "3 models with fixed settings; + one embedding pass per query",
           "gbm: prompt features (cost)": "fixed settings"}
print(f"===== fit timing {POOL}: train {len(tr)}, calibration {len(ca)}, {M} routes, prefill features {X.shape[1]}, threads {os.environ.get('NTHREADS', '16')}")
for kk, sec in T.items():
    print(f"  {kk:<78} {sec:8.1f} s   {CONFIGS[kk]}", flush=True)
json.dump({"pool": POOL, "n_train": int(len(tr)), "seconds": T, "configs": CONFIGS},
          open(Path(os.environ.get("OUT_DIR", Path(__file__).parent)) / f"fit_timing_{POOL.replace('-', '').lower()}.json", "w"), indent=1)
print("DONE", flush=True)

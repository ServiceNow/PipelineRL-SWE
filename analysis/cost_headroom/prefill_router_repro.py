"""Faithful reimplementation of the prefill router's SUCCESS pipeline (Varshney et al., arXiv 2603.20895, Sec. 3.1) vs our linear success
readouts (NEW_PATH 4.A.68), pinned, same encoder (Qwen3-4B), same splits, same cost readout for both.
Theirs, as described: upper half of the encoder's layers (here the stored layers >= L/2: 18, 22, 25, 29, 32, 36), last-token vs mean
pooling, PCA to d in {50, 100, 200, 300}; (layer, pooling, d) chosen PER TARGET by 5-fold stratified CV with L2 logistic regression on
TRAIN; features = concatenation of each target's PCA features; SharedTrunkNet = joint multi-output MLP (2 x 256 ReLU, dropout 0.1),
BCEWithLogits on each route's success rate over its draws, Adam, early stopping on CALIBRATION (their 15% validation split); 10 seeds,
the 5 best by validation BCE averaged. Unstated details (hidden size, depth, dropout, lr, patience, logistic C) are our choices.
Ours: the paper's per-route L2 logistic on all 8 layers x {mean, last}, C chosen on calibration, Platt-calibrated.
Reported on TEST: mean success AUC and log loss per route; routing with OUR cost readout for both: cost saved by our success vs theirs
at matched accuracy (and each vs median pricing). Usage: REASON_ROOT=.../reason_pinned python prefill_router_repro.py LCB|Omni|MMLU-Pro
"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / "tmlr_free_analyses.py").read().split("# ---------------------------------------------------------------- 1.")[0]
     .replace('print(f"===== {POOL}', 'print(f"===== prefill-router repro {POOL}'))
sp_ = json.loads((old / "split_manifest.json").read_text()); ca = np.array([idx[str(p)] for p in sp_["calibration_problem_ids"]])
z = np.load(feat, allow_pickle=True); zid = {str(p): i for i, p in enumerate(z["problem_ids"])}; rows = [zid[p] for p in ids]
layers = list(map(int, z["layers"])); upper = [j for j, l_ in enumerate(layers) if l_ >= max(layers) / 2]
okd = t["final_outcome"] & v; rate = np.where(v, okd, 0).sum(2) / cnt          # per-route success rate over valid draws (soft target)
ybin = (rate >= .5).astype(int)
DGRID = (50, 100, 200, 300)


def cv_auc(Xp, y):
    if y[tr].min() == y[tr].max():
        return 0.5
    aucs = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xp[tr], y[tr]):
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xp[tr][a], y[tr][a])
        if 0 < y[tr][b].mean() < 1:
            aucs.append(roc_auc_score(y[tr][b], m.decision_function(Xp[tr][b])))
    return float(np.mean(aucs)) if aucs else 0.5


cache = {}
def pca_feats(j, pk, d):
    key = (j, pk, d)
    if key not in cache:
        A = z[pk][rows, j, :].astype(np.float32); sc = StandardScaler().fit(A[tr])
        pc = PCA(d, random_state=0).fit(sc.transform(A[tr])); cache[key] = pc.transform(sc.transform(A))
    return cache[key]


chosen = []
for k in range(M):
    best = None
    for j in upper:
        for pk in ("last", "mean"):
            for d in [d_ for d_ in DGRID if d_ < len(tr)]:          # Omni has 275 training problems
                a = cv_auc(pca_feats(j, pk, d), ybin[:, k])
                if best is None or a > best[0]:
                    best = (a, j, pk, d)
    chosen.append(best); print(f"  {slots[k]}: layer {layers[best[1]]} {best[2]} PCA {best[3]} (CV AUC {best[0]:.3f})", flush=True)
Xc = np.concatenate([pca_feats(j, pk, d) for _, j, pk, d in chosen], 1).astype(np.float32)
sc = StandardScaler().fit(Xc[tr]); Xc = sc.transform(Xc).astype(np.float32)


def train_net(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xc.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(.1), torch.nn.Linear(256, 256),
                              torch.nn.ReLU(), torch.nn.Dropout(.1), torch.nn.Linear(256, M))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4); lossf = torch.nn.BCEWithLogitsLoss()
    Xt, Yt = torch.tensor(Xc[tr]), torch.tensor(rate[tr], dtype=torch.float32); Xv, Yv = torch.tensor(Xc[ca]), torch.tensor(rate[ca], dtype=torch.float32)
    best, state, bad = 1e9, None, 0
    for ep in range(400):
        net.train(); perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 64):
            i = perm[b:b + 64]; opt.zero_grad(); lossf(net(Xt[i]), Yt[i]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(Xv), Yv))
        if vl < best - 1e-5:
            best, state, bad = vl, {kk: vv.clone() for kk, vv in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 20:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        return best, torch.sigmoid(net(torch.tensor(Xc))).numpy()


nets = sorted((train_net(s) for s in range(10)), key=lambda x: x[0])[:5]
P_theirs = np.clip(np.mean([p for _, p in nets], 0), 1e-4, 1 - 1e-4)
yb_ev = (q[ev] > .5).astype(int)


def metrics(Pm):
    aucs = [roc_auc_score(yb_ev[:, k], Pm[ev, k]) for k in range(M) if 0 < yb_ev[:, k].mean() < 1]
    ll = -np.mean(yb_ev * np.log(Pm[ev]) + (1 - yb_ev) * np.log(1 - Pm[ev]))
    return float(np.mean(aucs)), float(ll)


def saved_P(Pa, Ca, Pb, Cb, ii):
    def fr(Pm, C):
        pts = []
        for V in VALUES:
            m = (V * Pm[ii] - C[ii]).argmax(1); kk = np.arange(len(ii)); pts.append((paid[ii][kk, m].mean(), q[ii][kk, m].mean()))
        return hull(pts)
    Ha, Hb = fr(Pa, Ca), fr(Pb, Cb); lo, hi = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1])
    T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    return 1 - float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T]))))


rng = np.random.default_rng(0); BS = [ev[rng.integers(0, len(ev), len(ev))] for _ in range(200)]
res = dict(pool=POOL, chosen=[dict(route=s, layer=layers[c[1]], pooling=c[2], pca=c[3], cv_auc=c[0]) for s, c in zip(slots, chosen)])
for nm, Pm in (("ours_linear", P), ("prefill_router", P_theirs)):
    a, l_ = metrics(Pm); res[nm] = dict(auc=a, logloss=l_, vs_median=saved_P(Pm, C_ours, Pm, C_med, ev))
    print(f"  {nm:<15} test AUC {a:.3f}  log loss {l_:.3f}  | with our cost readout, saves {res[nm]['vs_median']*100:+.1f}% vs median pricing", flush=True)
g = saved_P(P, C_ours, P_theirs, C_ours, ev); b = [saved_P(P, C_ours, P_theirs, C_ours, bb) for bb in BS]
res["ours_vs_theirs"] = [g, *np.percentile(b, [2.5, 97.5]).tolist()]
print(f"  our linear success vs their pipeline (same cost readout): ours saves {g*100:+.1f}% [{np.percentile(b,2.5)*100:+.1f}, {np.percentile(b,97.5)*100:+.1f}]", flush=True)
json.dump(res, open(Path(__file__).parent / f"prefill_router_repro_{POOL.replace('-', '').lower()}{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)

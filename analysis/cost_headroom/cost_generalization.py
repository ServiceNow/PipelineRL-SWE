"""Where a simple cost readout might beat the external cost estimators beyond full-label in-domain fits (TMLR), pinned, billed.
Cost-only comparisons: every arm routes with the SAME success predictions (the target pool's own paper success readouts), so only
the cost estimator differs, as in the label-efficiency figure. Arms (each predicts log output length per route, smearing, then a
per-route level match to its training mean, as fresh_baselines.py / baseline_cost_heads.py):
  ours          StandardScaler + RidgeCV on the frozen Qwen3-4B-Instruct prefill (mean + last token, 8 layers)
  fromsuccess   (our ablation) ridge on the success logits (+ squares) of the success readouts
  median, mean  constant per route (median is the paper rule)
  gbm           prompt-feature gradient boosting (baseline_cost_heads.text_features; latencyrouting-style)
  mixllm        MixLLM-style: jina-embeddings-v2-base-code -> MLP + random forest + kNN, averaged
  knn           CARROT-style kNN on the same embedding, nn chosen on calibration from {1, 3, 5, 10, 20}
  zerorouter    ZeroRouter pricing: D-dim 2PL IRT on the training outcomes, stage 2 = prefill -> PCA -> ridge onto (log a, b),
                per-route mean length in K quantile bins of s = a^T b; (D, K) chosen on calibration from {1, 5} x {5, 10, 20}
Modes:
  labels POOL     label efficiency: every arm refit on n training problems (10, 20, 50, 100, 200, all; 5 seeds, same subsets in
                  every arm); test = the pool's evaluated set (Omni / MMLU-Pro: the 1,000 / 6,500 test problems)
  transfer POOL   new benchmark: every arm trained on OTHER pools' training splits, applied to POOL's test set, (a) same-domain
                  sources (coding: LCB APPS BCB CC; reasoning: Omni500 MMLU-Pro AIME SuperGPQA BBEH) and (b) all other pools;
                  level either zero-shot (sources' level) or matched per route on m = 10 random target training problems
                  (5 seeds, same problems in every arm). Reference rows: in-domain ours and in-domain median (full target train).
                  Omni uses its 500-problem pool (Instruct prefill; test 150), since the 1,000 Omni test set has Thinking features.
                  The baselines' configurations (zerorouter D, K; knn nn) are chosen on the TARGET calibration set: generous to them.
Metric: cost saved at matched accuracy vs the target's in-domain median rule (full training split); ours - arm with a paired
bootstrap over test problems (200 resamples; resample b uses seed b mod 5).
  fulltransfer POOL  a whole router moved: success AND cost estimators trained on other pools, with every choice (C, Platt, nn, D, K)
                  made on the SOURCE calibration splits (no target labels; level-from-10 variant touches only the cost level). Arms:
                  ours (prefill success + prefill cost), prefill_router (prefill success + median cost: the prefill router's rule),
                  mixllm (embedding success, same logistic recipe, + MixLLM cost), knn (CARROT kNN success + cost), zerorouter (its
                  own IRT success + bin pricing). Reference: the target's in-domain median rule with its own readouts.
  pooled POOL     (a) cost readout trained on target train + other pools vs target train only (+ full-router version, success refit
                  pooled vs in-domain, C / Platt on the target calibration split); (b) sources + n target labels vs target-only n
                  (n = 10..200, 5 seeds; level matched to the n problems; unweighted and target rows x10); (c) why transfer works:
                  out-of-pool log-length R2 per route, between-pool share of log-length variance, cross-route agreement of pool means.
  whytransfer POOL  why the dedicated readout transfers and pricing from success does not: in-domain vs transferred R2 and saving
                  for ours / fromsuccess / one-feature mean-logit map / ORACLE difficulty map (true solve rate), raw and level-matched;
                  success-space shift and log length by true-difficulty bin per pool.
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned OUT_DIR=... python cost_generalization.py labels|transfer|fulltransfer|pooled|whytransfer POOL
"""
import json, os, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import read_predictions
from decompose import R, hull, cost_at
from billed import RATE
from baseline_cost_heads import rich, text_features
import torch
from zr_dimsweep import fit_stage1
torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))

REAL = "/mnt/llmd/results/exps/aristides/reason"
POOLS = {   # tensors dir, Qwen3-4B-Instruct-2507 prefill (all: layers 9..36, the same 8); "fresh" = evaluated on the expanded test set
    "LCB": ("pool_v2_tensors_5rung", f"{REAL}/pv2_scout_prefill_1756715297/scout.npz", None),
    "APPS": ("apps_tensors", f"{REAL}/apps_probe/instruct.npz", None),
    "BCB": ("bcb_tensors_5r", f"{REAL}/bcb_scout_prefill.npz", None),
    "CC": ("cc_tensors", f"{REAL}/cc_pool/scout_prefill.npz", None),
    "Omni500": ("omni500_tensors", f"{REAL}/omni500_probe/instruct.npz", None),
    "Omni": ("omni500_tensors", None, "omni500"),          # labels mode only (Thinking prefill, like the paper's Omni readouts)
    "MMLU-Pro": ("mmlupro_tensors", None, "mmlupro"),
    "AIME": ("aime_tensors", f"{REAL}/aime_probe/instruct.npz", None),
    "SuperGPQA": ("supergpqa_tensors", f"{REAL}/supergpqa_probe/instruct.npz", None),
    "BBEH": ("bbeh_tensors", f"{REAL}/bbeh_probe/instruct.npz", None)}
DOMAIN = {"coding": ["LCB", "APPS", "BCB", "CC"], "reasoning": ["Omni500", "MMLU-Pro", "AIME", "SuperGPQA", "BBEH"]}
VALUES = np.geomspace(1e-7, 1, 300); sig = lambda z: 1 / (1 + np.exp(-z))
rate_of = lambda s: RATE["oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")]
ARMS = ["ours", "fromsuccess", "median", "mean", "gbm", "mixllm", "knn", "zerorouter"]
NBOOT = int(os.environ.get("NBOOT", "200")); SEEDS = int(os.environ.get("SEEDS", "5"))


def load(pool):
    name, feat, ds = POOLS[pool]; old = R / name
    F = R / "expanded_eval_20261001" / ds if ds else old; feat = Path(feat) if feat else F / "prefill_combined.npz"
    t = np.load(F / "tensors.npz", allow_pickle=True)
    ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"])); idx = {p: i for i, p in enumerate(ids)}; M = len(slots)
    sp = json.loads((old / "split_manifest.json").read_text())
    tr, ca, te = [np.array([idx[str(p)] for p in sp[k + "_problem_ids"]]) for k in ("train", "calibration", "test")]
    n_old = len(np.load(old / "tensors.npz", allow_pickle=True)["problem_ids"]); ev = np.arange(n_old, len(ids)) if ds else te
    v = t["valid"].astype(bool); n = v.sum(2); cnt = np.maximum(n, 1); succ = np.where(v, t["final_outcome"], 0).sum(2)
    q = succ / cnt; L = np.maximum(np.where(v, t["completion_tokens"], 0).sum(2) / cnt, 1); I = np.where(v, t["prompt_tokens"], 0).sum(2) / cnt
    rates = np.array([rate_of(s) for s in slots]); paid = I * rates[:, 0] + L * rates[:, 1]
    if ds:
        import glob
        for f in glob.glob(f"{R}/math_expand_20261001/{ds}/*_d0.jsonl"):
            for l in open(f):
                r = json.loads(l)
                if r.get("finish_reason") != "error" and r.get("usage_cost") is not None and r["problem_id"] in idx and r["route_label"] in slots:
                    paid[idx[r["problem_id"]], slots.index(r["route_label"])] = r["usage_cost"]
        P = read_predictions(F / "success_preds.jsonl", ids, "p_successes", M); P[:n_old] = read_predictions(old / "content_preds.jsonl", ids[:n_old], "p_successes", M)
        e = np.load(F / "text_embeddings.npz", allow_pickle=True); eid = {str(p): i for i, p in enumerate(e["problem_ids"])}; E = e["jina"][[eid[p] for p in ids]]
    else:
        P = read_predictions(old / "content_preds.jsonl", ids, "p_successes", M); E = np.load(old / "emb_jina_code.npy")
    ok = lambda ii: ii[(v[ii].sum(2) > 0).all(1)]
    texts = [str(json.loads(l)["problem_statement"]) for l in open(F / "problems.jsonl")]
    d = dict(pool=pool, ids=ids, slots=slots, M=M, tr=tr[(n[tr] > 0).all(1)], ca=ok(ca), ev=ok(ev), n=n, succ=succ, q=q, L=L, I=I, rates=rates,
             paid=paid, P=np.clip(P, 1e-4, 1 - 1e-4), X=rich(feat, ids), E=np.asarray(E, np.float32), TF=np.array([text_features(x) for x in texts], float))
    fv = v.argmax(2)[..., None]; d["y0"] = np.take_along_axis(np.asarray(t["final_outcome"]).astype(int), fv, 2)[..., 0]
    CT = np.full(v.shape[:2] + (16,), np.nan); CT[:, :, :v.shape[2]] = np.where(v, t["completion_tokens"], np.nan); d["CT"] = CT
    d["med"] = np.array([np.median(t["completion_tokens"][d["tr"], k][v[d["tr"], k]]) for k in range(M)])   # the paper rule (over draws)
    return d


def curve(d, Pm, C, ii):
    m = (VALUES[:, None, None] * Pm[ii][None] - C[ii][None]).argmax(2); r = np.arange(len(ii))[None]
    return hull(list(zip(d["paid"][ii][r, m].mean(1), d["q"][ii][r, m].mean(1))))


def saved(H, H0):
    lo, hi = max(H[0][1], H0[0][1]), min(H[-1][1], H0[-1][1])
    if hi <= lo:
        return np.nan
    T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    return 1 - float(np.exp(np.nanmean(np.log([cost_at(H, x) / cost_at(H0, x) for x in T]))))


cost_of = lambda d, tk: d["I"] * d["rates"][:, 0] + tk * d["rates"][:, 1]


def fit_predict(arm, S, T, cfg=None):
    """Train on S (dict with X, E, TF, lg, Y [N, M], succ, n) and predict tokens [N_T, M] for T (dict with the same features)."""
    M = S["Y"].shape[1]; out = np.zeros((len(T["X"]), M))
    if arm in ("median", "mean"):
        c = np.nanmedian(S["CT"].transpose(1, 0, 2).reshape(M, -1), 1) if arm == "median" else np.exp(S["Y"]).mean(0)   # median over draws
        return np.repeat(c[None], len(T["X"]), 0)
    if arm == "zerorouter":
        D, K = cfg; Ns = len(S["X"]); pc = min(256, Ns - 1)
        la, bb, th = fit_stage1(S["succ"], S["n"], D, 0)
        pca = PCA(pc, random_state=0).fit(S["X"]); Zs = pca.transform(S["X"]); sd = Zs.std(0) + 1e-6; Zs /= sd; Zt = pca.transform(T["X"]) / sd
        pr = RidgeCV(alphas=np.geomspace(1, 1e5, 11)).fit(Zs, np.c_[la, bb]).predict(Zt)
        s_tr = (np.exp(la) * bb).sum(1); s_te = (np.exp(pr[:, :D]) * pr[:, D:]).sum(1)
        e = np.quantile(s_tr, np.linspace(0, 1, K + 1)[1:-1]); bs, bt = np.searchsorted(e, s_tr), np.searchsorted(e, s_te)
        Ls = np.exp(S["Y"])
        tab = np.array([[Ls[bs == j, k].mean() if (bs == j).any() else Ls[:, k].mean() for j in range(K)] for k in range(M)])
        return tab.T[bt]
    if arm == "ours":
        sc = StandardScaler().fit(S["X"]); Xs, Xt = sc.transform(S["X"]), sc.transform(T["X"])
        fit = lambda k: RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(Xs, S["Y"][:, k], sample_weight=S.get("w"))
    elif arm == "fromsuccess":
        Fs, Ft = np.c_[S["lg"], S["lg"] ** 2], np.c_[T["lg"], T["lg"] ** 2]
        sc = StandardScaler().fit(Fs); Xs, Xt = sc.transform(Fs), sc.transform(Ft)
        fit = lambda k: RidgeCV(alphas=np.geomspace(1e-3, 1e4, 15)).fit(Xs, S["Y"][:, k])
    elif arm in ("oracle_diff", "meanlogit_lin"):
        if arm == "oracle_diff":                  # leaky upper bound for difficulty-only pricing: the problem's true solve rate
            Fs, Ft = (np.c_[v_["q"].mean(1), v_["q"].mean(1) ** 2] for v_ in (S, T))
        else:                                     # one feature, linear: no extrapolating polynomial in success space
            Fs, Ft = S["lg"].mean(1)[:, None], T["lg"].mean(1)[:, None]
        sc = StandardScaler().fit(Fs); Xs, Xt = sc.transform(Fs), sc.transform(Ft)
        fit = lambda k: RidgeCV(alphas=np.geomspace(1e-3, 1e4, 15)).fit(Xs, S["Y"][:, k])
    elif arm == "gbm":
        Xs, Xt = S["TF"], T["TF"]
        fit = lambda k: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=min(10, max(2, len(Xs) // 5)), random_state=0).fit(Xs, S["Y"][:, k])
    elif arm in ("mixllm", "knn"):
        sc = StandardScaler().fit(S["E"]); Xs, Xt = sc.transform(S["E"]), sc.transform(T["E"])
        if arm == "knn":
            fit = lambda k: KNeighborsRegressor(min(cfg, len(Xs)), weights="distance").fit(Xs, S["Y"][:, k])
        else:
            class Avg:
                def __init__(s_, k):
                    s_.ms = [RandomForestRegressor(300, min_samples_leaf=3, n_jobs=int(os.environ.get("NTHREADS", "8")), random_state=0),
                             KNeighborsRegressor(min(15, len(Xs)), weights="distance")]
                    if len(Xs) >= 20:            # early_stopping needs a validation split
                        s_.ms.append(MLPRegressor(hidden_layer_sizes=(128,), alpha=1e-2, max_iter=500, early_stopping=True, random_state=0))
                    for m_ in s_.ms:
                        m_.fit(Xs, S["Y"][:, k])

                def predict(s_, X_):
                    return np.mean([m_.predict(X_) for m_ in s_.ms], 0)
            fit = Avg
    for k in range(M):
        mdl = fit(k); ys, yt = mdl.predict(Xs), mdl.predict(Xt)
        sm = np.mean(np.exp(S["Y"][:, k] - ys)); es, et = np.exp(ys) * sm, np.exp(yt) * sm
        out[:, k] = et * np.exp(S["Y"][:, k]).mean() / es.mean()
    return out


def auc(y, s_):
    r = np.argsort(np.argsort(s_)) + 1; npos = y.sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / max(npos * (len(y) - npos), 1)


def success_linear(key, S, Ca, T):
    """the paper's success readout (activation_content_preds.py --rich --select-C): per route, L2 logistic on standardized features,
    binomial over all draws, C chosen by AUC on the calibration rows (Ca), Platt-calibrated on Ca. Fitted on S only."""
    from sklearn.linear_model import LogisticRegression
    sc = StandardScaler().fit(S[key]); Xs, Xc, Xt = sc.transform(S[key]), sc.transform(Ca[key]), sc.transform(T[key])
    scale = max(1, S[key].shape[1] // 2560); out = np.zeros((len(Xt), S["succ"].shape[1]))
    for k in range(out.shape[1]):
        yb = S["y0"][:, k]; best = (1e-3, -1.0)
        for cand in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
            if 0 < yb.mean() < 1 and 0 < Ca["y0"][:, k].mean() < 1:
                v_ = auc(Ca["y0"][:, k], LogisticRegression(max_iter=2000, C=cand / scale).fit(Xs, yb).decision_function(Xc))
                if v_ > best[1]:
                    best = (cand, v_)
        s_, n_ = S["succ"][:, k], S["n"][:, k]; w = np.r_[s_, n_ - s_]; kw = w > 0
        clf = LogisticRegression(max_iter=2000, C=best[0] / scale).fit(np.vstack([Xs, Xs])[kw], np.r_[np.ones(len(Xs)), np.zeros(len(Xs))][kw], sample_weight=w[kw])
        lo_c, lo_t = clf.decision_function(Xc), clf.decision_function(Xt)
        pl = LogisticRegression(max_iter=2000, C=1e6).fit(lo_c[:, None], Ca["y0"][:, k]) if 0 < Ca["y0"][:, k].mean() < 1 else None
        out[:, k] = pl.predict_proba(lo_t[:, None])[:, 1] if pl is not None else sig(lo_t)
    return np.clip(out, 1e-4, 1 - 1e-4)


def logloss(y, p_):
    p_ = np.clip(p_, 1e-4, 1 - 1e-4); return float(-np.mean(y * np.log(p_) + (1 - y) * np.log(1 - p_)))


def success_knn(S, Ca, T):
    """CARROT-style: success = mean success rate of the nn nearest training problems (embedding space); nn by calibration log loss"""
    sc = StandardScaler().fit(S["E"]); Xs, Xc, Xt = sc.transform(S["E"]), sc.transform(Ca["E"]), sc.transform(T["E"]); M = S["q"].shape[1]
    ll = {nn: logloss(Ca["y0"], KNeighborsRegressor(nn).fit(Xs, S["q"]).predict(Xc)) for nn in (5, 10, 20, 50) if nn <= len(Xs)}
    nn = min(ll, key=ll.get)
    return np.clip(KNeighborsRegressor(nn).fit(Xs, S["q"]).predict(Xt), .02, .98), nn


def zerorouter_full(S, Ca, T):
    """ZeroRouter's own success (IRT latent read from the prefill) and bin pricing, fitted on S; D by calibration log loss, K by
    calibration squared error of log length"""
    pc = min(256, len(S["X"]) - 1); pca = PCA(pc, random_state=0).fit(S["X"]); sd = pca.transform(S["X"]).std(0) + 1e-6
    Zs, Zc, Zt = pca.transform(S["X"]) / sd, pca.transform(Ca["X"]) / sd, pca.transform(T["X"]) / sd; best = None
    for D in (1, 5):
        la, bb, th = fit_stage1(S["succ"], S["n"], D, 0); rg = RidgeCV(alphas=np.geomspace(1, 1e5, 11)).fit(Zs, np.c_[la, bb])
        lat = lambda Z_, rg=rg, D=D: (np.exp(rg.predict(Z_)[:, :D]), rg.predict(Z_)[:, D:])
        Ac, Bc = lat(Zc); Pc = sig((Ac[:, None, :] * (th[None] - Bc[:, None, :])).sum(-1)); l_ = logloss(Ca["y0"], Pc)
        if best is None or l_ < best[0]:
            best = (l_, D, la, bb, th, lat)
    _, D, la, bb, th, lat = best; At, Bt = lat(Zt); Pt = sig((At[:, None, :] * (th[None] - Bt[:, None, :])).sum(-1))
    s_tr = (np.exp(la) * bb).sum(1); Ac, Bc = lat(Zc); s_c, s_t = (Ac * Bc).sum(1), (At * Bt).sum(1); Ls = np.exp(S["Y"]); M = Ls.shape[1]; bk = None
    for K in (5, 10, 20):
        e = np.quantile(s_tr, np.linspace(0, 1, K + 1)[1:-1]); bs_ = np.searchsorted(e, s_tr)
        tab = np.array([[Ls[bs_ == j, k].mean() if (bs_ == j).any() else Ls[:, k].mean() for j in range(K)] for k in range(M)])
        err = float(np.mean((np.log(tab.T[np.searchsorted(e, s_c)]) - Ca["Y"]) ** 2))
        if bk is None or err < bk[0]:
            bk = (err, K, tab.T[np.searchsorted(e, s_t)])
    return np.clip(Pt, 1e-4, 1 - 1e-4), bk[2], (D, bk[1])


def view(d, ii):
    lg = np.log(d["P"] / (1 - d["P"]))
    return dict(X=d["X"][ii], E=d["E"][ii], TF=d["TF"][ii], lg=lg[ii], Y=np.log(d["L"][ii]), succ=d["succ"][ii], n=d["n"][ii], CT=d["CT"][ii], y0=d["y0"][ii], q=d["q"][ii])


def stack(views, weights=None):
    out_ = {k: np.concatenate([w[k] for w in views]) for k in views[0] if k != "w"}
    if weights is not None:                       # per-view row weight (ridge sample_weight)
        out_["w"] = np.concatenate([np.full(len(w["X"]), x, float) for w, x in zip(views, weights)])
    return out_


def r2_routes(d, tk, ii):
    y = np.log(d["L"][ii]); e = np.log(np.maximum(tk[ii], 1))
    return [float(1 - ((y[:, k] - e[:, k]) ** 2).sum() / ((y[:, k] - y[:, k].mean()) ** 2).sum()) for k in range(y.shape[1])]


def choose(d, S, cal_view):
    """calibration choice of zerorouter (D, K) and knn nn, by saving vs median on the target-side calibration problems"""
    H0 = curve(d, d["P"], cost_of(d, np.repeat(d["med"][None], len(d["ids"]), 0)), d["ca"]); best = {}
    full = np.zeros((len(d["ids"]), d["M"]))
    for arm, grid in (("zerorouter", [(D, K) for D in (1, 5) for K in (5, 10, 20)]), ("knn", [1, 3, 5, 10, 20])):
        sc = {}
        for cfg in grid:
            if arm == "knn" and cfg > len(S["X"]):
                continue
            tk = full.copy(); tk[d["ca"]] = fit_predict(arm, S, cal_view, cfg); sc[cfg] = saved(curve(d, d["P"], cost_of(d, tk), d["ca"]), H0)
        best[arm] = max(sc, key=lambda c: -np.inf if np.isnan(sc[c]) else sc[c])
    return best


def evaluate(d, preds, ev, Ps=None):
    """preds: {arm: [list over seeds of tokens [N, M]]}; Ps: optional {arm: success [N, M]} (default: the target's own readouts)
    -> saving vs the target's in-domain median rule, ours - arm with a paired bootstrap"""
    Pa = lambda a: Ps[a] if Ps and a in Ps else d["P"]
    Cm = cost_of(d, np.repeat(d["med"][None], len(d["ids"]), 0)); H0 = curve(d, d["P"], Cm, ev)
    pt = {a: [saved(curve(d, Pa(a), cost_of(d, tk), ev), H0) for tk in r] for a, r in preds.items()}
    rb = np.random.default_rng(0); res = {}
    BS = [ev[rb.integers(0, len(ev), len(ev))] for _ in range(NBOOT)]; H0b = [curve(d, d["P"], Cm, ii) for ii in BS]
    bs = {a: np.array([saved(curve(d, Pa(a), cost_of(d, r[b % len(r)]), ii), H0b[b]) for b, ii in enumerate(BS)]) for a, r in preds.items()}
    for a in preds:
        res[a] = dict(vs_median=float(np.nanmean(pt[a])), sd_seeds=float(np.nanstd(pt[a])))
        if a != "ours" and "ours" in preds:
            dd = (bs["ours"] - bs[a]) * 100
            res[a]["ours_minus_it"] = [float((np.nanmean(pt["ours"]) - np.nanmean(pt[a])) * 100), *map(float, np.nanpercentile(dd, [2.5, 97.5]))]
    return res


def show(tag, res):
    print(f"  {tag}: " + "  ".join(f"{a} {r['vs_median']*100:+5.1f}" for a, r in res.items()), flush=True)
    print("      ours - arm: " + "  ".join(f"{a} {r['ours_minus_it'][0]:+5.1f} [{r['ours_minus_it'][1]:+.1f}, {r['ours_minus_it'][2]:+.1f}]"
                                   for a, r in res.items() if "ours_minus_it" in r), flush=True)


MODE, POOL = sys.argv[1], sys.argv[2]; out = {"mode": MODE, "pool": POOL}
if MODE == "labels":
    d = load(POOL); N = len(d["ids"]); tr = d["tr"]; T = view(d, np.arange(N)); rng = np.random.default_rng(0)
    print(f"===== label efficiency {POOL}: train {len(tr)}, calibration {len(d['ca'])}, test {len(d['ev'])}", flush=True)
    out["by_n"] = {}
    for nn in [10, 20, 50, 100, 200, len(tr)]:
        reps = 1 if nn == len(tr) else SEEDS; subs = [tr if nn == len(tr) else rng.choice(tr, nn, replace=False) for _ in range(reps)]
        preds = {a: [] for a in ARMS}; cfgs = []
        for sub in subs:
            S = view(d, sub); c = choose(d, S, view(d, d["ca"])); cfgs.append(c)
            for a in ARMS:
                preds[a].append(fit_predict(a, S, T, c.get(a)))
        res = evaluate(d, preds, d["ev"]); res["_configs"] = [{k: list(v) if isinstance(v, tuple) else v for k, v in c.items()} for c in cfgs]
        out["by_n"][str(nn)] = res; show(f"n={nn:<4} ({reps} seeds)", {a: r for a, r in res.items() if a != "_configs"})
elif MODE == "whytransfer":
    # Why does the dedicated readout transfer across benchmarks while pricing from success does not, if in-domain it is mostly
    # difficulty? Cost-only (target success readouts). Arms in-domain (target train) and transferred (same-domain sources):
    #   ours, fromsuccess (paper ablation: logits + squares), meanlogit_lin (one linear feature), oracle_diff (TRUE solve rate,
    #   linear + square: upper bound for difficulty-only pricing). Each transferred arm raw and with its level matched to the
    #   target TEST mean per route (leaky; isolates shape from level). Test log-length R2 per route.
    # Diagnostics: per pool, mean predicted success logit and true solve rate (covariate shift in success space), and mean log
    # length per route within true-difficulty bins (does the same difficulty mean the same length on every benchmark?).
    assert POOL != "Omni", "uses Omni500"
    d = load(POOL); N = len(d["ids"]); T = view(d, np.arange(N)); S_in = view(d, d["tr"]); ev = d["ev"]
    dom = "coding" if POOL in DOMAIN["coding"] else "reasoning"; same = [p for p in DOMAIN[dom] if p != POOL]
    print(f"===== why transfer, target {POOL} (test {len(ev)}); sources {same}", flush=True)
    srcv = {}
    for p in same:
        s = load(p); srcv[p] = view(s, s["tr"]); del s
    SRC = stack([srcv[p] for p in same]); ARMSW = ["ours", "fromsuccess", "meanlogit_lin", "oracle_diff"]
    tk_in = {a: fit_predict(a, S_in, T) for a in ARMSW}; tk_tr = {a: fit_predict(a, SRC, T) for a in ARMSW}
    lvl = lambda x: x * (d["L"][ev].mean(0) / x[ev].mean(0))[None]
    preds = {f"{a}|in": [x] for a, x in tk_in.items()} | {f"{a}|xfer": [x] for a, x in tk_tr.items()} | {f"{a}|xfer_lvl": [lvl(x)] for a, x in tk_tr.items()}
    preds = {"ours": preds.pop("ours|in")} | preds
    r = evaluate(d, preds, ev); r2 = {a: r2_routes(d, x[0], ev) for a, x in preds.items()}
    out["saving"] = r; out["r2"] = r2
    for a in preds:
        print(f"  {a:<20} saving {r[a]['vs_median']*100:+6.1f}   R2 per route " + " ".join(f"{x:+.2f}" for x in r2[a]), flush=True)
    lg_ = lambda v_: float(v_["lg"].mean(1).mean()); qd = lambda v_: float(v_["q"].mean(1).mean())
    diag = {p: dict(mean_logit=lg_(v_), sd_logit=float(v_["lg"].mean(1).std()), mean_true_rate=qd(v_)) for p, v_ in [(POOL, S_in)] + list(srcv.items())}
    bins = [0, .2, .4, .6, .8, 1.01]; bl = {}
    for p, v_ in [(POOL, S_in)] + list(srcv.items()):
        dd = v_["q"].mean(1); bl[p] = [[float(v_["Y"][(dd >= lo) & (dd < hi), k].mean()) if ((dd >= lo) & (dd < hi)).sum() >= 5 else None
                                       for lo, hi in zip(bins, bins[1:])] for k in range(d["M"])]
    out["diag"] = diag; out["loglen_by_true_difficulty_bin"] = dict(bins=bins, routes=d["slots"], by_pool=bl)
    print("  success-space shift (mean logit / sd / true solve rate): " + "  ".join(f"{p} {x['mean_logit']:+.2f}/{x['sd_logit']:.2f}/{x['mean_true_rate']:.2f}" for p, x in diag.items()), flush=True)
    for k, rt in enumerate(d["slots"]):
        print(f"  mean log length by true solve-rate bin {bins[:-1]} for {rt}: " + "  ".join(f"{p} " + " ".join("  -  " if x is None else f"{x:5.2f}" for x in bl[p][k]) for p in bl), flush=True)
elif MODE == "pooled":
    # (2) pooled training: target train + other pools vs target train only; (3) transfer + n target labels vs target-only n;
    # (4) why transfer works: out-of-pool length R2 per route, and how much of log length is a pool-level effect
    assert POOL != "Omni", "uses Omni500 (Instruct prefill like every other pool)"
    d = load(POOL); N = len(d["ids"]); T = view(d, np.arange(N)); S_in = view(d, d["tr"]); Ca_t = view(d, d["ca"]); ev = d["ev"]
    dom = "coding" if POOL in DOMAIN["coding"] else "reasoning"
    same, allo = [p for p in DOMAIN[dom] if p != POOL], [p for p in DOMAIN["coding"] + DOMAIN["reasoning"] if p != POOL]
    print(f"===== pooled / transfer+n / why, target {POOL} (train {len(d['tr'])}, test {len(ev)}); same-domain sources {same}", flush=True)
    cache = {}
    for p in allo:
        s = load(p); cache[p] = (view(s, s["tr"]), view(s, s["ca"])); del s
    SRC, SRC_all = stack([cache[p][0] for p in same]), stack([cache[p][0] for p in allo])
    # ---- (2) cost-only: in-domain vs transfer vs pooled (success = the target's own readouts)
    tk = {"ours": fit_predict("ours", S_in, T), "transfer_same": fit_predict("ours", SRC, T),
          "pooled_same": fit_predict("ours", stack([S_in, cache[same[0]][0]] + [cache[p][0] for p in same[1:]]), T),
          "pooled_all": fit_predict("ours", stack([S_in] + [cache[p][0] for p in allo]), T)}
    for a in ("pooled_same", "pooled_all"):          # pooled level: match the target's own training mean (its labels are in the fit)
        tk[a] = tk[a] * (d["L"][d["tr"]].mean(0) / tk[a][d["tr"]].mean(0))[None]
    r2 = {a: r2_routes(d, x, ev) for a, x in tk.items()}
    rA = evaluate(d, {a: [x] for a, x in tk.items()}, ev); out["cost_only"] = dict(saving=rA, r2=r2)
    show("cost-only (target success readouts)", rA)
    print("      test log-length R2 per route " + str(d["slots"]) + ": " + "  ".join(f"{a} " + " ".join(f"{x:.2f}" for x in r) for a, r in r2.items()), flush=True)
    # ---- (2b) full router: success refit in-domain vs pooled (C and Platt on the TARGET calibration split in both)
    P_in = success_linear("X", S_in, Ca_t, T); P_pool = success_linear("X", stack([S_in, SRC]), Ca_t, T)
    aucs = {nm: float(np.mean([auc(d["y0"][ev, k], Pm[ev, k]) for k in range(d["M"])])) for nm, Pm in (("in_domain", P_in), ("pooled", P_pool), ("paper_readouts", d["P"]))}
    rB = evaluate(d, {"ours": [tk["ours"]], "pooled_same": [tk["pooled_same"]]}, ev, {"ours": P_in, "pooled_same": P_pool})
    out["full"] = dict(saving=rB, success_auc=aucs); show("full router, refit in-domain (ours) vs pooled same-domain", rB)
    print(f"      success AUC: " + "  ".join(f"{a} {x:.3f}" for a, x in aucs.items()), flush=True)
    # ---- (2c) which side transfers (BCB / CC anomaly): source-trained success (C / Platt on SOURCE calibration) x in-domain or source cost
    P_src = success_linear("X", SRC, stack([cache[p][1] for p in same]), T)
    rD = evaluate(d, {"ours": [tk["ours"]], "src_success": [tk["ours"]], "src_success_src_cost": [tk["transfer_same"]], "src_cost": [tk["transfer_same"]]},
                  ev, {"src_success": P_src, "src_success_src_cost": P_src})
    bll = lambda Pm: float(-np.mean(d["q"][ev] * np.log(Pm[ev]) + (1 - d["q"][ev]) * np.log(1 - Pm[ev])))      # vs per-problem success rate
    cal = {nm: dict(auc=float(np.mean([auc(d["y0"][ev, k], Pm[ev, k]) for k in range(d["M"])])), logloss=bll(Pm),
                    mean_pred=Pm[ev].mean(0).round(3).tolist()) for nm, Pm in (("paper_readouts", d["P"]), ("in_domain_refit", P_in), ("pooled", P_pool), ("source_only", P_src))}
    cal["true_rate"] = d["q"][ev].mean(0).round(3).tolist()
    out["which_side"] = dict(saving=rD, calibration=cal); show("which side transfers (ours = target success + in-domain cost)", rD)
    print("      success on test: " + "  ".join(f"{nm} AUC {c['auc']:.3f} logloss {c['logloss']:.3f}" for nm, c in cal.items() if nm != "true_rate")
          + f"   mean pred / true rate per route: {cal['source_only']['mean_pred']} / {cal['true_rate']}", flush=True)
    # ---- (3) transfer + n target labels (cost-only, as the label-efficiency figure)
    rng = np.random.default_rng(0); out["plus_n"] = {}
    for nn in [10, 20, 50, 100, 200]:
        if nn >= len(d["tr"]):
            continue
        preds = {"ours": [], "target_only": [], "transfer_level_n": [], "src_plus_n_w10": []}
        for _ in range(SEEDS):
            sub = rng.choice(d["tr"], nn, replace=False); St = view(d, sub); lv = lambda x: x * (d["L"][sub].mean(0) / x[sub].mean(0))[None]
            preds["ours"].append(lv(fit_predict("ours", stack([St, SRC]), T)))                    # sources + n target rows
            preds["src_plus_n_w10"].append(lv(fit_predict("ours", stack([St, SRC], [10.0, 1.0]), T)))
            preds["target_only"].append(fit_predict("ours", St, T))
            preds["transfer_level_n"].append(lv(tk["transfer_same"]))
        rC = evaluate(d, preds, ev); out["plus_n"][str(nn)] = rC; show(f"n={nn:<3} target labels: ours = sources + n", rC)
    # ---- (4) why: share of log-length variance that is between pools (per route), and whether routes agree on which pools are long
    pools_ = [POOL] + allo; Ys = [np.log(d["L"][d["tr"]])] + [cache[p][0]["Y"] for p in allo]
    allY = np.concatenate(Ys); between = [float(np.var(np.concatenate([np.full(len(y_), y_[:, k].mean()) for y_ in Ys])) / np.var(allY[:, k])) for k in range(d["M"])]
    means = np.array([y_.mean(0) for y_ in Ys]); cc = np.corrcoef(means.T)
    out["why"] = dict(pools=pools_, between_pool_share=between, pool_route_mean_loglen=means.tolist(),
                      mean_cross_route_corr_of_pool_means=float(cc[np.triu_indices(d["M"], 1)].mean()))
    print(f"  why: between-pool share of log-length variance per route {[round(x, 2) for x in between]}; mean cross-route correlation of pool means "
          f"{out['why']['mean_cross_route_corr_of_pool_means']:.2f}", flush=True)
elif MODE == "fulltransfer":
    # a whole router moved to a new benchmark: success AND cost estimators trained on other pools; every choice (C, nn, D, K, Platt)
    # made on the SOURCE pools' calibration splits, so no target label is used except in the level-from-10 variant (cost level only)
    assert POOL != "Omni", "transfer uses Omni500 (Instruct prefill like every other pool)"
    d = load(POOL); N = len(d["ids"]); T = view(d, np.arange(N)); rng = np.random.default_rng(0)
    dom = "coding" if POOL in DOMAIN["coding"] else "reasoning"
    srcs = {"same_domain": [p for p in DOMAIN[dom] if p != POOL], "all_other": [p for p in DOMAIN["coding"] + DOMAIN["reasoning"] if p != POOL]}
    print(f"===== full transfer to {POOL} (test {len(d['ev'])}); same-domain sources {srcs['same_domain']}", flush=True)
    S_in = view(d, d["tr"]); r_in = evaluate(d, {"ours": [fit_predict("ours", S_in, T)]}, d["ev"]); out["in_domain_ours"] = r_in["ours"]
    print(f"  in-domain ours (target readouts) {r_in['ours']['vs_median']*100:+.1f}", flush=True)
    cache = {}; lvl_sets = [rng.choice(d["tr"], 10, replace=False) for _ in range(SEEDS)]
    for sname, plist in srcs.items():
        for p in plist:
            if p not in cache:
                s = load(p); cache[p] = (view(s, s["tr"]), view(s, s["ca"])); del s
        S, Ca = stack([cache[p][0] for p in plist]), stack([cache[p][1] for p in plist])
        P_lin = success_linear("X", S, Ca, T); P_emb = success_linear("E", S, Ca, T); P_knn, nn = success_knn(S, Ca, T)
        P_zr, tok_zr, cfg_zr = zerorouter_full(S, Ca, T)
        Ps = {"ours": P_lin, "prefill_router": P_lin, "mixllm": P_emb, "knn": P_knn, "zerorouter": P_zr}
        zs = {"ours": fit_predict("ours", S, T), "prefill_router": fit_predict("median", S, T), "mixllm": fit_predict("mixllm", S, T),
              "knn": fit_predict("knn", S, T, nn), "zerorouter": tok_zr}
        aucs = {a: float(np.mean([auc(d["y0"][d["ev"], k], Ps[a][d["ev"], k]) for k in range(d["M"])])) for a in ("ours", "mixllm", "knn", "zerorouter")}
        aucs["target_readouts"] = float(np.mean([auc(d["y0"][d["ev"], k], d["P"][d["ev"], k]) for k in range(d["M"])]))
        r0 = evaluate(d, {a: [x] for a, x in zs.items()}, d["ev"], Ps)
        lm = {a: [x * (d["L"][ls].mean(0) / x[ls].mean(0))[None] for ls in lvl_sets] for a, x in zs.items()}
        r1 = evaluate(d, lm, d["ev"], Ps)
        out[sname] = dict(sources=plist, n_source=int(len(S["X"])), knn_nn=nn, zerorouter_DK=list(cfg_zr), test_auc=aucs, zero_shot=r0, level10=r1)
        print(f"  {sname}: test success AUC " + " ".join(f"{a} {x:.3f}" for a, x in aucs.items()), flush=True)
        show(f"{sname} ({len(S['X'])} source problems) zero-shot", r0); show(f"{sname} level from 10 target problems", r1)
elif MODE == "transfer":
    assert POOL != "Omni", "transfer uses Omni500 (Instruct prefill like every other pool)"
    d = load(POOL); N = len(d["ids"]); T = view(d, np.arange(N)); rng = np.random.default_rng(0)
    dom = "coding" if POOL in DOMAIN["coding"] else "reasoning"
    srcs = {"same_domain": [p for p in DOMAIN[dom] if p != POOL], "all_other": [p for p in DOMAIN["coding"] + DOMAIN["reasoning"] if p != POOL]}
    print(f"===== transfer to {POOL} (test {len(d['ev'])}); same-domain sources {srcs['same_domain']}", flush=True)
    cache = {}
    # in-domain reference (full target train)
    S_in = view(d, d["tr"]); c_in = choose(d, S_in, view(d, d["ca"]))
    ind = {a: [fit_predict(a, S_in, T, c_in.get(a))] for a in ARMS}; r_in = evaluate(d, ind, d["ev"]); out["in_domain"] = r_in
    show("in-domain (full target train)", r_in)
    lvl_sets = [rng.choice(d["tr"], 10, replace=False) for _ in range(SEEDS)]
    for sname, plist in srcs.items():
        for p in plist:
            if p not in cache:
                s = load(p); cache[p] = view(s, s["tr"]); del s
        S = stack([cache[p] for p in plist]); c = choose(d, S, view(d, d["ca"]))     # baselines' configs chosen on the TARGET calibration set (generous)
        zs = {a: fit_predict(a, S, T, c.get(a)) for a in ARMS}
        r0 = evaluate(d, {a: [x] for a, x in zs.items()}, d["ev"])
        lm = {a: [x * (d["L"][ls].mean(0) / x[ls].mean(0))[None] for ls in lvl_sets] for a, x in zs.items()}   # level from 10 target problems
        r1 = evaluate(d, lm, d["ev"])
        out[sname] = dict(sources=plist, n_source=int(len(S["X"])), configs={k: list(v) if isinstance(v, tuple) else v for k, v in c.items()},
                          zero_shot=r0, level10=r1)
        show(f"{sname} ({len(S['X'])} source problems) zero-shot", r0); show(f"{sname} level from 10 target problems", r1)
out_dir = Path(os.environ.get("OUT_DIR", Path(__file__).parent))
json.dump(out, open(out_dir / f"cost_generalization_{MODE}_{POOL.replace('-', '').lower()}{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
print("DONE", flush=True)

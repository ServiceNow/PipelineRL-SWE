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
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned OUT_DIR=... python cost_generalization.py labels|transfer POOL
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
        fit = lambda k: RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(Xs, S["Y"][:, k])
    elif arm == "fromsuccess":
        Fs, Ft = np.c_[S["lg"], S["lg"] ** 2], np.c_[T["lg"], T["lg"] ** 2]
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


def view(d, ii):
    lg = np.log(d["P"] / (1 - d["P"]))
    return dict(X=d["X"][ii], E=d["E"][ii], TF=d["TF"][ii], lg=lg[ii], Y=np.log(d["L"][ii]), succ=d["succ"][ii], n=d["n"][ii], CT=d["CT"][ii])


def stack(views):
    return {k: np.concatenate([w[k] for w in views]) for k in views[0]}


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


def evaluate(d, preds, ev):
    """preds: {arm: [list over seeds of tokens [N, M]]} -> saving vs median, ours - arm with a paired bootstrap"""
    Cm = cost_of(d, np.repeat(d["med"][None], len(d["ids"]), 0)); H0 = curve(d, d["P"], Cm, ev)
    pt = {a: [saved(curve(d, d["P"], cost_of(d, tk), ev), H0) for tk in r] for a, r in preds.items()}
    rb = np.random.default_rng(0); res = {}
    BS = [ev[rb.integers(0, len(ev), len(ev))] for _ in range(NBOOT)]; H0b = [curve(d, d["P"], Cm, ii) for ii in BS]
    bs = {a: np.array([saved(curve(d, d["P"], cost_of(d, r[b % len(r)]), ii), H0b[b]) for b, ii in enumerate(BS)]) for a, r in preds.items()}
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

"""Every cost baseline on the FRESH problems at BILLED prices (NEW_PATH 4.A.43). One protocol for all arms:
heads fitted on the ORIGINAL train split only (frozen), realized cost = billed usage_cost, predictions priced at effective billed
$/M; routing argmax_m V p_m - c_m; test frontier over V on the fresh problems.
Arms (cost estimator; success = our prefill success readout unless stated):
  ours          frozen 4B-prefill cost readout (archived heads, verified reconstruction)
  median        training-median output per route (the paper rule)
  mean          training-mean output per route
  fromsuccess   per-route ridge of log length on our success logits (+ squares), fitted on train
  zr-bins       ZeroRouter-style: K=10 quantile bins of our shared difficulty (mean success logit), per-route mean train length
  gbm           prompt-feature gradient boosting (baseline_cost_heads.text_features)
  zerorouter    full ZeroRouter reproduction: stage-1 2PL IRT on the 5 routes' train outcomes, stage 2 = 4B prefill -> PCA -> ridge,
                THEIR success model and bin-lookup pricing; (D, K) chosen on original CALIBRATION from {1,5} x {5,10,20}
Reported: (a) cost saved vs median at matched accuracy over the shared band; (b) DIRECT ours vs each arm: cost saved by ours at
matched accuracy over the band both reach; paired problem bootstrap (300) for both. Unweighted.
Usage: python fresh_baselines.py [--pool LCB]
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import POOLS, read_predictions
from decompose import MK, R, hull, cost_at
from baseline_cost_heads import text_features
from billed import RATE                                       # same fit as provider_routing.billed_rates, without running that analysis
from zr_dimsweep import fit_stage1

VALUES = np.geomspace(1e-7, 1, 300); sig = lambda z: 1 / (1 + np.exp(-z))
rate_of = lambda s: RATE["oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")]
# --pool LCB: the same arms on an ORIGINAL pool's own split (train fits, calibration picks ZeroRouter's config, TEST is evaluated);
# realized cost = realized tokens x billed rates (no per-call usage_cost there). NEW_PATH 4.A.59.
ORIG = {"LCB": ("pool_v2_tensors_5rung", "cost_preds_probe.jsonl", "/mnt/llmd/results/exps/aristides/reason/pv2_scout_prefill_1756715297/scout.npz"),
        "BCB": ("bcb_tensors_5r", "cost_preds_probe.jsonl", "/mnt/llmd/results/exps/aristides/reason/bcb_scout_prefill.npz"),
        "APPS": ("apps_tensors", "cost_preds_probe.jsonl", "/mnt/llmd/results/exps/aristides/reason/apps_probe/instruct.npz"),
        "AIME": ("aime_tensors", "cost_preds_probe_instruct.jsonl", "/mnt/llmd/results/exps/aristides/reason/aime_probe/instruct.npz"),
        "CC": ("cc_tensors", "cost_preds_probe.jsonl", "/mnt/llmd/results/exps/aristides/reason/cc_pool/scout_prefill.npz"),
        "SuperGPQA": ("supergpqa_tensors", "cost_preds_probe_instruct.jsonl", "/mnt/llmd/results/exps/aristides/reason/supergpqa_probe/instruct.npz"),
        "BBEH": ("bbeh_tensors", "cost_preds_probe_instruct.jsonl", "/mnt/llmd/results/exps/aristides/reason/bbeh_probe/instruct.npz")}
POOL = sys.argv[sys.argv.index("--pool") + 1] if "--pool" in sys.argv else None
out = {}
for ds in ([POOL] if POOL else ("omni500", "mmlupro")):
    if POOL:
        label = POOL; name, cost_file, feat = ORIG[POOL]; old = F = R / name; t = np.load(F / "tensors.npz", allow_pickle=True)
    else:
        label = "MMLU-Pro" if ds == "mmlupro" else "Omni"; name, cost_file = POOLS[label]; old = R / name
        F = R / "expanded_eval_20261001" / ds; t = np.load(F / "tensors.npz", allow_pickle=True)
    ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"])); idx = {p: i for i, p in enumerate(ids)}; M = len(slots)
    sp = json.loads((old / "split_manifest.json").read_text()); tr, ca = [np.array([idx[str(p)] for p in sp[k + "_problem_ids"]]) for k in ("train", "calibration")]
    n_old = len(np.load(old / "tensors.npz", allow_pickle=True)["problem_ids"]); fresh = np.arange(n_old, len(ids))
    if POOL:
        fresh = np.array([idx[str(p)] for p in sp["test_problem_ids"]])      # "fresh" = the evaluated problems
    v = t["valid"].astype(bool); cnt = np.maximum(v.sum(2), 1); succ = np.where(v, t["final_outcome"], 0).sum(2); n = v.sum(2)
    q = succ / cnt; L = np.where(v, t["completion_tokens"], 0).sum(2) / cnt; I = np.where(v, t["prompt_tokens"], 0).sum(2) / cnt
    rates = np.array([rate_of(s) for s in slots]); paid = I * rates[:, 0] + L * rates[:, 1]
    for f in ([] if POOL else glob.glob(f"{R}/math_expand_20261001/{ds}/*_d0.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("finish_reason") != "error" and r.get("usage_cost") is not None and r["problem_id"] in idx and r["route_label"] in slots:
                paid[idx[r["problem_id"]], slots.index(r["route_label"])] = r["usage_cost"]
    fresh = fresh[(v[fresh].sum(2) > 0).all(1)]
    if POOL:
        learned = read_predictions(old / cost_file, ids, "expected_costs", M); P = read_predictions(old / "content_preds.jsonl", ids, "p_successes", M)
    else:
        learned = read_predictions(F / "paper_cost_preds.jsonl", ids, "expected_costs", M)
        learned[:n_old] = read_predictions(old / cost_file, ids[:n_old], "expected_costs", M)
        P = read_predictions(F / "success_preds.jsonl", ids, "p_successes", M)
        P[:n_old] = read_predictions(old / "content_preds.jsonl", ids[:n_old], "p_successes", M)
    P = np.clip(P, 1e-4, 1 - 1e-4); Lg = np.log(P / (1 - P))
    asg = np.array([[MK[s][0] / 1e6, MK[s][1] / 1e6] for s in slots])
    tok = {"ours": np.maximum((learned - I * asg[:, 0]) / asg[:, 1], 1)}
    tok["median"] = np.repeat([[np.median(t["completion_tokens"][tr, k][v[tr, k]]) for k in range(M)]], len(ids), 0)
    tok["mean"] = np.repeat([L[tr].mean(0)], len(ids), 0)
    Y = np.log(np.maximum(L, 1)); FS = np.c_[Lg, Lg ** 2]
    tok["fromsuccess"] = np.stack([np.exp(make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-3, 1e4, 15))).fit(FS[tr], Y[tr, k]).predict(FS))
                                   * np.mean(np.exp(Y[tr, k] - make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-3, 1e4, 15))).fit(FS[tr], Y[tr, k]).predict(FS[tr]))) for k in range(M)], 1)
    dbar = Lg.mean(1); e = np.quantile(dbar[tr], np.linspace(0, 1, 11)[1:-1]); b = np.searchsorted(e, dbar)
    tok["zr-bins"] = np.stack([np.array([L[tr][b[tr] == j, k].mean() if (b[tr] == j).any() else L[tr, k].mean() for j in range(10)])[b] for k in range(M)], 1)
    texts = [json.loads(l)["problem_statement"] for l in (F / "problems.jsonl").read_text().splitlines()]
    TF = np.array([text_features(x) for x in texts], float)
    tok["gbm"] = np.stack([np.exp(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=10, random_state=0).fit(TF[tr], Y[tr, k]).predict(TF)) for k in range(M)], 1)
    if (old / "cost_preds_mixllm.jsonl").exists() and POOL:          # MixLLM-style embedding ensemble (baseline_cost_heads.py --only mixllm)
        tok["mixllm"] = np.maximum((read_predictions(old / "cost_preds_mixllm.jsonl", ids, "expected_costs", M) - I * asg[:, 0]) / asg[:, 1], 1)
    for k_ in ("fromsuccess", "zr-bins", "gbm"):                      # level-match each route to its train mean (as baseline_cost_heads)
        tok[k_] = tok[k_] * (L[tr].mean(0) / tok[k_][tr].mean(0))
    cost = {a: I * rates[:, 0] + x * rates[:, 1] for a, x in tok.items()}
    if POOL:
        cost["oracle"] = paid.copy()                                  # headroom: each problem priced at its realized cost
    succp = {a: P for a in cost}
    # ---- full ZeroRouter, configuration chosen on calibration
    z = np.load(feat if POOL else F / "prefill_combined.npz", allow_pickle=True); zid = {str(p): i for i, p in enumerate(z["problem_ids"])}
    X = np.concatenate([z[k_].reshape(len(z[k_]), -1) for k_ in ("mean", "last")], 1)[[zid[p] for p in ids]].astype(np.float32); del z
    Z = PCA(256, random_state=0).fit(X[tr]).transform(X); Z /= Z[tr].std(0) + 1e-6; del X

    def front(p_, c_, ii):
        pts = []
        for V in VALUES:
            m = (V * p_[ii] - c_[ii]).argmax(1); pts.append((paid[ii][np.arange(len(ii)), m].mean(), q[ii][np.arange(len(ii)), m].mean()))
        return hull(pts)

    def saved(pa, ca_, pb, cb, ii):                                   # cost saved by arm a vs arm b at matched accuracy
        Ha, Hb = front(pa, ca_, ii), front(pb, cb, ii); lo, hi = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1])
        T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
        return 1 - float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T]))))
    best = None
    for D in (1, 5):
        la, bb, th = fit_stage1(succ[tr], n[tr], D, 0); pr = RidgeCV(alphas=np.geomspace(1, 1e5, 11)).fit(Z[tr], np.c_[la, bb]).predict(Z)
        A = np.exp(pr[:, :D]); B = pr[:, D:]; A[tr] = np.exp(la); B[tr] = bb; Pz = sig((A[:, None, :] * (th[None] - B[:, None, :])).sum(-1))
        s = (A * B).sum(1)
        for K in (5, 10, 20):
            ee = np.quantile(s[tr], np.linspace(0, 1, K + 1)[1:-1]); bn = np.searchsorted(ee, s)
            tab = np.array([[L[tr][bn[tr] == j, k].mean() if (bn[tr] == j).any() else L[tr, k].mean() for j in range(K)] for k in range(M)])
            Cz = I * rates[:, 0] + tab.T[bn] * rates[:, 1]; g = saved(Pz, Cz, P, cost["median"], ca)
            if best is None or g > best[0]:
                best = (g, D, K, Pz, Cz)
    succp["zerorouter"], cost["zerorouter"] = best[3], best[4]
    print(f"\n===== {label} {'test' if POOL else 'fresh'} n={len(fresh)} (billed prices); ZeroRouter config chosen on calibration: D={best[1]}, K={best[2]}")
    rng = np.random.default_rng(0); BS = [fresh[rng.integers(0, len(fresh), len(fresh))] for _ in range(300)]
    res = {}
    for a in cost:
        g_med = saved(succp[a], cost[a], P, cost["median"], fresh) if a != "median" else 0.0
        g_dir = saved(P, cost["ours"], succp[a], cost[a], fresh) if a != "ours" else 0.0
        bm = [saved(succp[a], cost[a], P, cost["median"], bb_) for bb_ in BS] if a != "median" else [0.0]
        bd = [saved(P, cost["ours"], succp[a], cost[a], bb_) for bb_ in BS] if a != "ours" else [0.0]
        res[a] = dict(vs_median=g_med, vs_median_ci=list(np.percentile(bm, [2.5, 97.5])), ours_vs_it=g_dir, ours_vs_it_ci=list(np.percentile(bd, [2.5, 97.5])))
        print(f"  {a:<12} saved vs median {g_med*100:+6.1f}% [{np.percentile(bm,2.5)*100:+.1f}, {np.percentile(bm,97.5)*100:+.1f}]"
              f"   | ours saves vs it {g_dir*100:+6.1f}% [{np.percentile(bd,2.5)*100:+.1f}, {np.percentile(bd,97.5)*100:+.1f}]", flush=True)
    out[label] = res
json.dump(out, open(Path(__file__).parent / f"{'pool_baselines_' + POOL if POOL else 'fresh_baselines'}{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)

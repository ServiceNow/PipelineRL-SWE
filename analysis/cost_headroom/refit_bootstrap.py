"""Refit bootstrap (NEW_PATH 4.A.49): the paper's intervals resample TEST problems only, conditional on the fitted readouts. Here each
replicate also resamples the ORIGINAL TRAIN problems with replacement and refits every cost estimator on them:
  ours                 archived recipe (StandardScaler fitted on the resample -> RidgeCV over 1e1..1e7 with efficient LOO ->
                       Duan smearing -> train-mean level match), on the full 4B prefill features (mean+last, 8 layers); solved
                       in the dual (kernel form), exact for ridge; the alpha choice uses centred LOO
  median               training-median output per route
  cost-from-success    per-route ridge of log length on success logits (+ squares)
  difficulty bins      10 quantile bins of mean success logit, per-route mean train length
Success predictions are held fixed (shared by every arm). Billed prices, realized = usage_cost (as fresh_baselines.py).
Replicate 0 refits on the original train set and is checked against the archived predictions.
Reported per arm: cost saved by ours vs it on fresh problems at matched accuracy; CI from test-only resampling (B) and from
train+test resampling (B).
Usage: python refit_bootstrap.py [B]
"""
import glob, json, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import POOLS, read_predictions
from decompose import MK, R, hull, cost_at
from baseline_cost_heads import rich
from billed import RATE

B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
VALUES = np.geomspace(1e-7, 1, 300); ALPHAS = np.geomspace(1e1, 1e7, 13)
rate_of = lambda s: RATE["oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")]


def kernel_ridge(Wb, Wt, Y):
    """Ridge with intercept on rows Wb (already centred by the scaler), targets Y [nb, M]; alpha per column by LOO.
    Returns predictions on Wb and Wt."""
    K = Wb @ Wb.T; lam, U = np.linalg.eigh(K.astype(np.float64)); lam = np.maximum(lam, 0); Kt = (Wt @ Wb.T).astype(np.float64)
    mu = Y.mean(0); Yc = Y - mu; UtY = U.T @ Yc; pb, pt = np.zeros_like(Y), np.zeros((len(Wt), Y.shape[1]))
    for k in range(Y.shape[1]):
        best = None
        for a in ALPHAS:
            d = 1 / (lam + a); c = U @ (d * UtY[:, k])                  # LOO with an unpenalised intercept: r_i / (1 - H_ii)
            h = 1 / len(Y) + (U ** 2) @ (lam * d); loo = np.mean((a * c / (1 - h)) ** 2)
            if best is None or loo < best[0]:
                best = (loo, c)
        c = best[1]; pb[:, k] = mu[k] + K @ c; pt[:, k] = mu[k] + Kt @ c
    return pb, pt


out = {}
for ds in ("omni500", "mmlupro"):
    label = "MMLU-Pro" if ds == "mmlupro" else "Omni"; name, cost_file = POOLS[label]; old = R / name
    F = R / "expanded_eval_20261001" / ds; t = np.load(F / "tensors.npz", allow_pickle=True)
    ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"])); idx = {p: i for i, p in enumerate(ids)}; M = len(slots)
    sp = json.loads((old / "split_manifest.json").read_text()); tr = np.array([idx[str(p)] for p in sp["train_problem_ids"]])
    n_old = len(np.load(old / "tensors.npz", allow_pickle=True)["problem_ids"]); fresh = np.arange(n_old, len(ids))
    v = t["valid"].astype(bool); cnt = np.maximum(v.sum(2), 1)
    q = np.where(v, t["final_outcome"], 0).sum(2) / cnt; L = np.where(v, t["completion_tokens"], 0).sum(2) / cnt; I = np.where(v, t["prompt_tokens"], 0).sum(2) / cnt
    rates = np.array([rate_of(s) for s in slots]); paid = I * rates[:, 0] + L * rates[:, 1]
    for f in glob.glob(f"{R}/math_expand_20261001/{ds}/*_d0.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("finish_reason") != "error" and r.get("usage_cost") is not None and r["problem_id"] in idx and r["route_label"] in slots:
                paid[idx[r["problem_id"]], slots.index(r["route_label"])] = r["usage_cost"]
    fr = fresh[(v[fresh].sum(2) > 0).all(1)]
    assert (v[tr].sum(2) > 0).all(), "train problem without a draw"
    P = np.clip(read_predictions(F / "success_preds.jsonl", ids, "p_successes", M), 1e-4, 1 - 1e-4)
    P[:n_old] = np.clip(read_predictions(old / "content_preds.jsonl", ids[:n_old], "p_successes", M), 1e-4, 1 - 1e-4); Lg = np.log(P / (1 - P))
    learned = read_predictions(F / "paper_cost_preds.jsonl", ids, "expected_costs", M); learned[:n_old] = read_predictions(old / cost_file, ids[:n_old], "expected_costs", M)
    asg = np.array([[MK[s][0] / 1e6, MK[s][1] / 1e6] for s in slots]); tok_arch = np.maximum((learned - I * asg[:, 0]) / asg[:, 1], 1)
    X = rich(F / "prefill_combined.npz", ids); Y = np.log(np.maximum(L, 1)); FS = np.c_[Lg, Lg ** 2]
    med_draws = [t["completion_tokens"][:, k] for k in range(M)]

    def fit_all(trb):
        mu = X[trb].mean(0); sd = X[trb].std(0); sd[sd == 0] = 1
        Wb = (X[trb] - mu) / sd; Wt = (X[fr] - mu) / sd
        pb, pt = kernel_ridge(Wb, Wt, Y[trb]); tok = {}
        sm = np.mean(np.exp(Y[trb] - pb), 0); lvl = L[trb].mean(0) / (np.exp(pb) * sm).mean(0)
        o = np.zeros((len(ids), M)); o[fr] = np.exp(pt) * sm * lvl; tok["ours"] = o
        tok["median"] = np.repeat([[np.median(med_draws[k][trb][v[trb, k]]) for k in range(M)]], len(ids), 0)
        fs = []
        for k in range(M):
            m_ = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-3, 1e4, 15))).fit(FS[trb], Y[trb, k]); yh = m_.predict(FS)
            fs.append(np.exp(yh) * np.mean(np.exp(Y[trb, k] - yh[trb])))
        tok["cost-from-success"] = np.stack(fs, 1)
        dbar = Lg.mean(1); e = np.quantile(dbar[trb], np.linspace(0, 1, 11)[1:-1]); b = np.searchsorted(e, dbar)
        tok["difficulty bins"] = np.stack([np.array([L[trb][b[trb] == j, k].mean() if (b[trb] == j).any() else L[trb, k].mean() for j in range(10)])[b] for k in range(M)], 1)
        for a in ("cost-from-success", "difficulty bins"):
            tok[a] = tok[a] * (L[trb].mean(0) / tok[a][trb].mean(0))
        return {a: I * rates[:, 0] + x * rates[:, 1] for a, x in tok.items()}

    def front(c, ii):
        pts = []
        for V in VALUES:
            m = (V * P[ii] - c[ii]).argmax(1); pts.append((paid[ii][np.arange(len(ii)), m].mean(), q[ii][np.arange(len(ii)), m].mean()))
        return hull(pts)

    def saved(ca, cb, ii):
        Ha, Hb = front(ca, ii), front(cb, ii); lo, hi = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1])
        T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
        return 1 - float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T]))))
    t0 = time.time(); C0 = fit_all(tr)
    rel = np.abs((C0["ours"][fr] - I[fr] * rates[:, 0]) / rates[:, 1] / tok_arch[fr] - 1)
    print(f"\n===== {label}: n_train {len(tr)}, fresh {len(fr)}; replicate-0 refit vs archived token forecasts: median rel err {np.median(rel):.2e}, max {rel.max():.2e} ({time.time()-t0:.0f}s)", flush=True)
    arms = [a for a in C0 if a != "ours"]
    point = {a: saved(C0["ours"], C0[a], fr) for a in arms}
    rng = np.random.default_rng(0); test_only = {a: [] for a in arms}; refit = {a: [] for a in arms}
    for bi in range(B):
        te_b = fr[rng.integers(0, len(fr), len(fr))]; trb = tr[rng.integers(0, len(tr), len(tr))]
        Cb = fit_all(trb)
        for a in arms:
            test_only[a].append(saved(C0["ours"], C0[a], te_b)); refit[a].append(saved(Cb["ours"], Cb[a], te_b))
        if bi % 20 == 19:
            print(f"  replicate {bi+1}/{B} ({time.time()-t0:.0f}s)", flush=True)
    res = {}
    for a in arms:
        to, rf = np.percentile(test_only[a], [2.5, 97.5]), np.percentile(refit[a], [2.5, 97.5])
        res[a] = dict(point=point[a], test_only_ci=list(to), refit_ci=list(rf), refit_median=float(np.median(refit[a])))
        print(f"  ours saves vs {a:<18} {point[a]*100:+6.1f}%   test-only [{to[0]*100:+.1f}, {to[1]*100:+.1f}]   train+test refit [{rf[0]*100:+.1f}, {rf[1]*100:+.1f}]", flush=True)
    out[label] = res
    del X
json.dump(out, open(Path(__file__).parent / f"refit_bootstrap{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)

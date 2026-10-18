"""Per-query hard budgets on the 10 training sets, against calling deepseek-v4-flash alone (NEW_PATH 4.A.114). budget_cap.py measured
hard caps on the manifest split only (old 150 / 300-problem Omni-MATH / MMLU-Pro tests) and against median / constant rules only;
here the full test sets, the training-set protocol, single routes as references, and a DIRECT readout of whether a call fits.
HARD cap X (USD per query, fixed in advance): the call gets max_tokens = (X - input cost) / output rate (effective billed rates); a
draw longer than that is cut off and fails. Realized per (problem, route): mean over valid draws of [solved AND length <= cap].
Caps: deepseek-v4-flash's per-draw cost at the 50/70/80/90/95th percentiles over the non-test problems (train + calibration, the
same in every training set; no test labels).
Arms (every router uses the same refitted success readouts P; each sends a query to the route with the highest score):
  ds_alone       deepseek-v4-flash on every query
  single_train   the single route most accurate under the cap on the training problems
  single_test    the single route most accurate under the cap on the TEST problems (an optimistic reference)
  median         P x [route's training median length <= cap]   (the paper rule)
  constant       P x P(L <= cap) from the route's training length distribution (same for every query)
  derived        P x P(L <= cap) from our cost readout: the ridge prediction plus the empirical training residuals
  direct_fit     P x a logistic readout of [length <= cap] on the prefill (per route and cap; C by calibration AUC, Platt on calibration)
  direct_joint   a logistic readout of [solved AND length <= cap] on the prefill (same recipe; no separate P)
  oracle         P x the problem's own fraction of draws within the cap (leaky ceiling for any length predictor)
Nothing predicted to fit: the route with the most room relative to its training median (as budget_cap.py).
Reported per cap: test accuracy and mean spend (input + min(length, cap) x rate) per arm, and paired differences (direct_fit,
direct_joint, derived, oracle) - (ds_alone, median, single_train) with NBOOT test resamples (rng 0, the same in every set).
Usage: REASON_ROOT=.../reason_pinned OUT_DIR=... python budget_draws.py LCB|Omni|MMLU-Pro S0 S1
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
POOL_, S0, S1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); sys.argv = [sys.argv[0], "budget", POOL_]
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
from sklearn.linear_model import LogisticRegression
OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent)); QS = (50, 70, 80, 90, 95)
name_, _, dsx = POOLS[POOL_]; t = np.load((R / "expanded_eval_20261001" / dsx if dsx else R / name_) / "tensors.npz", allow_pickle=True)
v = t["valid"].astype(bool); ok = (t["final_outcome"].astype(bool) & v); ct = t["completion_tokens"].astype(float); pt = t["prompt_tokens"].astype(float)


def readout(S, Ca, T, cnt, n_, y0):
    """success_linear's recipe for an arbitrary binary label (per-draw counts cnt / n_, first-draw label y0) with a guard for routes
    whose training or calibration label is (nearly) constant: then the smoothed training rate."""
    sc = StandardScaler().fit(S["X"]); Xs, Xc, Xt = sc.transform(S["X"]), sc.transform(Ca["X"]), sc.transform(T["X"])
    scale = max(1, S["X"].shape[1] // 2560); out = np.zeros((len(Xt), cnt.shape[1])); ytr, yca = y0[S["ii"]], y0[Ca["ii"]]
    for k in range(out.shape[1]):
        s_, m_ = cnt[S["ii"], k], n_[S["ii"], k]; rate = (s_.sum() + 1) / (m_.sum() + 2)
        if min(s_.sum(), (m_ - s_).sum()) < 5 or min(yca[:, k].sum(), (1 - yca[:, k]).sum()) < 2:
            out[:, k] = rate; continue
        best = (1e-3, -1.0)
        for cand in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
            if 0 < ytr[:, k].mean() < 1:
                v_ = auc(yca[:, k], LogisticRegression(max_iter=2000, C=cand / scale).fit(Xs, ytr[:, k]).decision_function(Xc))
                if v_ > best[1]:
                    best = (cand, v_)
        w = np.r_[s_, m_ - s_]; kw = w > 0
        clf = LogisticRegression(max_iter=2000, C=best[0] / scale).fit(np.vstack([Xs, Xs])[kw], np.r_[np.ones(len(Xs)), np.zeros(len(Xs))][kw], sample_weight=w[kw])
        pl = LogisticRegression(max_iter=2000, C=1e6).fit(clf.decision_function(Xc)[:, None], yca[:, k])
        out[:, k] = pl.predict_proba(clf.decision_function(Xt)[:, None])[:, 1]
    return np.clip(out, 1e-4, 1 - 1e-4)


res = {"pool": POOL_, "qs": QS, "seeds": {}}
for s in range(S0, S1):
    os.environ["DRAW"] = str(s); d = load(POOL_); ids = d["ids"]; N = len(ids); tr, ca, ev = d["tr"], d["ca"], d["ev"]; M = d["M"]
    assert list(map(str, t["problem_ids"])) == ids; ds = d["slots"].index("dsv4f"); rin, rout = d["rates"][:, 0], d["rates"][:, 1]
    cost_d = np.where(v, pt * rin[None, :, None] + ct * rout[None, :, None], np.nan)                       # uncapped cost per draw
    nontest = np.r_[tr, ca]                                                                          # the same problems in every set
    caps = np.nanpercentile(cost_d[nontest, ds][v[nontest, ds]], QS); res["caps"] = caps.tolist()
    S, Ca, T = (dict(view(d, ii), ii=ii) for ii in (tr, ca, np.arange(N)))
    tok = fit_predict("ours", S, T); MU = np.log(np.maximum(tok, 1)); LOGL = np.log(np.maximum(ct, 1))
    RES = [np.sort((LOGL[tr, k] - MU[tr, k][:, None])[v[tr, k]]) for k in range(M)]
    trL = [np.sort(ct[tr, k][v[tr, k]]) for k in range(M)]; med = np.array([np.median(x) for x in trL])
    inp = np.nan_to_num(np.nanmean(np.where(v, pt, np.nan), 2)); n_ = v.sum(2); first = v.argmax(2)[..., None]
    rb = np.random.default_rng(0); BS = [np.arange(len(ev))] + [rb.integers(0, len(ev), len(ev)) for _ in range(NBOOT)]
    r_ = {"n_train": len(tr), "n_test": len(ev), "caps": {}}
    for q_, X in zip(QS, caps):
        cap = np.maximum((X - inp * rin) / rout, 0.0)                                                   # [N, M] tokens
        fitd = v & (ct <= cap[..., None]); jd = fitd & ok
        acc_m = np.where(n_ > 0, jd.sum(2) / np.maximum(n_, 1), 0.0)                                      # realized, per (problem, route)
        spend_m = np.nanmean(np.where(v, pt * rin[None, :, None] + np.minimum(ct, cap[..., None]) * rout[None, :, None], np.nan), 2)
        y_fit = np.take_along_axis(fitd, first, 2)[..., 0].astype(int); y_joint = np.take_along_axis(jd, first, 2)[..., 0].astype(int)
        Pfit = readout(S, Ca, T, fitd.sum(2), n_, y_fit); Pjoint = readout(S, Ca, T, jd.sum(2), n_, y_joint)
        derived = np.stack([np.searchsorted(RES[k], np.log(np.maximum(cap[:, k], 1e-9)) - MU[:, k], side="right") / len(RES[k]) for k in range(M)], 1)
        constant = np.stack([np.searchsorted(trL[k], cap[:, k], side="right") / len(trL[k]) for k in range(M)], 1)
        scores = {"median": d["P"] * (med[None] <= cap), "constant": d["P"] * constant, "derived": d["P"] * derived,
                  "direct_fit": d["P"] * Pfit, "direct_joint": Pjoint, "oracle": d["P"] * np.where(n_ > 0, fitd.sum(2) / np.maximum(n_, 1), 0)}
        tr_acc = acc_m[np.r_[tr, ca]].mean(0); choice = {"ds_alone": np.full(N, ds), "single_train": np.full(N, int(tr_acc.argmax())),
                                                          "single_test": np.full(N, int(acc_m[ev].mean(0).argmax()))}
        fb = (cap / np.maximum(med[None], 1)).argmax(1)
        for a, sc in scores.items():
            sc = np.where(cap > 0, sc, -1); m = sc.argmax(1); choice[a] = np.where(sc.max(1) <= 1e-6, fb, m)
        A = {a: acc_m[ev, m[ev]] for a, m in choice.items()}; C_ = {a: spend_m[ev, m[ev]] for a, m in choice.items()}
        row = {a: dict(acc=float(A[a].mean() * 100), spend_per_1k=float(np.nanmean(C_[a]) * 1e3), off_ds=float((choice[a][ev] != ds).mean() * 100))
               for a in A}
        row["single_train_route"] = d["slots"][int(tr_acc.argmax())]; row["single_test_route"] = d["slots"][int(acc_m[ev].mean(0).argmax())]
        row["auc_fit_ds"] = auc(y_fit[ev, ds], Pfit[ev, ds]) if 0 < y_fit[ev, ds].mean() < 1 else float("nan")
        row["auc_derived_ds"] = auc(y_fit[ev, ds], derived[ev, ds]) if 0 < y_fit[ev, ds].mean() < 1 else float("nan")
        row["diffs"] = {}
        for a in ("direct_fit", "direct_joint", "derived", "oracle"):
            for b in ("ds_alone", "median", "single_train", "derived"):
                if a == b:
                    continue
                bs = np.array([(A[a][ii] - A[b][ii]).mean() for ii in BS]) * 100
                row["diffs"][f"{a}-{b}"] = dict(point=float(bs[0]), bs=bs[1:].tolist())
        r_["caps"][str(q_)] = row
        g = lambda a: f"{a} {row[a]['acc']:5.1f}"
        print(f"  set {s} cap p{q_} (${X * 1e3:.2f}/1k-equiv): " + " ".join(g(a) for a in A) + f" | AUC fit ds: direct {row['auc_fit_ds']:.3f} "
              f"derived {row['auc_derived_ds']:.3f} | direct_fit-ds {row['diffs']['direct_fit-ds_alone']['point']:+.1f} "
              f"direct_joint-ds {row['diffs']['direct_joint-ds_alone']['point']:+.1f} oracle-ds {row['diffs']['oracle-ds_alone']['point']:+.1f}", flush=True)
    res["seeds"][s] = r_
    json.dump(res, open(OUT / f"budget_draws_{POOL_.lower().replace('-', '')}_{S0}_{S1}{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
print("DONE", flush=True)

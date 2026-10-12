"""TMLR robustness additions (NEW_PATH 4.A.86), pinned, billed. Reuses cost_generalization.py (load / view / fit_predict / choose /
evaluate). Every arm routes with the SAME success predictions unless stated; cost estimators as in cost_generalization.py.
Modes:
  deploy POOL     deployable policies: each arm picks, on the CALIBRATION split, the cheapest V reaching each of 10 accuracy targets
                  (spanning the band every arm reaches on calibration); that V is applied once to the test set. Per arm and target:
                  test accuracy, test $ per query, raw saving vs the median rule's deployed policy at the same target, saving vs the
                  median rule's TEST frontier at the arm's achieved accuracy ("matched"), loss vs the arm's own test frontier, and
                  the miss |test accuracy - target|, and the saving vs the median rule's DEPLOYED curve (its 10 deployed points,
                  interpolated) at the arm's achieved accuracy (headline). ours - arm with a paired bootstrap over test problems (V fixed).
  resplit POOL S0 S1   the whole pipeline (success readouts refit with the paper recipe, every cost arm refit, configurations
                  chosen on calibration) on random re-splits of the pool's problems with the original train / calibration sizes,
                  seeds S0..S1-1. Per split: saving vs median for every arm and the oracle (headroom), ours - arm with a paired
                  bootstrap, success AUC. LCB's paper split is temporal; its re-splits are random.
  pricegap POOL   price-gap sweep: route prices r_m(alpha) = g * (r_m / g)^alpha (g = geometric mean over routes; input and output
                  alike), alpha 0 (flat) .. 5; costs recomputed from tokens (no billed overrides); savings vs median of ours,
                  pricing from success, and the oracle (headroom), with the max / min output-price ratio at each alpha.
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned OUT_DIR=... python tmlr_strengthen.py deploy|resplit|pricegap POOL [S0 S1]
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
_src = open(Path(__file__).parent / "cost_generalization.py").read()
exec(_src.split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent)); TAG = os.environ.get("RESULT_TAG", "")
MODE_, POOL_ = sys.argv[1], sys.argv[2]; res = {"mode": MODE_, "pool": POOL_}
ARMS8 = ["ours", "median", "mean", "fromsuccess", "gbm", "mixllm", "knn", "zerorouter"]


def choices(d, Pm, Cm, ii):
    return (VALUES[:, None, None] * Pm[ii][None] - Cm[ii][None]).argmax(2)            # [nV, n]


if MODE_ == "deploy":
    d = load(POOL_); N = len(d["ids"]); tr, ca, ev = d["tr"], d["ca"], d["ev"]
    S, T = view(d, tr), view(d, np.arange(N)); cfg = choose(d, S, view(d, ca))
    C = {a: cost_of(d, fit_predict(a, S, T, cfg.get(a))) if a != "median" else cost_of(d, np.repeat(d["med"][None], N, 0)) for a in ARMS8}
    q, paid = d["q"], d["paid"]
    sel = {}
    for a in ARMS8:
        mc, mt = choices(d, d["P"], C[a], ca), choices(d, d["P"], C[a], ev)
        rc, rt = np.arange(len(ca))[None], np.arange(len(ev))[None]
        sel[a] = dict(acc_c=q[ca][rc, mc].mean(1), cost_c=paid[ca][rc, mc].mean(1), Q=q[ev][rt, mt], Pd=paid[ev][rt, mt])   # Q, Pd: [nV, n_ev]
    lo = max(s["acc_c"].min() for s in sel.values()); hi = min(s["acc_c"].max() for s in sel.values())
    targets = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 10)
    pickV = {a: [None if not (s["acc_c"] >= t - 1e-12).any() else int(np.flatnonzero(s["acc_c"] >= t - 1e-12)[np.argmin(s["cost_c"][s["acc_c"] >= t - 1e-12])]) for t in targets] for a, s in sel.items()}

    def stats(ii):
        out_ = {}
        Hm = hull(list(zip(sel["median"]["Pd"][:, ii].mean(1), sel["median"]["Q"][:, ii].mean(1))))
        dm = np.array([[sel["median"]["Q"][v_, ii].mean(), sel["median"]["Pd"][v_, ii].mean()] for v_ in pickV["median"] if v_ is not None])
        dm = dm[np.argsort(dm[:, 0])]; dm[:, 1] = np.maximum.accumulate(dm[:, 1])          # the median rule's DEPLOYED curve
        dep = lambda acc: np.interp(acc, dm[:, 0], dm[:, 1]) if dm[0, 0] <= acc <= dm[-1, 0] else np.nan
        for a, s in sel.items():
            Ha = hull(list(zip(s["Pd"][:, ii].mean(1), s["Q"][:, ii].mean(1)))); rows = []
            for j, t in enumerate(targets):
                v_, vm = pickV[a][j], pickV["median"][j]
                if v_ is None or vm is None:
                    rows.append([np.nan] * 8); continue
                acc, cst = s["Q"][v_, ii].mean(), s["Pd"][v_, ii].mean(); accm, cstm = sel["median"]["Q"][vm, ii].mean(), sel["median"]["Pd"][vm, ii].mean()
                cm, ca_ = cost_at(Hm, acc), cost_at(Ha, acc)
                rows.append([acc, cst, 1 - cst / cstm, acc - accm, (1 - cst / cm) if np.isfinite(cm) else np.nan,
                             (cst / ca_ - 1) if np.isfinite(ca_) else np.nan, abs(acc - t), 1 - cst / dep(acc)])
            out_[a] = np.array(rows)
        return out_
    pt = stats(np.arange(len(ev))); rb = np.random.default_rng(0)
    BS = [stats(rb.integers(0, len(ev), len(ev))) for _ in range(NBOOT)]
    print(f"===== deployable policies {POOL_}: calibration {len(ca)} -> test {len(ev)}; targets {targets.round(3).tolist()}", flush=True)
    names = ["test_acc", "usd_per_query", "raw_vs_median", "acc_minus_median", "vs_median_test_frontier", "own_frontier_loss", "miss",
             "vs_median_deployed"]
    for a in ARMS8:
        m = np.nanmean(pt[a], 0); r_ = dict(zip(names, map(float, m))); r_["per_target"] = pt[a].tolist()
        if a != "ours":
            for col, key in ((4, "ours_minus_it_frontier"), (7, "ours_minus_it_deployed")):
                dd = np.array([np.nanmean(b["ours"][:, col] - b[a][:, col]) for b in BS]) * 100
                r_[key] = [float(np.nanmean(pt["ours"][:, col] - pt[a][:, col]) * 100), *map(float, np.nanpercentile(dd, [2.5, 97.5]))]
        res[a] = r_
        e_ = r_.get("ours_minus_it_deployed"); extra = f"  ours - it (deployed) {e_[0]:+5.1f} [{e_[1]:+.1f}, {e_[2]:+.1f}]" if e_ else ""
        print(f"  {a:12s} vs median deployed {m[7]*100:+5.1f}%  vs median test frontier {m[4]*100:+5.1f}%  raw {m[2]*100:+5.1f}%  acc - median {m[3]*100:+5.2f} pt  own-frontier loss {m[5]*100:+4.1f}%"
              f"  miss {m[6]*100:.2f} pt  ${m[1]*1e3:.3f}/1k queries{extra}", flush=True)
    res["targets"] = targets.tolist(); res["cfg"] = {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}

elif MODE_ == "resplit":
    s0, s1 = int(sys.argv[3]), int(sys.argv[4]); d = load(POOL_); N = len(d["ids"])
    U = np.concatenate([d["tr"], d["ca"], d["ev"]]); ntr, nca = len(d["tr"]), len(d["ca"]); res["splits"] = {}
    print(f"===== re-splits {POOL_}: {len(U)} problems, train {ntr}, calibration {nca}, test {len(U) - ntr - nca}; seeds {s0}..{s1 - 1}", flush=True)
    for s in range(s0, s1):
        perm = np.random.default_rng(1000 + s).permutation(U); tr2, ca2, ev2 = perm[:ntr], perm[ntr:ntr + nca], perm[ntr + nca:]
        d2 = dict(d); d2.update(tr=tr2, ca=ca2, ev=ev2)
        d2["med"] = np.array([np.nanmedian(d["CT"][tr2, k]) for k in range(d["M"])])
        A = dict(X=d["X"], y0=d["y0"], succ=d["succ"], n=d["n"])
        sub = lambda ii: {k: v[ii] for k, v in A.items()}
        d2["P"] = success_linear("X", sub(tr2), sub(ca2), sub(np.arange(N)))
        aucs = [float(auc(d["y0"][ev2, k], d2["P"][ev2, k])) for k in range(d["M"])]
        S, T = view(d2, tr2), view(d2, np.arange(N)); cfg = choose(d2, S, view(d2, ca2))
        preds = {a: [fit_predict(a, S, T, cfg.get(a))] for a in ARMS8}; preds["oracle"] = [d["L"]]
        r_ = evaluate(d2, preds, ev2); r_["auc"] = aucs; r_["cfg"] = {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}
        res["splits"][s] = r_
        print(f"  seed {s}: AUC {np.round(aucs, 3).tolist()}  " + "  ".join(f"{a} {r_[a]['vs_median']*100:+5.1f}" for a in preds), flush=True)
        print("      ours - arm: " + "  ".join(f"{a} {r_[a]['ours_minus_it'][0]:+5.1f} [{r_[a]['ours_minus_it'][1]:+.1f}, {r_[a]['ours_minus_it'][2]:+.1f}]"
                                       for a in preds if a != "ours"), flush=True)
        json.dump(res, open(OUT / f"tmlr_strengthen_resplit_{POOL_.lower().replace('-', '')}_{s0}_{s1}{TAG}.json", "w"), indent=1, default=float)

elif MODE_ == "pricegap":
    d = load(POOL_); N = len(d["ids"]); S, T = view(d, d["tr"]), view(d, np.arange(N))
    tk = {"ours": fit_predict("ours", S, T), "fromsuccess": fit_predict("fromsuccess", S, T), "oracle": d["L"]}
    g = np.exp(np.log(d["rates"]).mean(0)); res["rows"] = []
    print(f"===== price-gap sweep {POOL_} (test {len(d['ev'])}); routes {d['slots']}; actual output $/M {np.round(d['rates'][:, 1] * 1e6, 3).tolist()}", flush=True)
    for alpha in (0, .25, .5, .75, 1, 1.5, 2, 3, 4, 5):
        rates = g * (d["rates"] / g) ** alpha; d2 = dict(d); d2["rates"] = rates; d2["paid"] = d["I"] * rates[:, 0] + d["L"] * rates[:, 1]
        r_ = evaluate(d2, {a: [v] for a, v in tk.items()}, d["ev"]); ratio = float(rates[:, 1].max() / rates[:, 1].min())
        r_["alpha"] = alpha; r_["out_price_ratio"] = ratio; res["rows"].append(r_)
        print(f"  alpha {alpha:4.2f}  out-price ratio {ratio:7.1f}x   headroom {r_['oracle']['vs_median']*100:+5.1f}  ours {r_['ours']['vs_median']*100:+5.1f}"
              f"  fromsuccess {r_['fromsuccess']['vs_median']*100:+5.1f}   ours - fromsuccess {r_['fromsuccess']['ours_minus_it'][0]:+5.1f} "
              f"[{r_['fromsuccess']['ours_minus_it'][1]:+.1f}, {r_['fromsuccess']['ours_minus_it'][2]:+.1f}]", flush=True)

if MODE_ != "resplit":
    json.dump(res, open(OUT / f"tmlr_strengthen_{MODE_}_{POOL_.lower().replace('-', '')}{TAG}.json", "w"), indent=1, default=float)
print("DONE", flush=True)

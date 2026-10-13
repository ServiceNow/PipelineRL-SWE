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
  resplit_tc POOL S0 S1   (NEW_PATH 4.A.88) draw-preserving re-splits: only the paper's train + calibration problems (2-16 draws per
                  route) are reshuffled into train / calibration of the original sizes; the test set is the paper's. Seed -1 = the paper
                  split through the same pipeline (matched control for `resplit`, whose success readouts are refit).
  splitdraw POOL  (NEW_PATH 4.A.88) a noise-free bound on the headroom: on test problems with >= 2 valid draws on every route, each
                  (problem, route)'s draws are split at random into a pricing half A and an evaluation half B (REPS splits); every
                  arm is scored on B's mean outcome and cost. oracle_realized prices at B's own mean length (the paper's headroom
                  oracle, which also knows B's sampling noise); oracle_heldout prices at A's mean length (achievable by an
                  expected-cost predictor, noisy with |A| draws). Same training fits (ours, median) in every split.
  draws POOL S0 S1  (NEW_PATH 4.A.92) the paper's numbers averaged over training draws: for each seed (DRAW=s of
                  decompose.split_manifest: train + calibration reshuffled, test kept; -1 = the paper split) the success readouts are
                  refitted and every cost arm of tab:fresh / tab:pools / tab:allest is fitted: ours, median, mean, fromsuccess, zr-bins
                  (ten bins of the mean success logit), gbm, mixllm, knn, zr-pricing (ZeroRouter bins with our success),
                  zerorouter (its own success and bins, zerorouter_full), ft-<enc> (FT_DIR/ft_cost_<pool>_<enc>_d<s>.jsonl if
                  present), oracle (realized length). Per arm: saving vs median and the DIRECT saving of ours vs the arm, point and
                  NBOOT test resamples (rng 0, the same resamples in every seed), saved for draws_aggregate.py.
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned OUT_DIR=... python tmlr_strengthen.py deploy|resplit|resplit_tc|pricegap|splitdraw|draws POOL [S0 S1]
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

elif MODE_ in ("resplit", "resplit_tc"):
    s0, s1 = int(sys.argv[3]), int(sys.argv[4]); d = load(POOL_); N = len(d["ids"]); tc = MODE_ == "resplit_tc"
    U = np.concatenate([d["tr"], d["ca"]] + ([] if tc else [d["ev"]])); ntr, nca = len(d["tr"]), len(d["ca"]); res["splits"] = {}
    print(f"===== {'draw-preserving ' if tc else ''}re-splits {POOL_}: {len(U)} problems reshuffled, train {ntr}, calibration {nca}, test "
          f"{len(d['ev']) if tc else len(U) - ntr - nca}; seeds {s0}..{s1 - 1}{' (-1 = paper split)' if s0 < 0 else ''}", flush=True)
    for s in range(s0, s1):
        if s < 0:
            tr2, ca2, ev2 = d["tr"], d["ca"], d["ev"]
        elif tc:
            perm = np.random.default_rng(2000 + s).permutation(U); tr2, ca2, ev2 = perm[:ntr], perm[ntr:], d["ev"]
        else:
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
        json.dump(res, open(OUT / f"tmlr_strengthen_{MODE_}_{POOL_.lower().replace('-', '')}_{s0}_{s1}{TAG}.json", "w"), indent=1, default=float)

elif MODE_ == "draws":
    s0, s1 = int(sys.argv[3]), int(sys.argv[4]); res["seeds"] = {}; FT = Path(os.environ["FT_DIR"]) if os.environ.get("FT_DIR") else None
    for s in range(s0, s1):
        os.environ["DRAW"] = str(s); d = load(POOL_); N = len(d["ids"]); tr, ca, ev = d["tr"], d["ca"], d["ev"]
        S, T, Ca = view(d, tr), view(d, np.arange(N)), view(d, ca); cfg = choose(d, S, Ca)
        tk = {a: fit_predict(a, S, T, cfg.get(a)) for a in ARMS8 if a != "zerorouter"}; tk["zr-pricing"] = fit_predict("zerorouter", S, T, cfg["zerorouter"])
        dbar = np.log(d["P"] / (1 - d["P"])).mean(1); e_ = np.quantile(dbar[tr], np.linspace(0, 1, 11)[1:-1]); b_ = np.searchsorted(e_, dbar)
        zb = np.stack([np.array([d["L"][tr][b_[tr] == j, k].mean() if (b_[tr] == j).any() else d["L"][tr, k].mean() for j in range(10)])[b_] for k in range(d["M"])], 1)
        tk["zr-bins"] = zb * (d["L"][tr].mean(0) / zb[tr].mean(0))          # level-matched to the train mean, as fresh_baselines.py
        Pz, tk["zerorouter"], zcfg = zerorouter_full(S, Ca, T); Ps = {a: d["P"] for a in tk}; Ps["zerorouter"] = Pz
        for f_ in sorted(FT.glob(f"ft_cost_{POOL_}_*_d{s}.jsonl")) if FT and s >= 0 else []:
            tk["ft-" + f_.stem.split("_")[3]] = read_predictions(f_, d["ids"], "pred_tokens", d["M"]); Ps["ft-" + f_.stem.split("_")[3]] = d["P"]
        tk["oracle"] = d["L"]; Ps["oracle"] = d["P"]
        C = {a: cost_of(d, x) for a, x in tk.items()}; Cm = cost_of(d, np.repeat(d["med"][None], N, 0))
        rb = np.random.default_rng(0); BS = [ev] + [ev[rb.integers(0, len(ev), len(ev))] for _ in range(NBOOT)]
        Hs = [{a: curve(d, Ps[a], C[a], ii) for a in C} | {"median": curve(d, d["P"], Cm, ii)} for ii in BS]
        r_ = {}
        for a in C:
            vm = np.array([saved(H[a], H["median"]) for H in Hs]) * 100; vd = np.array([saved(H["ours"], H[a]) for H in Hs]) * 100
            r_[a] = dict(vs_median=float(vm[0]), vs_median_bs=vm[1:].tolist(), ours_vs_it=float(vd[0]), ours_vs_it_bs=vd[1:].tolist())
        r_["_auc"] = [float(auc(d["y0"][ev, k], d["P"][ev, k])) for k in range(d["M"])]; r_["_cfg"] = {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}
        r_["_cfg"]["zerorouter_full"] = list(zcfg); r_["_n"] = dict(train=len(tr), calibration=len(ca), test=len(ev))
        res["seeds"][s] = r_
        print(f"  seed {s:2d}: vs median " + "  ".join(f"{a} {r_[a]['vs_median']:+5.1f}" for a in C), flush=True)
        print("           ours vs it " + "  ".join(f"{a} {r_[a]['ours_vs_it']:+5.1f} [{np.percentile(r_[a]['ours_vs_it_bs'], 2.5):+.1f}, {np.percentile(r_[a]['ours_vs_it_bs'], 97.5):+.1f}]"
                                         for a in C if a != "ours"), flush=True)
        json.dump(res, open(OUT / f"tmlr_strengthen_draws_{POOL_.lower().replace('-', '')}_{s0}_{s1}{TAG}.json", "w"), indent=1, default=float)

elif MODE_ == "splitdraw":
    d = load(POOL_); N = len(d["ids"]); REPS = int(os.environ.get("REPS", "20"))
    name, feat, ds = POOLS[POOL_]; assert not ds, "splitdraw needs a pool evaluated on its own multi-draw split"
    t = np.load(R / name / "tensors.npz", allow_pickle=True); v = t["valid"].astype(bool)
    yo, ct, pt = np.asarray(t["final_outcome"]).astype(float), t["completion_tokens"].astype(float), t["prompt_tokens"].astype(float)
    ev = d["ev"][(v[d["ev"]].sum(2) >= 2).all(1)]
    S, T = view(d, d["tr"]), view(d, np.arange(N)); ours = fit_predict("ours", S, T)
    print(f"===== split-draw headroom {POOL_}: {len(ev)} of {len(d['ev'])} test problems with >= 2 draws on every route; {REPS} splits; "
          f"draws per route (median) {np.median(v[ev].sum(2), 0).tolist()}", flush=True)
    full = evaluate(d, {"ours": [ours], "oracle": [d["L"]]}, ev); res["all_draws"] = full
    print(f"  all draws (paper protocol) on these problems: headroom {full['oracle']['vs_median']*100:+5.1f}  ours {full['ours']['vs_median']*100:+5.1f}", flush=True)
    rows = []; res["reps"] = []
    for rep in range(REPS):
        rg = np.random.default_rng(3000 + rep); A = np.zeros(v.shape, bool); B = np.zeros(v.shape, bool)
        for i in ev:
            for m in range(d["M"]):
                k = rg.permutation(np.flatnonzero(v[i, m])); A[i, m, k[:len(k) // 2]] = True; B[i, m, k[len(k) // 2:]] = True
        mean_ = lambda x, W: np.where(W, x, 0).sum(2) / np.maximum(W.sum(2), 1)
        d2 = dict(d); q2, L2, I2 = d["q"].copy(), d["L"].copy(), d["I"].copy(); LA = d["L"].copy()
        q2[ev], L2[ev], I2[ev] = mean_(yo, B)[ev], np.maximum(mean_(ct, B)[ev], 1), mean_(pt, B)[ev]; LA[ev] = np.maximum(mean_(ct, A)[ev], 1)
        d2.update(q=q2, L=L2, I=I2, paid=I2 * d["rates"][:, 0] + L2 * d["rates"][:, 1])
        r_ = evaluate(d2, {"ours": [ours], "oracle_realized": [L2], "oracle_heldout": [LA]}, ev); res["reps"].append(r_)
        rows.append([r_[a]["vs_median"] for a in ("ours", "oracle_realized", "oracle_heldout")])
        print(f"  split {rep:2d}: ours {rows[-1][0]*100:+5.1f}  oracle realized {rows[-1][1]*100:+5.1f}  held-out {rows[-1][2]*100:+5.1f}   "
              f"ours - held-out oracle {r_['oracle_heldout']['ours_minus_it'][0]:+5.1f} [{r_['oracle_heldout']['ours_minus_it'][1]:+.1f}, {r_['oracle_heldout']['ours_minus_it'][2]:+.1f}]", flush=True)
    m_, sd_ = np.nanmean(rows, 0) * 100, np.nanstd(rows, 0) * 100
    res["summary"] = dict(ours=[m_[0], sd_[0]], oracle_realized=[m_[1], sd_[1]], oracle_heldout=[m_[2], sd_[2]],
                          capture_realized=m_[0] / m_[1], capture_heldout=m_[0] / m_[2])
    print(f"  MEAN over splits: ours {m_[0]:.1f} +- {sd_[0]:.1f}; headroom realized {m_[1]:.1f} +- {sd_[1]:.1f}, held-out {m_[2]:.1f} +- {sd_[2]:.1f}; "
          f"capture {m_[0] / m_[1]:.2f} (realized) / {m_[0] / m_[2]:.2f} (held-out)", flush=True)

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

if MODE_ not in ("resplit", "resplit_tc", "draws"):
    json.dump(res, open(OUT / f"tmlr_strengthen_{MODE_}_{POOL_.lower().replace('-', '')}{TAG}.json", "w"), indent=1, default=float)
print("DONE", flush=True)

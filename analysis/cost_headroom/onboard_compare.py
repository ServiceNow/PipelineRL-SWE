"""Onboarding a NEW route from k labelled problems: ours vs ZeroRouter vs kNN vs naive, pinned, billed (TMLR; supersedes the
unpinned list-price 4.A.31 / 4.A.33 comparison). Same loading, test sets and billed prices as tmlr_free_analyses.py.
Hold out route h; every OTHER route keeps the paper's full readouts (success + cost) in every arm, so only the new route's
estimates differ. k random training problems of h (20 draws, the SAME anchors in every arm):
  ours      cost = shared level (mean predicted log length of the other routes) + 1 offset + smearing; success = logistic in the
            shared difficulty (mean logit of the other routes' success readouts), 2 parameters (onboard_full.py's recipe)
  zr        ZeroRouter (2601.06220) as in fresh_baselines.py: D-dim 2PL IRT on the OTHER routes' train outcomes, stage 2 = prefill
            (mean + last) -> PCA(256) -> ridge onto (log alpha, b); new route: theta fitted on the k anchors (their stage-1
            positions), cost = per-bin mean output of the anchors in K quantile bins of s = alpha^T b (empty bin -> anchors' mean)
  zr-dopt   zr with their D-optimal-style anchors (top-k Fisher information sum_d alpha_d^2 p (1 - p) at theta = 0), one draw
  knn       CARROT-style: success and length of route h = mean over the nn nearest anchors in the same PCA(256) prefill space
  naive     constant success rate + median output of the k anchors
  ours-dopt ours on zr-dopt's anchors (anchor choice is separate from the estimator; paired with zr-dopt)
Configurations chosen on CALIBRATION per k (mean over held-out routes and draws), then applied once to test: zr / zr-dopt
(D, K) in {1, 5} x {5, 10, 20}; knn nn in {1, 3, 5, 10} (<= k). zr's best-on-test configuration is reported as an upper bound.
Metric: cost saved at matched accuracy vs the FULL pool priced at training-median length (fixed reference; same as before);
intervals for ours - arm: paired bootstrap over test problems (200 resamples), resample b paired with anchor draw b mod 20.
HOLD=groups: hold out a whole MODEL at a time (gpt-oss-20b low+medium, gpt-oss-120b medium+high, deepseek-v4-flash), so no
sibling effort of the new model stays in the pool; all its routes are onboarded from the same k problems.
Usage: [HOLD=groups] REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned OUT_DIR=... python onboard_compare.py LCB|Omni|MMLU-Pro|SuperGPQA|BBEH
"""
import json, os, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / "tmlr_free_analyses.py").read().split("# ---------------------------------------------------------------- 1.")[0])
import torch
from zr_dimsweep import fit_stage1, fit_theta
torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))

sig = lambda z: 1 / (1 + np.exp(-z))
KS, NDRAW, NBOOT = [5, 10, 20, 50], 20, int(os.environ.get("NBOOT", "200"))
if os.environ.get("SMOKE"):
    KS, NDRAW, NBOOT = [5, 20], 2, 3
ca = np.array([idx[str(p)] for p in sp["calibration_problem_ids"]]); ca = ca[(v[ca].sum(2) > 0).all(1)]
succ, n = np.where(v, t["final_outcome"], 0).sum(2).astype(float), v.sum(2).astype(float)
lg = np.log(P / (1 - P)); LT = np.log(tok); C_full = C_ours
X = rich(feat, ids); Z = PCA(256, random_state=0).fit(X[tr]).transform(X); Z /= Z[tr].std(0) + 1e-6; del X


def curve(Pm, C, ii):
    m = (VALUES[:, None, None] * Pm[ii][None] - C[ii][None]).argmax(2); r = np.arange(len(ii))[None]
    return hull(list(zip(paid[ii][r, m].mean(1), q[ii][r, m].mean(1))))


def gain(Pm, C, ii, H0):
    H = curve(Pm, C, ii); lo, hi = max(H[0][1], H0[0][1]), min(H[-1][1], H0[-1][1])
    if hi <= lo:
        return np.nan
    T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    return 1 - float(np.exp(np.nanmean(np.log([cost_at(H, x) / cost_at(H0, x) for x in T]))))


def put(G, pc):
    """G: held-out routes; pc = (P cols [N, |G|], C cols [N, |G|])"""
    Pm, C = P.copy(), C_full.copy(); Pm[:, G] = np.clip(pc[0], 1e-4, 1 - 1e-4); C[:, G] = pc[1]; return Pm, C


# HOLD=groups: hold out a whole MODEL (both efforts of a gpt-oss size) so no sibling route stays in the pool; default: one route at a time
GROUPS = ([[slots.index(x) for x in g] for g in (["oss20lo", "oss20md"], ["oss120md", "oss120hi"], ["dsv4f"])]
          if os.environ.get("HOLD") == "groups" else [[h] for h in range(M)])
GN = ["+".join(slots[h] for h in G) for G in GROUPS]
stackc = lambda lst: (np.stack([a for a, _ in lst], 1), np.stack([b for _, b in lst], 1))
H0_te, H0_ca = curve(P, C_med, ev), curve(P, C_med, ca)
full = gain(P, C_full, ev, H0_te)
print(f"===== onboarding {POOL}: test n={len(ev)}, calibration n={len(ca)}, train n={len(tr)}; all routes on full readouts {full*100:+.1f}% "
      f"vs median; held out: {GN}", flush=True)
CFG_ZR = [(D, K) for D in (1, 5) for K in (5, 10, 20)]
# per (group, k): list over draws of {arm: (P cols, C cols)}; zr arms keep every configuration so calibration can choose
cols, dopt, without = {}, {}, {}
for gi, G in enumerate(GROUPS):
    others = [m for m in range(M) if m not in G]; trh = tr[(n[tr][:, G] > 0).all(1)]
    Pw, Cw = P.copy(), C_full.copy(); Pw[:, G] = 1e-4; Cw[:, G] = 1e9; without[GN[gi]] = gain(Pw, Cw, ev, H0_te)
    lvl, dbar = LT[:, others].mean(1), lg[:, others].mean(1)
    lat = {}
    for D in (1, 5):
        la, bb, th = fit_stage1(succ[tr][:, others], n[tr][:, others], D, 0)
        pr = RidgeCV(alphas=np.geomspace(1, 1e5, 11)).fit(Z[tr], np.c_[la, bb]).predict(Z)
        A, B = np.exp(pr[:, :D]), pr[:, D:]; A[tr], B[tr] = np.exp(la), bb; s = (A * B).sum(1)
        bins = {K: np.searchsorted(np.quantile(s[tr], np.linspace(0, 1, K + 1)[1:-1]), s) for K in (5, 10, 20)}
        p0 = sig((A[trh] * (0 - B[trh])).sum(1)); fisher = (A[trh] ** 2).sum(1) * p0 * (1 - p0)
        lat[D] = (A, B, bins, trh[np.argsort(-fisher)])

    def zr_cols(an, D):
        A, B, bins, _ = lat[D]; per = {cfg: [] for cfg in [(D, K) for K in bins]}
        for h in G:
            th_ = fit_theta(A[an], B[an], succ[an, h], n[an, h], D); ph = sig((A * (th_[None] - B)).sum(1))
            for K, bn in bins.items():
                tab = np.array([L[an, h][bn[an] == j].mean() if (bn[an] == j).any() else L[an, h].mean() for j in range(K)])
                per[(D, K)].append((ph, I[:, h] * rates[h, 0] + tab[bn] * rates[h, 1]))
        return {cfg: stackc(v_) for cfg, v_ in per.items()}

    def ours_cols(an):
        o = []
        for h in G:
            off = np.mean(np.log(L[an, h]) - lvl[an]); sm = np.mean(np.exp(np.log(L[an, h]) - (lvl[an] + off)))
            yk = np.concatenate([t["final_outcome"][i, h][v[i, h]] for i in an]).astype(int); Xk = np.repeat(dbar[an], n[an, h].astype(int))
            ph = LogisticRegression(C=1.0).fit(Xk[:, None], yk).predict_proba(dbar[:, None])[:, 1] if len(set(yk)) == 2 else np.full(len(ids), yk.mean())
            o.append((ph, I[:, h] * rates[h, 0] + np.exp(lvl + off) * sm * rates[h, 1]))
        return stackc(o)

    rng = np.random.default_rng(1000 + gi)
    for k in KS:
        draws = []
        for _ in range(NDRAW):
            an = rng.choice(trh, k, replace=False); o = {"ours": ours_cols(an)}
            nv = []
            for h in G:
                yk = np.concatenate([t["final_outcome"][i, h][v[i, h]] for i in an]).astype(int)
                nv.append((np.full(len(ids), np.clip(yk.mean(), .02, .98)), I[:, h] * rates[h, 0] + np.median(L[an, h]) * rates[h, 1]))
            o["naive"] = stackc(nv)
            for D in (1, 5):
                for cfg, pc in zr_cols(an, D).items():
                    o[("zr",) + cfg] = pc
            d2 = (Z[an] ** 2).sum(1)[None] - 2 * Z @ Z[an].T; order = np.argsort(d2, 1)      # + |z|^2 is constant per row
            for nn in (1, 3, 5, 10):
                if nn <= k:
                    nb = an[order[:, :nn]]
                    o[("knn", nn)] = stackc([(np.clip(q[nb, h].mean(1), .02, .98), I[:, h] * rates[h, 0] + L[nb, h].mean(1) * rates[h, 1]) for h in G])
            draws.append(o)
        cols[(gi, k)] = draws
        dopt[(gi, k)] = {("zr-dopt",) + cfg: pc for D in (1, 5) for cfg, pc in zr_cols(lat[D][3][:k], D).items()}
        dopt[(gi, k)].update({("ours-dopt", D): ours_cols(lat[D][3][:k]) for D in (1, 5)})     # ours on the SAME selected anchors
    print(f"  prepared held-out {GN[gi]} (without it: {without[GN[gi]]*100:+.1f}%)", flush=True)

# ---- choose configurations on calibration, per k
chosen = {}
for k in KS:
    zr_c = {cfg: np.nanmean([gain(*put(G, d[("zr",) + cfg]), ca, H0_ca) for h, G in enumerate(GROUPS) for d in cols[(h, k)]]) for cfg in CFG_ZR}
    dp_c = {cfg: np.nanmean([gain(*put(G, dopt[(h, k)][("zr-dopt",) + cfg]), ca, H0_ca) for h, G in enumerate(GROUPS)]) for cfg in CFG_ZR}
    kn_c = {nn: np.nanmean([gain(*put(G, d[("knn", nn)]), ca, H0_ca) for h, G in enumerate(GROUPS) for d in cols[(h, k)]]) for nn in (1, 3, 5, 10) if nn <= k}
    chosen[k] = {"zr": max(zr_c, key=zr_c.get), "zr-dopt": max(dp_c, key=dp_c.get), "knn": max(kn_c, key=kn_c.get)}
    print(f"  k={k}: calibration picks zr D,K={chosen[k]['zr']}  zr-dopt D,K={chosen[k]['zr-dopt']}  knn nn={chosen[k]['knn']}", flush=True)


def arms_of(h, k, j):
    d, c = cols[(h, k)][j], chosen[k]
    return {"ours": d["ours"], "zr": d[("zr",) + c["zr"]], "zr-dopt": dopt[(h, k)][("zr-dopt",) + c["zr-dopt"]],
            "knn": d[("knn", c["knn"])], "naive": d["naive"], "ours-dopt": dopt[(h, k)][("ours-dopt", c["zr-dopt"][0])]}


ARMS = ["ours", "zr", "zr-dopt", "knn", "naive", "ours-dopt"]
res = {"pool": POOL, "n_test": int(len(ev)), "n_cal": int(len(ca)), "routes": slots, "held_out": GN, "full": full, "without": without, "by_k": {}}
rb = np.random.default_rng(0); BS = [ev[rb.integers(0, len(ev), len(ev))] for _ in range(NBOOT)]; H0_bs = [curve(P, C_med, ii) for ii in BS]
for k in KS:
    NG = len(GROUPS)
    pt_ = {a: np.array([[gain(*put(G, arms_of(h, k, j)[a]), ev, H0_te) for j in range(NDRAW)] for h, G in enumerate(GROUPS)]) for a in ARMS}
    zr_best = max(np.nanmean([[gain(*put(G, cols[(h, k)][j][("zr",) + cfg]), ev, H0_te) for j in range(NDRAW)] for h, G in enumerate(GROUPS)]) for cfg in CFG_ZR)
    bs = {a: np.array([[gain(*put(G, arms_of(h, k, b % NDRAW)[a]), ii, H0b) for h, G in enumerate(GROUPS)] for b, (ii, H0b) in enumerate(zip(BS, H0_bs))])
          for a in ARMS}
    row = {"chosen": {a: list(c) if isinstance(c, tuple) else c for a, c in chosen[k].items()}, "zr_best_on_test": float(zr_best), "arms": {}}
    for a in ARMS:
        r_ = dict(mean=float(np.nanmean(pt_[a])), by_route={GN[h]: float(np.nanmean(pt_[a][h])) for h in range(NG)},
                  sd_over_draws=float(np.nanstd(np.nanmean(pt_[a], 0))))
        if a != "ours":
            dd = np.nanmean(bs["ours"] - bs[a], 1) * 100
            r_["ours_minus_it"] = [float(np.nanmean(pt_["ours"] - pt_[a]) * 100), *map(float, np.nanpercentile(dd, [2.5, 97.5]))]
            r_["ours_minus_it_by_route"] = {GN[h]: float(np.nanmean(pt_["ours"][h] - pt_[a][h]) * 100) for h in range(NG)}
        row["arms"][a] = r_
    dd = np.nanmean(bs["ours-dopt"] - bs["zr-dopt"], 1) * 100
    row["oursdopt_minus_zrdopt"] = [float(np.nanmean(pt_["ours-dopt"] - pt_["zr-dopt"]) * 100), *map(float, np.nanpercentile(dd, [2.5, 97.5]))]
    res["by_k"][str(k)] = row
    print(f"  k={k:<3} " + "  ".join(f"{a} {row['arms'][a]['mean']*100:+5.1f}%" for a in ARMS) + f"   (zr best-on-test {zr_best*100:+.1f}%)", flush=True)
    for a in ARMS[1:]:
        m_, lo_, hi_ = row["arms"][a]["ours_minus_it"]
        print(f"        ours - {a:<8} {m_:+5.1f} pt [{lo_:+.1f}, {hi_:+.1f}]   by route: "
              + " ".join(f"{s} {x:+.1f}" for s, x in row["arms"][a]["ours_minus_it_by_route"].items()), flush=True)
    m_, lo_, hi_ = row["oursdopt_minus_zrdopt"]
    print(f"        same selected anchors: ours-dopt - zr-dopt {m_:+5.1f} pt [{lo_:+.1f}, {hi_:+.1f}]", flush=True)
out_dir = Path(os.environ.get("OUT_DIR", Path(__file__).parent))
json.dump(res, open(out_dir / f"onboard_compare_{POOL.replace('-', '').lower()}{'_groups' if os.environ.get('HOLD') == 'groups' else ''}{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
print("DONE", flush=True)

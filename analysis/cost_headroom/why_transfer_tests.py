"""Falsification tests for the explanation of 4.A.77: "difficulty is a benchmark-specific proxy for the work a problem needs, because
WHY a problem is hard differs between benchmarks; the dedicated readout reads cues of the work itself, which mean the same thing across
benchmarks." Each test states the prediction that would refute the explanation. Pinned, train/test splits, cost-only where routing is
involved (every arm uses the target pool's own success readouts). Loading / readouts from cost_generalization.py.
  slopes   (a) the solve-rate -> log-length relation differs between benchmarks: per pool and route, OLS slope and intercept on the
             TRAIN problems. (b) the transfer error of ORACLE-difficulty pricing grows with how different the target's slope is from its
             sources' (Spearman over target x route). REFUTED IF slopes are similar across pools, or (b) shows no relation.
           (c) a "work" oracle -- the problem's realized length on the cheapest route, gpt-oss-20b-low -- transferred the same way
             predicts the other routes' lengths about as well as in-domain, while oracle difficulty does not. REFUTED IF work transfers
             no better than difficulty.
           (d) where the target looks unlike its sources in prefill space (domain-classifier AUC; nearest-neighbour distance ratio), the
             dedicated readout keeps less of its in-domain saving under transfer (Spearman over targets). REFUTED IF no relation.
  gaps     Same difficulty, different length: for every pair of pools in a domain, problems are matched by true solve rate (8 quantile
           bins of the pair); per route and bin, the ACTUAL log-length gap between the two pools vs the gap PREDICTED by the dedicated
           readout trained on all OTHER pools (neither of the pair). Difficulty-only pricing predicts a zero gap by construction.
           REFUTED IF matched-difficulty gaps are small, or the readout's predicted gaps do not track them.
  subjects Within one benchmark and ONE set of success readouts (removes the per-benchmark logit-scale confound): leave-one-subject-out
           on MMLU-Pro (categories), SuperGPQA (disciplines), BBEH (tasks). Each problem's cost comes from a model fitted without its
           subject; ours vs pricing from success vs oracle difficulty, R2 and saving vs in-domain all-subject fits. REFUTED IF the
           difficulty arms hold up across subjects as well as the dedicated readout does.
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned OUT_DIR=... python why_transfer_tests.py slopes|gaps|subjects
"""
import json, os, sys
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, str(Path(__file__).parent))
_src = open(Path(__file__).parent / "cost_generalization.py").read()
exec(_src.split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
TEST = sys.argv[1]; OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent)); TAG = os.environ.get("RESULT_TAG", "")
REAL = Path("/mnt/llmd/results/exps/aristides/reason")
POOLS9 = DOMAIN["coding"] + DOMAIN["reasoning"]; res = {"test": TEST}


def ols(x, y):
    A = np.c_[np.ones(len(x)), x]; b = np.linalg.lstsq(A, y, rcond=None)[0]; return b      # intercept, slope


def r2(y, e):
    return float(1 - ((y - e) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def poly_fit_pred(xs, ys, xt):                       # y ~ x + x^2, least squares (the oracle maps)
    A = lambda x: np.c_[np.ones(len(x)), x, x ** 2]
    return A(xt) @ np.linalg.lstsq(A(xs), ys, rcond=None)[0]


if TEST == "slopes":
    D = {p: load(p) for p in POOLS9}; M = D["LCB"]["M"]; slots = D["LCB"]["slots"]
    sl = {p: [ols(d["q"][d["tr"]].mean(1), np.log(d["L"][d["tr"], k])).tolist() for k in range(M)] for p, d in D.items()}
    res["slopes"] = sl
    print("(a) slope of log length on true solve rate (train), per route " + str(slots))
    for p in POOLS9:
        print(f"    {p:<10} " + "  ".join(f"{b:+5.2f} (int {a:5.2f})" for a, b in sl[p]), flush=True)
    rows = []; out_b = {}
    for tg in POOLS9:
        dom = "coding" if tg in DOMAIN["coding"] else "reasoning"; srcs = [p for p in DOMAIN[dom] if p != tg]; d = D[tg]; ev = d["ev"]
        xs = np.concatenate([D[p]["q"][D[p]["tr"]].mean(1) for p in srcs]); xt = d["q"][ev].mean(1)
        w_s = np.concatenate([np.log(D[p]["L"][D[p]["tr"], 0]) for p in srcs]); w_t = np.log(d["L"][ev, 0])          # work oracle: oss20lo length
        per = []
        for k in range(M):
            ys = np.concatenate([np.log(D[p]["L"][D[p]["tr"], k]) for p in srcs]); yt = np.log(d["L"][ev, k])
            od_x = r2(yt, poly_fit_pred(xs, ys, xt)); od_in = r2(yt, poly_fit_pred(d["q"][d["tr"]].mean(1), np.log(d["L"][d["tr"], k]), xt))
            wk_x = r2(yt, poly_fit_pred(w_s, ys, w_t)) if k else np.nan
            wk_in = r2(yt, poly_fit_pred(np.log(d["L"][d["tr"], 0]), np.log(d["L"][d["tr"], k]), w_t)) if k else np.nan
            mism = abs(sl[tg][k][1] - np.mean([sl[p][k][1] for p in srcs]))
            per.append(dict(route=slots[k], oracle_diff_in=od_in, oracle_diff_xfer=od_x, work_in=wk_in, work_xfer=wk_x, slope_mismatch=mism))
            rows.append((mism, od_in - od_x))
        out_b[tg] = per
        print(f"    {tg:<10} oracle difficulty R2 in/xfer " + " ".join(f"{x['oracle_diff_in']:+.2f}/{x['oracle_diff_xfer']:+.2f}" for x in per)
              + "   work oracle (oss20lo length) in/xfer " + " ".join(f"{x['work_in']:+.2f}/{x['work_xfer']:+.2f}" for x in per[1:]), flush=True)
    rho = spearmanr([a for a, _ in rows], [b for _, b in rows]); res["by_target"] = out_b
    res["slope_mismatch_vs_difficulty_transfer_drop"] = dict(spearman=float(rho.correlation), p=float(rho.pvalue), n=len(rows))
    wk = [(x["work_in"] - x["work_xfer"], x["oracle_diff_in"] - x["oracle_diff_xfer"]) for per in out_b.values() for x in per[1:]]
    res["mean_r2_drop"] = dict(work=float(np.mean([a for a, _ in wk])), oracle_difficulty=float(np.mean([b for _, b in wk])))
    print(f"(b) Spearman(slope mismatch, oracle-difficulty R2 drop under transfer) = {rho.correlation:+.2f} (p {rho.pvalue:.3f}, n {len(rows)})")
    print(f"(c) mean R2 drop under transfer: work oracle {res['mean_r2_drop']['work']:+.2f}, oracle difficulty {res['mean_r2_drop']['oracle_difficulty']:+.2f}")
    # (d) dissimilarity vs retention of the dedicated readout's saving (cost-only transfer, 4.A.73 outputs)
    ret, auc_, nnr = [], [], []
    for tg in POOLS9:
        f = OUT / f"cost_generalization_transfer_{tg.replace('-', '').lower()}{TAG}.json"
        if not f.exists():
            continue
        j = json.load(open(f)); ind, xf = j["in_domain"]["ours"]["vs_median"], j["same_domain"]["zero_shot"]["ours"]["vs_median"]
        dom = "coding" if tg in DOMAIN["coding"] else "reasoning"; srcs = [p for p in DOMAIN[dom] if p != tg]; d = D[tg]
        Xs = np.concatenate([D[p]["X"][D[p]["tr"]] for p in srcs]); Xt = d["X"][d["ev"]]
        pca = PCA(64, random_state=0).fit(Xs); Zs, Zt = pca.transform(Xs), pca.transform(Xt); sd = Zs.std(0) + 1e-6; Zs, Zt = Zs / sd, Zt / sd
        rng = np.random.default_rng(0); it = rng.choice(len(Zt), min(len(Zt), len(Zs)), replace=False)
        Zc = np.r_[Zs, Zt[it]]; yc = np.r_[np.zeros(len(Zs)), np.ones(len(it))]
        a_ = float(np.mean(cross_val_score(LogisticRegression(max_iter=2000, C=0.1), Zc, yc, cv=5, scoring="roc_auc")))
        nn = NearestNeighbors(n_neighbors=2).fit(Zs); dt = nn.kneighbors(Zt, 1)[0][:, 0]; ds = nn.kneighbors(Zs, 2)[0][:, 1]
        ret.append(xf / ind if ind > 0.05 else np.nan); auc_.append(a_); nnr.append(float(np.median(dt) / np.median(ds)))
        print(f"    {tg:<10} retention {xf*100:+5.1f}/{ind*100:+5.1f} = {ret[-1]:+.2f}   domain AUC {a_:.3f}   NN distance ratio {nnr[-1]:.2f}", flush=True)
    ok = [i for i, x in enumerate(ret) if np.isfinite(x)]
    if len(ok) >= 4:
        r1, r2_ = spearmanr([ret[i] for i in ok], [auc_[i] for i in ok]), spearmanr([ret[i] for i in ok], [nnr[i] for i in ok])
        res["dissimilarity_vs_retention"] = dict(spearman_auc=float(r1.correlation), spearman_nn=float(r2_.correlation), n=len(ok))
        print(f"(d) Spearman(retention, domain AUC) = {r1.correlation:+.2f}; Spearman(retention, NN ratio) = {r2_.correlation:+.2f} (n {len(ok)}; targets "
              "with in-domain saving > 5%)")

elif TEST == "gaps":
    D = {p: load(p) for p in POOLS9}; M = D["LCB"]["M"]; rows = []; fs_rows = []
    for dom, plist in DOMAIN.items():
        for A, B in combinations(plist, 2):
            others = [p for p in POOLS9 if p not in (A, B)]
            S = stack([view(D[p], D[p]["tr"]) for p in others])
            pr = {p: np.log(fit_predict("ours", S, view(D[p], np.arange(len(D[p]["ids"]))))) for p in (A, B)}
            pf = {p: np.log(fit_predict("fromsuccess", S, view(D[p], np.arange(len(D[p]["ids"]))))) for p in (A, B)}
            ia, ib = np.r_[D[A]["tr"], D[A]["ev"]], np.r_[D[B]["tr"], D[B]["ev"]]
            qa, qb = D[A]["q"][ia].mean(1), D[B]["q"][ib].mean(1); edges = np.quantile(np.r_[qa, qb], np.linspace(0, 1, 9)[1:-1])
            ba, bb = np.searchsorted(edges, qa), np.searchsorted(edges, qb)
            for k in range(M):
                ya, yb = np.log(D[A]["L"][ia, k]), np.log(D[B]["L"][ib, k])
                for j in range(8):
                    ma, mb = ba == j, bb == j
                    if ma.sum() >= 5 and mb.sum() >= 5:
                        rows.append((dom, A, B, k, j, ya[ma].mean() - yb[mb].mean(), pr[A][ia][ma, k].mean() - pr[B][ib][mb, k].mean()))
                        fs_rows.append(pf[A][ia][ma, k].mean() - pf[B][ib][mb, k].mean())
            g = [(r[5], r[6]) for r in rows if r[1] == A and r[2] == B]
            print(f"  {A:>9} vs {B:<10} matched-difficulty gap, mean |actual| {np.mean([abs(a) for a, _ in g]):.2f} log units; "
                  f"corr(actual, predicted) {pearsonr(*zip(*g))[0]:+.2f}", flush=True)
    act, pred = np.array([r[5] for r in rows]), np.array([r[6] for r in rows]); fsp = np.array(fs_rows)
    slope = float(np.polyfit(act, pred, 1)[0])
    res.update(n_cells=len(rows), mean_abs_actual_gap=float(np.abs(act).mean()), corr_ours=float(pearsonr(act, pred)[0]), slope_ours=slope,
               corr_fromsuccess=float(pearsonr(act, fsp)[0]), share_gap_explained_ours=float(1 - ((act - pred) ** 2).mean() / (act ** 2).mean()),
               share_gap_explained_fromsuccess=float(1 - ((act - fsp) ** 2).mean() / (act ** 2).mean()),
               cells=[dict(domain=r[0], A=r[1], B=r[2], route=int(r[3]), bin=int(r[4]), actual=float(r[5]), predicted=float(r[6])) for r in rows])
    print(f"ALL: {len(rows)} (pair, route, difficulty-bin) cells; mean |actual gap| {res['mean_abs_actual_gap']:.2f} log units (x{np.exp(res['mean_abs_actual_gap']):.2f}); "
          f"ours: corr {res['corr_ours']:+.2f}, slope {slope:.2f}, share of squared gap explained {res['share_gap_explained_ours']:+.2f}; "
          f"pricing from success: corr {res['corr_fromsuccess']:+.2f}, share {res['share_gap_explained_fromsuccess']:+.2f}; difficulty-only: 0 by construction")

elif TEST == "subjects":
    subj = {}
    for f in [*REAL.glob("math_pool/*/problems.jsonl"), *Path(__file__).parent.glob("expansion_20261001/*_tasks*.jsonl")]:
        for l in open(f):
            if l.strip():
                r = json.loads(l); subj.setdefault(str(r["problem_id"]), r.get("subject"))
    for P_ in ("MMLU-Pro", "SuperGPQA", "BBEH"):
        d = load(P_); N = len(d["ids"]); T = view(d, np.arange(N)); ev = d["ev"]; sj = np.array([str(subj.get(p)) for p in d["ids"]])
        print(f"===== {P_}: {len(set(sj[d['tr']]))} subjects in train, {(sj == 'None').sum()} problems without a subject", flush=True)
        arms = ["ours", "fromsuccess", "meanlogit_lin", "oracle_diff"]
        inn = {a: fit_predict(a, view(d, d["tr"]), T) for a in arms}; loso = {a: np.ones((N, d["M"])) for a in arms}
        for s_ in sorted(set(sj)):
            trs = d["tr"][sj[d["tr"]] != s_]; tgt = np.where(sj == s_)[0]
            if len(trs) < 50 or not len(tgt):
                continue
            for a in arms:
                x = fit_predict(a, view(d, trs), T); loso[a][tgt] = x[tgt]
        preds = {"ours": [inn["ours"]]} | {f"{a}|in": [inn[a]] for a in arms[1:]} | {f"{a}|loso": [loso[a]] for a in arms}
        r_ = evaluate(d, preds, ev); r2s = {a: r2_routes(d, x[0], ev) for a, x in preds.items()}
        res[P_] = dict(saving=r_, r2=r2s, n_subjects=int(len(set(sj))))
        for a in preds:
            print(f"  {a:<20} saving {r_[a]['vs_median']*100:+6.1f}   R2 per route " + " ".join(f"{x:+.2f}" for x in r2s[a])
                  + ("" if a == "ours" else f"   ours(in) - it {r_[a]['ours_minus_it'][0]:+5.1f} [{r_[a]['ours_minus_it'][1]:+.1f}, {r_[a]['ours_minus_it'][2]:+.1f}]"), flush=True)
json.dump(res, open(OUT / f"why_transfer_tests_{TEST}{TAG}.json", "w"), indent=1, default=float)
print("DONE", flush=True)

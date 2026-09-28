"""WHY is per-problem output length predictable from a 4B prefill on LCB but not on CodeContests / TACO / BCB?

Per pool and route, target y = log of the problem's mean OUTPUT tokens (what the market-priced cost head predicts).
  ceiling     reliability of the per-problem mean given draw noise: k*ICC / (1 + (k-1)*ICC), k = valid draws/problem.
              The best R2 ANY predictor can reach against the observed mean (label noise, not predictor weakness).
  head R2     the real 4B head, log space, test split (cost_preds_market.jsonl; output part recovered exactly).
  diff->len   how much of y the problem's TRUE difficulty explains: R2 of y on a cubic in the route's own solve
              rate and the pool-mean solve rate (5-fold CV). The "difficulty channel": if length is mostly
              difficulty, a probe that predicts difficulty predicts length.
  probe diff  how well the probe predicts that difficulty: Spearman(success head p_m, true solve rate), test split.
  meta        R2 of y on surface metadata (log statement length + platform/difficulty one-hots), 5-fold CV.
  runaway     share of draws within 2% of the route's max observed output (the generation cap), and the
              share of problems whose draws straddle it (some capped, some not) -- near-random events.
Usage: python why_predictable.py pool[:cost_file] ...
"""
import json, sys, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R


def cv_r2(X, y, k=5, seed=0):
    X = np.c_[np.ones(len(y)), X]; rng = np.random.default_rng(seed); f = rng.permutation(len(y)) % k; pred = np.zeros(len(y))
    for i in range(k):
        tr, te = f != i, f == i
        w, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=1e-6); pred[te] = X[te] @ w
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def pool(spec):
    name, _, cfile = spec.partition(":"); cfile = cfile or "cost_preds_market.jsonl"
    D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
    S = [str(s) for s in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
    v = t["valid"].astype(bool); ok = (t["final_outcome"] & t["valid"]).astype(float)
    pt, ct = t["prompt_tokens"].astype(float), t["completion_tokens"].astype(float)
    n = v.sum(2); avail = n > 0
    Q = np.where(avail, (ok * v).sum(2) / np.maximum(n, 1), np.nan); qbar = np.nanmean(Q, 1)
    outm = np.where(avail, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan)
    inp = np.nanmean(np.where(v, pt, np.nan), 2)
    sp = json.load(open(D / "split_manifest.json")); te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
    lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"] for l in open(D / "content_preds.jsonl")}
    lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"] for l in open(D / cfile)}
    P = np.array([lp[p][:len(S)] for p in pids]); LC = np.array([lc[p][:len(S)] for p in pids])
    probs = [json.loads(l) for l in open(D / "problems.jsonl")]; pmeta = {str(r["problem_id"]): r for r in probs}
    stmt = np.array([len(str(pmeta.get(p, {}).get("problem_statement", ""))) for p in pids], float)
    cats = []
    for key in ("platform", "difficulty"):
        vals = [str(pmeta.get(p, {}).get(key, "")) for p in pids]; u = sorted(set(vals))
        if 1 < len(u) < 30:
            cats.append(np.array([[x == c for c in u[1:]] for x in vals], float))
    meta = np.c_[np.log1p(stmt)] if not cats else np.c_[np.log1p(stmt), np.concatenate(cats, 1)]
    rows = {}
    for m, s in enumerate(S):
        a = avail[:, m]; y = np.log(np.maximum(outm[:, m], 1.0))
        lv = np.log(np.maximum(np.where(v[:, m], ct[:, m], np.nan), 1.0))
        tot = np.nanvar(lv[a]); within = np.nanmean(np.nanvar(lv[a], 1)) if v.shape[2] > 1 else np.nan
        icc = 1 - within / tot if tot > 0 and np.isfinite(within) else np.nan
        k = n[a, m].mean(); rel = k * icc / (1 + (k - 1) * icc) if np.isfinite(icc) else np.nan
        # head's predicted output tokens (market head: cost = inp*pin + out*pout, in USD)
        pout = MK[s][1] / 1e6; pin = MK[s][0] / 1e6
        yo = np.log(np.maximum((LC[:, m] - np.nan_to_num(inp[:, m]) * pin) / pout, 1.0))
        tt = te[a[te]]
        head = 1 - ((y[tt] - yo[tt]) ** 2).sum() / ((y[tt] - y[tt].mean()) ** 2).sum()
        q = Q[a, m]; qb = qbar[a]
        diff = cv_r2(np.c_[q, q ** 2, q ** 3, qb, qb ** 2, qb ** 3], y[a])
        rho = spearmanr(P[tt, m], Q[tt, m]).correlation
        mt = cv_r2(meta[a], y[a])
        cap = np.nanmax(np.where(v[:, m], ct[:, m], np.nan)); hit = v[:, m] & (ct[:, m] >= 0.98 * cap)
        straddle = (hit.any(1) & (v[:, m] & ~hit).any(1))[a].mean() if v.shape[2] > 1 else np.nan
        rows[s] = dict(k=float(k), icc=float(icc), ceiling=float(rel), head_r2=float(head), diff_r2=float(diff),
                       probe_diff_rho=float(rho), meta_r2=float(mt), runaway=float(hit[a].sum() / v[a, m].sum()),
                       straddle=float(straddle), cap=float(cap), sd_y=float(y[a].std()))
    return name, rows


def main():
    res = dict(pool(s) for s in sys.argv[1:])
    json.dump(res, open("analysis/cost_headroom/why_predictable.json", "w"), indent=1, default=float)
    print("y = log mean OUTPUT tokens per problem. ceiling = best achievable R2 given draw noise; head = the 4B head (test);"
          "\ndiff->len = R2 from TRUE difficulty (CV); probe diff = Spearman(success head, true solve rate), test;"
          "\nmeta = R2 from statement length + platform/difficulty labels (CV); runaway = draws at the cap; straddle = problems"
          " with some draws capped and some not")
    print(f"{'pool / route':<34}{'k':>5}{'ICC':>6}{'ceil':>6}{'head':>7}{'diff->len':>10}{'probe diff':>11}{'meta':>6}{'runaway':>9}{'straddle':>9}{'sd y':>6}")
    for name, rows in res.items():
        for s, d in rows.items():
            print(f"{name[:22] + ' / ' + s:<34}{d['k']:5.1f}{d['icc']:6.2f}{d['ceiling']:6.2f}{d['head_r2']:+7.2f}{d['diff_r2']:10.2f}"
                  f"{d['probe_diff_rho']:11.2f}{d['meta_r2']:6.2f}{d['runaway']*100:8.1f}%{d['straddle']*100:8.1f}%{d['sd_y']:6.2f}")


if __name__ == "__main__":
    main()

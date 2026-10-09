"""Does the prefill's length signal beyond measured difficulty encode WORK (length-specific) or just FINER difficulty?
Per pool (LCB 341 / Omni-MATH 1,000 / MMLU-Pro 6,500 test problems, pinned):
  predicted level   l(x) = mean over the five reasoning routes of the dedicated readout's predicted log length (fitted on train)
  measured difficulty D(x) = every reasoning route's TRUE solve rate and its square (10 features; all draws)
  length residual   r(x) = l(x) - OLS(l ~ D)        : what the readout says about length beyond measured difficulty
  success residual  s(x) = mean predicted success logit - OLS(. ~ D)   : positive control -- finer difficulty by construction
Held-out models never used to fit anything: the five NON-REASONING routes (deepseek thinking off, Llama-3.1-8B, Qwen3-30B-A3B-Instruct,
Llama-3.3-70B, Qwen3-235B-A22B-Instruct; one draw each, same problems). For each, partial correlations given D (both sides
residualised on D) of r and s with (a) its log output length and (b) its success.
  Prediction if r is WORK:              r ~ held-out LENGTH (+), r ~ held-out SUCCESS ~ 0
  Prediction if r is FINER DIFFICULTY:  r ~ held-out SUCCESS (-: long-for-its-difficulty problems fail more), like -s does
  s is the power check: it should predict held-out success (+) if the test can detect finer difficulty at all.
Also: on problems EVERY reasoning route solves on every draw (solve rate 1), test R2 of the predicted log length per route, and the same
partial correlations restricted to them. Paired bootstrap (500) over problems for every correlation.
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned OUT_DIR=... python residual_work_test.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
_src = open(Path(__file__).parent / "cost_generalization.py").read()
exec(_src.split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
NR = Path("/mnt/llmd/results/exps/aristides/reason/nonreason_eval_20261009"); OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent))
rng = np.random.default_rng(0); res = {}


def resid(y, X):
    A = np.c_[np.ones(len(X)), X]; return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def pcorr(a, b, X, ok):
    """partial correlation of a and b given X, on rows ok, with a bootstrap 95% interval"""
    a, b, X = a[ok], b[ok], X[ok]; ra, rb = resid(a, X), resid(b, X); c = float(np.corrcoef(ra, rb)[0, 1]); bs = []
    for _ in range(500):
        i = rng.integers(0, len(a), len(a)); bs.append(np.corrcoef(resid(a[i], X[i]), resid(b[i], X[i]))[0, 1])
    return [c, *map(float, np.nanpercentile(bs, [2.5, 97.5]))], int(ok.sum())


for P_, ds in (("LCB", "lcb"), ("Omni", "omni500"), ("MMLU-Pro", "mmlupro")):
    d = load(P_); N = len(d["ids"]); T = view(d, np.arange(N)); ev = d["ev"]
    tok = fit_predict("ours", view(d, d["tr"]), T); lp = np.log(tok)
    t = np.load(NR / ds / "tensors.npz", allow_pickle=True); nid = {str(p): i for i, p in enumerate(t["problem_ids"])}
    rows = np.array([nid.get(p, -1) for p in d["ids"]]); nslots = list(map(str, t["model_slots"]))
    nv = t["valid"][:, :, 0].astype(bool); ny = (t["final_outcome"][:, :, 0] & t["valid"][:, :, 0]).astype(float)
    nl = np.log(np.maximum(t["completion_tokens"][:, :, 0].astype(float), 1))
    q = d["q"][ev]; D = np.c_[q, q ** 2]; l = lp[ev].mean(1); lg = np.log(d["P"] / (1 - d["P"]))[ev].mean(1)
    r_, s_ = resid(l, D), resid(lg, D); out = {"n_test": int(len(ev)), "corr_r_s": float(np.corrcoef(r_, s_)[0, 1]), "heldout": {}}
    print(f"===== {P_}: test {len(ev)}; corr(length residual, success residual) = {out['corr_r_s']:+.2f}", flush=True)
    # sanity: the reasoning routes' own lengths (in-sample routes) and the predicted level
    own = [pcorr(l, np.log(d["L"][ev, k]), D, np.ones(len(ev), bool))[0][0] for k in range(d["M"])]
    print(f"  sanity: partial corr(r, reasoning route log length | D) {' '.join(f'{x:+.2f}' for x in own)}", flush=True)
    allsolved = (q == 1).all(1); out["n_all_solved"] = int(allsolved.sum())
    for m, nm in enumerate(nslots):
        ri = rows[ev]; ok = (ri >= 0) & nv[np.maximum(ri, 0), m]
        yl, ys = np.where(ok, nl[np.maximum(ri, 0), m], 0.0), np.where(ok, ny[np.maximum(ri, 0), m], 0.0)
        e = {"r_vs_length": pcorr(l, yl, D, ok), "r_vs_success": pcorr(l, ys, D, ok), "s_vs_length": pcorr(lg, yl, D, ok),
             "s_vs_success": pcorr(lg, ys, D, ok), "acc": float(ys[ok].mean())}
        if allsolved.sum() >= 40:
            e["allsolved_r_vs_length"] = pcorr(l, yl, D, ok & allsolved); e["allsolved_r_vs_success"] = pcorr(l, ys, D, ok & allsolved)
        out["heldout"][nm] = e
        f = lambda k: f"{e[k][0][0]:+.2f} [{e[k][0][1]:+.2f}, {e[k][0][2]:+.2f}]"
        print(f"  {nm:<10} (acc {e['acc']:.2f}, n {e['r_vs_length'][1]}): r~length {f('r_vs_length')}  r~success {f('r_vs_success')}   |   "
              f"s~length {f('s_vs_length')}  s~success {f('s_vs_success')}"
              + (f"   | all-solved (n {e['allsolved_r_vs_length'][1]}): r~length {f('allsolved_r_vs_length')}  r~success {f('allsolved_r_vs_success')}"
                 if "allsolved_r_vs_length" in e else ""), flush=True)
    if allsolved.sum() >= 40:
        y_ = np.log(d["L"][ev][allsolved]); e_ = lp[ev][allsolved]
        r2a = [float(1 - ((y_[:, k] - e_[:, k] - (y_[:, k] - e_[:, k]).mean()) ** 2).sum() / ((y_[:, k] - y_[:, k].mean()) ** 2).sum()) for k in range(d["M"])]
        out["allsolved_r2_per_route_level_free"] = r2a
        print(f"  problems every reasoning route always solves: {allsolved.sum()}; predicted log length R2 within them (level-free) "
              + " ".join(f"{x:+.2f}" for x in r2a), flush=True)
    res[P_] = out
json.dump(res, open(OUT / f"residual_work_test{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
print("DONE", flush=True)

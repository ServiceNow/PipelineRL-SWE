"""Shared length alone, two ways, on one protocol (NEW_PATH 4.A.133): does pricing with only a query's predicted SHARED length keep
most of the cost probe's saving (paper appendix: 93% on LCB) or lose ~10 points (SCX-style row of tab:allest)?
Per training set (DRAW=s, the tab:allest protocol, billed costs, saved() at matched accuracy vs median pricing, same success probes):
  ours        per-candidate cost probe
  ours-level  our probe's predictions reduced to their shared part: mean over candidates of (log prediction - candidate's training
              mean log length), plus each candidate's training mean; rescaled to the training mean like every estimator
  scx         one ridge onto the candidate-centred mean log length (extra_estimators_draws.py), plus each candidate's training mean
  level-true  the actual shared part with training-mean offsets (an oracle reference)
Usage: REASON_ROOT=.../reason_pinned python level_vs_scx.py POOL [S0 S1] -> level_vs_scx_<pool>_pinned.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
POOL_ = sys.argv[1]; S0, S1 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 10)
sys.argv = [sys.argv[0], "levelscx", POOL_]
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])


def finish(d, yh):
    """log-length predictions [N, M] -> tokens, smeared and rescaled to each candidate's training mean (the shared post-processing)"""
    tr, Y = d["tr"], np.log(d["L"]); tk = np.zeros_like(d["L"])
    for m in range(d["M"]):
        sm = np.mean(np.exp(Y[tr, m] - yh[tr, m])); es, et = np.exp(yh[tr, m]) * sm, np.exp(yh[:, m]) * sm
        tk[:, m] = et * d["L"][tr, m].mean() / es.mean()
    return tk


def scx(d):
    tr = d["tr"]; Y = np.log(d["L"]); c = Y[tr].mean(0); ybar = (Y - c).mean(1)
    sc = StandardScaler().fit(d["X"][tr]); Xs, Xa = sc.transform(d["X"][tr]), sc.transform(d["X"])
    z = RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(Xs, ybar[tr]).predict(Xa)
    return finish(d, c[None] + z[:, None])


res = {"pool": POOL_, "seeds": {}}
for s in range(S0, S1):
    os.environ["DRAW"] = str(s); d = load(POOL_); N = len(d["ids"]); tr, ev = d["tr"], d["ev"]; Y = np.log(d["L"]); c = Y[tr].mean(0)
    ours = fit_predict("ours", view(d, tr), view(d, np.arange(N))); lo = np.log(np.maximum(ours, 1))
    lvl = (lo - c[None]).mean(1, keepdims=True)
    tk = {"ours": ours, "ours-level": finish(d, c[None] + lvl), "scx": scx(d), "level-true": finish(d, c[None] + (Y - c[None]).mean(1, keepdims=True))}
    H0 = curve(d, d["P"], cost_of(d, np.repeat(d["med"][None], N, 0)), ev)
    sv = {a: 100 * saved(curve(d, d["P"], cost_of(d, x), ev), H0) for a, x in tk.items()}
    r2l = lambda x: float(1 - ((Y[ev].mean(1) - np.log(np.maximum(x[ev], 1)).mean(1)) ** 2).sum() / ((Y[ev].mean(1) - Y[ev].mean()) ** 2).sum())
    res["seeds"][s] = dict(saved_vs_median=sv, level_r2={a: r2l(x) for a, x in tk.items()})
    print(f"set {s}: " + "  ".join(f"{a} {v:.1f}" for a, v in sv.items()) + "  | shared-length R2 " + " ".join(f"{a} {v:.2f}" for a, v in res["seeds"][s]["level_r2"].items()), flush=True)
A = {a: float(np.mean([r["saved_vs_median"][a] for r in res["seeds"].values()])) for a in tk}; res["mean"] = A
print("MEAN " + "  ".join(f"{a} {v:.1f}" for a, v in A.items()) + f"  | ours-level keeps {A['ours-level'] / A['ours'] * 100:.0f}%, scx {A['scx'] / A['ours'] * 100:.0f}%")
json.dump(res, open(Path(__file__).parent / f"level_vs_scx_{POOL_.lower().replace('-', '')}_pinned.json", "w"), indent=1)

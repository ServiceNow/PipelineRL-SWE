"""Cost-accuracy frontiers for Figure 2's left panel on any dataset of the cost_generalization pipeline (the tab:allest protocol, one
training set): our success probes shared by both arms, our cost probe vs median pricing, billed costs, the V sweep and its upper
concave hull, and connectors at three accuracies of the interior 90% of the shared band (as paper_nowai/make_billed_curves.py).
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned python frontier_curves.py APPS [POOL ...] -> frontier_curves_pinned.json (merged)
"""
import json, sys
from pathlib import Path
WANT = sys.argv[1:]
__file__ = str(Path(__file__).resolve()); _cg = Path(__file__).parent / "cost_generalization.py"; sys.argv = ["x", "labels", WANT[0]]
exec(open(_cg).read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
outf = Path(__file__).parent / "frontier_curves_pinned.json"; res = json.load(open(outf)) if outf.exists() else {}
for P in WANT:
    d = load(P); N = len(d["ids"]); ev = d["ev"]; rr = np.arange(len(ev))[None]
    tk = fit_predict("ours", view(d, d["tr"]), view(d, np.arange(N)))
    cost = {"learned": cost_of(d, tk), "median": cost_of(d, np.repeat(d["med"][None], N, 0))}
    out = {"n": int(len(ev)), "curves": {}}; H = {}
    for arm, c in cost.items():
        ch = (VALUES[:, None, None] * d["P"][ev][None] - c[ev][None]).argmax(2)
        acc, spend = d["q"][ev][rr, ch].mean(1), d["paid"][ev][rr, ch].mean(1); H[arm] = hull(list(zip(spend.tolist(), acc.tolist())))
        out["curves"][arm] = dict(accuracy=acc.tolist(), mean_cost_usd=spend.tolist(), hull_cost_usd=[h[0] for h in H[arm]], hull_accuracy=[h[1] for h in H[arm]])
    lo, hi = max(H["learned"][0][1], H["median"][0][1]), min(H["learned"][-1][1], H["median"][-1][1])
    band = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12); out["band"] = [float(lo), float(hi)]
    out["saved_band_mean"] = saved(H["learned"], H["median"])
    out["connectors"] = [dict(accuracy=float(x), ours_cost_usd=float(cost_at(H["learned"], x)), median_cost_usd=float(cost_at(H["median"], x)),
                              saved=float(1 - cost_at(H["learned"], x) / cost_at(H["median"], x))) for x in band[[1, 5, 10]]]
    res[P] = out
    print(f"{P}: n={len(ev)} band {lo:.3f}-{hi:.3f} saved {out['saved_band_mean']*100:.1f}%  connectors "
          + ", ".join(f"{c['accuracy']*100:.1f}%: {c['saved']*100:.0f}%" for c in out["connectors"]))
json.dump(res, open(outf, "w"), indent=1)
